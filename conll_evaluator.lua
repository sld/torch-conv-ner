require 'torch'
require 'nn'

cuda = false
mode = 'sent'

module('conll_evaluator', package.seeall)

local function to_cuda(x) return cuda and x:cuda() or x end

local function predict(net, criterion, samples, labels)
  local pred = net:forward(samples)
  local err = criterion:forward(pred, labels)
  local inds, _ = nil
  _, inds = torch.max(pred, 2)
  return inds, err
end

local function make_label_index_str_map(labelMapFile)
  local label_index_str_map = {}
  for line in io.lines(labelMapFile) do
    local label_string, label_index = string.match(line, "([^\t]+)\t([^\t]+)")
    label_index_str_map[tonumber(label_index)] = label_string
  end

  return label_index_str_map
end

local function write_predicts_to_file(predicts, predicts_file, label_index_str_map)
  local i = 1
  local predict_id = nil
  for i = 1, predicts:size()[1] do
    predict_id = predicts[i][1]
    local predicted_label = label_index_str_map[predict_id]
    predicts_file:write(predicted_label..'\n')
  end
end

local function make_predicts_file_and_get_loss(torchTestFilename, label_index_str_map, net, torchComprenoDataFilename)
  local test = torch.load(torchTestFilename)

  local compreno_data = nil
  if torchComprenoDataFilename then
    compreno_data = torch.load(torchComprenoDataFilename)
  end

  local predicts_filename = torchTestFilename..'-predicts'
  local i = 1
  local predicts_file = io.open(predicts_filename, 'w')
  local criterion = nn.CrossEntropyCriterion()
  to_cuda(criterion)

  local test_set_size = nil
  local batch_size = nil
  if mode == 'sent' or mode == 'compreno' then
    test_set_size = table.getn(test.labels)
    batch_size = 1
  elseif mode == 'win' then
    test_set_size = test.labels:size()[1]
    batch_size = 64
  end
  local batches_count = math.floor(test_set_size / batch_size)

  local total_err = 0
  local subset_data = nil
  local subset_test_data = nil
  local subset_compreno_data = nil
  local total_time = sys.clock()
  for i = 1, test_set_size, batch_size do
    local start = i
    local finish = i + batch_size - 1
    local per_batch_time = sys.clock()
    if finish > test_set_size then finish = test_set_size end
    local subset_labels = nil

    if mode == 'win' then
      subset_data = to_cuda(test.data[{{start, finish}}])
      subset_labels = to_cuda(test.labels[{{start, finish}}])
    elseif mode == 'sent' then
      assert(batch_size == 1, 'Batch size should equal to 1 for sent mode')
      subset_test_data = to_cuda(test.data[i])
      subset_data = subset_test_data
      subset_labels = to_cuda(test.labels[i])
    elseif mode == 'compreno' then
      assert(batch_size == 1, 'Batch size should equal to 1 for sent mode')
      subset_test_data = to_cuda(test.data[i])
      subset_compreno_data = compreno_data.data[i]
      subset_data = {subset_test_data, subset_compreno_data}
      subset_labels = to_cuda(test.labels[i])
    end

    local predicts, err = predict(net, criterion, subset_data, subset_labels)
    total_err = total_err + err
    write_predicts_to_file(predicts, predicts_file, label_index_str_map)

    per_batch_time = sys.clock() - per_batch_time
    io.write(string.format('\r%.3f percent complete; %.3f sec per batch', i/test_set_size, per_batch_time))
    io.flush()
  end
  total_err = total_err / batches_count
  total_time = sys.clock() - total_time
  print("Eval set loss: "..total_err)
  print("Time: "..total_time)
  predicts_file:close()
  return predicts_filename, total_err
end

local function trim(s)
  return s:match "^%s*(.-)%s*$"
end

local function run_cmd(cmd)
  local handle = io.popen(cmd)
  local result = handle:read("*a")
  handle:close()
  return trim(result)
end

local function make_conll_file_from_predicts(predicts_filename, original_test_filename)
  local cmd = 'python utils/torch-to-conll-converter.py '..predicts_filename..' '..original_test_filename
  return run_cmd(cmd)
end

local function get_metrics(conll_file)
  local result = run_cmd('cat '..conll_file..' | bin/conlleval')
  print(result)

  local i = 1
  local metrics = {}
  for line in string.gmatch(result, '[^\r\n]+') do
    if i == 2 then
      local pattern = '(%d+.%d+)'
      for metr in string.gmatch(line, pattern) do
        table.insert(metrics, metr)
      end
      break
    end
    i = i + 1
  end
  pr = tonumber(metrics[2])
  rec = tonumber(metrics[3])
  f1 = tonumber(metrics[4])
  return {Pr=pr, Rec=rec, FB1=f1}
end

function evaluate(filename, label_map_file, original_test_filename, net, compreno_data_filename)
  net:evaluate()
  local label_index_str_map = make_label_index_str_map(label_map_file)
  local predicts_filename, loss = make_predicts_file_and_get_loss(filename, label_index_str_map, net, compreno_data_filename)
  local conll_file = make_conll_file_from_predicts(predicts_filename, original_test_filename)
  local metrics = get_metrics(conll_file)
  metrics['Loss'] = loss
  net:training()
  return metrics
end
