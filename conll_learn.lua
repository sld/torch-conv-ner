require 'torch'
require 'nn'
require 'conll_evaluator'
require 'optim'
require 'conll_utils'
require 'models/convolution_model'
local nninit = require 'nninit'

cuda = false
mode = 'sent'

module('conll_learn', package.seeall)

-- net_info = {net=xx, criterion=xx, save_prefix=xx}
-- train_info = {train_set=xx, batch_size=xx, epochs=xx}
-- test_info = {torch_test_filename=xx, label_map_file=xx, original_test_filename=xx, evaluate_freq=xx}
local function train_model(net_info, train_info, test_info, with_separate_lr)
  --- split training data into batches
  local data_batches = {}
  local label_batches = {}

  local batch_size = train_info['batch_size']
  local train_compreno_data = nil
  local train_set_size = nil

  if mode == 'sent' then
    train_set_size = table.getn(train_info['train_set'].labels)
  elseif mode == 'win' then
    train_set_size = train_info['train_set'].labels:size()[1]
  elseif mode == 'compreno' then
    train_set_size = table.getn(train_info['train_set'].labels)
    train_compreno_data = conll_utils.deepcopy(train_info['compreno_set'].data)
  end
  -- train_set_size = 100

  local batches_count = math.floor(train_set_size / batch_size)

  local net = net_info['net']
  local criterion = net_info['criterion']

  local train_data = nil
  train_data = conll_utils.deepcopy(train_info['train_set'].data)

  local train_labels = conll_utils.deepcopy(train_info['train_set'].labels)
  local rand_inds = torch.randperm(train_set_size)
  local best_f1 = 0
  local best_f1_epoch = 1

  local function on_train(epoch, currentError, perEpochTime)
    local fb1 = 0
    local file = io.open('snapshots/torch-log-v'..net_info['version'], 'a')
    local log_str = 'Epoch: '..epoch..'\tTrainLoss: '..currentError..'\tEpochTime: '..perEpochTime..'\n'
    file:write(log_str)
    print(log_str)

    if (epoch == 1 or epoch % test_info['evaluate_freq'] == 0 or epoch == train_info['epochs']) then
      local metrics = conll_evaluator.evaluate(test_info['torch_test_filename'],
      test_info['label_map_file'], test_info['original_test_filename'], net:clone(), test_info['compreno_set_filename'])
      fb1 = metrics['FB1']
      log_str = 'Epoch: '..epoch..'\tTestLoss: '..metrics['Loss']..'\tTestF1: '..fb1..'\tTestPr: '..metrics['Pr']..'\tTestRec: '..metrics['Rec']..'\n'
      file:write(log_str)
      print(log_str)
    end
    if fb1 > best_f1 then
      best_f1 = fb1
      best_f1_epoch = epoch
      conll_utils.save_model(net, net_info['save_prefix'], epoch)
    end
    log_str = "BestF1Epoch: "..best_f1_epoch..'\n'
    print(log_str)
    file:write(log_str)
    file:close()
  end

  local config = { learningRate = 0.01 }

  if with_separate_lr then
    local params_lr_m = net:clone()
    local params_lr = params_lr_m:getParameters()
    params_lr:fill(1)
    local gaz_size = 15
    local conv_fan_in = 3 * (50 + 5 + 30 + 4 * gaz_size)
    params_lr_m:get(4).weight:fill(1 / (conv_fan_in))
    params_lr_m:get(7).weight:fill(1 / 300)
    params_lr_m:get(10).weight:fill(1 / 300)
    config['learningRates'] = params_lr
  end

  -- local weights, dw = net:getParameters()
  -- local lrs = conll_utils.to_cuda(torch.Tensor():typeAs(weights):resizeAs(weights))

  local layerOptimConfig = {}
  local layerParameters = {}
  local layerGradParameters = {}
  local layer = nil

  if mode == 'compreno' then
    local parallelLayer = net:get(1)
    local parameters, gradParameters = parallelLayer:get(1):getParameters()
    table.insert(layerParameters, parameters)
    table.insert(layerGradParameters, gradParameters)
    table.insert(layerOptimConfig, conll_utils.copytable(config))

    parameters, gradParameters = parallelLayer:get(2):getParameters()
    table.insert(layerParameters, parameters)
    table.insert(layerGradParameters, gradParameters)
    table.insert(layerOptimConfig, conll_utils.copytable(config))
    for i = 2, net:size() do
      layer = net:get(i)
      parameters, gradParameters = layer:getParameters()
      table.insert(layerParameters, parameters)
      table.insert(layerGradParameters, gradParameters)
      table.insert(layerOptimConfig, conll_utils.copytable(config))
    end
  else
    for i = 1, net:size() do
      layer = net:get(i)
      local parameters, gradParameters = layer:getParameters()
      table.insert(layerParameters, parameters)
      table.insert(layerGradParameters, gradParameters)
      table.insert(layerOptimConfig, conll_utils.copytable(config))
    end
  end

  for epoch = 1, train_info['epochs'] do
    local per_epoch_time = sys.clock()

    local epoch_err = 0

    rand_inds = torch.randperm(train_set_size)
    for i = 1, train_set_size do
      train_data[i] = conll_utils.to_cuda(train_info['train_set'].data[rand_inds[i]])
      if mode == 'compreno' then
        train_compreno_data[i] = train_info['compreno_set'].data[rand_inds[i]]
      end
      train_labels[i] = conll_utils.to_cuda(train_info['train_set'].labels[rand_inds[i]])
    end

    local subset_data = nil
    local subset_labels = nil
    local per_batch_time = nil

    for i = 1, train_set_size, batch_size do
      for j = 1, #layerGradParameters do
        layerGradParameters[j]:zero()
      end

      per_batch_time = sys.clock()
      if mode == 'sent' then
        subset_labels = train_labels[i]
        subset_data = train_data[i]
      elseif mode == 'win' then
        local start = i
        local finish = i + batch_size - 1
        if finish > train_set_size then finish = train_set_size end
        subset_labels = conll_utils.to_cuda(train_labels[{{start, finish}}])
        subset_data = conll_utils.to_cuda(train_data[{{start, finish}}])
      elseif mode == 'compreno' then
        subset_labels = train_labels[i]
        subset_data = {train_data[i], train_compreno_data[i]}
      end
      local output = net:forward(subset_data)

      local err = criterion:forward(output, subset_labels)

      epoch_err = epoch_err + err
      local df_do = criterion:backward(output, subset_labels)
      net:backward(subset_data, df_do)

      for i = 1, #layerParameters do
        -- If this layer has parameters to be optimized
        if layerParameters[i]:nDimension() ~= 0 then
          -- Perform optimization
          local feval = function(x)
            return _, layerGradParameters[i]
          end
          optim.sgd(feval, layerParameters[i], layerOptimConfig[i])
        end
      end

      per_batch_time = sys.clock() - per_batch_time
      io.write(string.format('\r%.3f percent complete; %.3f sec per batch; %.3f batch err;', i/train_set_size, per_batch_time, err))
      io.flush()
    end
    io.write('\n')
    per_epoch_time = sys.clock() - per_epoch_time
    epoch_err = epoch_err / batches_count
    on_train(epoch, epoch_err, per_epoch_time)
  end
end

local function make_net(embeddings_path, compreno_svd_path, model_name_raw)
  local model_name = model_name_raw:gsub('-subset', '')
  model_name = model_name:gsub('-full', '')
  if model_name == 'convolution' then
    return convolution_model.make_net(embeddings_path)
  end
end

local function build_net_by_using_old_net(net, compreno_svd_path, model_name_raw)
  local model_name = model_name_raw:gsub('-subset', '')
  model_name = model_name:gsub('-full', '')
  if model_name == 'convolution' then
    error('build_net_by_using_old_net is not implemented for convolution model')
  end
end

local function get_learning_data(model_name)
  local train_set_files = nil
  local torch_test_filename = nil
  local original_test_filename = nil

  train_set_files = { 'data/conll2003/eng.train.iobes.torch' }
  torch_test_filename = 'data/conll2003/eng.testb.test.iobes.torch'
  original_test_filename = 'data/conll2003/eng.testb.test.iobes'

  return torch_test_filename, original_test_filename, train_set_files
end

local function run()
  local seed = 2016
  torch.manualSeed(seed)

  local cmd = torch.CmdLine()
  cmd:option('-cuda', false, 'whether to use gpu')
  cmd:option('-dump', '', 'load from dump net')
  cmd:option('-batch_size', 1, 'batch size')
  cmd:option('-version', '21', 'version is used to identify net and log filenames')
  cmd:option('-mode', 'sent', 'win|sent mode')
  cmd:option('-model_name', '', 'experiment-only-compreno|sparse-compreno|without-compreno|compreno-svd-win|compreno-two-nets|deafult model loading')
  cmd:option('-with_separate_lr', false, 'with separate lr for each layer')
  cmd:option('-useOldNetParams', false, 'buld new net by using old net weights')
  local params = cmd:parse(arg)
  print(params)
  mode = params.mode
  cuda = params.cuda
  conll_evaluator.cuda = cuda
  conll_utils.cuda = cuda
  conll_evaluator.mode = mode
  if(cuda) then
    require 'cunn'
    print('using GPU')
  end

  local embeddings_path = 'data/embeddings/senna.torch'
  local compreno_svd_path = 'data/conll-abbyy/conll/compreno-vectors.torch'
  local version = params.version
  local save_prefix = 'snapshots/smotri-net-v'..version
  local net, criterion = nil

  net, criterion = make_net(embeddings_path, compreno_svd_path, params.model_name)

  if params.dump ~= '' then
    net = torch.load(params.dump)
  end

  if params.useOldNetParams then
    net = build_net_by_using_old_net(net, compreno_svd_path, params.model_name)
  end

  net:training()
  local net_info = { net=net, criterion=criterion, save_prefix=save_prefix, version=version }

  local torch_test_filename, original_test_filename, train_set_files = get_learning_data(params.model_name)

  local batch_size = params.batch_size
  local epochs = 1000
  local train_set = conll_utils.get_set_from_files(train_set_files)

  local compreno_set_file = nil
  local compreno_set = nil
  local torch_compreno_test_filename = nil
  if mode == 'compreno' then
    compreno_set_files = {'data/conll2003/eng.train.compreno.torch', 'data/conll2003/eng.testa.dev.compreno.torch'}
    compreno_set = conll_utils.get_set_from_files(compreno_set_files)
    torch_compreno_test_filename = 'data/conll2003/eng.testb.test.compreno.torch'
  end

  local train_info = { train_set=train_set, batch_size=batch_size,
    epochs=epochs, compreno_set=compreno_set }

  local log_file = io.open('snapshots/torch-log-v'..net_info['version'], 'a')
  log_file:write(cmd:string('', params, {})..'\n')
  log_file:write(net:__tostring__()..'\n')
  log_file:close()

  local label_map_file = 'data/conll2003/label-map.index'
  local evaluate_freq = 2
  local test_info = { torch_test_filename=torch_test_filename,
    label_map_file=label_map_file, original_test_filename=original_test_filename,
    evaluate_freq=evaluate_freq, test_set=torch.load(torch_test_filename),
    compreno_set_filename=torch_compreno_test_filename }
  train_model(net_info, train_info, test_info, with_separate_lr)
end

run()
