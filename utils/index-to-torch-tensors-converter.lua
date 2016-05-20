require 'torch'
require 'nn'

cmd = torch.CmdLine()
cmd:option('-inFile', '', 'input file')
cmd:option('-len', '', 'uniform length for each sequence')
cmd:option('-outFile', '', 'out torch features file')
cmd:option('-isCapDataset', false, 'to form splitted dataset')
cmd:option('-mode', '', 'win or sent mode')
cmd:option('-batchedMode', false, 'save batched or plain version of sent dataset')
cmd:option('-comprenoInFile', '', 'compreno features input file')
cmd:option('-comprenoOutFile', '', 'out torch compreno features file')
local params = cmd:parse(arg)

print(params)

function string:split(inSplitPattern, outResults)
  if not outResults then
    outResults = {}
  end

  local theStart = 1
  local theSplitStart, theSplitEnd = string.find( self, inSplitPattern, theStart )
  while theSplitStart do
    table.insert( outResults, string.sub( self, theStart, theSplitStart-1 ) )
    theStart = theSplitEnd + 1
    theSplitStart, theSplitEnd = string.find( self, inSplitPattern, theStart )
  end
  table.insert( outResults, string.sub( self, theStart ) )
  return outResults
end

local function calc_num_lines(file)
  local numLines = 0

  for _ in io.lines(file) do
    numLines = numLines + 1
  end
  print(string.format('num input lines = %d',numLines))

  return numLines
end

local function get_data_and_labels_win(file, numLines, win_len)
  local labels = torch.Tensor(numLines)
  local features_count = win_len
  local data = torch.Tensor(numLines, features_count)

  local lineIdx = 0
  for line in io.lines(file) do
    lineIdx = lineIdx + 1
    local splitted = line:split('\t')
    local label = tonumber(splitted[1])
    local ids = splitted[2]:split(' ')
    local ids_size = table.getn(ids)
    labels[lineIdx] = label

    for i = 1, features_count do
      local id = tonumber(ids[i])
      data[lineIdx][i] = id
    end
  end

  return data, labels
end

local function get_data_for_compreno(file)
  print('Simple Compreno')
  local data = {}
  local lineIdx = 0

  for line in io.lines(file) do
    lineIdx = lineIdx + 1
    local splitted = line:split(" ")
    local splitted_size = table.getn(splitted)
    local splitted_tensor = torch.ones(1, splitted_size, 2)
    data[lineIdx] = splitted_tensor
    for j = 1, splitted_size do
      data[lineIdx][1][j][1] = tonumber(splitted[j])
      data[lineIdx][1][j][2] = 1
    end
  end

  return data
end

local function get_data_for_compreno_dense(file)
  print('Simple Dense Compreno')
  local data = torch.ByteTensor(176229, 83951)
  local lineIdx = 0

  for line in io.lines(file) do
    lineIdx = lineIdx + 1
    local splitted = line:split('\t')
    local label = splitted[1]
    local indexes = splitted[2]:split(' ')
    local indexes_size = table.getn(indexes)

    for i = 1, indexes_size do
      local index = tonumber(indexes[i])
      data[lineIdx][index] = 1
    end
  end

  return data
end

local function get_data_for_compreno_batched(file, batch_line_idxs)
  print('BATCHED Compreno')
  local lineIdx = 0
  local data_lines = {}
  for line in io.lines(file) do
    lineIdx = lineIdx + 1
    local splitted = line:split(" ")
    local splitted_size = table.getn(splitted)
    local splitted_tensor = torch.ones(splitted_size, 2)
    data_lines[lineIdx] = splitted_tensor
    for j = 1, splitted_size do
      data_lines[lineIdx][j][1] = tonumber(splitted[j])
      data_lines[lineIdx][j][2] = 1
    end
  end

  local data = {}

  for i = 1, #batch_line_idxs do
    local batch_lines = batch_line_idxs[i]
    for j = 1, batch_lines:size()[1] do
      line_idx = batch_lines[j]
      if data[i] == nil then
        data[i] = { data_lines[line_idx] }
      else
        table.insert(data[i], data_lines[line_idx])
      end
    end
  end

  return data
end

local function get_data_and_labels_for_sent(file, numLines, batchedMode, keep_features)
  local labels = torch.totable(torch.Tensor(numLines))
  local data = {}
  local grouped_data = {}
  local grouped_labels = {}
  local grouped_line_idxs = {}
  local batched_line_idxs = {}
  local sent_lens = {}
  local lineIdx = 0
  local parallel_features = 8
  local features_count = parallel_features

  local keep_features_size = nil
  if keep_features then
    keep_features_size = table.getn(keep_features)
    features_count = keep_features_size
  end

  for line in io.lines(file) do
    lineIdx = lineIdx + 1

    local ctr = 0
    local splitted = line:split("\t")
    local label_idx = splitted[1]
    labels[lineIdx] = label_idx
    splitted = splitted[2]:split(" ")

    local size = table.getn(splitted)
    local wordsCount = size / parallel_features

    data[lineIdx] = torch.Tensor(parallel_features, wordsCount)

    for j = 0, parallel_features - 1 do
      for k = 1, wordsCount do
        data[lineIdx][j + 1][k] = splitted[wordsCount * j + k]
      end
    end

    if keep_features then
      local keep_data = torch.Tensor(keep_features_size, wordsCount)
      for l = 1, keep_features_size do
        keep_data[l] = data[lineIdx][keep_features[l]]
      end
      data[lineIdx] = keep_data
    end

    if not batchedMode then
      data[lineIdx] = data[lineIdx]:resize(1, features_count, wordsCount)
      labels[lineIdx] = torch.Tensor(1)
      labels[lineIdx][1] = label_idx
    end

    -- Group data by sentence length
    local wordsCountWithoutPadding = wordsCount - 2
    if grouped_data[wordsCountWithoutPadding] == nil then
      grouped_data[wordsCountWithoutPadding] = {}
      grouped_line_idxs[wordsCountWithoutPadding] = {}
      grouped_labels[wordsCountWithoutPadding] = {}
      table.insert(grouped_data[wordsCountWithoutPadding], data[lineIdx])
      table.insert(grouped_labels[wordsCountWithoutPadding], labels[lineIdx])
      table.insert(grouped_line_idxs[wordsCountWithoutPadding], lineIdx)
      sent_lens[wordsCountWithoutPadding] = 1
    else
      table.insert(grouped_data[wordsCountWithoutPadding], data[lineIdx])
      table.insert(grouped_labels[wordsCountWithoutPadding], labels[lineIdx])
      table.insert(grouped_line_idxs[wordsCountWithoutPadding], lineIdx)
      sent_lens[wordsCountWithoutPadding] = sent_lens[wordsCountWithoutPadding] + 1
    end
  end

  -- Make batches from grouped data
  if batchedMode then
    local batch_size = 33
    local batched_data = {}
    local batched_labels = {}
    batched_line_idxs = {}
    for sent_len, total_count in pairs(sent_lens) do
      for i = 1, total_count, batch_size do
        local batch_data = torch.Tensor(batch_size, features_count, sent_len + 2)
        local batch_labels = torch.Tensor(batch_size)
        local batch_line_idxs = torch.Tensor(batch_size)
        for j = 1, batch_size do
          if (i + j - 1) > total_count then
            batch_data = batch_data:resize(j - 1, features_count, sent_len + 2)
            batch_labels = batch_labels:resize(j - 1)
            batch_line_idxs = batch_line_idxs:resize(j - 1)
            break
          end
          batch_data[j] = grouped_data[sent_len][i + j - 1]
          batch_labels[j] = grouped_labels[sent_len][i + j - 1]
          batch_line_idxs[j] = grouped_line_idxs[sent_len][i + j - 1]
        end
        table.insert(batched_data, batch_data)
        table.insert(batched_labels, batch_labels)
        table.insert(batched_line_idxs, batch_line_idxs)
      end
    end

    local total_check_size = 0
    for sent_len, total_count in pairs(sent_lens) do
      total_check_size = total_check_size + total_count
    end

    local total_batched_data_size = 0
    for batch_num, tensor in pairs(batched_data) do
      total_batched_data_size = total_batched_data_size + tensor:size()[1]
    end
    print(total_batched_data_size, total_check_size, table.getn(batched_labels))

    data = batched_data
    labels = batched_labels
  end

  return data, labels, batched_line_idxs
end

local function run_win(params)
  local numLines = calc_num_lines(params.inFile)
  local data, labels = get_data_and_labels_win(params.inFile, numLines, params.len)

  if params.isCapDataset then
    data:resize(numLines, 2, 5)
  end

  local stuff = {
    labels = labels,
    data = data
  }

  torch.save(params.outFile, stuff)
end

local function run_sent_one(params, keep_features)
  local numLines = calc_num_lines(params.inFile)
  local data, labels, line_idxs = get_data_and_labels_for_sent(params.inFile, numLines, params.batchedMode, keep_features)

  local stuff = {
    labels = labels,
    data = data
  }

  torch.save(params.outFile, stuff)

  return line_idxs, labels
end

local function run_compreno_processing(params, line_idxs, labels)
  local data = nil
  if table.getn(line_idxs) ~= 0 then
    data = get_data_for_compreno_batched(params.comprenoInFile, line_idxs)
  else
    data = get_data_for_compreno(params.comprenoInFile)
  end

  local stuff = {
    labels = labels,
    data = data
  }

  torch.save(params.comprenoOutFile, stuff)
end

if params.mode == 'win' then
  run_win(params)
elseif params.mode == 'sentence-convolution' then
  line_idxs, labels = run_sent_one(params)
elseif params.mode == 'without-compreno' then
  keep_features = {1, 2, 3, 4, 5, 6, 7}
  line_idxs, labels = run_sent_one(params, keep_features)
elseif params.mode == 'experiment-only-compreno' then
  keep_features = {2, 3, 8}
  line_idxs, labels = run_sent_one(params, keep_features)
elseif params.mode == 'dense-autoencoder-compreno' then
  data = get_data_for_compreno_dense(params.inFile)
  torch.save(params.outFile, data)
elseif params.mode == 'comp_cap_pos_gaz_model' then
  keep_features = {2, 3, 4, 5, 6, 7, 8}
  line_idxs, labels = run_sent_one(params, keep_features)
elseif params.mode == 'compreno-pos' then
  keep_features = {3, 8}
  line_idxs, labels = run_sent_one(params, keep_features)
end

if params.comprenoInFile ~= '' then
  run_compreno_processing(params, line_idxs, labels)
end
