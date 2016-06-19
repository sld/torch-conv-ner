require 'torch'
require 'nn'
require 'conll_utils'
require 'models/convolution_model'
require 'models/compreno_model'
local nninit = require 'nninit'

module('convolution_with_compreno_model', package.seeall)

function make_net(embeddings_path)
  local raw_compreno_sparse_net, _ = compreno_model.make_net()
  local raw_conv_net, _ = convolution_model.make_net(embeddings_path)

  local hidden_units = 300
  local net_conv = nn.Sequential()

  for i = 1, 8 do net_conv:add(raw_conv_net:get(i)) end

  net_conv = conll_utils.to_cuda(net_conv)

  local compreno_sparse_net = nn.Sequential()
  for i = 1, 6 do compreno_sparse_net:add(raw_compreno_sparse_net:get(i)) end
  if conll_utils.cuda then
    compreno_sparse_net:add(nn.Copy('torch.DoubleTensor', 'torch.CudaTensor'))
  end

  local parallel_net = nn.ParallelTable()
  parallel_net:add(net_conv)
  parallel_net:add(compreno_sparse_net)

  local net = nn.Sequential()
  net:add(parallel_net)
  net:add(conll_utils.to_cuda(nn.JoinTable(2)))
  local sparse_output = 256
  local conv_output = 300
  local joined_hidden_units = sparse_output + conv_output
  net:add(conll_utils.to_cuda(nn.Linear(joined_hidden_units, hidden_units):init('weight', nninit.uniform, -1.0 / math.sqrt(hidden_units), 1.0 / math.sqrt(hidden_units))))
  net:add(conll_utils.to_cuda(nn.Dropout()))
  net:add(conll_utils.to_cuda(nn.HardTanh()))
  net:add(conll_utils.to_cuda(raw_conv_net:get(10)))

  local criterion = nn.CrossEntropyCriterion()
  criterion = conll_utils.to_cuda(criterion)

  print(net)

  return net, criterion
end

function build_net_by_using_old_net(old_conv_net, old_compreno_sparse_net)
  local hidden_units = 300
  local net_conv = nn.Sequential()

  for i = 1, 8 do net_conv:add(old_conv_net:get(i)) end

  net_conv = conll_utils.to_cuda(net_conv)

  local compreno_sparse_net = nn.Sequential()
  for i = 1, 6 do compreno_sparse_net:add(old_compreno_sparse_net:get(i)) end
  if conll_utils.cuda then
    compreno_sparse_net:add(nn.Copy('torch.DoubleTensor', 'torch.CudaTensor'))
  end

  local parallel_net = nn.ParallelTable()
  parallel_net:add(net_conv)
  parallel_net:add(compreno_sparse_net)

  local net = nn.Sequential()
  net:add(parallel_net)
  net:add(conll_utils.to_cuda(nn.JoinTable(2)))
  local sparse_output = 256
  local conv_output = 300
  local joined_hidden_units = sparse_output + conv_output
  net:add(conll_utils.to_cuda(nn.Linear(joined_hidden_units, hidden_units):init('weight', nninit.uniform, -1.0 / math.sqrt(hidden_units), 1.0 / math.sqrt(hidden_units))))
  net:add(conll_utils.to_cuda(nn.Dropout()))
  net:add(conll_utils.to_cuda(nn.HardTanh()))
  net:add(conll_utils.to_cuda(old_conv_net:get(10)))

  print('Built from old net', net)

  return net
end
