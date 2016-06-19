require 'torch'
require 'nn'
require 'conll_utils'
local nninit = require 'nninit'

module('compreno_model', package.seeall)

function make_net(embeddings_path, compreno_svd_path)
  local net = nn.Sequential()

  local hidden_units = 256
  local num_classes = 17
  net:add(nn.SparseLinear(83951, hidden_units))
  net:add(nn.Dropout())
  net:add(nn.HardTanh())
  net:add(nn.Linear(hidden_units, hidden_units):init('weight', nninit.xavier))
  net:add(nn.Dropout())
  net:add(nn.HardTanh())
  net:add(nn.Linear(hidden_units, num_classes):init('weight', nninit.xavier))

  local criterion = nn.CrossEntropyCriterion()
  criterion = conll_utils.to_cuda(criterion)

  print(net)

  return net, criterion
end
