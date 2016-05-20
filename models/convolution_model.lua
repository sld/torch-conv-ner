require 'torch'
require 'nn'
require 'conll_utils'
local nninit = require 'nninit'

module('convolution_model', package.seeall)

function make_net(embeddings_path)
  local hidden_units = 300
  local num_classes = 17

  local data = torch.load(embeddings_path)
  local vocab_size = data.data:size(1)
  local embedding_dim = data.data:size(2)
  local emb_lookup_table = nn.LookupTable(vocab_size, embedding_dim)
  emb_lookup_table.weight = data.data

  local cap_lookup_table = nn.LookupTable(5, 5):init('weight', nninit.uniform, -1.0, 1.0)
  local pos_lookup_table = nn.LookupTable(250, 30):init('weight', nninit.uniform, -1.0, 1.0)
  local gazetteer_loc_lookup_table = nn.LookupTable(10, 15):init('weight', nninit.uniform, -1.0, 1.0)
  local gazetteer_misc_lookup_table = nn.LookupTable(10, 15):init('weight', nninit.uniform, -1.0, 1.0)
  local gazetteer_org_lookup_table = nn.LookupTable(10, 15):init('weight', nninit.uniform, -1.0, 1.0)
  local gazetteer_per_lookup_table = nn.LookupTable(10, 15):init('weight', nninit.uniform, -1.0, 1.0)

  local gaz_size = 15
  local total_vec_size = embedding_dim + 5 + 30 + 4 * gaz_size

  local parallel_lookup = nn.ParallelTable()
  parallel_lookup:add(emb_lookup_table)
  parallel_lookup:add(cap_lookup_table)
  parallel_lookup:add(pos_lookup_table)
  parallel_lookup:add(gazetteer_loc_lookup_table)
  parallel_lookup:add(gazetteer_misc_lookup_table)
  parallel_lookup:add(gazetteer_org_lookup_table)
  parallel_lookup:add(gazetteer_per_lookup_table)

  local sp = nn.SplitTable(1, 2)
  sp.updateGradInput = function() end

  local net_conv = nn.Sequential()
  net_conv:add(sp)
  net_conv:add(parallel_lookup)
  net_conv:add(nn.JoinTable(3))
  local conv_fan_in = 3 * total_vec_size
  net_conv:add(nn.TemporalConvolution(total_vec_size, hidden_units, 3, 1):init('weight', nninit.uniform, -1.0 / math.sqrt(conv_fan_in), 1.0 / math.sqrt(conv_fan_in)))
  net_conv:add(nn.Max(2))
  net_conv:add(nn.Dropout())
  net_conv:add(nn.Linear(hidden_units, hidden_units):init('weight', nninit.uniform, -1.0 / math.sqrt(hidden_units), 1.0 / math.sqrt(hidden_units)))
  net_conv:add(nn.Dropout())
  net_conv:add(nn.HardTanh())
  net_conv:add(nn.Linear(hidden_units, num_classes):init('weight', nninit.uniform, -1.0 / math.sqrt(hidden_units), 1.0 / math.sqrt(hidden_units)))

  local criterion = nn.CrossEntropyCriterion()
  criterion = conll_utils.to_cuda(criterion)

  net_conv = conll_utils.to_cuda(net_conv)

  print(net_conv)

  return net_conv, criterion
end
