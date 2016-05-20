require 'torch'
module('conll_utils', package.seeall)

cuda = false

function save_model(net, prefix, epoch)
  local save_model_name = prefix.."-best-f1"..".torch"
  torch.save(save_model_name, net)
end

function shallowcopy(orig)
  local orig_type = type(orig)
  local copy
  if orig_type == 'table' then
    copy = {}
    for orig_key, orig_value in pairs(orig) do
        copy[orig_key] = orig_value
    end
  else -- number, string, boolean, etc
    copy = orig
  end
  return copy
end

function deepcopy(orig)
  local orig_type = type(orig)
  local copy
  if orig_type == 'table' then
    copy = {}
    for orig_key, orig_value in next, orig, nil do
      copy[deepcopy(orig_key)] = deepcopy(orig_value)
    end
    setmetatable(copy, deepcopy(getmetatable(orig)))
  elseif orig_type == 'userdata' then
    copy = orig:clone()
  else -- number, string, boolean, etc
    copy = orig
  end
  return copy
end

function subrange(t, first, last)
  local sub = {}
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end

function copytable(mytable)
  newtable = {}
  for k, v in pairs(mytable) do
    newtable[k] = v
  end
  return newtable
end

function get_set_from_files(files)
  local set = nil
  for _, file in pairs(files) do
    if set == nil then
      set = torch.load(file)
    else
      local new_set = torch.load(file)
      for _, v in ipairs(new_set.data) do
        table.insert(set.data, v)
      end
      for _, v in ipairs(new_set.labels) do
        table.insert(set.labels, v)
      end
    end
  end

  print('union set size is ', table.getn(set), table.getn(set.data), table.getn(set.labels))

  return set
end

function to_cuda(x) return cuda and x:cuda() or x end
