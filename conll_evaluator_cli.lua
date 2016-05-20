require 'torch'
require 'nn'
require 'conll_evaluator'

local cmd = torch.CmdLine()
cmd:option('-cuda', false, 'whether to use gpu')
cmd:option('-test', 'data/conll2003/eng.testa.torch','torch format test file list')
cmd:option('-model', '', 'file with trained model')
cmd:option('-mode', 'sent', 'sent|win mode')
cmd:option('-labelMap', 'data/conll2003/label-map.index','file containing map from label strings to index. needed for entity level evaluation')
cmd:option('-originalTestFilename', '', 'original conll test filename')
cmd:option('-comprenoDataFilename', '', 'compreno torch data filename')


local params = cmd:parse(arg)

if(params.cuda) then
  require 'cunn'
  print('using GPU')
end

local comprenoDataFilename = nil
if params.comprenoDataFilename ~= '' then
  comprenoDataFilename = params.comprenoDataFilename
end

local net = torch.load(params.model)

conll_evaluator.cuda = params.cuda
conll_evaluator.mode = params.mode
conll_evaluator.evaluate(params.test, params.labelMap, params.originalTestFilename, net, comprenoDataFilename)
