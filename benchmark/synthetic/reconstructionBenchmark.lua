require 'cunn'
require 'optim'
local matio = require 'matio'
local data = dofile('../benchmark/synthetic/data.lua')

-----------------------------
--------parameters-----------
local params = {}
--params.bgVal = 0
params.name = 'name'
params.gpu = 3
params.imgSizeY = 64
params.imgSizeX = 64
params.synset = 2958343 --chair:3001627, aero:2691156, car:2958343
params.flipPred = 0
params.gridSizeX = 32
params.gridSizeY = 32
params.gridSizeZ = 32
params.nImgs = 2
params.disp = 0
params.evalSet = 'val'
params.numTrainIter = 10000

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end

if params.disp == 0 then params.disp = false end
if params.flipPred == 0 then params.flipPred = false end

params.imgSize = torch.Tensor({params.imgSizeX, params.imgSizeY})
params.gridSize = torch.Tensor({params.gridSizeX, params.gridSizeY, params.gridSizeZ})
params.synset = '0' .. tostring(params.synset) --to resolve string/number issues in passing bash arguments
params.snapshotDir = '../cachedir/snapshots/shapenet/' .. params.name
params.modelsDataDir = '../cachedir/blenderRenderPreprocess/' .. params.synset .. '/'
params.voxelsDir = '../cachedir/shapenet/modelVoxels/' .. params.synset .. '/'
params.imgSize = torch.Tensor({params.imgSizeX, params.imgSizeY})
params.gridSize = torch.Tensor({params.gridSizeX, params.gridSizeY, params.gridSizeZ})
print(params)
-----------------------------
-----------------------------
params.saveDir = '../cachedir/resultsDir/shapenet/' .. params.name .. '_' .. tostring(params.numTrainIter) .. '_' .. params.evalSet

paths.mkdir(params.saveDir)
cutorch.setDevice(params.gpu)

local splitUtil = dofile('../benchmark/synthetic/splits.lua')
local testModels = splitUtil.getSplit(params.synset)[params.evalSet]
local dataLoader = data.dataLoader(params.modelsDataDir, params.voxelsDir, params.nImgs, params.imgSize, params.gridSize, testModels)

local predNet = torch.load(params.snapshotDir .. '/iter'.. params.numTrainIter .. '.t7')
predNet = predNet:cuda()
predNet:evaluate()

local counter = 1
for modelId =1,#testModels do
    print('modelId : ' .. tostring(modelId))
    local imgs, gtVol = dataLoader:forward()
    imgs = imgs:clone():cuda()
    local predOut = predNet:forward(imgs):clone():float()
    if(params.flipPred) then
        predOut = 1-predOut
    end
    for ix = 1, params.nImgs do
        matio.save(paths.concat(params.saveDir, tostring(counter) .. '.mat'),{gtName=testModels[modelId], gtVol=gtVol,volume=predOut[ix][1],img=dataLoader.imgsOriginal[ix]:clone():double()})
        counter = counter + 1
    end
end