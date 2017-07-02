require 'cunn'
require 'cudnn'
require 'optim'
local matio = require 'matio'
local data = dofile('../benchmark/pascal/data.lua')
local cropUtil = dofile('../utils/cropUtils.lua')
-----------------------------
--------parameters-----------
local params = {}
--params.bgVal = 0
params.name = 'name'
params.gpu = 1
params.suffix = ''
params.imgSizeY = 64
params.imgSizeX = 64
params.synset = 2958343 --chair:3001627, aero:2691156, car:2958343
params.flipPred = 1
params.gridSizeX = 32
params.gridSizeY = 32
params.gridSizeZ = 32
params.bs = 4
params.numTrainIter = 20000
params.evalSet = 'val'

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end

if params.flipPred == 0 then params.flipPred = false end

params.snapshotDir = '../cachedir/snapshots/pascal/' .. params.name
params.imgSize = torch.Tensor({params.imgSizeX, params.imgSizeY})
params.gridSize = torch.Tensor({params.gridSizeX, params.gridSizeY, params.gridSizeZ})
params.synset = '0' .. tostring(params.synset) --to resolve string/number issues in passing bash arguments
params.imgSize = torch.Tensor({params.imgSizeX, params.imgSizeY})
params.gridSize = torch.Tensor({params.gridSizeX, params.gridSizeY, params.gridSizeZ})
params.useTrain = (params.evalSet == 'train')

local synsetToClass = {s_03001627 = 'chair', s_02691156 = 'aeroplane', s_02958343 = 'car'}
params.class = synsetToClass['s_' .. params.synset]
params.voxelsDir = '../cachedir/pascal/modelVoxels/' .. params.class .. '/'
params.suffix = params.suffix .. '_' .. params.class  .. '_' .. params.evalSet
print(params)
-----------------------------
-----------------------------
params.saveDir = '../cachedir/resultsDir/pascal/' .. params.name .. '_'  .. tostring(params.numTrainIter) .. params.suffix

paths.mkdir(params.saveDir)
cutorch.setDevice(params.gpu)

local function BuildArray(...)
  local arr = {}
  for v in ... do
    arr[#arr + 1] = v
  end
  return arr
end

local predDirCs3dDir = '../cachedir/pascal/modelVoxelsCs3d/' .. params.class .. '/'
local testModels = {}
local pascalStatesDir

if(params.useTrain) then
    pascalStatesDir = '../cachedir/pascal/camera/' .. params.class .. '/train/'
    testModels = BuildArray(paths.files(pascalStatesDir,'.mat'))
else
    -- we'll only evaluate on instances where the 'Category-Specific Deformable Models' have available reconstructions
    pascalStatesDir = '../cachedir/pascal/camera/' .. params.class .. '/val/'
    testModelsAll = BuildArray(paths.files(pascalStatesDir,'.mat'))
    for ix = 1,#testModelsAll do
        if(paths.filep(paths.concat(predDirCs3dDir, testModelsAll[ix]))) then testModels[#testModels+1] = testModelsAll[ix] end
    end
end

local dataLoader = data.dataLoader(pascalStatesDir, params.voxelsDir, params.bs, params.imgSize, params.gridSize, testModels)

local predNet = torch.load(params.snapshotDir .. '/iter'.. params.numTrainIter .. '.t7')
predNet = predNet:cuda()
predNet:evaluate()

local counter = 1
if(params.disp) then disp = require 'display' end
for iter = 1,torch.ceil(#testModels/params.bs) do
    print('iter : ' .. tostring(iter))
    local imgs, gtVols = unpack(dataLoader:forward())
    imgs = cropUtil.resnetPreprocess(imgs)

    imgs = imgs:clone():cuda()
    local predOut = predNet:forward(imgs):clone():float()
    if(params.flipPred) then
        predOut = 1-predOut
    end
    for ix = 1, params.bs do
        matio.save(paths.concat(params.saveDir, tostring(counter) .. '.mat'),{gtName=testModels[counter], gtVol=gtVols[ix][1], volume=predOut[ix][1]})
        counter = (counter%#testModels) + 1
    end
end