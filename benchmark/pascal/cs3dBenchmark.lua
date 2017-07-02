require 'cunn'
require 'optim'
local matio = require 'matio'
local data = dofile('../benchmark/pascal/data.lua')

-----------------------------
--------parameters-----------
local params = {}
--params.bgVal = 0
params.evalSet = 'val'
params.suffix = ''
params.imgSizeY = 64
params.imgSizeX = 64
params.synset = 2958343 --chair:3001627, aero:2691156, car:2958343
params.flipPred = 0
params.gridSizeX = 32
params.gridSizeY = 32
params.gridSizeZ = 32
params.bs = 4
params.disp = 0

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end

if params.disp == 0 then params.disp = false end
if params.flipPred == 0 then params.flipPred = false end

params.synset = '0' .. tostring(params.synset) --to resolve string/number issues in passing bash arguments
params.imgSize = torch.Tensor({params.imgSizeX, params.imgSizeY})
params.gridSize = torch.Tensor({params.gridSizeX, params.gridSizeY, params.gridSizeZ})

local synsetToClass = {s_03001627 = 'chair', s_02691156 = 'aeroplane', s_02958343 = 'car'}
params.class = synsetToClass['s_' .. params.synset]
params.name = 'cs3d_' .. params.class
params.voxelsDir = '../cachedir/pascal/modelVoxels/' .. params.class .. '/'
params.predDir = '../cachedir/pascal/modelVoxelsCs3d/' .. params.class .. '/'

print(params)
-----------------------------
-----------------------------
params.saveDir = '../cachedir/resultsDir/pascal/' .. params.name .. '_' .. params.evalSet  ..params.suffix
paths.mkdir(params.saveDir)

local function BuildArray(...)
  local arr = {}
  for v in ... do
    arr[#arr + 1] = v
  end
  return arr
end

local pascalStatesDir = '../cachedir/pascal/camera/' .. params.class .. '/' .. params.evalSet .. '/'
local testModelsAll = BuildArray(paths.files(pascalStatesDir,'.mat'))
local testModels = {}
for ix = 1,#testModelsAll do
    if(paths.filep(paths.concat(params.predDir, testModelsAll[ix]))) then testModels[#testModels+1] = testModelsAll[ix] end
end

local dataLoader = data.dataLoader(pascalStatesDir, params.voxelsDir, params.bs, params.imgSize, params.gridSize, testModels)

local counter = 1

for iter = 1,torch.ceil(#testModels/params.bs) do
    print('iter : ' .. tostring(iter))
    local imgs, gtVols, modelNames = unpack(dataLoader:forward())
    for ix = 1, params.bs do
        local pred = matio.load(paths.concat(params.predDir, testModels[counter]), {'Volume'});
        if(pred) then
            pred = pred.Volume:typeAs(gtVols)
            matio.save(paths.concat(params.saveDir, tostring(counter) .. '.mat'),{gtName=testModels[counter], gtVol=gtVols[ix][1], volume=pred})
        end
        counter = (counter%#testModels) + 1
    end
end