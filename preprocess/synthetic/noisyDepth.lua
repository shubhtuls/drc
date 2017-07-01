-- example usage: synset=3001627 th synthetic/noisyDepth.lua
local params = {}
params.synset = 2958343 --chair:3001627, aero:2691156, car:2958343
params.nImgs = 10
params.noise = 0.2
for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end

local maxSavedDepth = 10
local nImgs = params.nImgs
params.synset = '0' .. tostring(params.synset) --to resolve string/number issues in passing bash arguments
params.modelsDataDir = '../cachedir/blenderRenderPreprocess/' .. params.synset .. '/'
local noiseRange = params.noise / maxSavedDepth --dividing because images are saved with values in [0,1] instead of [0,maxDepth]

require 'image'
local maxSavedDepth = 10
local function BuildArray(...)
  local arr = {}
  for v in ... do
    arr[#arr + 1] = v
  end
  return arr
end
-------------------------------
-------------------------------
local modelNames = BuildArray(paths.files(params.modelsDataDir,'...'))
for mId = 1,#modelNames do
    local modelDir = paths.concat(params.modelsDataDir, modelNames[mId])
    for ix=1,nImgs do
        local dIm = image.load(paths.concat(modelDir, 'depth_' .. tostring(ix-1) .. '.png'))
        local noiseIm = torch.Tensor(dIm:size()):uniform(-noiseRange, noiseRange)
        noiseIm:cmul(torch.lt(dIm,1):double())
        dIm = dIm + noiseIm
        image.save(paths.concat(modelDir, 'noise0pt2_depth' .. '_' .. tostring(ix-1) .. '.png'), dIm)
    end
end