local M = {}
local dUtil = dofile('../utils/depthUtils.lua')
local cropUtils = dofile('../utils/cropUtils.lua')
local matio = require 'matio'
require 'image'
-------------------------------
-------------------------------
local function BuildArray(...)
  local arr = {}
  for v in ... do
    arr[#arr + 1] = v
  end
  return arr
end
-------------------------------
-------------------------------
local dataLoader = {}
dataLoader.__index = dataLoader

setmetatable(dataLoader, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

function dataLoader.new(stateFilesDir, bs, nRays, imgSize, modelNames)
    local self = setmetatable({}, dataLoader)
    self.bs = bs
    self.nRays = nRays
    self.imgSize = imgSize
    self.stateFilesDir = stateFilesDir
    self.modelNames = modelNames
    self.multiClass = (torch.type(modelNames[1]) == 'table')
    return self
end

function dataLoader:forward()
    
    local nRays = self.nRays
    local imgs = torch.Tensor(self.bs, 3, self.imgSize[1], self.imgSize[2]):fill(0)
    local depths = torch.Tensor(self.bs, nRays)
    local origins = torch.Tensor(self.bs, nRays, 3)
    local directions = torch.Tensor(self.bs, nRays, 3)
    
    for b = 1,self.bs do
        local modelName
        if(self.multiClass) then
            local cId = torch.random(1,#self.modelNames)
            local mId = torch.random(1,#self.modelNames[cId])
            modelName = self.modelNames[cId][mId]
        else
            local mId = torch.random(1,#self.modelNames)
            modelName = self.modelNames[mId]
            --print(modelName)
        end
        
        local state = matio.load(paths.concat(self.stateFilesDir, modelName), {'mask','bbox','cameraScale','cameraRot','translation','im'})
        local bbox = state.bbox:view(4)
        local dirSamples, orgSamples, maskSamples = dUtil.samplePascalOrthographicRays(state.mask:double(), bbox, state.cameraScale:view(1), state.cameraRot, state.translation:view(2), nRays, 0.25)
        
        local imgRgb = state.im:transpose(2,3):transpose(1,2):double()
        imgRgb = imgRgb/255
        
        imgRgb = cropUtils.cropImg(imgRgb:double(),torch.Tensor({bbox[1],bbox[2],bbox[1]-1+bbox[3],bbox[2]-1+bbox[4]}), 0.15, 0.15)
        
        imgRgb = image.scale(imgRgb,self.imgSize[1],self.imgSize[2])
        if(torch.random(0,1) == 1) then imgRgb = image.hflip(imgRgb) end
        imgs[b] = imgRgb
        
        depths[b]:copy(1-maskSamples)
        
        local refSymmetry = torch.Tensor(nRays):bernoulli()*2 - 1
        dirSamples:narrow(2,2,1):copy(dirSamples:narrow(2,2,1):clone():cmul(refSymmetry))
        orgSamples:narrow(2,2,1):copy(orgSamples:narrow(2,2,1):clone():cmul(refSymmetry))

        directions[b]:copy(dirSamples)
        origins[b]:copy(orgSamples)
        
    end
    
    local bgLabel = depths:clone()
    return {imgs, {origins, directions, depths}, bgLabel}
end
-------------------------------
-------------------------------
M.dataLoader = dataLoader
return M