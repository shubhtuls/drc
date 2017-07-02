local M = {}
local dUtil = dofile('../utils/depthUtils.lua')
local matio = require 'matio'
local cropUtils = dofile('../utils/cropUtils.lua')
require 'image'
-------------------------------
-------------------------------
local dataLoader = {}
dataLoader.__index = dataLoader

setmetatable(dataLoader, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

function dataLoader.new(stateFilesDir, voxelsDir, bs, imgSize, voxelSize, modelNames)
    local self = setmetatable({}, dataLoader)
    self.bs = bs
    self.imgSize = imgSize
    self.stateFilesDir = stateFilesDir
    self.voxelsDir = voxelsDir
    self.voxelSize = voxelSize
    self.modelNames = modelNames

    self.mId = 0
    return self
end

function dataLoader:forward()
    
    local imgs = torch.Tensor(self.bs, 3, self.imgSize[1], self.imgSize[2]):fill(0)
    local gtVols = torch.Tensor(self.bs, 1, self.voxelSize[1], self.voxelSize[2], self.voxelSize[3]):fill(0)
    
    for b = 1,self.bs do
        local modelNamesBatch = {}
        self.mId = (self.mId % #self.modelNames)+1
        local mId = self.mId
        modelNamesBatch[b] = self.modelNames[mId]
        local state = matio.load(paths.concat(self.stateFilesDir, self.modelNames[mId]), {'subtype','bbox','im'})
        local bbox = state.bbox:view(4)
        local subtype = state.subtype:view(1)
        local voxelFile = paths.concat(self.voxelsDir, tostring(subtype[1]) .. '.mat')
        local voxelGt = matio.load(voxelFile,{'Volume'})['Volume']:typeAs(imgs)

        local imgRgb = state.im:transpose(2,3):transpose(1,2):double()
        imgRgb = imgRgb/255
        imgRgb = cropUtils.cropImg(imgRgb:double(),torch.Tensor({bbox[1],bbox[2],bbox[1]-1+bbox[3],bbox[2]-1+bbox[4]}), 0.15, 0)
        imgRgb = image.scale(imgRgb,self.imgSize[1],self.imgSize[2])
        imgs[b] = imgRgb
        gtVols[b][1] = voxelGt
    end
    
    return {imgs, gtVols, modelNamesBatch}
end
-------------------------------
-------------------------------
M.dataLoader = dataLoader
return M