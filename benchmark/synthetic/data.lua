local M = {}
require 'image'
local matio = require 'matio'
-------------------------------
-------------------------------
local dataLoader = {}
dataLoader.__index = dataLoader

setmetatable(dataLoader, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

function dataLoader.new(imgsDir, voxelsDir, nImgs, imgSize, voxelSize, modelNames)
    local self = setmetatable({}, dataLoader)
    self.nImgs = nImgs
    self.imgSize = imgSize
    self.imgsDir = imgsDir
    self.voxelSize = voxelSize
    self.voxelsDir = voxelsDir
    self.modelNames = modelNames
    self.imgsOriginal = torch.Tensor(self.nImgs, 3, 224, 224):fill(0)
    self.mId = 1
    return self
end

function dataLoader:forward()
    local imgs = torch.Tensor(self.nImgs, 3, self.imgSize[1], self.imgSize[2]):fill(0)
    local mId = self.mId
    local imgsDir = paths.concat(self.imgsDir, self.modelNames[mId])
    local nImgs = self.nImgs
    local voxelFile = paths.concat(self.voxelsDir, self.modelNames[mId] .. '.mat')
    local voxels = matio.load(voxelFile,{'Volume'})['Volume']:typeAs(imgs)
    voxels = voxels:transpose(3,2) --to nullify things done by blender when rendering
    
    for b = 1,nImgs do
        local inpImgNum = (b-1)
        local imgOrig = image.load(string.format('%s/render_%d.png',imgsDir,inpImgNum))
        
        local imgRgb = image.scale(imgOrig:clone(),self.imgSize[2], self.imgSize[1])
        local alphaMask = imgRgb[4]:repeatTensor(3,1,1)
        imgRgb = torch.cmul(imgRgb:narrow(1,1,3),alphaMask) + 1 - alphaMask
        imgs[b] = imgRgb:clone()

        local alphaMask = imgOrig[4]:repeatTensor(3,1,1)
        imgOrig = torch.cmul(imgOrig:narrow(1,1,3),alphaMask) + 1 - alphaMask
        self.imgsOriginal[b] = imgOrig:clone()
    end
    
    self.mId = self.mId%(#self.modelNames)+1
    return imgs, voxels
    
end
-------------------------------
-------------------------------
M.dataLoader = dataLoader
return M