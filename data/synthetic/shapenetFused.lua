local M = {}
require 'image'
local matio = require 'matio'
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

function dataLoader.new(imgsDir, voxelsDir, bs, imgSize, voxelSize, nImgs, modelNames)
    local self = setmetatable({}, dataLoader)
    self.bs = bs
    self.imgSize = imgSize
    self.imgsDir = imgsDir
    self.voxelSize = voxelSize
    self.voxelsDir = voxelsDir
    self.nImgs = nImgs
    self.modelNames = modelNames
    return self
end

function dataLoader:forward()
    local imgs = torch.Tensor(self.bs, 3, self.imgSize[1], self.imgSize[2]):fill(0)
    local voxelProbs = torch.Tensor(self.bs, 1, self.voxelSize[1], self.voxelSize[2], self.voxelSize[3]):fill(0)
    local weights = torch.Tensor(self.bs, 1, self.voxelSize[1], self.voxelSize[2], self.voxelSize[3]):fill(0)
    
    for b = 1,self.bs do
        local mId = torch.random(1,#self.modelNames)
        local imgsDir = paths.concat(self.imgsDir, self.modelNames[mId])
        local nImgs = self.nImgs or #BuildArray(paths.files(imgsDir,'.mat'))
        local inpImgNum = torch.random(0,nImgs-1)
        local imgRgb = image.scale(image.load(string.format('%s/render_%d.png',imgsDir,inpImgNum)),self.imgSize[2], self.imgSize[1])
        local alphaMask = imgRgb[4]:repeatTensor(3,1,1)
        imgRgb = torch.cmul(imgRgb:narrow(1,1,3),alphaMask) + 1 - alphaMask
        imgs[b] = imgRgb
        
        local voxelFile = paths.concat(self.voxelsDir, self.modelNames[mId] .. '.mat')
        local model = matio.load(voxelFile,{'emptyCount','occCount'})
        local normalizer = (model.emptyCount + model.occCount):typeAs(voxelProbs)
        weights[b][1] = (torch.gt(normalizer,0)):typeAs(weights)
        normalizer = normalizer + (torch.eq(normalizer,0):typeAs(normalizer))
        local probs = model.occCount:clone():typeAs(normalizer)
        probs:cdiv(normalizer)
        voxelProbs[b][1] = probs
    end
    return imgs, voxelProbs, weights
end
-------------------------------
-------------------------------
M.dataLoader = dataLoader
return M