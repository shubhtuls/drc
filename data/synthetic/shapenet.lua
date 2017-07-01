local M = {}
require 'image'
local cropUtils = dofile('../utils/cropUtils.lua')
local dUtils = dofile('../utils/depthUtils.lua')
local matio = require 'matio'
-------------------------------
-------------------------------
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
local dataLoader = {}
dataLoader.__index = dataLoader

setmetatable(dataLoader, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

function dataLoader.new(synsetDir, bs, nCams, nRaysPerCam, imgSize, minDisp, maskOnly, nImgs, modelNames, useNoise)
    local self = setmetatable({}, dataLoader)
    self.bs = bs
    self.nCams = nCams
    self.nRaysPerCam = nRaysPerCam
    self.imgSize = imgSize
    self.minDisp = minDisp
    self.synsetDir = synsetDir
    self.maskOnly = maskOnly
    self.nImgs = nImgs
    self.modelNames = modelNames
    self.depthPrefix = useNoise and 'noise0pt2_depth' or 'depth'
    return self
end

function dataLoader:forward()
    
    local nRays = self.nRaysPerCam*self.nCams
    local imgs = torch.Tensor(self.bs, 3, self.imgSize[1], self.imgSize[2]):fill(0)
    local depths = torch.Tensor(self.bs, nRays):float()
    local origins = torch.Tensor(self.bs, nRays, 3):float()
    local directions = torch.Tensor(self.bs, nRays, 3):float()
    for b = 1,self.bs do
        local mId = torch.random(1,#self.modelNames)
        local imgsDir = paths.concat(self.synsetDir, self.modelNames[mId])
        local nImgs = self.nImgs or #BuildArray(paths.files(imgsDir,'.mat'))
        local inpImgNum = torch.random(0,nImgs-1)

        local imgRgb = image.load(string.format('%s/render_%d.png',imgsDir,inpImgNum))
        if(self.bgImgsList) then
            -- useful for PASCAL VOC experiments, we'll set the bgImgsList externally
            imgRgb = cropUtils.blendBg(imgRgb, self.bgImgsList)
            imgRgb = image.scale(imgRgb,self.imgSize[2], self.imgSize[1])
        else
            imgRgb = image.scale(imgRgb,self.imgSize[2], self.imgSize[1])
            local alphaMask = imgRgb[4]:repeatTensor(3,1,1)
            imgRgb = torch.cmul(imgRgb:narrow(1,1,3),alphaMask) + 1 - alphaMask
        end
        imgs[b] = imgRgb
        local rPerm = torch.randperm(nImgs)
        
        for nc = 1,self.nCams do
            local numSamples =  self.nRaysPerCam
            local imgNum = rPerm[nc] - 1
            --local imgNum = (nc==1) and inpImgNum or torch.random(0,nImgs-1)
            local camData = matio.load(string.format('%s/camera_%d.mat',imgsDir,imgNum),{'pos','quat','K','extrinsic'})
            local dMap = (image.load(string.format('%s/%s_%d.png',imgsDir,self.depthPrefix,imgNum))*maxSavedDepth)
            local disparityIm = dMap:clone():pow(-1)
            local dispMask = torch.Tensor(dMap:size()):fill(1)
            
            local mMat = dUtils.inverseMotionMat(camData.extrinsic)
            
            local dirSamples, dSamples = dUtils.sampleRays(disparityIm, dispMask, camData.K, self.minDisp, numSamples)
            
            local orgSamples = torch.Tensor(4,numSamples):fill(0)
            orgSamples:narrow(1,4,1):fill(1)
            orgSamples = torch.mm(mMat, orgSamples:typeAs(mMat)):narrow(1,1,3):transpose(2,1)
            dirSamples = torch.mm(mMat, dirSamples:typeAs(mMat)):narrow(1,1,3):transpose(2,1)
            
            depths[b]:narrow(1,(nc-1)*numSamples+1,numSamples):copy(dSamples)
            directions[b]:narrow(1,(nc-1)*numSamples+1,numSamples):copy(dirSamples)
            origins[b]:narrow(1,(nc-1)*numSamples+1,numSamples):copy(orgSamples)
        end
    end
    local bgLabel = torch.gt(depths,10):double()
    if(self.maskOnly) then
        depths  = bgLabel:clone()
    end
    return {imgs, {origins, directions, depths}, bgLabel}
end
-------------------------------
-------------------------------
M.dataLoader = dataLoader
return M