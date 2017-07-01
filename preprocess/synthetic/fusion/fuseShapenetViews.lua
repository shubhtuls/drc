local M = {}
local dUtils = dofile('../utils/depthUtils.lua')
local matio = require 'matio'
local rpsem = require 'rpsem'
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
local fusion = {}
fusion.__index = fusion

setmetatable(fusion, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

function fusion.new(synsetDir, nImgs, maskOnly, geometry, useNoise)
    local self = setmetatable({}, fusion)
    self.nImgs = nImgs
    self.depthPrefix = useNoise and 'noise0pt2_depth' or 'depth'
    self.synsetDir = synsetDir
    self.modelNames = BuildArray(paths.files(self.synsetDir,'...'))
    
    self.geometry = geometry
    self.sz, self.nCells = unpack(geometry:size())
    self.gridSize = {}
    for d=1,self.sz:size(1) do
        self.gridSize[d] = self.sz[d]
    end
    self.maxBounds = self.geometry.maxBounds:clone():contiguous()
    self.minBounds = self.geometry.minBounds:clone():contiguous()
    local useProj = (self.geometry.focal ~= nil)
    self.focal = (useProj and self.geometry.focal or self.geometry.step):clone():contiguous()
    self.useProj = torch.Tensor({useProj and 1 or 0}):int()
    self.maskOnly = torch.Tensor({maskOnly and 1 or 0}):int()
    self.nRaysPerCam = 224*224
    self.emptyCount = torch.Tensor(torch.LongStorage(self.gridSize))
    self.occCount = torch.Tensor(torch.LongStorage(self.gridSize))
    
    return self
end

function fusion:fuseModel(imgsDir)
    local nRays = self.nRaysPerCam*self.nImgs
    local bs = 1
    local depths = torch.Tensor(bs, nRays):float()
    local origins = torch.Tensor(bs, nRays, 3):float()
    local directions = torch.Tensor(bs, nRays, 3):float()
    
    local nImgs = self.nImgs
    for nc = 1,self.nImgs do
        local numSamples =  self.nRaysPerCam
        local imgNum = (nc-1)
        local camData = matio.load(string.format('%s/camera_%d.mat',imgsDir,imgNum),{'pos','quat','K','extrinsic'})
        local dMap = (image.load(string.format('%s/%s_%d.png',imgsDir,self.depthPrefix,imgNum))*maxSavedDepth)
        local disparityIm = dMap:clone():pow(-1)
        local dispMask = torch.Tensor(dMap:size()):fill(1)
        local mMat = dUtils.inverseMotionMat(camData.extrinsic)

        local dirSamples, dSamples = dUtils.allImgRays(disparityIm, dispMask, camData.K, 0)

        local orgSamples = torch.Tensor(4,numSamples):fill(0)
        orgSamples:narrow(1,4,1):fill(1)
        orgSamples = torch.mm(mMat, orgSamples:typeAs(mMat)):narrow(1,1,3):transpose(2,1)
        dirSamples = torch.mm(mMat, dirSamples:typeAs(mMat)):narrow(1,1,3):transpose(2,1)

        depths[1]:narrow(1,(nc-1)*numSamples+1,numSamples):copy(dSamples)
        directions[1]:narrow(1,(nc-1)*numSamples+1,numSamples):copy(dirSamples)
        origins[1]:narrow(1,(nc-1)*numSamples+1,numSamples):copy(orgSamples)
    end

    if(self.maskOnly[1]==1) then
        depths  = torch.gt(depths,10):double()
    end
    self.emptyCount = self.emptyCount:contiguous():view(1,self.nCells):fill(0)
    self.occCount = self.occCount:contiguous():view(1,self.nCells):fill(0)
    rpsem.computeRayFusion(self, origins:contiguous():double(), directions:contiguous():double(), depths:contiguous():double())
    return self.emptyCount:reshape(torch.LongStorage(self.gridSize)), self.occCount:reshape(torch.LongStorage(self.gridSize))
end

function fusion:fuseAll(saveDir)
    for mId = 1,#self.modelNames do
        local imgsDir = paths.concat(self.synsetDir, self.modelNames[mId])
        local emptyCount, occCount = self:fuseModel(imgsDir)
        matio.save(paths.concat(saveDir,self.modelNames[mId] .. '.mat'),{emptyCount=emptyCount:int(),occCount=occCount:int()})
    end
end
-------------------------------
-------------------------------

M.fusion = fusion
return M