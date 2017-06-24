local M = {}
local rpsem = require 'rpsem'
-------------------------------
-------------------------------
local rayPotential = {}
rayPotential.__index = rayPotential

setmetatable(rayPotential, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

-- uses a bins stucture
function rayPotential.new(geometry, maxDepth, nClasses, bgDist)
    local self = setmetatable({}, rayPotential)
    self.geometry = geometry
    self.sz, self.nCells = unpack(geometry:size())
    self.nClasses = nClasses
    
    self.maxBounds = self.geometry.maxBounds:clone():contiguous()
    self.minBounds = self.geometry.minBounds:clone():contiguous()
    self.focal = self.geometry.focal:clone():contiguous()
    
    self.bgDist = bgDist:clone():contiguous()
    self.maxDepth = torch.Tensor({maxDepth})
    self.useProj = torch.Tensor({1}):int()
    
    return self
end

function rayPotential:forward(_preds, rays)
    local origins, dirs, depths, classIds = unpack(rays)
    --print(dirs[1])
    local _predsGeom, _predsClass = unpack(_preds)
    local bs = origins:size(1)
    local nrays = origins:size(2)
    
    self.predsGeom = _predsGeom:contiguous():view(bs, self.nCells)
    self.predsSem = _predsClass:permute(1,3,4,5,2):contiguous():view(bs, self.nCells, self.nClasses)
    
    self.gradPredsGeom = torch.Tensor(bs, self.nCells):fill(0)
    self.gradPredsSem = torch.Tensor(bs, self.nCells, self.nClasses):fill(0)
    
    self.E_psi = torch.Tensor(bs, nrays):fill(0):contiguous()
    rpsem.computeRpSem(self, origins:contiguous():double(), dirs:contiguous():double(), depths:contiguous():double(), (classIds-1):int():contiguous())
    return self.E_psi
end

function rayPotential:backward(_preds, rays)
    return {self.gradPredsGeom:reshape(_preds[1]:size()), self.gradPredsSem:permute(1,3,2):contiguous():reshape(_preds[2]:size())}
end

-------------------------------
-------------------------------
M.rayPotential = rayPotential
return M