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
function rayPotential.new(geometry, maxDepth, maskOnly)
    local self = setmetatable({}, rayPotential)
    local maskOnly = maskOnly or false
    self.geometry = geometry
    self.sz, self.nCells = unpack(geometry:size())
    
    self.maxBounds = self.geometry.maxBounds:clone():contiguous()
    self.minBounds = self.geometry.minBounds:clone():contiguous()
    local useProj = (self.geometry.focal ~= nil)
    self.focal = (useProj and self.geometry.focal or self.geometry.step):clone():contiguous()
    
    self.maxDepth = torch.Tensor({maxDepth})
    self.useProj = torch.Tensor({useProj and 1 or 0}):int()
    self.maskOnly = torch.Tensor({maskOnly and 1 or 0}):int()
    --print(self.maskOnly)
    return self
end

function rayPotential:forward(_predsGeom, rays)
    local origins, dirs, depths = unpack(rays)
    --print(dirs[1])
    local bs = origins:size(1)
    local nrays = origins:size(2)
    
    self.predsGeom = _predsGeom:contiguous():view(bs, self.nCells):double()
    self.gradPredsGeom = torch.Tensor(bs, self.nCells):fill(0)
    
    self.E_psi = torch.Tensor(bs, nrays):fill(0):contiguous()
    if(#rays==3) then
        rpsem.computeRpGeom(self, origins:contiguous():double(), dirs:contiguous():double(), depths:contiguous():double())
    else
        local weights = rays[4]
        rpsem.computeRpGeomWeighted(self, origins:contiguous():double(), dirs:contiguous():double(), depths:contiguous():double(), weights:contiguous():double())
    end
    return self.E_psi
end

function rayPotential:backward(_preds, rays)
    return self.gradPredsGeom:reshape(_preds:size())
end

-------------------------------
-------------------------------
M.rayPotential = rayPotential
return M