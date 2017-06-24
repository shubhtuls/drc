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
function rayPotential.new(geometry, bgColor)
    local self = setmetatable({}, rayPotential)
    self.geometry = geometry
    self.sz, self.nCells = unpack(geometry:size())
    
    self.maxBounds = self.geometry.maxBounds:clone():contiguous()
    self.minBounds = self.geometry.minBounds:clone():contiguous()
    self.focal = self.geometry.step:clone():contiguous()
    
    self.bgColor = bgColor:clone():contiguous()
    self.useProj = torch.Tensor({0}):int()
    
    return self
end

function rayPotential:forward(_preds, rays)
    local origins, dirs, colors = unpack(rays)
    --print(dirs[1])
    local _predsGeom, _predsColor = unpack(_preds)
    local bs = origins:size(1)
    local nrays = origins:size(2)
    
    self.predsGeom = _predsGeom:contiguous():view(bs, self.nCells):double()
    self.predsColor = _predsColor:permute(1,3,4,5,2):contiguous():view(bs, self.nCells, 3):double()
    
    self.gradPredsGeom = torch.Tensor(bs, self.nCells):fill(0)
    self.gradPredsColor = torch.Tensor(bs, self.nCells, 3):fill(0)
    
    self.E_psi = torch.Tensor(bs, nrays):fill(0):contiguous()
    rpsem.computeRpColor(self, origins:contiguous():double(), dirs:contiguous():double(), colors:contiguous():double())
    return self.E_psi
end

function rayPotential:backward(_preds, rays)
    return {self.gradPredsGeom:reshape(_preds[1]:size()), self.gradPredsColor:permute(1,3,2):contiguous():reshape(_preds[2]:size())}
end

-------------------------------
-------------------------------
M.rayPotential = rayPotential
return M