local M = {}
-------------------------------
-------------------------------
local gridNd = {}
gridNd.__index = gridNd

setmetatable(gridNd, {
    __call = function (cls, ...)
        return cls.new(...)
    end,
})

function gridNd.new(minBounds, maxBounds, focal)
    local self = setmetatable({}, gridNd)
    self.minBounds = minBounds
    self.maxBounds = maxBounds
    self.focal = focal
    self.Nd = focal:size(1)
    --print(self)
    return self
end

function gridNd:size()
    local sz = torch.Tensor(self.Nd):fill(0)
    local Nd = self.Nd
    local numCells = 1
    for d = 1,(Nd-1) do
        sz[d] = (self.maxBounds[d] - self.minBounds[d])
        numCells = numCells*sz[d]
    end
    -- change z-bin function below
    sz[Nd] = (torch.log(self.maxBounds[Nd]) - torch.log(self.minBounds[Nd]))/self.focal[Nd]
    numCells = numCells*sz[Nd]
    return {sz,numCells}
end

function gridNd:cellIndex(point)
    local Nd = self.Nd
    local inds = torch.Tensor(self.Nd)
    for d=1,(self.Nd-1) do
        local pixel = self.focal[d]*point[d]/point[self.Nd]
        --print(d,pixel,point[d],point[self.Nd])
        if (pixel >= self.maxBounds[d]) or (pixel < self.minBounds[d]) then
            return nil
        end
        inds[d] = 1+torch.floor(pixel-self.minBounds[d])
    end
    if(point[self.Nd] >= self.maxBounds[Nd]) or (point[self.Nd] < self.minBounds[Nd]) then
        return nil
    end
    local z = torch.log(point[Nd]/self.minBounds[Nd])/self.focal[Nd]
    inds[self.Nd] = 1+torch.floor(z)
    return inds
end

function gridNd:gridIndToPoint(inds)
    -- inds are 1-indexed
    -- assumed the index is valid
    local Nd = self.Nd
    local point = torch.Tensor(self.Nd)
    point[Nd] = torch.exp((inds[Nd]-1)*self.focal[Nd])*self.minBounds[Nd]
    for d=1,(self.Nd-1) do
        local pixel = inds[d] - 1 + self.minBounds[d]
        point[d] = pixel*point[self.Nd]/self.focal[d]
    end
    return point
end

function gridNd:initRayPoint(origin, direction)
    --print(origin, direction)
    local t_min = tonumber('inf')
    local eps = 1e-4
    local point = origin + (eps)*direction
    if(not (self:cellIndex(point) == nil)) then
        return {point, self:cellIndex(point)}
    end
    local Nd = self.Nd
    for d=1,self.Nd do
        if(torch.abs(direction[d]) > 0) then
            local boundaryVals = {self.minBounds[d], self.maxBounds[d]}
            for ix=1,2 do
                local bVal = boundaryVals[ix]
                local t
                if(d==Nd) then
                    t = (bVal - origin[d])/direction[d]
                else
                    t = (bVal*point[Nd] - self.focal[d]*point[d])/(direction[d]*self.focal[d] - direction[Nd]*bVal)
                end
                if(t > 0) and (t < t_min) then
                    local candPoint = point + (t+eps)*direction
                    if not (self:cellIndex(candPoint) == nil) then
                        t_min = t
                    end
                end
            end
        end
    end
    --print(t_min)
    if(t_min == tonumber('inf')) then
        return {origin, nil}
    end
    local newPoint = point + (t_min+eps)*direction
    local newCell = self:cellIndex(newPoint)
    --print(origin, direction, newPoint)
    return {newPoint, newCell}
end

-- assumes point is already slightly 'inside' a cell
function gridNd:nextRayPoint(point, direction)
    local t_min = tonumber('inf')
    local eps = 1e-4
    local Nd = self.Nd
    for d=1,(Nd-1) do
        local fx = self.focal[d]
        local pixel = fx*point[d]/point[Nd] --projection equation
        local p_f = torch.floor(pixel) --possible neighboring bin boundary
        local p_c = torch.ceil(pixel) --possible neighboring bin boundary
        local diff
        
        local t1 = (p_f*point[Nd] - fx*point[d])/(direction[d]*fx - direction[Nd]*p_f)
        local t2 = (p_c*point[Nd] - fx*point[d])/(direction[d]*fx - direction[Nd]*p_c)
        --print(d,t1,t2)
        
        if (t1>0) and (t1 < t_min) then
            t_min = t1
        end
        if (t2>0) and (t2 < t_min) then
            t_min = t2
        end
    end
    local z = torch.log(point[Nd]/self.minBounds[Nd])/self.focal[Nd]
    local zNext
    if(direction[Nd] > 0) then zNext = torch.ceil(z) else zNext = torch.floor(z) end
    local Z_next = torch.exp(zNext*self.focal[Nd])*self.minBounds[Nd]
    local tZ = (Z_next - point[Nd])/direction[Nd]
    --print(3, tZ)
    if (tZ>0) and (tZ < t_min) then
        t_min = tZ
    end
    local newPoint = point + (t_min+eps)*direction
    local newCell = self:cellIndex(newPoint)
    return {newPoint, newCell}
end

function gridNd:traceRay(origin, direction)
    local pointSeq = {}
    local cellSeq = {}
    local depthSeq = {}
    local point, cell = unpack(self:initRayPoint(origin, direction))
    local nc = 1
    while(cell ~= nil) do
        pointSeq[nc] = point
        cellSeq[nc] = cell
        depthSeq[nc] = torch.norm(point-origin)
        --print(depthSeq[nc])
        nc = nc+1
        --print(point, cell)
        point, cell = unpack(self:nextRayPoint(point, direction))
    end
    pointSeq[nc] = point
    return {cellSeq, depthSeq, pointSeq}
end
-------------------------------
-------------------------------
M.gridNd = gridNd
return M