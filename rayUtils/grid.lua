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

function gridNd.new(minBounds, maxBounds, step)
    local self = setmetatable({}, gridNd)
    self.minBounds = minBounds
    self.maxBounds = maxBounds
    self.step = step
    return self
end

function gridNd:size()
    local sz = torch.Tensor(self.step:size()):fill(0)
    local numCells = 1
    for d = 1,sz:size(1) do
        sz[d] = (self.maxBounds[d] - self.minBounds[d])/self.step[d]
        numCells = numCells*sz[d]
    end
    return {sz,numCells}
end

function gridNd:cellIndex(point)
    local inds = torch.Tensor(point:size(1))
    for d=1,point:size(1) do
        if (point[d] >= self.maxBounds[d]) or (point[d] < self.minBounds[d]) then
            return nil
        end
        inds[d] = 1+torch.floor((point[d]-self.minBounds[d])/self.step[d])
    end
    return inds
end

function gridNd:gridIndToPoint(inds)
    -- inds are 1-indexed
    -- assumed the index is valid
    local Nd = inds:size(1)
    local point = torch.Tensor(Nd)
    for d=1,Nd do
        point[d] = (inds[d] - 1)*self.step[d] + self.minBounds[d]
    end
    return point
end

-- assumes point is already slightly 'inside' a cell
function gridNd:nextRayPoint(point, direction)
    local t_min = tonumber('inf')
    local eps = 1e-4
    for d=1,point:size(1) do
        if(torch.abs(direction[d]) > 0) then
            local diff
            if(direction[d] > 0) then -- increasing coord
                diff = self.step[d] - (point[d] - self.minBounds[d])%self.step[d]
            else
                diff = (point[d] - self.minBounds[d])%self.step[d]
            end
            local t = diff/torch.abs(direction[d])
            if (t < t_min) then
                t_min = t
            end
        end
    end
    local newPoint = point + (t_min+eps)*direction
    local newCell = self:cellIndex(newPoint)
    return {newPoint, newCell}
end

function gridNd:initRayPoint(origin, direction)
    local t_min = tonumber('inf')
    local eps = 1e-4
    local point = origin + (eps)*direction
    if(not (self:cellIndex(point) == nil)) then
        return {point, self:cellIndex(point)}
    end
    
    for d=1,point:size(1) do
        if(torch.abs(direction[d]) > 0) then
            local boundaryVals = {self.minBounds[d], self.maxBounds[d]}
            for ix=1,2 do
                local bVal = boundaryVals[ix]
                local t = (bVal - point[d])/direction[d]
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
    return {newPoint, newCell}
end

function gridNd:traceRay(origin, direction)
    local pointSeq = {}
    local cellSeq = {}
    local depthSeq = {}
    local point, cell = unpack(self:initRayPoint(origin, direction))
    local nc = 1
    while(not (cell == nil)) do
        pointSeq[nc] = point
        cellSeq[nc] = cell
        depthSeq[nc] = torch.norm(point-origin)
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