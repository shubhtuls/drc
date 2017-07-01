require 'json'
require 'image'
local M = {}
----------
local function loadDisparity(dispDir, camDir, imgIdDisp, imgIdCamera)
    local dispMap = image.load(paths.concat(dispDir, imgIdDisp .. '_disparity.png'))*(torch.pow(2,16)-1)
    dispMap = torch.round(dispMap)
    local dispMask = torch.ne(dispMap, 0)
    dispMap = (dispMap:float() - 1)/256
    local cam =  json.load(paths.concat(camDir, imgIdCamera .. '_camera.json'))
    local B  = cam['extrinsic']['baseline']
    local fx = cam['intrinsic']['fx']
    local dispMap = dispMap/(fx * B)
    return dispMap, dispMask
end
----------------
local function loadCameraMat(camDir, imgId)
    local filler = torch.Tensor({0, 0, 0, 1}):reshape(1,4)
    local cam =  json.load(paths.concat(camDir, imgId .. '_camera.json'))
    local fx = cam['intrinsic']['fx']
    local fy = cam['intrinsic']['fy']
    local u0 = cam['intrinsic']['u0']
    local v0 = cam['intrinsic']['v0']
    local intMat  = torch.Tensor({fx, 0, u0, 0, 
                        0, fy, v0, 0, 
                        0,  0, 1, 0, 
                        0,  0, 0, 1}):reshape(4,4)
    
    local cs2ours = torch.Tensor({0, -1,  0,
                        0,  0, -1,
                        1,  0,  0}):reshape(3,3)

    local ours2cs = cs2ours:transpose(2,1)
    
    local x_c2v = cam['extrinsic']['x']
    local y_c2v = cam['extrinsic']['y']
    local z_c2v = cam['extrinsic']['z']
    local T_c2v = torch.Tensor({x_c2v, y_c2v, z_c2v}):reshape(3,1)
    local T_c2v = torch.mm(cs2ours, T_c2v)
    
    local cy = torch.cos(cam['extrinsic']['yaw'])
    local sy = torch.sin(cam['extrinsic']['yaw'])
    local cp = torch.cos(cam['extrinsic']['pitch'])
    local sp = torch.sin(cam['extrinsic']['pitch'])
    local cr = torch.cos(cam['extrinsic']['roll'])
    local sr = torch.sin(cam['extrinsic']['roll'])
    local R_c2v = torch.Tensor({cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr,
                      sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr,
                        -sp,          cp*sr,          cp*cr}):reshape(3,3)
    
    local R_c2v = torch.mm(cs2ours,torch.mm(R_c2v,ours2cs))
    
    local extMat_c2v = torch.cat(R_c2v, T_c2v, 2)
    local extMat_c2v = torch.cat(extMat_c2v, filler, 1)

    local R_v2c = R_c2v:transpose(2,1)
    local T_v2c = -torch.mm(R_v2c, T_c2v)
    local extMat_v2c = torch.cat(R_v2c, T_v2c,2)
    local extMat_v2c = torch.cat(extMat_v2c, filler, 1)
    
    return intMat, extMat_c2v, extMat_v2c
    
end
----------------
local function getRotFromYaw(yaw)
    local rotMat = torch.Tensor({torch.cos(yaw), 0, torch.sin(yaw),
                              0, 1, 0,
                      -torch.sin(yaw), 0, torch.cos(yaw)}):reshape(3,3)
    return rotMat
end

local function loadMotionMat(dataDir, ids)
    local trMat = {}
    local time = 1/17
    local filler = torch.Tensor({0, 0, 0, 1}):reshape(1,4)
    for ix=1,#ids do
        local odomFile = paths.concat(dataDir, ids[ix] .. '_vehicle.json')
        local odom = json.load(odomFile)
        local yaw = -odom['yawRate'] * time
        
        local tx = odom['speed'] * odom['yawRate'] * (time^2)/2.
        local ty = 0
        local tz = odom['speed'] * time - odom['speed'] * (odom['yawRate']^2) * (time^3)/6
        local tVec = torch.Tensor({tx, ty, tz}):reshape(3,1)
        
        local rotMat = getRotFromYaw(yaw)
        local T = torch.cat(rotMat, tVec, 2)
        T =  torch.cat(T, filler, 1)
        trMat[ix] = T
    end
    return trMat
end

local function integrateMotion(motionMats)
    -- returns M_t+K_to_t = M_t+1_to_t * ... * M_t+K_to_t+K-1
    -- motionMats contain M_t+k_to_t+k-1
    local transform = torch.eye(4)
    local K = #motionMats
    for k=1,K do
        transform = torch.mm(transform, motionMats[k])
    end
    return transform
end

local function inverseMotionMat(motionMat)
    local filler = torch.Tensor({0, 0, 0, 1}):reshape(1,4):typeAs(motionMat)
    local rot = motionMat[{{1,3},{1,3}}]
    local trans = motionMat[{{1,3},{4}}]
    local rotT = rot:transpose(2,1)
    --print(trans:size())
    --print(rotT:size())
    local transT = -torch.mm(rotT, trans)
    local T = torch.cat(rotT, transT, 2)
    T =  torch.cat(T, filler, 1)
    return T
end

local function integrateBackwardMotion(motionMats)
    -- returns M_t_to_t+K = M_t+K_to_t+K-1 * ... * M_t+1_to_t
    -- motionMats contain M_t+k_to_t+k-1
    local transform = torch.eye(4)
    local K = #motionMats
    for k=1,K do
        transform = torch.mm(inverseMotionMat(motionMats[k]), transform)
    end
    return transform
end


-- in normal usage, srcFrame : 25, targetFrame : 19
local function motionMatSeq(srcImgNum, targetImgNum, seqId, vehicleDir)
    local idInit = targetImgNum
    local idFinal = srcImgNum-1
    if(srcImgNum < targetImgNum) then
        idInit = srcImgNum
        idFinal = targetImgNum-1
    end
    local imgSeq = {}
    for ix = 0,(idFinal-idInit) do
        imgSeq[ix] = seqId .. string.format("_%06d", ix+idInit)
    end
    local mMats = loadMotionMat(vehicleDir, imgSeq)
    local motion = integrateMotion(mMats)
    if(srcImgNum < targetImgNum) then motion = inverseMotionMat(motion) end
    return motion
end
----------------
local function pixelTo3Dcoord(instrinsic, Xs, Ys, Ds)
    local fx = instrinsic[1][1]
    local fy = instrinsic[2][2]
    local u0 = instrinsic[1][3]
    local v0 = instrinsic[2][3]
    local Zs = torch.cdiv(torch.ones(Ds:size()),Ds)
    local Xs = torch.cmul((Xs - u0)/fx, Zs)
    local Ys = torch.cmul((Ys - v0)/fy, Zs)
    local filler = torch.Tensor(Ds:size()):fill(1)
    return torch.cat(Xs, torch.cat(Ys, torch.cat(Zs, filler)))
end
----------------
local function imgToPoints(disparityIm, dispMask, img_rgb, instrinsic, sampleStep, minDisparity, scaleFactor)
    local scaleFactor = scaleFactor or 1
    local imH = disparityIm:size(2)
    local imW = disparityIm:size(3)
    local returnPixels = (img_rgb ~= nil)
    local disparityIm = disparityIm:reshape(imH,imW)
    local dispMask = dispMask:reshape(imH,imW)
    dispMask = torch.cmul(dispMask:clone(), torch.gt(disparityIm, minDisparity):typeAs(dispMask))

    local nPointsMax = torch.round(imH/sampleStep+1)*torch.round(imW/sampleStep+1)
    local pointsX = torch.Tensor(nPointsMax,1):fill(0)
    local pointsY = torch.Tensor(nPointsMax,1):fill(0)
    local pointsD = torch.Tensor(nPointsMax,1):fill(0)
    local selectedPixels = torch.Tensor(3,nPointsMax):fill(0)
    
    local nPoints = 0
    for y=1,imH,sampleStep do
        for x=1,imW,sampleStep do
            if(dispMask[y][x]>0) then
                nPoints = nPoints+1
                pointsX[nPoints] = x*scaleFactor
                pointsY[nPoints] = y*scaleFactor
                pointsD[nPoints] = disparityIm[y][x]
                if(returnPixels) then
                    selectedPixels[{{1,3},nPoints}] = img_rgb[{{1,3},y,x}]
                end
            end
        end
    end
    --print(nPoints)
    local pointsLifted = pixelTo3Dcoord(instrinsic, pointsX:narrow(1,1,nPoints), pointsY:narrow(1,1,nPoints), pointsD:narrow(1,1,nPoints)):transpose(2,1)
    if(returnPixels) then
        selectedPixels = selectedPixels:narrow(2,1,nPoints)
        return pointsLifted, selectedPixels
    else
        return pointsLifted
    end
end
----------------
local function allImgRays(disparityIm, dispMask, instrinsic, minDisp, scaleFactor)
    local scaleFactor = scaleFactor or 1
    local imH = disparityIm:size(2)
    local imW = disparityIm:size(3)
    local disparityIm = disparityIm:reshape(imH,imW)
    local dispMask = dispMask:reshape(imH,imW)
    dispMask = torch.cmul(dispMask, torch.gt(disparityIm, minDisp):typeAs(dispMask))
    local numSamples = torch.gt(dispMask,0):sum()
    local dirSamples = torch.Tensor(4,numSamples):fill(0)
    local pointsX = torch.Tensor(numSamples,1):fill(0)
    local pointsY = torch.Tensor(numSamples,1):fill(0)
    local pointsD = torch.Tensor(numSamples,1):fill(0)
    
    local ns = 0
    for x=1,imW do
        for y=1,imH do
            if(dispMask[y][x]>0) then
                ns = ns+1
                pointsX[ns] = x*scaleFactor
                pointsY[ns] = y*scaleFactor
                pointsD[ns] = disparityIm[y][x]
            end
        end
    end
    local pixelCoords = torch.cat(pointsY, pointsX)/scaleFactor
    local pointsLifted = pixelTo3Dcoord(instrinsic, pointsX, pointsY, pointsD):transpose(2,1):narrow(1,1,3):clone()
    local dSamples = pointsLifted:clone():pow(2):sum(1):pow(0.5)
    pointsLifted:cdiv(dSamples:repeatTensor(3,1))
    dirSamples:narrow(1,1,3):copy(pointsLifted)
    return dirSamples, dSamples, pixelCoords
end
----------------
local function sampleRays(disparityIm, dispMask, instrinsic, minDisp, numSamples, scaleFactor)
    local scaleFactor = scaleFactor or 1
    local imH, imW
    if(disparityIm:nDimension() > 2) then
        imH = disparityIm:size(2)
        imW = disparityIm:size(3)
    else
        imH = disparityIm:size(1)
        imW = disparityIm:size(2)
    end
    local disparityIm = disparityIm:clone():reshape(imH,imW)
    local dispMask = dispMask:clone():reshape(imH,imW)
    dispMask = torch.cmul(dispMask, torch.gt(disparityIm, minDisp):typeAs(dispMask))
    
    local dirSamples = torch.Tensor(4,numSamples):fill(0)
    local pointsX = torch.Tensor(numSamples,1):fill(0)
    local pointsY = torch.Tensor(numSamples,1):fill(0)
    local pointsD = torch.Tensor(numSamples,1):fill(0)
    
    local ns = 0
    while(ns < numSamples) do
        local x = torch.random(1,imW)
        local y = torch.random(1,imH)
        if(dispMask[y][x]>0) then
            ns = ns+1
            pointsX[ns] = x*scaleFactor
            pointsY[ns] = y*scaleFactor
            pointsD[ns] = disparityIm[y][x]
        end
    end
    local pixelCoords = torch.cat(pointsY, pointsX)/scaleFactor
    local pointsLifted = pixelTo3Dcoord(instrinsic, pointsX, pointsY, pointsD):transpose(2,1):narrow(1,1,3):clone()
    local dSamples = pointsLifted:clone():pow(2):sum(1):pow(0.5)
    pointsLifted:cdiv(dSamples:repeatTensor(3,1))
    dirSamples:narrow(1,1,3):copy(pointsLifted)
    return dirSamples, dSamples, pixelCoords
end
----------------
local function sampleRaysColor(colorIm, instrinsic, numSamples, scaleFactor)
    local scaleFactor = scaleFactor or 1
    local imH, imW
    imH = colorIm:size(2)
    imW = colorIm:size(3)
    
    local dirSamples = torch.Tensor(4,numSamples):fill(0)
    local pointsX = torch.Tensor(numSamples,1):fill(0)
    local pointsY = torch.Tensor(numSamples,1):fill(0)
    local pointsColor = torch.Tensor(numSamples,3):fill(0)
    
    local ns = 0
    while(ns < numSamples) do
        local x = torch.random(1,imW)
        local y = torch.random(1,imH)
        ns = ns+1
        pointsX[ns] = x*scaleFactor
        pointsY[ns] = y*scaleFactor
        pointsColor[ns]:copy(colorIm[{{},y,x}])
    end
    local pixelCoords = torch.cat(pointsY, pointsX)/scaleFactor
    local pointsLifted = pixelTo3Dcoord(instrinsic, pointsX, pointsY, pointsX:clone():fill(1)):transpose(2,1):narrow(1,1,3):clone()
    local dSamples = pointsLifted:clone():pow(2):sum(1):pow(0.5)
    pointsLifted:cdiv(dSamples:repeatTensor(3,1))
    dirSamples:narrow(1,1,3):copy(pointsLifted)
    return dirSamples, pointsColor, pixelCoords
end
----------------
local function samplePascalOrthographicRays(maskIm, bbox, camScale, camRot, camTrans, numSamples, boxPadding)
    local rotationCorrection = torch.Tensor({
            1,0,0,
            0,0,1,
            0,-1,0
        }):reshape(3,3) --some difference in coordinate frame due to blender
    local camRot = torch.mm(camRot,rotationCorrection)
    local Rinv = camRot:clone():transpose(2,1)
    local imH = maskIm:size(1)
    local imW = maskIm:size(2)
    local boxPadding = boxPadding or 1
    
    local maxHW = torch.Tensor({bbox[3],bbox[4]}):max()
    local xRangeMin = torch.round(torch.Tensor({1, bbox[1]-boxPadding*maxHW}):max())
    local yRangeMin = torch.round(torch.Tensor({1, bbox[2]-boxPadding*maxHW}):max())
    
    local yRangeMax = torch.round(torch.Tensor({imH, bbox[2]+(1+boxPadding)*maxHW-1}):min())
    local xRangeMax = torch.round(torch.Tensor({imW, bbox[1]+(1+boxPadding)*maxHW-1}):min())
    
    
    local dirSamples = torch.Tensor(3,numSamples):fill(0)
    local maskSamples = torch.Tensor(numSamples):fill(0)
    
    dirSamples:narrow(1,3,1):fill(1)
    dirSamples = torch.mm(Rinv,dirSamples)
    
    local orgSamples = torch.Tensor(3,numSamples):fill(0)
    
    local ns = 0
    while(ns < numSamples) do
        local x = torch.random(xRangeMin,xRangeMax)
        local y = torch.random(yRangeMin,yRangeMax)
        ns = ns+1
        maskSamples[ns] = maskIm[y][x]
        orgSamples[1][ns] = (1/camScale[1])*(x-camTrans[1])
        orgSamples[2][ns] = (1/camScale[1])*(y-camTrans[2])
    end
    orgSamples = torch.mm(Rinv,orgSamples)
    orgSamples = orgSamples - 100*dirSamples
    return dirSamples:transpose(1,2):clone(), orgSamples:transpose(1,2):clone(), maskSamples
end
----------------
M.loadDisparity = loadDisparity
M.loadCameraMat = loadCameraMat
M.loadMotionMat = loadMotionMat
M.integrateMotion = integrateMotion
M.inverseMotionMat = inverseMotionMat
M.motionMatSeq = motionMatSeq
M.pixelTo3Dcoord = pixelTo3Dcoord
M.imgToPoints = imgToPoints
M.allImgRays = allImgRays
M.sampleRays = sampleRays
M.sampleRaysColor = sampleRaysColor
M.samplePascalOrthographicRays = samplePascalOrthographicRays
return M