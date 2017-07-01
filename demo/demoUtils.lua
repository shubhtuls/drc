local M = {}
----------------------------
local cubeV = torch.Tensor({{0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {0.0, 1.0, 1.0}, {1.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 0.0}, {1.0, 1.0, 1.0}})
cubeV = 2*cubeV - 1
local cubeF = torch.Tensor({{1,  7,  5 }, {1,  3,  7 }, {1,  4,  3 }, {1,  2,  4 }, {3,  8,  7 }, {3,  4,  8 }, {5,  7,  8 }, {5,  8,  6 }, {1,  5,  6 }, {1,  6,  2 }, {2,  6,  8 }, {2,  8,  4}})
------------------------
local function voxelsToMesh(predVol, thresh)
    local vCounter = 1
    local fCounter = 1
    local totPoints = torch.gt(predVol, thresh):sum()
    
    local vAll = cubeV:repeatTensor(totPoints, 1)
    local fAll = cubeF:repeatTensor(totPoints, 1)
    
    local fOffset = torch.repeatTensor(torch.linspace(0,12*totPoints-1,12*totPoints),3,1):transpose(1,2)
    fOffset = torch.floor(fOffset/12)*8
    fAll = fAll + fOffset
    
    for x=1,predVol:size(1) do
        for y=1,predVol:size(2) do
            for z=1,predVol:size(3) do
                if predVol[x][y][z] > thresh then
                    vAll:narrow(1,vCounter,8):narrow(2,1,1):add(x)
                    vAll:narrow(1,vCounter,8):narrow(2,2,1):add(y)
                    vAll:narrow(1,vCounter,8):narrow(2,3,1):add(z)
                    vCounter = vCounter+8
                    fCounter = fCounter+12
                end
            end
        end
    end
    return vAll:mul(1/32):add(-0.5), fAll
end

------------------------
local function writeObj(meshFile, vertices, faces)
    local mtlFile = string.split(meshFile, '.obj$')[1] .. '.mtl'
    local meshFileHandle = io.open(meshFile, 'w')
    local cVal = 0.2
    local mtlFileHandle = io.open(mtlFile, 'w')
    mtlFileHandle:write('newmtl m0\n')
    mtlFileHandle:write(string.format('Ka %f %f %f\n', cVal, cVal, cVal))
    mtlFileHandle:write(string.format('Kd %f %f %f\n', cVal, cVal, cVal))
    mtlFileHandle:write('Ks 1 1 1\n')
    mtlFileHandle:write(string.format('illum %d\n',9))
    mtlFileHandle:close()
    
    mtlFile = mtlFile:split('/')
    mtlFile = mtlFile[#mtlFile]
    meshFileHandle:write(string.format('mtllib %s\n usemtl m0\n',mtlFile))
    for vx = 1,vertices:size(1) do
        meshFileHandle:write(string.format('v %f %f %f\n', vertices[vx][1], vertices[vx][2], vertices[vx][3]))
    end
    for fx = 1,faces:size(1) do
        meshFileHandle:write(string.format('f %d %d %d\n', faces[fx][1], faces[fx][2], faces[fx][3]))
    end
    meshFileHandle:close()
end
------------------------
local function renderMesh(blenderExec, blendFile, meshFile, pngFile)
    local command = string.format('bash renderer/render.sh %s %s %s %s',blenderExec, blendFile, meshFile, pngFile)
    --print(command)
    os.execute(command)
end
------------------------
local function RGBAtoRGB(img)
    local alphaMask = img[4]:repeatTensor(3,1,1)
    local imgOut = torch.cmul(img:clone():narrow(1,1,3),alphaMask) + 1 - alphaMask
    return imgOut
end
------------------------
M.voxelsToMesh = voxelsToMesh
M.writeObj = writeObj
M.renderMesh = renderMesh
M.RGBAtoRGB = RGBAtoRGB
return M