torch.manualSeed(1)
require 'cunn'
require 'optim'
local data = dofile('../data/synthetic/shapenetFused.lua')
local netBlocks = dofile('../nnutils/netBlocks.lua')
local netInit = dofile('../nnutils/netInit.lua')
local vUtil = dofile('../utils/visUtils.lua')
-----------------------------
--------parameters-----------
local params = {}
--params.bgVal = 0
params.name = 'shapenetFused'
params.gpu = 1
params.useNoise = 0
params.batchSize = 8
params.imgSizeY = 64
params.imgSizeX = 64
params.synset = 2958343 --chair:3001627, aero:2691156, car:2958343

params.gridSizeX = 32
params.gridSizeY = 32
params.gridSizeZ = 32

params.imsave = 0
params.disp = 0
params.bottleneckSize = 100
params.visIter = 100
params.nConvEncLayers = 5
params.nConvDecLayers = 4
params.nImgs = 5
params.nConvEncChannelsInit = 8
params.numTrainIter = 10000

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end

if params.disp == 0 then params.display = false end
if params.imsave == 0 then params.imsave = false end
if params.useNoise == 0 then params.useNoise = false end
params.visDir = '../cachedir/visualization/' .. params.name
params.snapshotDir = '../cachedir/snapshots/shapenet/' .. params.name
params.imgSize = torch.Tensor({params.imgSizeX, params.imgSizeY})
params.gridSize = torch.Tensor({params.gridSizeX, params.gridSizeY, params.gridSizeZ})
params.synset = '0' .. tostring(params.synset) --to resolve string/number issues in passing bash arguments
params.modelsDataDir = '../cachedir/blenderRenderPreprocess/' .. params.synset .. '/'
params.voxelsDir = '../cachedir/fusionPreprocess'.. (params.useNoise and 'Noise0pt2/' or '/') .. params.synset .. '/nIm_' .. tostring(params.nImgs) .. '/'
print(params)
-----------------------------
-----------------------------
paths.mkdir(params.visDir)
paths.mkdir(params.snapshotDir)
cutorch.setDevice(params.gpu)
local fout = io.open(paths.concat(params.snapshotDir,'log.txt'), 'w')
for k,v in pairs(params) do
    fout:write(string.format('%s : %s\n',tostring(k),tostring(v)))
end
fout:flush()
-----------------------------
----------LossComp-----------
local lossFunc = nn.BCECriterion()
-----------------------------
----------Encoder------------
local encoder, nOutChannels = netBlocks.convEncoderSimple2d(params.nConvEncLayers,params.nConvEncChannelsInit,3,true) --output is nConvEncChannelsInit*pow(2,nConvEncLayers-1) X imgSize/pow(2,nConvEncLayers)
local featSpSize = params.imgSize/torch.pow(2,params.nConvEncLayers)
--print(featSpSize)
local bottleneck = nn.Sequential():add(nn.Reshape(nOutChannels*featSpSize[1]*featSpSize[2],1,1,true))
local nInputCh = nOutChannels*featSpSize[1]*featSpSize[2]
for nLayers=1,2 do --fc for joint reasoning
    bottleneck:add(nn.SpatialConvolution(nInputCh,params.bottleneckSize,1,1)):add(nn.SpatialBatchNormalization(params.bottleneckSize)):add(nn.LeakyReLU(0.2, true))
    nInputCh = params.bottleneckSize
end
encoder:add(bottleneck)
encoder:apply(netInit.weightsInit)
--print(encoder)
---------------------------------
----------World Decoder----------
local featSpSize = params.gridSize/torch.pow(2,params.nConvDecLayers)
local decoder  = nn.Sequential():add(nn.SpatialConvolution(params.bottleneckSize,nOutChannels*featSpSize[1]*featSpSize[2]*featSpSize[3],1,1,1)):add(nn.SpatialBatchNormalization(nOutChannels*featSpSize[1]*featSpSize[2]*featSpSize[3])):add(nn.ReLU(true)):add(nn.Reshape(nOutChannels,featSpSize[1],featSpSize[2],featSpSize[3], true))
decoder:add(netBlocks.convDecoderSimple3d(params.nConvDecLayers,nOutChannels,params.nConvEncChannelsInit,1,true))
decoder:apply(netInit.weightsInit)
-----------------------------
----------Recons-------------
local splitUtil = dofile('../benchmark/synthetic/splits.lua')
local trainModels = splitUtil.getSplit(params.synset)['train']
--print(#trainModels)
local dataLoader = data.dataLoader(params.modelsDataDir, params.voxelsDir, params.batchSize, params.imgSize, params.gridSize, params.nImgs, trainModels)
local netRecons = nn.Sequential():add(encoder):add(decoder)
netRecons = netRecons:cuda()
lossFunc = lossFunc:cuda()
--print(encoder)
--print(decoder)
local err = 0

-- Optimization parameters
local optimState = {
   learningRate = 0.0001,
   beta1 = 0.9,
}

local netParameters, netGradParameters = netRecons:getParameters()
local tm = torch.Timer()
local data_tm = torch.Timer()
local imgs, pred, rays

-- fX required for training
local fx = function(x)
    tm:reset(); tm:resume()
    netGradParameters:zero()
    data_tm:reset(); data_tm:resume()
    imgs, voxelProbs, lossMask = dataLoader:forward()
    data_tm:stop()
    --print('Data loaded')
    
    imgs = imgs:cuda()
    voxelProbs = voxelProbs:cuda()
    lossMask = lossMask:cuda()
    pred = netRecons:forward(imgs)
    err = lossFunc:forward(pred, voxelProbs)
    local gradPred = lossFunc:backward(pred, voxelProbs):clone():cmul(lossMask)
    netRecons:backward(imgs, gradPred)
    tm:stop()
    return err, netGradParameters
end
--print(netRecons)
-----------------------------
----------Training-----------
if(params.display) then disp = require 'display' end
for iter=1,params.numTrainIter do
    print(iter,err)
    --print(('Data/Total time : %f/%f'):format(data_tm:time().real,tm:time().real))
    fout:write(string.format('%d %f\n',iter,err))
    fout:flush()
    if(iter%params.visIter==0) then
        local dispVar = pred:clone()
        if(params.disp == 1) then
            disp.image(imgs, {win=10, title='inputIm'})
            disp.image(dispVar:max(3):squeeze(), {win=1, title='predX'})
            disp.image(dispVar:max(4):squeeze(), {win=2, title='predY'})
            disp.image(dispVar:max(5):squeeze(), {win=3, title='predZ'})
        end
        if(params.imsave == 1) then
            vUtils.imsave(imgs, params.visDir .. '/inputIm'.. iter .. '.png')
            vUtils.imsave(dispVar:max(3):squeeze(), params.visDir.. '/predX' .. iter .. 'step'.. forwIter .. '.png')
            vUtils.imsave(dispVar:max(4):squeeze(), params.visDir.. '/predY' .. iter .. 'step'.. forwIter .. '.png')
            vUtils.imsave(dispVar:max(5):squeeze(), params.visDir.. '/predZ' .. iter .. 'step'.. forwIter .. '.png')
        end
    end
    if(iter%5000)==0 then
        torch.save(params.snapshotDir .. '/iter'.. iter .. '.t7', netRecons)
    end
    optim.adam(fx, netParameters, optimState)
end