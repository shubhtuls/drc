local bgImgsDir = '/home/eecs/shubhtuls/cachedir/drc/SUN2012pascalformat/JPEGImages/'
-- directory for copying random background textures from, downloaded from http://groups.csail.mit.edu/vision/SUN/releases/SUN2012pascalformat.tar.gz

-- Shapenet 3D Only Training
-- disp=0 gpu=1 useResNet=1 batchSizeShapenet=16 shapenetWeight=1 pascalWeight=0 name=SNet th pascal/pascal.lua

-- Pascal Only Training
-- disp=0 gpu=1 useResNet=1 batchSizePascal=16 shapenetWeight=0 bgWtPascal=0.2 pascalWeight=1 name=p3d th pascal/pascal.lua

torch.manualSeed(1)
require 'cunn'
require 'optim'
require 'cudnn'

local gUtils = dofile('../rayUtils/grid.lua')
local netBlocks = dofile('../nnutils/netBlocks.lua')
local rp = dofile('../rayUtils/rpgeomWrapper.lua')
local netInit = dofile('../nnutils/netInit.lua')
local vUtil = dofile('../utils/visUtils.lua')
local cropUtil = dofile('../utils/cropUtils.lua')
local splitUtil = dofile('../benchmark/synthetic/splits.lua')
-----------------------------
--------parameters-----------
local gridBoundShapenet = 0.5 --parameter fixed according to shapenet models' size
local gridBoundPascal = 0.5 --parameter fixed according to shapenet models' size
local bgDepth = 10.0 --parameter fixed according to rendering used

local params = {}
params.name = 'pascalDebug'
params.gpu = 1
params.useImagenet = 1
params.learningRate = 0.0001
params.batchSizeShapenet = 16
params.batchSizePascal = 16
params.pascalWeight = 1
params.shapenetWeight = 1
params.nImgs = 5
params.imgSizeY = 64
params.imgSizeX = 64
params.bgWtPascal = 0.2
params.useResNet = 1
params.synset = 1 --chair:3001627, aero:2691156, car:2958343, all:1

params.gridSizeX = 32
params.gridSizeY = 32
params.gridSizeZ = 32

params.imsave = 0
params.disp = 0
params.maskOnly = 0
params.nRaysTot = 3000
params.bottleneckSize = 100
params.visIter = 100
params.nConvEncLayers = 5
params.nFcLayers = 2
params.nConvDecLayers = 4
params.nConvEncChannelsInit = 8
params.numTrainIter = 25000

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end

if params.disp == 0 then params.display = false end
if params.maskOnly == 0 then params.maskOnly = false end
if params.imsave == 0 then params.imsave = false end
if params.useImagenet == 0 then params.useImagenet = false end
if params.useResNet == 0 then params.useResNet = false end

params.visDir = '../cachedir/visualization/' .. params.name
params.snapshotDir = '../cachedir/snapshots/pascal/' .. params.name
params.imgSize = torch.Tensor({params.imgSizeX, params.imgSizeY})
params.gridSize = torch.Tensor({params.gridSizeX, params.gridSizeY, params.gridSizeZ})
params.synset = '0' .. tostring(params.synset) --to resolve string/number issues in passing bash arguments
local synsetToClasses = {
    s_01 = {'chair','aeroplane','car'},
    s_03001627 = {'chair'},
    s_02691156 = {'aeroplane'},
    s_02958343 = {'car'}
}
params.classes = synsetToClasses['s_' .. params.synset]

params.modelsDataDir = '../cachedir/blenderRenderPreprocess/'
params.voxelsDir = '../cachedir/shapenet/modelVoxels/'
print(params)
if(params.synset == '01') then
    params.synsets = {'03001627','02691156','02958343'}
else
    params.synsets = {params.synset}
end
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
--------------------------------------
----------LossComp Shapenet-----------
local lossFuncShapenet = nn.BCECriterion()
lossFuncShapenet = lossFuncShapenet:float()

local minBoundsPascal = torch.Tensor({-1,-1,-1})*gridBoundPascal
local maxBoundsPascal = torch.Tensor({1,1,1})*gridBoundPascal
local stepPascal = torch.Tensor({2/params.gridSizeX,2/params.gridSizeY,2/params.gridSizeZ})*gridBoundPascal
local gridPascal = gUtils.gridNd(minBoundsPascal, maxBoundsPascal, stepPascal)
local lossFuncPascal = rp.rayPotential(gridPascal, bgDepth, true)
-----------------------------
----------Encoder------------

local encoder, nOutChannels
if(not params.useResNet) then
    encoder, nOutChannels = netBlocks.convEncoderSimple2d(params.nConvEncLayers,params.nConvEncChannelsInit,3,true) --output is nConvEncChannelsInit*pow(2,nConvEncLayers-1) X imgSize/pow(2,nConvEncLayers)
    local featSpSize = params.imgSize/torch.pow(2,params.nConvEncLayers)
    --print(featSpSize)
    local bottleneck = nn.Sequential():add(nn.Reshape(nOutChannels*featSpSize[1]*featSpSize[2],1,1,true))
    local nInputCh = nOutChannels*featSpSize[1]*featSpSize[2]
    for nLayers=1,params.nFcLayers do --fc for joint reasoning
        bottleneck:add(nn.SpatialConvolution(nInputCh,params.bottleneckSize,1,1)):add(nn.SpatialBatchNormalization(params.bottleneckSize)):add(nn.LeakyReLU(0.2, true))
        nInputCh = params.bottleneckSize
    end
    encoder:add(bottleneck)
    encoder:apply(netInit.weightsInit)
else
    encoder = nn.Sequential()
    -- downloaded from https://github.com/facebook/fb.resnet.torch/tree/master/pretrained
    local resNet = torch.load('../cachedir/initModels/resnet-18.t7')
    for mx=1,8 do
        encoder:add(resNet.modules[mx])
    end
    nOutChannels = 512
    local featSpSize = params.imgSize/32
    local bottleneck = nn.Sequential():add(nn.Reshape(nOutChannels*featSpSize[1]*featSpSize[2],1,1,true))
    local nInputCh = nOutChannels*featSpSize[1]*featSpSize[2]
    for nLayers=1,2 do --fc for joint reasoning
        bottleneck:add(nn.SpatialConvolution(nInputCh,params.bottleneckSize,1,1)):add(nn.SpatialBatchNormalization(params.bottleneckSize)):add(nn.LeakyReLU(0.2, true))
        nInputCh = params.bottleneckSize
    end
    bottleneck:apply(netInit.weightsInit)
    encoder:add(bottleneck)
end

---------------------------------
----------World Decoder----------
local featSpSize = params.gridSize/torch.pow(2,params.nConvDecLayers)
local decoder  = nn.Sequential():add(nn.SpatialConvolution(params.bottleneckSize,nOutChannels*featSpSize[1]*featSpSize[2]*featSpSize[3],1,1,1)):add(nn.SpatialBatchNormalization(nOutChannels*featSpSize[1]*featSpSize[2]*featSpSize[3])):add(nn.ReLU(true)):add(nn.Reshape(nOutChannels,featSpSize[1],featSpSize[2],featSpSize[3],true))
decoder:add(netBlocks.convDecoderSimple3d(params.nConvDecLayers,nOutChannels,params.nConvEncChannelsInit,1,true))
decoder:apply(netInit.weightsInit)

-----------------------------
------Shapenet Data Layer----
local splitUtil = dofile('../benchmark/synthetic/splits.lua')
local trainModels = {}
for s=1,#params.synsets do
    local trainModels_s = splitUtil.getSplit(params.synsets[s])['train']    
    for m=1,#trainModels_s do
        trainModels[#trainModels+1] = params.synsets[s] .. '/' .. trainModels_s[m]
    end
end
local dataShapenet = dofile('../data/synthetic/shapenetVoxels.lua')
local dataLoaderShapenet = dataShapenet.dataLoader(params.modelsDataDir, params.voxelsDir, params.batchSizeShapenet, params.imgSize, params.gridSize, trainModels)

local function BuildArray(...)
  local arr = {}
  for v in ... do
    arr[#arr + 1] = v
  end
  return arr
end

local bgImgsList = BuildArray(paths.files(bgImgsDir,'.jpg'))
for ix=1,#bgImgsList do
    bgImgsList[ix] = paths.concat(bgImgsDir, bgImgsList[ix])
end
dataLoaderShapenet.bgImgsList = bgImgsList
----------------------------
-------Pascal Data Layer----
local nRaysTot = params.nRaysTot
local dataPascal = dofile('../data/pascal/pascalOrthographic.lua')
local pascalStatesDir = '../cachedir/pascal/camera/'
local pascalModelNames = {}

for c = 1,#params.classes do
    local pascalModelNames_c = BuildArray(paths.files(paths.concat(pascalStatesDir, params.classes[c], 'train') ,'.mat'))
    for ix = 1,#pascalModelNames_c do
        pascalModelNames[#pascalModelNames+1] = params.classes[c] .. '/train/' .. pascalModelNames_c[ix]
    end
    if(params.useImagenet) then
        local imagenetModelNames_c = BuildArray(paths.files(paths.concat(pascalStatesDir, params.classes[c], 'imagenet') ,'.mat'))
        for ix = 1,#imagenetModelNames_c do
            pascalModelNames[#pascalModelNames+1] = params.classes[c] .. '/imagenet/' .. imagenetModelNames_c[ix]
        end
    end
end
local dataLoaderPascal = dataPascal.dataLoader(pascalStatesDir, params.batchSizePascal, nRaysTot, params.imgSize, pascalModelNames)

-----------------------------
----------Recons-------------
local netRecons = nn.Sequential():add(encoder):add(decoder)
netRecons = netRecons:cuda()
local err = 0
local errShapenet = 0
local errPascal = 0

-- Optimization parameters
local optimState = {
   learningRate = params.learningRate,
   beta1 = 0.9,
}

local netParameters, netGradParameters = netRecons:getParameters()
local imgsShapenet, predShapenet, raysShapenet, bgLabelShapenet, gtShapenet
local imgsPascal, predPascal, raysPascal
local imgsAll, predAll

--local loss_tm = torch.Timer()
local data_tm_snet = torch.Timer(); data_tm_snet:stop()
local data_tm_pascal = torch.Timer(); data_tm_pascal:stop()
local tot_tm = torch.Timer(); tot_tm:stop()

-- fX required for training
local fxSeparate = function(x)
    netGradParameters:zero()
    tot_tm:reset(); tot_tm:resume()

    if(params.shapenetWeight > 0) then
        data_tm_snet:reset(); data_tm_snet:resume()
        imgsShapenet, gtShapenet = dataLoaderShapenet:forward()
        gtShapenet = (1-gtShapenet):float()
        imgsShapenet = cropUtil.resnetPreprocess(imgsShapenet)        
        data_tm_snet:stop()

        imgsShapenet = imgsShapenet:cuda()
        predShapenet = netRecons:forward(imgsShapenet):clone():float()
        errShapenet = lossFuncShapenet:forward(predShapenet, gtShapenet)
        local gradPredShapenet = lossFuncShapenet:backward(predShapenet, gtShapenet):cuda()
        netRecons:backward(imgsShapenet, params.shapenetWeight*gradPredShapenet)
    end

    if(params.pascalWeight > 0) then
        data_tm_pascal:reset(); data_tm_pascal:resume()
        imgsPascal, raysPascal, bgLabelPascal = unpack(dataLoaderPascal:forward())
        data_tm_pascal:stop()
        raysPascal[4] = params.bgWtPascal*bgLabelPascal + (1-bgLabelPascal) --higher weight to fg rays
        imgsPascal = cropUtil.resnetPreprocess(imgsPascal)
        imgsPascal = imgsPascal:cuda()
        predPascal = netRecons:forward(imgsPascal):clone():float()
        errPascal = lossFuncPascal:forward(predPascal, raysPascal):mean()
        local gradPredPascal = lossFuncPascal:backward(predPascal, raysPascal):cuda()*(1/params.batchSizePascal)*(1/nRaysTot)
        netRecons:backward(imgsPascal, params.pascalWeight*gradPredPascal)
    end

    if (params.pascalWeight > 0) and (params.shapenetWeight > 0) then
        imgsAll = torch.cat(imgsShapenet, imgsPascal, 1)
        predAll = torch.cat(predShapenet, predPascal, 1)
    elseif (params.pascalWeight == 0) and (params.shapenetWeight > 0) then
        imgsAll = imgsShapenet
        predAll = predShapenet
    elseif (params.pascalWeight > 0) and (params.shapenetWeight == 0) then
        imgsAll = imgsPascal
        predAll = predPascal
    end

    tot_tm:stop()

    err = params.shapenetWeight*errShapenet + params.pascalWeight*errPascal
    return err, netGradParameters
end

-----------------------------
----------Training-----------
if(params.disp) then disp = require 'display' end
for iter=1,params.numTrainIter do
--for iter=1,1 do
    print(iter, err, errPascal, errShapenet)
    fout:write(string.format('%d %f\n',iter,err))
    fout:flush()
    if(iter%params.visIter==0) then
        local imgs, pred

        imgs = cropUtil.undoResnetPreprocess(imgsAll)
        pred = predAll

        local dispVar = 1-pred:clone()
        if(params.disp == 1) then
            disp.image(imgs, {win=10, title='inputIm'})
            disp.image(dispVar:max(3):squeeze(), {win=1, title='predX'})
            disp.image(dispVar:max(4):squeeze(), {win=2, title='predY'})
            disp.image(dispVar:max(5):squeeze(), {win=3, title='predZ'})
        end
        if(params.imsave == 1) then
            vUtils.imsave(imgs, params.visDir .. '/inputIm'.. iter .. '.png')
            vUtils.imsave(dispVar:max(3):squeeze(), params.visDir.. '/predX' .. iter .. '.png')
            vUtils.imsave(dispVar:max(4):squeeze(), params.visDir.. '/predY' .. iter .. '.png')
            vUtils.imsave(dispVar:max(5):squeeze(), params.visDir.. '/predZ' .. iter .. '.png')
        end
    end
    if(iter%5000)==0 then
        torch.save(params.snapshotDir .. '/iter'.. iter .. '.t7', netRecons)
    end
    optim.adam(fxSeparate, netParameters, optimState)
end
