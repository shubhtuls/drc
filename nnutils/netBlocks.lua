require 'nn'
require 'nngraph'
local M = {}

function M.convEncoderSimple1d(nLayers, nChannelsInit, nInputChannels, useBn)
    local nInputChannels = nInputChannels or 3
    local nChannelsInit = nChannelsInit or 8
    local useBn = useBn~=false and true
    local nOutputChannels = nChannelsInit
    local encoder = nn.Sequential()
    
    for l=1,nLayers do
        encoder:add(nn.TemporalConvolution(nInputChannels, nOutputChannels, 3, 1))
        encoder:add(nn.LeakyReLU(0.2, true))
        encoder:add(nn.TemporalMaxPooling(2, 2))
        nInputChannels = nOutputChannels
        nOutputChannels = nOutputChannels*2
    end
    return encoder, nOutputChannels/2 -- division by two offsets the mutiplication in last iteration
end

function M.convEncoderSimple2d(nLayers, nChannelsInit, nInputChannels, useBn)
    local nInputChannels = nInputChannels or 3
    local nChannelsInit = nChannelsInit or 8
    local useBn = useBn~=false and true
    local nOutputChannels = nChannelsInit
    local encoder = nn.Sequential()
    
    for l=1,nLayers do
        encoder:add(nn.SpatialConvolution(nInputChannels, nOutputChannels, 3, 3, 1, 1, 1, 1))
        if useBn then encoder:add(nn.SpatialBatchNormalization(nOutputChannels)) end
        encoder:add(nn.LeakyReLU(0.2, true))
        encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2))
        nInputChannels = nOutputChannels
        nOutputChannels = nOutputChannels*2
    end
    return encoder, nOutputChannels/2 -- division by two offsets the mutiplication in last iteration
end

function M.convEncoderComplex2d(nLayers, nChannelsInit, nInputChannels, useBn)
    local nInputChannels = nInputChannels or 3
    local nChannelsInit = nChannelsInit or 8
    local useBn = useBn~=false and true
    local nOutputChannels = nChannelsInit
    local encoder = nn.Sequential()
    
    for l=1,nLayers do
        encoder:add(nn.SpatialConvolution(nInputChannels, nOutputChannels, 3, 3, 1, 1, 1, 1))
        if useBn then encoder:add(nn.SpatialBatchNormalization(nOutputChannels)) end
        encoder:add(nn.LeakyReLU(0.2, true))
        
        encoder:add(nn.SpatialConvolution(nOutputChannels, nOutputChannels, 3, 3, 1, 1, 1, 1))
        if useBn then encoder:add(nn.SpatialBatchNormalization(nOutputChannels)) end
        encoder:add(nn.LeakyReLU(0.2, true))
        
        encoder:add(nn.SpatialMaxPooling(2, 2, 2, 2))
        nInputChannels = nOutputChannels
        nOutputChannels = nOutputChannels*2
    end
    return encoder, nOutputChannels/2 -- division by two offsets the mutiplication in last iteration
end

function M.convDecoderSimple2d(nLayers, nInputChannels, ndf, nFinalChannels, useBn)
    --adds nLayers deconv layers + 1 conv layer
    local nFinalChannels = nFinalChannels or 3
    local ndf = ndf or 8 --channels in penultimate layer
    local useBn = useBn~=false and true
    local nOutputChannels = ndf*torch.pow(2,nLayers-1)
    local decoder = nn.Sequential()
    for l=1,nLayers do
        decoder:add(nn.SpatialFullConvolution(nInputChannels, nOutputChannels, 4, 4, 2, 2, 1, 1))
        if useBn then decoder:add(nn.SpatialBatchNormalization(nOutputChannels)) end
        decoder:add(nn.ReLU(true))
        nInputChannels = nOutputChannels
        nOutputChannels = nOutputChannels/2
    end
    decoder:add(nn.SpatialConvolution(ndf, nFinalChannels, 3, 3, 1, 1, 1, 1))
    decoder:add(nn.Tanh()):add(nn.AddConstant(1)):add(nn.MulConstant(0.5))
    return decoder
end

function M.convDecoderComplex2d(nLayers, nInputChannels, ndf, nFinalChannels, useBn)
    --adds nLayers deconv-conv layers + 1 final conv layer
    local nFinalChannels = nFinalChannels or 3
    local ndf = ndf or 8 --channels in penultimate layer
    local useBn = useBn~=false and true
    local nOutputChannels = ndf*torch.pow(2,nLayers-1)
    local decoder = nn.Sequential()
    for l=1,nLayers do
        decoder:add(nn.SpatialFullConvolution(nInputChannels, nOutputChannels, 4, 4, 2, 2, 1, 1))
        if useBn then decoder:add(nn.SpatialBatchNormalization(nOutputChannels)) end
        decoder:add(nn.LeakyReLU(0.2, true))
        
        decoder:add(nn.SpatialConvolution(nOutputChannels, nOutputChannels, 3, 3, 1, 1, 1, 1))
        if useBn then decoder:add(nn.SpatialBatchNormalization(nOutputChannels)) end
        decoder:add(nn.ReLU(true))
        nInputChannels = nOutputChannels
        nOutputChannels = nOutputChannels/2
    end
    decoder:add(nn.SpatialConvolution(ndf, nFinalChannels, 3, 3, 1, 1, 1, 1))
    decoder:add(nn.Tanh()):add(nn.AddConstant(1)):add(nn.MulConstant(0.5))
    return decoder
end

function M.convDecoderSimple3d(nLayers, nInputChannels, ndf, nFinalChannels, useBn, normalizeOut)
    --adds nLayers deconv layers + 1 conv layer
    local nFinalChannels = nFinalChannels or 1
    local ndf = ndf or 8 --channels in penultimate layer
    local useBn = useBn~=false and true
    local normalizeOut = normalizeOut~=false and true
    local nOutputChannels = ndf*torch.pow(2,nLayers-1)
    local decoder = nn.Sequential()
    for l=1,nLayers do
        decoder:add(nn.VolumetricFullConvolution(nInputChannels, nOutputChannels, 4, 4, 4, 2, 2, 2, 1, 1, 1))
        if useBn then decoder:add(nn.VolumetricBatchNormalization(nOutputChannels)) end
        decoder:add(nn.ReLU(true))
        nInputChannels = nOutputChannels
        nOutputChannels = nOutputChannels/2
    end
    decoder:add(nn.VolumetricConvolution(ndf, nFinalChannels, 3, 3, 3, 1, 1, 1, 1, 1, 1))
    if(normalizeOut) then
        decoder:add(nn.Tanh()):add(nn.AddConstant(1)):add(nn.MulConstant(0.5))
    end
    return decoder
end

function M.VolumetricSoftMax(nC)
    -- input is B X C X H X W X D, output is also B X C X H X W X D but normalized across C
    local inpPred = -nn.Identity()
    local shift = inpPred - nn.Max(2) - nn.Unsqueeze(2) - nn.Replicate(nC,2)
    local shiftedInp = {inpPred, shift} - nn.CSubTable()
    local expInp = shiftedInp - nn.Exp()
    local denom = expInp - nn.Sum(2) - nn.Unsqueeze(2) - nn.Replicate(nC,2)
    local out = {expInp, denom} - nn.CDivTable()
    
    local gmod = nn.gModule({inpPred}, {out})
    return gmod
end

return M