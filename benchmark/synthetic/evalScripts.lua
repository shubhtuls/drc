-- sample usage: gpu=1 th synthetic/evalScripts.lua | bash
local params = {}
params.gpu = 1

for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end

local classes = {'aero','car','chair'}
local classToSynset = {chair='3001627', aero='2691156', car='2958343'}
local netNameSuffixes = {'voxels', 'mask_nIm5', 'depth_nIm5', 'depth_nIm5_noise', 'fuse_nIm5', 'fuse_nIm5_noise'}
local flipPreds = {0,1,1,1,0,0} -- the networks trained using the DRC loss predicted emptiness probabilities instead of occupancy probabilities, so we'll flip them before evaluating
local numTrainIters = {10000}
local evalSets = {'val','test'}

for c=1,#classes do
    local class = classes[c]
    local synset = classToSynset[class]
    for nx = 1,#netNameSuffixes do
        local netSuffix = netNameSuffixes[nx]
        for ix=1,#numTrainIters do
            local nIter = numTrainIters[ix]
            for ex=1,#evalSets do
                local cmd = string.format('evalSet=%s numTrainIter=%d flipPred=%d gpu=%d nIm=2 name=%s_%s synset=%s th synthetic/reconstructionBenchmark.lua', evalSets[ex], nIter, flipPreds[nx], params.gpu, class, netSuffix, synset)
                print(cmd)
            end
        end
    end
end