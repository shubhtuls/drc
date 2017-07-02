local params = {}
params.gpu = 1

for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end

local classes = {'aero','car','chair'}
local classToSynset = {chair='3001627', aero='2691156', car='2958343'}

-- the baseline trained on only SNet actually performs worse on real data when trained for longer
-- so, we'll evaluate it at 10000 iterations as well since it performs best then (this is the number reported in the paper)
local netNameSuffixes = {'p3d', 'SNet', 'SNet', 'p3dSNetCombined'}
local flipPreds = {1, 1, 1, 1}
local numTrainIters = {20000, 10000, 25000, 20000}
local evalSets = {'val','train'}

for c=1,#classes do
    local class = classes[c]
    local synset = classToSynset[class]
    for ex=1,#evalSets do
        for nx = 1,#netNameSuffixes do
            local netSuffix = netNameSuffixes[nx]
            local nIter = numTrainIters[nx]
            local cmd = string.format('evalSet=%s numTrainIter=%d flipPred=%d gpu=%d name=%s synset=%s th pascal/reconstructionBenchmark.lua', evalSets[ex], nIter, flipPreds[nx], params.gpu, netSuffix, synset)
            print(cmd)
        end

        local cmd = string.format('evalSet=%s synset=%s th pascal/cs3dBenchmark.lua', evalSets[ex], synset)
        print(cmd)
    end
end