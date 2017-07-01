-- example  usage: cd preprocess; useNoise=0 synset=3001627 th synthetic/fusion/shapenetFusion.lua
local params = {}
params.synset = 2958343 --chair:3001627, aero:2691156, car:2958343
params.nImgs = 5
params.useNoise = 0
for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end

if(params.useNoise == 0) then params.useNoise = false end
local nImgs = params.nImgs
local sf = dofile('../preprocess/synthetic/fusion/fuseShapenetViews.lua')
local gUtils = dofile('../rayUtils/grid.lua')
local gridBound = 0.5 --parameter fixed according to shapenet models' size
params.synset = '0' .. tostring(params.synset) --to resolve string/number issues in passing bash arguments
params.modelsDataDir = '../cachedir/blenderRenderPreprocess/' .. params.synset .. '/'
params.fusedModelsDir = '../cachedir/fusionPreprocess'.. (params.useNoise and 'Noise0pt2/' or '/') .. params.synset .. '/nIm_' .. tostring(nImgs) .. '/'
paths.mkdir(params.fusedModelsDir)

params.gridSizeX = 32
params.gridSizeY = 32
params.gridSizeZ = 32

local minBounds = torch.Tensor({-1,-1,-1})*gridBound
local maxBounds = torch.Tensor({1,1,1})*gridBound
local step = torch.Tensor({2/params.gridSizeX,2/params.gridSizeY,2/params.gridSizeZ})*gridBound
local grid = gUtils.gridNd(minBounds, maxBounds, step)
local fusion = sf.fusion(params.modelsDataDir, nImgs, false, grid, params.useNoise)
fusion:fuseAll(params.fusedModelsDir)