local M = {}
local matio = require 'matio'
matio.use_lua_strings = true
-------------------------------
-------------------------------
local function BuildArray(...)
  local arr = {}
  for v in ... do
    arr[#arr + 1] = v
  end
  return arr
end
-------------------------------
-------------------------------
local function getSplit(synset, saveDir, modelsDir)
    --paths.mkdir(saveDir)
    local saveDir = saveDir or ('../cachedir/splits/shapenet/')
    local modelsDir = modelsDir or paths.concat('../cachedir/blenderRenderPreprocess/', synset)
    local saveFile = paths.concat(saveDir, synset .. '.file')
    if(paths.filep(saveFile)) then
        local spFile = torch.DiskFile(saveFile, 'r')
        --local splits = matio.load(saveFile, {'train','val','test'})
        local splits = spFile:readObject()
        spFile:close()
        return splits
    end
    local modelNames = BuildArray(paths.files(modelsDir,'...'))
    local nModels =  #modelNames
    
    local randgen = torch.Generator()
    torch.manualSeed(randgen, 0)
    local rperm = torch.randperm(randgen, nModels)
    
    local train = {}
    local val = {}
    local test = {}
    
    local ix = 1
    local ntrain, nval, ntest
    
    ntrain = 1
    while(ix < 0.7*nModels) do
        train[ntrain] = modelNames[rperm[ix]]
        ntrain = ntrain + 1
        ix = ix + 1
    end
    
    nval = 1
    while(ix < 0.8*nModels) do
        val[nval] = modelNames[rperm[ix]]
        nval = nval + 1
        ix = ix + 1
    end
    
    ntest = 1
    while(ix <= nModels) do
        test[ntest] = modelNames[rperm[ix]]
        ntest = ntest + 1
        ix = ix + 1
    end
    
    local splits = {}
    print(saveFile)
    --matio.save(saveFile, {train=train,val=val,test=test})
    splits.train = train
    splits.val = val
    splits.test = test
    paths.mkdir(saveDir)
    local spFile = torch.DiskFile(saveFile, 'w')
    spFile:writeObject(splits)
    spFile:close()
    
    return splits
end
-------------------------------
-------------------------------
M.getSplit = getSplit
return M