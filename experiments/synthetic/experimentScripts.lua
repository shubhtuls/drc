-- sample usage: class=chair gpu=2 th synthetic/experimentScripts.lua | bash
local params = {}
params.class = 'car' --chair:3001627, aero:2691156, car:2958343
params.disp = 0
params.gpu = 1

for k,v in pairs(params) do params[k] = tonumber(os.getenv(k)) or os.getenv(k) or params[k] end
local classToSynset = {chair='3001627', aero='2691156', car='2958343'}
local synset = classToSynset[params.class]

-- ShapeNet Voxels Experiment
local cmd = string.format('gpu=%d name=%s_voxels disp=%d synset=%s th synthetic/shapenetVoxels.lua',params.gpu,params.class, params.disp, synset)
print(cmd)

-- Mask Experiment
local cmd = string.format('gpu=%d nImgs=5 maskOnly=1 name=%s_mask_nIm5 disp=%d synset=%s th synthetic/shapenet.lua',params.gpu,params.class, params.disp, synset)
print(cmd)

-- Depth Experiment
local cmd = string.format('gpu=%d nImgs=5 maskOnly=0 name=%s_depth_nIm5 disp=%d synset=%s th synthetic/shapenet.lua',params.gpu,params.class, params.disp, synset)
print(cmd)

-- Noisy Depth Experiment
local cmd = string.format('gpu=%d useNoise=1 nImgs=5 maskOnly=0 name=%s_depth_nIm5_noise disp=%d synset=%s th synthetic/shapenet.lua',params.gpu,params.class, params.disp, synset)
print(cmd)

-- Fusion Experiment
local cmd = string.format('gpu=%d name=%s_fuse_nIm5 disp=%d synset=%s th synthetic/shapenetFused.lua',params.gpu,params.class, params.disp, synset)
print(cmd)

-- Fusion Experiment (Noisy Depth)
local cmd = string.format('gpu=%d useNoise=1 name=%s_fuse_nIm5_noise disp=%d synset=%s th synthetic/shapenetFused.lua',params.gpu,params.class, params.disp, synset)
print(cmd)
