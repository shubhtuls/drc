require 'image'
local transforms = dofile('../utils/transforms.lua')
local M = {}
----------
local function cropImg(img, bbox, paddingFrac, jitterFrac)
    local jitterFrac = jitterFrac or 0
    local paddingFrac = paddingFrac or 0
    if(jitterFrac == 0 and paddingFrac == 0) then return image.crop(outImg, bbox[1]-1, bbox[2]-1, bbox[3], bbox[4]) end
    local H_box = bbox[4] - bbox[2] + 1
    local W_box = bbox[3] - bbox[1] + 1
    local H = img:size(2)
    local W = img:size(3)
    
    local Xmin, Ymin, Xmax, Ymax
    Xmin = bbox[1] + torch.uniform(-jitterFrac,jitterFrac)*W_box - paddingFrac*W_box
    Xmax = bbox[3] + torch.uniform(-jitterFrac,jitterFrac)*W_box + paddingFrac*W_box
    
    Ymin = bbox[2] + torch.uniform(-jitterFrac,jitterFrac)*H_box - paddingFrac*H_box
    Ymax = bbox[4] + torch.uniform(-jitterFrac,jitterFrac)*H_box + paddingFrac*H_box
    
    if(Xmin < 1) then Xmin = 1 end
    if(Ymin < 1) then Ymin = 1 end
    if(Xmax > W) then Xmax = W end
    if(Ymax > H) then Ymax = H end
    
    return image.crop(img, Xmin-1, Ymin-1, Xmax, Ymax)
end
----------
local function computeBbox(fgMask)
    local H = fgMask:size(1)
    local W = fgMask:size(2)
    
    local occX = torch.gt(fgMask:max(1), 0):double():cmul(torch.linspace(1,W,W)) - torch.eq(fgMask:max(1), 0):double()*W -- (-W in unocc columns)
    local occY = torch.gt(fgMask:max(2), 0):double():cmul(torch.linspace(1,H,H)) - torch.eq(fgMask:max(2), 0):double()*H -- (-H in unocc columns)
    
    local xMax = occX:max()
    local xMin = occX:clone():abs():min()
    
    local yMax = occY:max()
    local yMin = occY:clone():abs():min()
    
    return torch.Tensor({xMin, yMin, xMax, yMax})
end
----------
local function blendBg(fgImg, bgImgsList)
    local fgMask = torch.gt(fgImg[4],0):double()
    if(fgMask:sum() <= 0) then return fgImg:narrow(1,1,3):clone() end
    local bgId = bgImgsList[torch.random(1,#bgImgsList)]
    local bgImg = image.load(bgId)
    
    local maxHW_fg = fgMask:size(1)
    if(maxHW_fg < fgMask:size(2)) then maxHW_fg = fgMask:size(2) end
    
    local minHW_bg = bgImg:size(2)
    if(minHW_bg > bgImg:size(3)) then minHW_bg = bgImg:size(3) end
    
    if(minHW_bg < maxHW_fg) then
        local rsz_ratio = maxHW_fg/minHW_bg
        bgImg = image.scale(bgImg, torch.ceil(bgImg:size(2)*rsz_ratio), torch.ceil(bgImg:size(3)*rsz_ratio))
    end
    
    local initY = torch.random(0, bgImg:size(2) - fgImg:size(2))
    local initX = torch.random(0, bgImg:size(3) - fgImg:size(3))
    
    bgImg = image.crop(bgImg, initX, initY, initX + fgImg:size(3), initY + fgImg:size(2))
    if(bgImg:size(1) == 1) then bgImg = bgImg:repeatTensor(3,1,1) end
    
    local alphaMask = fgImg[4]:repeatTensor(3,1,1)
    --print(alphaMask:size(), fgImg:size(), bgImg:size())
    local outImg = fgImg:narrow(1,1,3):clone():cmul(alphaMask) + bgImg:clone():cmul(1-alphaMask)
    
    local bbox = computeBbox(fgMask:double())
    --print(bbox, outImg:size())
    --outImg = image.crop(outImg, bbox[1]-1, bbox[2]-1, bbox[3], bbox[4])
    outImg = cropImg(outImg, bbox, 0.15, 0.15)
    
    return outImg
    
end
----------
local function resnetPreprocess(imgs)
    local imgs = imgs:clone()
    local meanstd = {
        mean = { 0.485, 0.456, 0.406 },
       std = { 0.229, 0.224, 0.225 },
    }
    local imgTransform = transforms.ColorNormalize(meanstd)
    if imgs:nDimension() == 3 then
        return imgTransform(imgs)
    end
    
    for b = 1,imgs:size(1) do
        imgs[b]:copy(imgTransform(imgs[b]))
    end
    
    return imgs
end

local function undoResnetPreprocess(imgs)
  local imgs = imgs:clone()
  imgs:narrow(2,1,1):mul(.229):add(.485)
  imgs:narrow(2,2,1):mul(.224):add(.456)
  imgs:narrow(2,3,1):mul(.225):add(.406)
  return imgs
end

----------
M.blendBg = blendBg
M.cropImg = cropImg
M.resnetPreprocess = resnetPreprocess
M.undoResnetPreprocess = undoResnetPreprocess
return M