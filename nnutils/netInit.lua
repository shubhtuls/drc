local M = {}

function M.weightsInit(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

function M.weightsZeroInit(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:fill(0.0)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:fill(0) end
      if m.bias then m.bias:fill(0) end
   end
end

function M.stnInit(m, paramType)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      -- handle scale separately
      if paramType == 'scale' then
          m.bias:fill(1)
      end
      if paramType == 'translation' then
          m.bias:fill(0)
      end
      if paramType == 'rotation' then
          m.bias:fill(0)
      end
      if paramType == 'affine' then
          m.bias:fill(0)
          m.bias[1] = 1  --scaleX
          m.bias[5] = 1  --scaleY
      end
    end
end

return M