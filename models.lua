require 'nn'
require 'ShaveImage'

local M = {}


local function build_conv_block(dim, padding_type)
  local conv_block = nn.Sequential()
  local p = 0
  if padding_type == 'reflect' then
    conv_block:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  elseif padding_type == 'replicate' then
    conv_block:add(nn.SpatialReplicationPadding(1, 1, 1, 1))
  elseif padding_type == 'zero' then
    p = 1
  end
  conv_block:add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1, p, p))
  conv_block:add(nn.SpatialBatchNormalization(dim))
  conv_block:add(nn.ReLU(true))
  if padding_type == 'reflect' then
    conv_block:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
  elseif padding_type == 'replicate' then
    conv_block:add(nn.SpatialReplicationPadding(1, 1, 1, 1))
  end
  conv_block:add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1, p, p))
  conv_block:add(nn.SpatialBatchNormalization(dim))
  return conv_block
end


local function build_res_block(dim, padding_type)
  local conv_block = build_conv_block(dim, padding_type)
  local res_block = nn.Sequential()
  local concat = nn.ConcatTable()
  concat:add(conv_block)  
  if padding_type == 'none' or padding_type == 'reflect-start' then
    concat:add(nn.ShaveImage(2))
  else
    concat:add(nn.Identity())
  end
  res_block:add(concat):add(nn.CAddTable())
  return res_block
end


function M.build_model(opt)
  local arch = opt.arch:split(',')
  local prev_dim = 1
  local model = nn.Sequential()
  
  for i, v in ipairs(arch) do
    local first_char = string.sub(v, 1, 1)
    local layer, next_dim
    local needs_relu = true
    local needs_bn = true
    if first_char == 'c' then
      -- Convolution
      local f = tonumber(string.sub(v, 2, 2)) -- filter size
      local p = (f - 1) / 2 -- padding
      local s = tonumber(string.sub(v, 4, 4)) -- stride
      next_dim = tonumber(string.sub(v, 6))
      if opt.padding_type == 'reflect' then
        model:add(nn.SpatialReflectionPadding(p, p, p, p))
        p = 0
      elseif opt.padding_type == 'replicate' then
        model:add(nn.SpatialReplicationPadding(p, p, p, p))
        p = 0
      elseif padding_type == 'none' then
        p = 0
      end
      layer = nn.SpatialConvolution(prev_dim, next_dim, f, f, s, s, p, p)
    elseif first_char == 'f' then
      -- Full convolution
      local f = tonumber(string.sub(v, 2, 2)) -- filter size
      local p = (f - 1) / 2 -- padding
      local s = tonumber(string.sub(v, 4, 4)) -- stride
      local a = s - 1 -- adjustment
      next_dim = tonumber(string.sub(v, 6))
      layer = nn.SpatialFullConvolution(prev_dim, next_dim,
                                        f, f, s, s, p, p, a, a)
    elseif first_char == 'd' then
      -- Downsampling (strided convolution)
      next_dim = tonumber(string.sub(v, 2))
      layer = nn.SpatialConvolution(prev_dim, next_dim, 3, 3, 2, 2, 1, 1)
    elseif first_char == 'U' then
      -- Nearest-neighbor upsampling
      next_dim = prev_dim
      local scale = tonumber(string.sub(v, 2))
      layer = nn.SpatialUpSamplingNearest(scale)
    elseif first_char == 'u' then
      -- Learned upsampling (strided full-convolution)
      next_dim = tonumber(string.sub(v, 2))
      layer = nn.SpatialFullConvolution(prev_dim, next_dim, 3, 3, 2, 2, 1, 1, 1, 1)
    elseif first_char == 'C' then
      -- Non-residual conv block
      next_dim = tonumber(string.sub(v, 2))
      layer = build_conv_block(next_dim, opt.padding_type)
      needs_bn = false
      needs_relu = true
    elseif first_char == 'R' then
      -- Residual (non-bottleneck) block
      next_dim = tonumber(string.sub(v, 2))
      layer = build_res_block(next_dim, opt.padding_type)
      needs_bn = false
      needs_relu = false
    end
    model:add(layer)
    if i == #arch then
      needs_relu = false
      needs_bn = false
    end
    if needs_bn then
        model:add(nn.SpatialBatchNormalization(next_dim))
    end
    if needs_relu then
      model:add(nn.ReLU(true))
    end

    prev_dim = next_dim
  end

  model:add(nn.Tanh())
  

  return model
end


return M
