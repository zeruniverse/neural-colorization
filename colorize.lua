--[[
Deploy the model...

--]]
require 'torch'
require 'nn'
require 'image'
local utils = require 'utils'
require 'ShaveImage'


--[[
Use a trained feedforward model to stylize either a single image or an entire
directory of images.
--]]

local cmd = torch.CmdLine()

-- Model options
cmd:option('-model', 'model.t7')

-- Input / output options
cmd:option('-input_image', '')
cmd:option('-output_image', 'out.png')
cmd:option('-input_dir', '')
cmd:option('-output_dir', '')

-- GPU options
cmd:option('-gpu', -1)
cmd:option('-backend', 'cuda')
cmd:option('-use_cudnn', 1)


local function main()
  local opt = cmd:parse(arg)

  if (opt.input_image == '') and (opt.input_dir == '') then
    error('Must give exactly one of -input_image or -input_dir')
  end

  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)
  local ok, checkpoint = pcall(function() return torch.load(opt.model) end)
  if not ok then
    print('ERROR: Could not load model from ' .. opt.model)
    print('You may need to download the pretrained models by running')
    print('bash download_colorization_model.sh')
    return
  end
  local model = checkpoint.model
  model:evaluate()
  model:type(dtype)
  if use_cudnn then
    cudnn.convert(model, cudnn)
  end

  local function run_image(in_path, out_path)
    local img = image.load(in_path)
    local H, W = img:size(2), img:size(3)

    if img:size(1)>1 then
      img = image.rgb2y(img)
    end

    local img_pre = img:view(1, 1, H, W):type(dtype)
    local uv = model:forward(torch.add(img_pre,-0.5))
    uv = uv:type('torch.DoubleTensor'):view(2,uv:size(3),uv:size(4))
    uv[1]:mul(0.436)
    uv[2]:mul(0.615)
    img_pre = img_pre:type('torch.DoubleTensor'):view(1,H,W)
    uv = image.scale(uv,W,H)
    local img_out = torch.cat(img_pre,uv,1)
    img_out = image.yuv2rgb(img_out)
    
    print('Writing output image to ' .. out_path)
    local out_dir = paths.dirname(out_path)
    if not path.isdir(out_dir) then
      paths.mkdir(out_dir)
    end
    image.save(out_path, img_out)
  end


  if opt.input_dir ~= '' then
    if opt.output_dir == '' then
      error('Must give -output_dir with -input_dir')
    end
    for fn in paths.files(opt.input_dir) do
      if utils.is_image_file(fn) then
        local in_path = paths.concat(opt.input_dir, fn)
        local out_path = paths.concat(opt.output_dir, fn)
        run_image(in_path, out_path)
      end
    end
  elseif opt.input_image ~= '' then
    if opt.output_image == '' then
      error('Must give -output_image with -input_image')
    end
    run_image(opt.input_image, opt.output_image)
  end
end


main()
