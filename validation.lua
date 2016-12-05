--[[
To calculate test error. MSE of UV matrix
--]]
require 'torch'
require 'image'
require 'nn'
require 'DataLoader'
local utils = require 'utils'
require 'ShaveImage'

local cmd = torch.CmdLine()

-- Model options
cmd:option('-model', 'checkpoint.t7')
cmd:option('-h5_file', 'coco.h5')

-- Verify back_size images once
cmd:option('-batch_size', 30)

-- Backend options
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
cmd:option('-backend', 'cuda', 'cuda|opencl')


function main()
  local opt = cmd:parse(arg)

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
  local criterion = nn.MSECriterion():type(dtype)
  local loader = DataLoader(opt)

  -- Check loss on the validation set
  loader:reset('val')
  local val_loss=0.0
  for i=1, loader.num_minibatches['val'] do
    local x, y = loader:getBatch('val')
    x, y = x:type(dtype), y:type(dtype)
    local out = model:forward(x)
    for t = 1, y:size(1) do
      y[t][1]:mul(0.436)
      y[t][2]:mul(0.615)
      out[t][1]:mul(0.436)
      out[t][2]:mul(0.615)
    end
    val_loss = val_loss + criterion:forward(out,y)
    print(string.format('Iteration = %d / %d', i, loader.num_minibatches['val']))
  end
  val_loss = val_loss / loader.num_minibatches['val']
  -- val_loss above is for UV loss. Need YUV loss
  -- val_loss = val_loss * 2.0 / 3.0
  print(string.format('val loss = %f', val_loss))

end
main()
