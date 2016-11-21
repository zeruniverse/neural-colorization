require 'torch'
require 'optim'
require 'image'
require 'DataLoader'

local utils = require 'utils'
local models = require 'models'

local cmd = torch.CmdLine()


--[[
Train a feedforward style transfer model
--]]

-- Generic options
cmd:option('-arch', 'c9s1-32,d64,d128,R128,R128,R128,R128,R128,u64,u32,c9s1-2')
cmd:option('-h5_file', 'coco.h5')
cmd:option('-padding_type', 'reflect-start')
cmd:option('-resume_from_checkpoint', '')

-- Optimization
cmd:option('-num_iterations', 50000)
cmd:option('-max_train', -1)
cmd:option('-batch_size', 30)
cmd:option('-learning_rate', 1e-3)
cmd:option('-lr_decay_every', 3000)
cmd:option('-lr_decay_factor', 0.5)
cmd:option('-weight_decay', 0)

-- Checkpointing
cmd:option('-checkpoint_name', 'checkpoint')
cmd:option('-checkpoint_every', 1000)
cmd:option('-num_val_batches', 10)

-- Backend options
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
cmd:option('-backend', 'cuda', 'cuda|opencl')


function main()
  local opt = cmd:parse(arg)

  -- Figure out the backend
  local dtype, use_cudnn = utils.setup_gpu(opt.gpu, opt.backend, opt.use_cudnn == 1)

  -- Build the model
  local model = nil
  if opt.resume_from_checkpoint ~= '' then
    print('Loading checkpoint from ' .. opt.resume_from_checkpoint)
    model = torch.load(opt.resume_from_checkpoint).model:type(dtype)
  else
    print('Initializing model from scratch')
    model = models.build_model(opt):type(dtype)
  end
  if use_cudnn then cudnn.convert(model, cudnn) end
  model:training()
  print(model)

  local loader = DataLoader(opt)
  local params, grad_params = model:getParameters()
  local criterion = nn.MSECriterion():type(dtype)
  

  local function f(x)
    assert(x == params)
    grad_params:zero()
    
    -- x is y value, y is uv value
    local x, y = loader:getBatch('train')
    x, y = x:type(dtype), y:type(dtype)

    -- Run model forward
    local out = model:forward(x)
    local grad_out = nil

    -- This is a bit of a hack: if we are using reflect-start padding and the
    -- output is not the same size as the input, lazily add reflection padding
    -- to the start of the model so the input and output have the same size.
    if opt.padding_type == 'reflect-start' and x:size(3) ~= out:size(3) then
      local ph = (x:size(3) - out:size(3)) / 2
      local pw = (x:size(4) - out:size(4)) / 2
      local pad_mod = nn.SpatialReflectionPadding(pw, pw, ph, ph):type(dtype)
      model:insert(pad_mod, 1)
      out = model:forward(x)
    end

    local loss = criterion:forward(out,y)
    grad_out = criterion:backward(out, y)
    -- Run model backward
    model:backward(x, grad_out)

    -- Add regularization
    -- grad_params:add(opt.weight_decay, params)
 
    return loss, grad_params
  end


  local optim_state = {learningRate=opt.learning_rate}
  local train_loss_history = {}
  local val_loss_history = {}
  local val_loss_history_ts = {}

  for t = 1, opt.num_iterations do
    local epoch = t / loader.num_minibatches['train']

    local _, loss = optim.adam(f, params, optim_state)

    table.insert(train_loss_history, loss[1])

    print(string.format('Epoch %f, Iteration %d / %d, loss = %f',
          epoch, t, opt.num_iterations, loss[1]), optim_state.learningRate)

    if t % opt.checkpoint_every == 0 then
      -- Check loss on the validation set
      loader:reset('val')
      model:evaluate()
      local val_loss = 0
      print 'Running on validation set ... '
      local val_batches = opt.num_val_batches
      for j = 1, val_batches do
        local x, y = loader:getBatch('val')
        x, y = x:type(dtype), y:type(dtype)
        local out = model:forward(x)
        val_loss = val_loss + criterion:forward(out,y)
      end
      val_loss = val_loss / val_batches
      print(string.format('val loss = %f', val_loss))
      table.insert(val_loss_history, val_loss)
      table.insert(val_loss_history_ts, t)
      model:training()

      -- Save a JSON checkpoint
      local checkpoint = {
        opt=opt,
        train_loss_history=train_loss_history,
        val_loss_history=val_loss_history,
        val_loss_history_ts=val_loss_history_ts
      }
      local filename = string.format('%s.json', opt.checkpoint_name)
      paths.mkdir(paths.dirname(filename))
      utils.write_json(filename, checkpoint)

      -- Save a torch checkpoint; convert the model to float first
      model:clearState()
      if use_cudnn then
        cudnn.convert(model, nn)
      end
      model:float()
      checkpoint.model = model
      filename = string.format('%s.t7', opt.checkpoint_name)
      torch.save(filename, checkpoint)

      -- Convert the model back
      model:type(dtype)
      if use_cudnn then
        cudnn.convert(model, cudnn)
      end
      params, grad_params = model:getParameters()
    end

    if opt.lr_decay_every > 0 and t % opt.lr_decay_every == 0 then
      local new_lr = opt.lr_decay_factor * optim_state.learningRate
      optim_state = {learningRate = new_lr}
    end

  end

end


main()
