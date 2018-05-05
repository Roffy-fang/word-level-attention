require 'nn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTM = require 'misc.LSTMI'
local GRU = require 'misc.GRUI'
local MUT = require 'misc.MUTI'

-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.Imgrnn', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  local dropout_l = utils.getopt(opt, 'dropout_l', 0)
  local rnn_type = utils.getopt(opt, 'rnn_type', 'lstm')
  -- options for Language Model
  -- create the core lstm network. note +1 for both the START and END tokens
  if rnn_type == 'lstm' then
   -- self.core = LSTM.lstm(self.input_encoding_size, self.vocab_size + 1, self.rnn_size, self.num_layers, dropout_l, dropout_t, res_rnn,active, 0, false, 0)
   self.corel = LSTM.lstm(self.input_encoding_size, self.rnn_size, self.num_layers, dropout_l)
   self.corer = LSTM.lstm(self.input_encoding_size, self.rnn_size, self.num_layers, dropout_l)
   --self.corer = LSTM.lstm(self.rnn_size, self.rnn_size, self.num_layers, dropout_l)
  elseif rnn_type == 'rnn' then
   self.core = LSTM.rnn(self.input_encoding_size, self.rnn_size, self.num_layers, dropout_l)
  elseif rnn_type == 'gru' then
   self.corel = GRU.gru(self.input_encoding_size, self.rnn_size, self.num_layers, dropout_l)
   self.corer = GRU.gru(self.input_encoding_size, self.rnn_size, self.num_layers, dropout_l)
  elseif rnn_type == 'mut1' then
   self.core = MUT.mut1(self.input_encoding_size, self.rnn_size, self.num_layers, dropout_l)
  else
    assert(1==0, 'unsupport rnn type')
  end
  self:_createInitState(1) -- will be lazily resized later during forward passes
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  for h=1,self.num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the Imgrnn')
  self.clonesl = {self.corel}
  self.clonesr = {self.corer}
  for t=2,50 do
    self.clonesl[t] = self.corel:clone('weight', 'bias', 'gradWeight', 'gradBias')
    self.clonesr[t] = self.corer:clone('weight', 'bias', 'gradWeight', 'gradBias')
  end
end

function layer:getModulesList()
  return {self.corel,self.corer}
end

function layer:parameters()
  -- we only have two internal modules, return their params
  local p1,g1 = self.corel:parameters()
  local p2,g2 = self.corer:parameters()

  local params = {}
  for k,v in pairs(p1) do table.insert(params, v) end
  for k,v in pairs(p2) do table.insert(params, v) end
  
  local grad_params = {}
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end

  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)

  return params, grad_params
end

function layer:training()
  if self.clonesl == nil or  self.clonesr == nil  then  self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clonesl) do v:training() end
  for k,v in pairs(self.clonesr) do v:training() end
end

function layer:evaluate()
  if self.clonesl == nil or  self.clonesr == nil  then  self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clonesl) do v:evaluate() end
  for k,v in pairs(self.clonesr) do v:evaluate() end
end

--[[
takes a batch of images and runs the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxN LongTensor with integer elements 1..M, 
where D is sequence length and N is batch (so columns are sequences)
--]]
function layer:sample(inputs, opt)
  local img_x = inputs
  local batch_size = img_x:size(1)
  self:_createInitState(batch_size)
  local statel = self.init_state
  local stater = self.init_state

  local w, h = img_x:size(3),img_x:size(4)
--  img_x:resize(batch_size,self.input_encoding_size,w* h)
  
  local output=torch.zeros(batch_size, self.rnn_size, w, h):cuda()
  local upout={}
  for i=1, w do
    for j=1, h do
      local t = (i-1)*w + j
      local xt = img_x[{{},{},i,j}] 
     -- construct the inputs
       local inputs
       if j == 1  then 
         inputs = {xt,unpack(self.init_state)}
       else
         inputs = {xt,unpack(statel)}
       end
      -- forward the network
      local out = self.corel:forward(inputs)
      -- process the outputs
      --upout[t] = out[self.num_state+1] -- last element is the output vector
      output[{ {},{},i,j}] = out[self.num_state+1] -- last element is the output vector
      statel = {} -- the rest is state
      for i=1,self.num_state do table.insert(statel, out[i]) end
   end
 end

  for i=w,1,-1 do
    for j=h,1,-1 do
      local t = (i-1)*w + j
      local xt = img_x[{{},{},i,j}] 
     -- construct the inputs

       local inputs
       if j == h  then 
         inputs = {xt,unpack(self.init_state)}
       else
         inputs = {xt,unpack(stater)}
       end
      -- forward the network
      local out = self.corer:forward(inputs)
      -- process the outputs
      output[{ {},{},i,j}] =  output[{ {},{},i,j}]+ out[self.num_state+1] -- last element is the output vector
      stater = {} -- the rest is state
      for i=1,self.num_state do table.insert(stater, out[i]) end
   end
 end
  return output
end


--[[
input is a tuple of:
1. torch.Tensor of size NxK (K is dim of image code)
2. torch.LongTensor of size DxN, elements 1..M
   where M = opt.vocab_size and D = opt.seq_length

returns a (D+2)xNx(M+1) Tensor giving (normalized) log probabilities for the 
next token at every iteration of the LSTM (+2 because +1 for first dummy 
img forward, and another +1 because of START/END tokens shift)
--]]
function layer:updateOutput(input)
  local img_x = input
  if self.clonesl == nil or self.clonesr == nil then self:createClones() end -- lazily create clones on first forward pass

  local batch_size = img_x:size(1)
   self.w,self.h = img_x:size(3),img_x:size(4)
--  img_x:resize(batch_size,self.input_encoding_size,self.w* self.h)
  self.output:resize( batch_size, self.rnn_size,self.w,self.h)
  
  self:_createInitState(batch_size)

  self.tmax =self.w * self.h
  self.statel = {[0] = self.init_state}
  self.stater = {[self.tmax] = self.init_state}
  self.inputsl = {}
  self.inputsr = {}
  local upout = {}
  for i=1, self.w do
    for j=1, self.h do
     -- local xt = img_x[{{},{},i,j}] 
     -- construct the inputs
     local t = (i-1)*self.w + j
      local xt = img_x[{{},{},i,j}] 
      if j==1 then  
      self.inputsl[t] = {xt,unpack(self.statel[0])}
    else 
      self.inputsl[t] = {xt,unpack(self.statel[t-1])}
    end
      -- forward the network
  --    print(self.inputs[t])
      local out = self.clonesl[t]:forward(self.inputsl[t])
      -- process the outputs
      --upout[t] = out[self.num_state+1] -- last element is the output vector
      self.output[{ {},{},i,j}] = out[self.num_state+1] -- last element is the output vector
      self.statel[t] = {} -- the rest is state
      for i=1,self.num_state do table.insert(self.statel[t], out[i]) end
   end
 end

  for i= self.w, 1 ,-1 do
    for j=self.h, 1 ,-1 do
     -- local xt = img_x[{{},{},i,j}] 
     -- construct the inputs
     local t = (i-1)*self.w + j
      local xt = img_x[{{},{},i,j}] 
      if j==self.h then  
      self.inputsr[t] = {xt,unpack(self.stater[self.tmax])}
    else 
      self.inputsr[t] = {xt,unpack(self.stater[t])}
    end
     -- self.inputsr[t] = {xt,unpack(self.stater[t])}
      -- forward the network
     -- print(self.inputsr[t])
      local out = self.clonesr[t]:forward(self.inputsr[t])
      -- process the outputs
      self.output[{ {},{},i,j}] = self.output[{ {},{},i,j}] + out[self.num_state+1] -- last element is the output vector
      self.stater[t-1] = {} -- the rest is state
      for i=1,self.num_state do table.insert(self.stater[t-1], out[i]) end
   end
 end
  return self.output
end

--[[
gradOutput is an (D+2)xNx(M+1) Tensor.
--]]
function layer:updateGradInput(input, gradOutput)
  local dimg_x = input
  local batch_size = dimg_x:size(1)
  dimg_x:resize(batch_size,self.input_encoding_size,self.w, self.h)
  -- go backwards and lets compute gradients
  local dstatel = {[self.tmax] = self.init_state} -- this works when init_state is all zeros
  local dstater = {[0] = self.init_state} -- this works when init_state is all zeros
  local dupout = {}
 -- print(gradOutput:size())
  for i=1,self.w do
    for j=1,self.h do
    -- concat state gradients and output vector gradients at time step t
    local t = (i-1)*self.w + j
    local dout = {}
    if j == 1  then  
     for k=1,#dstater[t-1] do table.insert(dout, dstater[0][k]) end
    else
    for k=1,#dstater[t-1] do table.insert(dout, dstater[t-1][k]) end
    end
    table.insert(dout, gradOutput[{{},{},i,j}])
    local dinputs = self.clonesr[t]:backward(self.inputsr[t], dout)
    -- split the gradient to xt and to state
    --dupout[t] =dinputs[1]
    dimg_x[{{},{},i,j}] =dinputs[1]
    dstater[t] = {} -- copy over rest to state grad
    for k=2,self.num_state+1 do table.insert(dstater[t], dinputs[k]) end
    -- continue backprop of xt
    end
  end
  
  for i=self.w,1,-1 do
    for j=self.h,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local t = (i-1)*self.w + j
    local dout = {}
    if j == self.h  then  
     for k=1,#dstatel[t] do table.insert(dout, dstatel[self.tmax][k]) end
    else
     for k=1,#dstatel[t] do table.insert(dout, dstatel[t][k]) end
   end
    table.insert(dout, gradOutput[{{},{},i,j}])
    local dinputs = self.clonesl[t]:backward(self.inputsl[t], dout)
    -- split the gradient to xt and to state
    dimg_x[{{},{},i,j}] =  dimg_x[{{},{},i,j}] +  dinputs[1]
    dstatel[t-1] = {} -- copy over rest to state grad
    for k=2,self.num_state+1 do table.insert(dstatel[t-1], dinputs[k]) end
    
    -- continue backprop of xt
    end
  end
  self.gradInput = {dimg_x,torch.Tensor()}
  return self.gradInput
end

