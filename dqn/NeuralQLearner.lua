--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]
require 'optim'
require 'nnutils'
cons = require 'pl.pretty'

if not dqn then
    require 'initenv'
end

local nql = torch.class('dqn.NeuralQLearner')
--agent tweak indicators
local EXPLORE = 3
local GREEDY = 2
local MERGED = 1
local VANILA = 0

function nql:__init(args)
    self.state_dim  = args.state_dim -- State dimensionality.
    self.actions    = args.actions
--#########################################
    self.n_actions  = #args.actions
    self.objects    = args.game_objects
    self.n_objects  = #args.game_objects
    self.object_restrict_thresh = args.obj_bad_cmd_thresh or 0.5
    self.obj_drop_prob = args.obj_drop_prob or 0.5
    self.obj_thresh_acc = args.obj_thresh or 0
    self.obj_network = args.obj_net_file or 'conv_obj_net'
    self.obj_start = args.obj_start or args.learn_start or 0
    self.obj_lr = args.obj_lr or 0.0001
    self.AEN_sample_bias = args.AEN_sample_bias or 0
    assert(self.obj_start >= 1)
    assert(self.obj_thresh_acc >= 0 and self.obj_thresh_acc < 1)
    self.obj_sample = args.obj_sample or 0
    self.obj_max = args.obj_max or #args.game_objects
    assert(self.obj_sample <= self.n_objects and self.obj_sample >= 0)
    assert(self.obj_max <= self.n_objects and self.obj_max >= -1)
    self.parse_lable_scale = args.parse_lable_scale or 1
    self.double_elimination = args.double_elimination or 0
    self.AEN_n_filters = args.AEN_n_filters or 20
    self.shallow_exploration_flag = args.shallow_exploration_flag or 0
    self.shallow_elimination_flag = args.shallow_elimination_flag or 0
    if args.agent_tweak:match("greedy") then -- tweak option for large action space
        self.agent_tweak  = GREEDY
        print("greedy restriction tweak")
    elseif args.agent_tweak:match("explore") then
        self.agent_tweak  = EXPLORE
        print("exploration restriction tweak")
    elseif args.agent_tweak:match("merged") then
        self.agent_tweak  = MERGED
        print("greedy and exploration restriction tweak")
    else self.agent_tweak  = VANILA --vanilla
        print("vanila algo")
    end
    if self.agent_tweak == VANILA then
      self.shallow_elimination_flag =0
      self.shallow_exploration_flag =0
      self.double_elimination =0
    end
    if self.shallow_elimination_flag ==1 then
      self.lambda = args.lambda or 0.1
      self.beta       = args.elimination_beta or 10
      self.active_beta = self.beta
      self.n_features = self.AEN_n_filters*3 --account for bias
      self.elimination_freq = args.elimination_freq or 50000
      self.A = torch.CudaTensor(self.n_objects,self.n_features+1,self.n_features+1)
      self.A_init = torch.CudaTensor(self.n_objects,self.n_features+1,self.n_features+1)
      self.A_init:copy(torch.eye(self.n_features+1):mul(self.lambda):reshape(1,self.n_features+1,self.n_features+1):expand(self.n_objects,self.n_features+1,self.n_features+1))
      self.val_conf_buf = torch.CudaTensor(5,EVAL_STEPS)
    end

    self.objects_range=torch.range(1,self.n_objects):cudaInt()
    self.actions_range=torch.range(1,self.n_actions):cudaInt()

    --assert( self.obj_max+self.obj_sample > 0 or self.agent_tweak == VANILLA or self.agent_tweak == EXPLORE)
    self.sigmoid = cudnn.Sigmoid():cuda()
    self.softmax = cudnn.SoftMax():cuda()
--#########################################
    self.verbose    = args.verbose
    self.best       = args.best

    --- epsilon annealing
    self.ep_start   = args.ep or 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = args.ep_end or self.ep
    self.ep_endt    = args.ep_endt or 1000000

    ---- learning rate annealing
    self.lr_start       = args.lr or 0.01 --Learning rate.
    self.lr             = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost.
    self.minibatch_size = args.minibatch_size or 1
    self.valid_size     = args.valid_size or 500

    --- Q-learning parameters
    self.discount       = args.discount or 0.99 --Discount factor.
    self.update_freq    = args.update_freq or 1
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 0
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000
    self.hist_len       = args.hist_len or 1
    self.rescale_r      = args.rescale_r
    self.max_reward     = args.max_reward
    self.min_reward     = args.min_reward
    self.clip_delta     = args.clip_delta
    self.target_q       = args.target_q
    self.bestq          = 0

    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.hist_len*self.ncols,args.state_rows, args.state_cols}
    self.preproc        = args.preproc  -- name of preprocessing network
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1
    self.bufferSize     = args.bufferSize or 512
    self.transition_params = args.transition_params or {}

    self.network    = args.network or self:createNetwork()
    -- check whether there is a network file
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
              " is not a string!")
    end
    local msg, err = pcall(require, self.network)
    if not msg then
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, self.network)
        if not err_msg then
            error("Could not find network file ")
        end
        if self.best and exp.best_model then
            self.network = exp.best_model
        else
            self.network = exp.model
        end
    else
        print('Creating Agent Network from ' .. self.network)
        self.network = err
        self.network = self:network(args)

    end
--#########################################
--  init obj network
    self.Y_buff         = torch.CudaTensor(self.minibatch_size,self.n_objects) -- buffer for multiclass labels
    self.valid_Y_buff   = torch.CudaTensor(self.valid_size,self.n_objects)
    if self.agent_tweak ~= VANILA then
        local msg, err = pcall(require, self.obj_network)
        if not msg then
            -- try to load saved agent
            local err_msg, exp = pcall(torch.load, self.obj_network)
            if not err_msg then
                error("Could not find object network file ")
            end
        else
            print('Creating Object Network from ' .. self.obj_network)
            self.obj_network = err
            self.obj_network = self:obj_network(args)
        end
        -- set object network loss for multi lable learning
        --self.objNetLoss = nn.MultiLabelSoftMarginCriterion() --need to remove sigmoid activation from the network
        if self.shallow_elimination_flag == 1 then
          self.objNetLoss = nn.MSECriterion()
        else
          self.objNetLoss = nn.BCECriterion(torch.FloatTensor(self.n_objects):fill(self.parse_lable_scale))
        end
        self.optimState = {learningRate = self.obj_lr}--, nesterov = true, momentum = 0.8, dampening = 0} -- for obj network
        self.last_object_net_accuracy = 0
        if self.gpu and self.gpu >= 0 then
            self.obj_network:cuda()
            self.Y_buff:cuda()
            self.valid_Y_buff:cuda()
            self.objNetLoss:cuda()
            self.sigmoid:cuda()
            self.softmax:cuda()
            cudnn.convert(self.obj_network,cudnn)
        else
            self.obj_network:float()
        end
    --#########################################
        self.obj_w, self.obj_dw = self.obj_network:getParameters()
        print("number of parameterns in object network",#self.obj_w)
        --#########################################
        end
    -- end of object network init
--#########################################


    if self.gpu and self.gpu >= 0 then
        cudnn.benchmark = true
        cudnn.fastest = true
        self.network:cuda()
        cudnn.convert(self.network,cudnn)
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.tensor_type = torch.FloatTensor

    end

    if self.preproc ~= nil then
      -- Load preprocessing network.
      if not (type(self.preproc == 'string')) then
        error('The preprocessing is not a string')
      end
      msg, err = pcall(require, self.preproc)
      if not msg then
        error("Error loading preprocessing net")
      end
      self.preproc = err
      self.preproc = self:preproc()
      self.preproc:float()
    end

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Foat or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions,numObjects = self.n_objects,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize,
        sample_parse_buffer = self.agent_tweak ~= VANILA,
        AEN_sample_bias = self.AEN_sample_bias
    }

    self.transitions = dqn.TransitionTable(transition_args)

    self.numSteps = 0 -- Number of perceived states.
    self.lastState = nil
    self.lastAction = nil
--#########################################
    self.lastAction_o = nil
    self.lastAction_bad = nil
--#########################################
    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.

    self.q_max = 1
    self.r_max = 1
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    print("number of parameterns in state network",#self.w)


    self.deltas = self.dw:clone():fill(0)

    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)
    if self.shallow_elimination_flag==1 then
      self.obj_target_network    =  self.obj_network:clone()
    end

    if self.target_q then
        self.target_network = self.network:clone()
    end
end


function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    self.network = state.model
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.numSteps = 0
    print("RESET STATE SUCCESFULLY")
end


function nql:preprocess(rawstate)
    if self.preproc then

        return self.preproc:forward(rawstate:float())
                    :clone():reshape(self.state_dim)
    end
    return rawstate

end


function nql:getQUpdate(args)
    local s, a, r, s2, term, delta
    local q, q2, q2_max

    s = args.s
    a = args.a
    r = args.r
    s2 = args.s2
    term = args.term

    -- The order of calls to forward is a bit odd in order
    -- to avoid unnecessary calls (we only need 2).

    -- delta = r + (1-terminal) * gamma * max_a Q(s2, a) - Q(s, a)
    term = term:clone():float():mul(-1):add(1)

    local target_q_net
    if self.target_q then
        target_q_net = self.target_network
    else
        target_q_net = self.network
    end
    -- Compute max_a Q(s_2, a).
    local q2 = target_q_net:forward(s2):float()
    local q2_max-- = torch.FloatTensor((#q2)[1])
    if self.agent_tweak ~= VANILA and self.double_elimination == 1 and self.numSteps > self.obj_start then
      --output is a batch_sizeXn_objects needs to turn to batch_sizeXn_actions via elimination mask
      local batch_size = s2:size()[1]
      local AEN_prediction,uncertainty_bias-- =torch.CudaTensor(batch_size,self.n_objects)
      if self.shallow_elimination_flag == 1 then
        AEN_prediction,uncertainty_bias=self:batchAdaptiveElimination(s2)
        AEN_prediction:add(-uncertainty_bias)
      else
        AEN_prediction = self.obj_network:forward(s2):float()
      end

      local AEN_hard_prediction = AEN_prediction:ge(self.object_restrict_thresh):byte()

      local elimination_mask = nil
      if self.n_actions~=self.n_objects then
        elimination_mask = torch.ByteTensor(batch_size,self.n_actions-self.n_objects):zero() --non-AEN actions are always vavlid set signal to 0
        elimination_mask = elimination_mask:cat(AEN_hard_prediction) --we always assume AEN actions have the higher index range
      else
        elimination_mask = AEN_hard_prediction
      end
      q2[elimination_mask]=-1/0
    end
    q2_max = q2:max(2)

    -- Compute q2 = (1-terminal) * gamma * max_a Q(s2, a)
    local q2_gamma = q2_max:clone():mul(self.discount):cmul(term)

    local delta = r:clone():float()

    if self.rescale_r then
        delta:div(self.r_max)
    end
    delta:add(q2_gamma)

    -- q = Q(s,a)
    local q_all = self.network:forward(s):float()
    q = torch.FloatTensor(q_all:size(1))
    for i=1,q_all:size(1) do
        q[i] = q_all[i][a[i]]
    end
    delta:add(-1, q)

    if self.clip_delta then
        delta[delta:ge(self.clip_delta)] = self.clip_delta
        delta[delta:le(-self.clip_delta)] = -self.clip_delta
    end

    local targets = torch.zeros(self.minibatch_size, self.n_actions):float()
    for i=1,math.min(self.minibatch_size,a:size(1)) do
        targets[i][a[i]] = delta[i]
    end

    if self.gpu >= 0 then targets = targets:cuda() end

    return targets, delta, q2_max
end


function nql:qLearnMinibatch()
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    assert(self.transitions:size() > self.minibatch_size)

    local s, a, r, s2, term, s_for_obj, a_for_obj,a_o, bad_command = self.transitions:sample(self.minibatch_size)
    local targets, delta, q2_max = self:getQUpdate{s=s, a=a, r=r, s2=s2,
        term=term, update_qmax=true}
    -- zero gradients of parameters
    self.dw:zero()

    -- get new gradient
    self.network:backward(s, targets)

    -- add weight cost to gradient
    self.dw:add(-self.wc, self.w)

    -- compute linearly annealed learning rate
    local t = math.max(0, self.numSteps - self.learn_start)
    self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                self.lr_end
    self.lr = math.max(self.lr, self.lr_end)

    -- use gradients
    self.g:mul(0.95):add(0.05, self.dw)
    self.tmp:cmul(self.dw, self.dw)
    self.g2:mul(0.95):add(0.05, self.tmp)
    self.tmp:cmul(self.g, self.g)
    self.tmp:mul(-1)
    self.tmp:add(self.g2)
    self.tmp:add(0.01)
    self.tmp:sqrt()

    -- accumulate update
    self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
    self.w:add(self.deltas)
--#########################################
    -- now train obj network
    if self.agent_tweak ~= VANILA then
      self.obj_network:training()
      self:objLearnMiniBatch(s_for_obj,a_for_obj,a_o, bad_command)
      self.obj_network:evaluate()
    end
--#########################################
end

--#########################################
function nql:setYbuff(action_object, bad_command, validation)

  --@FIXME this is ugly, but will do for now
  if validation then
    self.valid_Y_buff = self.valid_Y_buff:zero() + 1 -- 2d buffer of size minibatchXnum_objects set 1 to unknown entries for validation
    local object_index
    for i=1, (#self.valid_Y_buff)[1] do -- expand labels
        object_index = action_object[i]
        self.valid_Y_buff[i][object_index] = bad_command[i]
    end

    else
      self.Y_buff = self.Y_buff:zero() + 0.5 -- 2d buffer of size minibatchXnum_objects set 0.5 to unknown entries
      local object_index
      for i=1, (#self.Y_buff)[1] do -- expand labels
          object_index = action_object[i]
          self.Y_buff[i][object_index] = bad_command[i]
      end
  end

  if self.gpu >=0 then
      if validation then self.valid_Y_buff:cuda()
      else self.Y_buff:cuda() end
  end
end


-- only object related samples in the batch, maybe add some normal samples?
function nql:objLearnMiniBatch(s,a,a_o,bad_command)
    assert(self.transitions:size() > self.minibatch_size)
    function feval()
        self.obj_dw:zero()
        self:setYbuff(a_o, bad_command)
        local grad_image = self.Y_buff:ne(0.5):cuda() -- maps which gradients we wish to keep
        local h_x = self.obj_network:forward(s):cuda()
        local J
        if self.shallow_elimination_flag == 1 then
          J = self.objNetLoss:forward(h_x, self.Y_buff)
        else
          J = self.objNetLoss:forward(h_x, self.Y_buff)/self.parse_lable_scale
        end
	    --zero out none informative gradients
	    local dJ_dh_x = torch.cmul(self.objNetLoss:backward(h_x, self.Y_buff),grad_image:float():cuda())
	    --local dJ_dh_x = self.objNetLoss:backward(h_x, self.Y_buff)--:cuda()
        self.obj_network:backward(s, dJ_dh_x) -- computes and updates gradTheta
	    return J, self.obj_dw
    end
     optim.adam(feval, self.obj_w, self.optimState)
end
--#########################################

function nql:sample_validation_data()
    print("sampling validation data")
    local s, a, r, s2, term,s_for_obj,a_for_obj,a_o,bad_command = self.transitions:sample(self.valid_size)

    self.valid_s    = s:clone()
    self.valid_a    = a:clone()
    self.valid_r    = r:clone()
    self.valid_s2   = s2:clone()
    self.valid_term = term:clone()
--#########################################
    if self.agent_tweak ~= VANILA then
      self.valid_s_for_obj = s_for_obj:clone()
      self.valid_a_for_obj = a_for_obj:clone()
      self.valid_a_o = a_o:clone()
      self.valid_bad_command = bad_command:clone()
      self:setYbuff(self.valid_a_o, self.valid_bad_command,true)
      local bad_parse_samples = bad_command:sum()
      print("Validation sample contains " .. bad_parse_samples/self.valid_size .. ' bad parse')
    end
    self.transitions:report(1)

--#########################################
end


function nql:compute_validation_statistics(ind)
    local targets, delta, q2_max = self:getQUpdate{s=self.valid_s,
        a=self.valid_a, r=self.valid_r, s2=self.valid_s2, term=self.valid_term}
    self.v_avg = self.q_max * q2_max:mean()
    self.tderr_avg = delta:clone():abs():mean()
--#########################################
    self.transitions:report(self.verbose)
    local PositivesCount = self.valid_Y_buff:eq(0):sum()

    if self.agent_tweak ~= VANILA then -- calc object net validation info
      local h_x = self.obj_target_network:forward(self.valid_s_for_obj)
      local J=0
      if self.shallow_elimination_flag == 1 then
        for i=1,self.valid_size do
            J=J+torch.abs(h_x[i][self.valid_a_o[i]]-self.valid_Y_buff[i][self.valid_a_o[i]])
        end
        J=J/self.valid_size
      else
        J = self.objNetLoss:forward(h_x, self.valid_Y_buff)/self.parse_lable_scale
      end
      local h_y = h_x:gt(self.object_restrict_thresh) -- calculate prediction
      local aen_mean,aen_std,sp_mean,sp_std = nil,nil,nil,nil
      if self.numSteps > self.obj_start then
        local eliminated = h_y:float():cuda():sum(2)
        aen_mean,aen_std = eliminated:mean(),eliminated:std()
        print(string.format('Elimination avg/std (no conf):         %.3f / %.3f ' ,aen_mean,aen_std ))
        EliminationBuff[ind] = aen_mean
        if self.shallow_elimination_flag == 1 then
            local shallow_pred,uncertainty_bias = self:batchAdaptiveElimination(self.valid_s_for_obj)
            shallow_pred = (shallow_pred - uncertainty_bias):gt(self.object_restrict_thresh):cuda()
            local fn_count = (shallow_pred -self.valid_Y_buff):eq(1):sum()
            local shallow_pred_sum=shallow_pred:float():sum(2)
            sp_mean,sp_std = shallow_pred_sum:mean(),shallow_pred_sum:std()
            print(string.format('Shallow Elimination avg/std: %.3f / %.3f, #FalsePositives: %.3f, #Positives: %.3f',sp_mean,sp_std,fn_count,PositivesCount))
            ShallowEliminationBuff[ind] = sp_mean
            ShallowEliminationFPBuff[ind] = fn_count
        end
      end
      --local p,of =  shallow_elimination TODO test how well shallow elimination is doing
      local sum = 0
      local false_neg = 0
      for i=1,self.valid_size do
          local object_index = self.valid_a_o[i]
          if h_y[i][object_index] == self.valid_bad_command[i] then
            sum = sum + 1
          else
            if self.valid_bad_command[i] == 0 then
              false_neg = false_neg + 1
            end
          end
      end
      --print("predicted lables vs validation sample\n",torch.cat(h_y:float(),self.valid_Y_buff:float()))
      local single_lable_acc = sum/self.valid_size
      print(string.format('AEN (no conf) accuracy: %.3f, Loss : %.3f, #FalsePositives: %.3f #Positives: %.3f',single_lable_acc,J,false_neg,PositivesCount))
      EliminationFPBuff[ind] = false_neg
      LossBuff[ind] = J
      AccBuff[ind] = single_lable_acc
      self.last_object_net_accuracy = single_lable_acc
      return {AEN_loss=J, AEN_single_accuracy=single_lable_acc,AEN_fn_count=false_neg,
      eliminated_actions_avg=aen_mean,adaptive_elimination_avg=sp_mean}
    end
--#########################################
end

--choose an action index based on last s,r,term
function nql:perceive(reward, rawstate, terminal, testing, testing_ep)
    -- Preprocess state (will be set to nil if terminal)
    --local state = self:preprocess(rawstate):float()
    local state = rawstate:float()
    local curState

    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, reward)
    end

    self.transitions:add_recent_state(state, terminal)

    local currentFullState = self.transitions:get_recent()

    --Store transition s, a, r, s',a_o,bad_command from last step
    if self.lastState and not testing then
			--add(s, a, r, term, a_o,bad_command)
        self.transitions:add(self.lastState, self.lastAction, reward,
                             self.lastTerminal, self.lastAction_o or 0, self.lastAction_bad)
    end

    if not testing and (
          self.numSteps == self.learn_start or (
            self.numSteps%(self.replay_memory/4) == 0 and
            self.numSteps <= self.replay_memory and
            self.numSteps > self.learn_start
          )
        ) then
        --sample validation data twice, initially when learning starts and again when we fill the entire replay memory
	      self:sample_validation_data()
    end

    curState= self.transitions:get_recent()
    curState = curState:resize(1, unpack(self.input_dims))
    -- Select action
    local actionIndex = 1
    local a_o

    if not terminal then
      	actionIndex = self:eGreedy(curState,testing, testing_ep)
      	a_o = self.actions[actionIndex].object or 0
    end

    self.transitions:add_recent_action(actionIndex,a_o)

    --Do some Q-learning updates
    if not testing then
      if self.numSteps >= self.learn_start then
        if self.shallow_elimination_flag == 1 and (self.numSteps == self.obj_start or
             self.numSteps > self.obj_start and self.numSteps % self.elimination_freq == 0) then
          self:elimination_update()
        end
        if self.numSteps % self.update_freq == 0 then
          for i = 1, self.n_replay do
            self:qLearnMinibatch()
          end
        end
      end

      self.numSteps = self.numSteps + 1
    end

    self.lastState = state:clone()
    self.lastAction = actionIndex
    self.lastAction_o = a_o
    self.lastTerminal = terminal
    -- self.lastAction_bad is updated externally after we try the command
    if self.target_q and self.numSteps % self.target_q == 1 then
        self.target_network = self.network:clone()
    end


    if not terminal then
        return actionIndex,a_o
    else
        return 0
    end
end


function nql:eGreedy(state, testing,testing_ep)
    if not testing then
	      self.ep = (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))
    else self.ep = testing_ep end
--#########################################
    local prediction,uncertainty_bias = nil,nil
    local hard_prediction = nil
    local actionIndex, a_o

    -- Turn single state into minibatch.  Needed for convolutional nets.
    if state:dim() == 2 then
	  assert(false, 'Input must be at least 3D')
      state = state:resize(1, state:size(1), state:size(2))
    end

    if self.gpu >= 0 then
      state = state:cuda()
    end
    -- Epsilon greedy version which cuts the chance to explore "bad actions" by half
    if self.agent_tweak ~= VANILA and self.last_object_net_accuracy > self.obj_thresh_acc and self.numSteps > self.obj_start then --start using object network insight
    --if self.agent_tweak ~= VANILA and  self.numSteps > self.obj_start then --start using object network insight
      if self.shallow_elimination_flag == 1 then
          --return 2 tensors: AE target net work prediction and confidence
          prediction = self.obj_target_network:forward(state):squeeze()
          local phi = torch.CudaTensor(1,self.n_features+1)
          phi:narrow(2,1,self.n_features):copy(self.obj_target_network.modules[3].output)
          phi:narrow(2,self.n_features+1,1):fill(1)
          uncertainty_bias = self:shallow_elimination(testing,prediction,phi)
          prediction = prediction-uncertainty_bias
      else
        --prediction = nn.Sigmoid():forward(self.obj_network:forward(state)):float() --for MLSML criterion
        prediction = self.obj_network:forward(state):squeeze() --for BCE cretirion network last layer is sigmoid.
        -- set self.object_restrict_thresh > 0.5 to consider high confidence predictions - relaxation for under-represented (s,a) pairs
      end
      hard_prediction = prediction:gt(self.object_restrict_thresh)
    end

    if torch.uniform() < self.ep then
  	  actionIndex = torch.random(1, self.n_actions) --choose at random
      -- prediction is always null for vanila, tweak 2 is only greedy action restriction so we also skip this part
      if prediction and self.agent_tweak ~= GREEDY then --restricted random action selection, else use standard exploration
        a_o = self.actions[actionIndex].object or 0	-- extract relevant object
        if a_o ~= 0 then --only for "take" actions use prediction to validate action
          if uncertainty_bias and self.shallow_exploration_flag == 1 then
            --shallow elimination exploration scheme aims at improving AEN confidence
            actionIndex = torch.multinomial(self.softmax:forward(uncertainty_bias:gt(self.object_restrict_thresh/2):cudaByte():cuda()),1):squeeze() + self.n_actions - self.n_objects
          else --naive exploration - will only follow AEN prediction
            repeat
              --choose stricktly take action at random - this is to avoid vanishing "take" actions from replay mem
              actionIndex = torch.random(self.n_actions - self.n_objects + 1, self.n_actions) -- assume take actions are always last
              a_o = self.actions[actionIndex].object -- extract relevant object
              --cons.dump(self.actions[actionIndex])
              assert(a_o)
              -- coin flip will determin if actions with positive hard prediction get through and returned to the agent
            until hard_prediction[a_o] == false or torch.uniform() > self.obj_drop_prob
          end

        end
      end

      return actionIndex
    else --use greedy agent policy and pass along the raw prediction for this state
        return self:greedy(state,prediction,hard_prediction)
    end
--#########################################
end

function nql:greedy(state,obj_net_prediction,obj_hard_pred)
    local q = self.network:forward(state):squeeze()
--#########################################
    --greedy action restriction segment
    local best_objects, soft_object_prediction,sampled_objects = nil,nil,nil

    --obj_net_prediction is always null for vanila, skip this part for strictly exploration tweak (no 3)
    if obj_net_prediction and self.agent_tweak ~= EXPLORE then --not nil only if we have started using object net insight
      if self.obj_max == -1 then
          --allow NQL to select an action from of all actions that are above the threshhold for the given state
          best_objects = self.objects_range[1-obj_hard_pred]
      elseif self.obj_max > 0 then
        --best AEN predictions over a fixed size subset of actions
        --most likely objects have the lowest values
        local _,sort_ind = obj_net_prediction:topk(self.obj_max)
        --best_objects = sort_ind[{{1,self.obj_max}}]
        best_objects=sort_ind
      end

      if self.obj_sample > 0 then
        --sample objects with bias to favor likely, this helps avoiding optimal action starvation
        --flip 1 to 0 and 0 to 1 and create probability distribution over the objects
        soft_object_prediction = self.softmax:forward(1 - obj_net_prediction)
        sampled_objects = torch.multinomial(soft_object_prediction, self.obj_sample)
        if best_objects == nil  then
          best_objects = sampled_objects
        else
          best_objects = best_objects:cat(sampled_objects)
        end
      end
    end
    if best_objects and best_objects:nDimension()>0 then
        local elimination_mask = torch.ByteTensor(self.n_actions):fill(1)
        for i = 1,best_objects:size()[1] do
            elimination_mask[best_objects[i]]=0
        end
        if self.n_actions~= self.n_objects then
            elimination_mask[{{1,self.n_actions-self.n_objects}}]=0
        end
        q[elimination_mask:cudaByte()]=-1/0
    end

    besta=self.actions_range[q:eq(q:max())]:totable()
    maxq=q[besta[1]]
  --#########################################
    self.bestq = maxq

    local r = torch.random(1, #besta)

    self.lastAction = besta[r]
    self.lastAction_o = self.actions[besta[r]].object or 0

    return besta[r]
end


function nql:createNetwork()
    local n_hid = 128
    local mlp = nn.Sequential()
    print("@debug create network")
    mlp:add(nn.Reshape(self.hist_len*self.ncols*self.state_dim))
    mlp:add(nn.Linear(self.hist_len*self.ncols*self.state_dim, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))

    return mlp
end


function nql:_loadNet()
    local net = self.network
    local obj_net = self.obj_network
    if self.gpu then
        net:cuda()
        obj_net:cuda()
    else
        net:float()
        obj_net:float()
    end
    return net,obj_net
end


function nql:init(arg)
    self.actions = arg.actions
    self.n_actions = #self.actions
    self.network, self.obj_network = self:_loadNet()
    -- Generate targets.
    self.transitions:empty()
end


function nql:report()
  print(get_weight_norms(self.network))
  print(get_grad_norms(self.network))
  if self.agent_tweak ~= VANILA then
    print(get_weight_norms(self.obj_network))
    print(get_grad_norms(self.obj_network))
  end
end

function nql:batchAdaptiveElimination(s)
  local batch_size=s:size()[1]
  local uncertainty_bias =torch.CudaTensor(batch_size,self.n_objects)
  local AEN_prediction = self.obj_target_network:forward(s)
  local phi = torch.CudaTensor(batch_size,self.n_features+1)
  phi:narrow(2,1,self.n_features):copy(self.obj_target_network.modules[3].output)
  phi:narrow(2,self.n_features+1,1):fill(1)
  for i=1,batch_size do --for each sample in batch
    local conf = nil
    conf = self:shallow_elimination(false, AEN_prediction[i],phi[i])
    uncertainty_bias[i] = conf

  end
  return AEN_prediction,uncertainty_bias
end

----- adaptive elimination, Tom&Nadav, 22/4 ------------------------------------

function nql:elimination_update()
  print('performing elimination_update')
  -- to do : get last layer number
  -- create target network for f
  -- next: eliminate based on phi^t theta +- sqrt(phi^tA^-1phi) > thresh
  collectgarbage()

  local start_t = sys.clock()sys.clock()
  -- initialization
  self.obj_target_network    =  self.obj_network:clone()
  self.obj_target_network:remove()
  self.obj_target_network:evaluate()
  local PHI = torch.CudaTensor(self.n_objects,self.n_features+1):zero()


  --pos_factor = (#self.transitions.take_action_index-#self.transitions.good_take_action_index+1)/(#self.transitions.good_take_action_index+1)
  self.A:copy(self.A_init)
  local replay_obj_index_table ,_=  self.transitions:getObjIndexTable()

  local i = 0
  local j = 1
  self.active_beta = self.beta*torch.log(#replay_obj_index_table)/torch.log(self.replay_memory)
  local ss_buff = torch.CudaTensor(self.minibatch_size,self.input_dims[1],self.input_dims[2],self.input_dims[3])
  local aa_buff = torch.Tensor(self.minibatch_size)
  local ee_buff = torch.Tensor(self.minibatch_size)

  while i < #replay_obj_index_table  do
    i=i+1
    local replay_ind = replay_obj_index_table[i]

    if self.transitions:isActionIndexSafeToGet(replay_ind) then
        local s, _, _,_, _,a_o,e = self.transitions:getByActionIndex(replay_ind)
        assert(a_o ~= 0,"should only see here AEN actions!")
        s = s:cuda()
        s = s:div(255):resize(1, unpack(self.input_dims))
        ss_buff[j]:copy(s)
        aa_buff[j]=a_o
        ee_buff[j]=e
        j=j+1
    end
    if j==(self.minibatch_size+1) then
      j=1
      local phi_buff =  torch.CudaTensor(self.minibatch_size,self.n_features+1)
      phi_buff:narrow(2,1,self.n_features):copy(self.obj_target_network:forward(ss_buff))
      phi_buff:narrow(2,self.n_features+1,1):fill(1)
      for t=1,self.minibatch_size do
        if ee_buff[t]==1 then -- bad action
          PHI[aa_buff[t]]:add(phi_buff[t])
        else -- good action
          local pos_factor = (self.transitions.action_histogram[2][aa_buff[t]] - self.transitions.action_histogram[1][aa_buff[t]] +1)/(self.transitions.action_histogram[1][aa_buff[t]] +1)
          phi_buff[t]:mul(pos_factor)
        end
        self.A[aa_buff[t]]:add(torch.mm(phi_buff:narrow(1,t,1):transpose(1,2),phi_buff:narrow(1,t,1)))--,phi:transpose(1,2)))
      end
    end
  end

  local A_Mat = torch.FloatTensor(self.n_objects,self.n_features+1,self.n_features+1)
  A_Mat:copy(self.A)
  for i = 1, self.A:size()[1] do
    self.A[i]:copy(torch.inverse(A_Mat[i]))
  end

  PHI = PHI:reshape(self.n_objects,self.n_features+1,1)
  local THETA = torch.bmm(self.A,PHI):squeeze()
  local weights = ((THETA:narrow(2,1,self.n_features)):reshape(self.n_objects,self.n_features))
  local biases  = ((THETA:narrow(2,self.n_features+1,1)):reshape(self.n_objects))


  self.obj_target_network    =  self.obj_network:clone()
  self.obj_target_network.modules[self.obj_target_network:size()].weight:copy(weights)
  self.obj_target_network.modules[self.obj_target_network:size()].bias:copy(biases)

  self.obj_target_network:evaluate()

  self.obj_network    =  self.obj_target_network:clone()

  if self.verbose > 1 then print ("shallow update took: " .. sys.clock()-start_t) end
end


function nql:shallow_elimination(testing,prediction,phi)
  --local start_t = sys.clock()
  phi = phi:reshape(1,self.n_features+1,1):expand(self.n_objects,self.n_features+1,1)
  local conf = torch.sqrt(torch.abs(self.active_beta*torch.bmm(phi:transpose(2,3),torch.bmm(self.A,phi)))):squeeze()
  if testing then
    self.val_conf_buf[1][eval_step]=prediction:mean() -- avg AEN eliminations
    self.val_conf_buf[2][eval_step]=torch.add(prediction,-conf):mean() -- avg hard eliminations with confidence
    self.val_conf_buf[3][eval_step]=conf:mean()
    self.val_conf_buf[4][eval_step]=conf:min()
    self.val_conf_buf[5][eval_step]=conf:max()
  end
  return conf-- self.sigmoid:forward(conf):mul(0.5):squeeze()
end
