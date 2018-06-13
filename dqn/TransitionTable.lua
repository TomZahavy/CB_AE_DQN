--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'image'

local trans = torch.class('dqn.TransitionTable')


function trans:__init(args)
    self.stateDim = args.stateDim
    self.numActions = args.numActions
    self.numObjects = args.numObjects
    self.histLen = args.histLen
    self.maxSize = args.maxSize or 1024^2
    self.bufferSize = args.bufferSize or 1024
    self.histType = args.histType or "linear"
    self.histSpacing = args.histSpacing or 1
    self.zeroFrames = args.zeroFrames or 1
    self.nonTermProb = args.nonTermProb or 1
    self.nonEventProb = args.nonEventProb or 1
    self.gpu = args.gpu
    self.numEntries = 0
    self.insertIndex = 0
    self.sample_parse_buffer = args.sample_parse_buffer
    self.histIndices = {}
    self.AEN_sample_bias = args.AEN_sample_bias or 0
    --successful parsed action in replay memory
    --played action counter in replay memory
    self.action_histogram = torch.zeros(2,self.numActions)
    local histLen = self.histLen
    if self.histType == "linear" then
        -- History is the last histLen frames.
        self.recentMemSize = self.histSpacing*histLen
        for i=1,histLen do
            self.histIndices[i] = i*self.histSpacing
        end
    elseif self.histType == "exp2" then
        -- The ith history frame is from 2^(i-1) frames ago.
        self.recentMemSize = 2^(histLen-1)
        self.histIndices[1] = 1
        for i=1,histLen-1 do
            self.histIndices[i+1] = self.histIndices[i] + 2^(7-i)
        end
    elseif self.histType == "exp1.25" then
        -- The ith history frame is from 1.25^(i-1) frames ago.
        self.histIndices[histLen] = 1
        for i=histLen-1,1,-1 do
            self.histIndices[i] = math.ceil(1.25*self.histIndices[i+1])+1
        end
        self.recentMemSize = self.histIndices[1]
        for i=1,histLen do
            self.histIndices[i] = self.recentMemSize - self.histIndices[i] + 1
        end
    end
    --transition table
    self.s = torch.ByteTensor(self.maxSize, self.stateDim):fill(0)
    self.a = torch.ShortTensor(self.maxSize):fill(0)
    self.a_o = torch.ShortTensor(self.maxSize):fill(0)
    self.bad_command  = torch.ByteTensor(self.maxSize):fill(0)
    self.r = torch.zeros(self.maxSize)
    self.t = torch.ByteTensor(self.maxSize):fill(0)
    self.action_encodings = torch.eye(self.numActions)

    -- Tables for storing the last histLen states.  They are used for
    -- constructing the most recent agent state more easily.
    self.recent_s = {}
    self.recent_a = {}
    self.recent_t = {}
    self.recent_a_o = {}
    self.recent_bad_command  = {}
    self.take_action_index = {}
    self.good_take_action_index = {}

    local s_size = self.stateDim*histLen
    --sampling buffers
    self.buf_bad_command  = torch.ByteTensor(self.bufferSize):fill(0)
    self.buf_s_for_obj    = torch.ByteTensor(self.bufferSize, s_size):fill(0)
    self.buf_a_for_obj      = torch.ShortTensor(self.bufferSize):fill(0)
    self.buf_a_o          = torch.ShortTensor(self.bufferSize):fill(0)

    self.buf_a      = torch.ShortTensor(self.bufferSize):fill(0)
    self.buf_r      = torch.zeros(self.bufferSize)
    self.buf_term   = torch.ByteTensor(self.bufferSize):fill(0)
    self.buf_s      = torch.ByteTensor(self.bufferSize, s_size):fill(0)
    self.buf_s2     = torch.ByteTensor(self.bufferSize, s_size):fill(0)


    if self.gpu and self.gpu >= 0 then
        self.gpu_s  = self.buf_s:float():cuda()
        self.gpu_s2 = self.buf_s2:float():cuda()
        self.gpu_s_for_obj = self.buf_s_for_obj:float():cuda()
    end
end

function trans:getObjIndexTable()
  return self.take_action_index,self.good_take_action_index
end

function trans:reset()
    self.numEntries = 0
    self.insertIndex = 0
    self.take_action_index = {}
    self.good_take_action_index = {}
end


function trans:size()
    return self.numEntries
end


function trans:empty()
    return self.numEntries == 0
end

function trans:fill_buffer()
    assert(self.numEntries >= self.bufferSize)
    -- clear CPU buffers
    self.buf_ind = 1
    local ind
    for buf_ind=1,self.bufferSize do
        local s, a, r, s2, term = self:sample_one(1)

        self.buf_s[buf_ind]:copy(s)
        self.buf_a[buf_ind] = a
        self.buf_r[buf_ind] = r
        self.buf_s2[buf_ind]:copy(s2)
        self.buf_term[buf_ind] = term
    end

    if self.sample_parse_buffer then
        --local obj_sample_histo = torch.zeros(2,self.numObjects+1)

        local size_take = #self.take_action_index
        local size_g_take = #self.good_take_action_index
        assert(size_g_take > 1 and size_take > 1)
        local ratio =size_g_take/size_take

        -- fill object related buffers
        for buf_ind=1,self.bufferSize do

            local s, a,r,s2,t,a_o,bad_command, index
            repeat
                if torch.uniform() < self.AEN_sample_bias then 	--prioritize good take actions to amend rep bias
                    index = self.good_take_action_index[torch.random(#self.good_take_action_index)]
                else
                    --sample any take action
                    index = self.take_action_index[torch.random(#self.take_action_index)]
                end
            until self:isActionIndexSafeToGet(index)-- for state concatination

            s, a, r, s2 ,t, a_o, bad_command = self:getByActionIndex(index) -- sample from the transition table
            assert(a_o ~= 0) --here we should not have none object action

            self.buf_s_for_obj[buf_ind]:copy(s)
            self.buf_a_for_obj[buf_ind] = a
            self.buf_a_o[buf_ind] = a_o
            self.buf_bad_command[buf_ind] = bad_command
            --obj_parse_histo[a_o] = obj_parse_histo[a_o] + 1 - bad_command
            --good parse samples in buffer
            --obj_sample_histo[1][a_o+1] = obj_sample_histo[1][a_o+1] + 1 - bad_command
            --total object samples in buffer
            --obj_sample_histo[2][a_o+1] = obj_sample_histo[2][a_o+1] + 1

        end
        --print("@DEBUG sample buffer:\negg,door,tree,leaves,nest,bag,bottle,rope, sword,lantern")
        --print(obj_sample_histo[{ {}, {2,11} }])
    end
    self.buf_s  = self.buf_s:float():div(255)
    self.buf_s2 = self.buf_s2:float():div(255)
    if self.sample_parse_buffer then
      self.buf_s_for_obj = self.buf_s_for_obj:float():div(255)
    end

    if self.gpu and self.gpu >= 0 then
        self.gpu_s:copy(self.buf_s):cuda()
        self.gpu_s2:copy(self.buf_s2):cuda()
        if self.sample_parse_buffer then
          self.gpu_s_for_obj:copy(self.buf_s_for_obj):cuda()
        end
    end
end

function trans:report(verbose)
    print("Total actions:  " .. self.numEntries .. " Take actions: " .. #self.take_action_index .. " Valid take actions: " .. #self.good_take_action_index)
    if verbose > 1 then
      print("replay action histogram: {good parse/action samples}\n", self.action_histogram) --[{ {},{1,30} }])
    end
end

function trans:sample_one()
    assert(self.numEntries > 1)
    local index
    local valid = false
    while not valid do
        -- start at 2 because of previous action
        index = torch.random(2, self.numEntries-self.recentMemSize)
        if self.t[index+self.recentMemSize-1] == 0 then
            valid = true
        end
        if self.nonTermProb < 1 and self.t[index+self.recentMemSize] == 0 and
            torch.uniform() > self.nonTermProb then
            -- Discard non-terminal states with probability (1-nonTermProb).
            -- Note that this is the terminal flag for s_{t+1}.
            valid = false
        end
        if self.nonEventProb < 1 and self.t[index+self.recentMemSize] == 0 and
            self.r[index+self.recentMemSize-1] == 0 and
            torch.uniform() > self.nonTermProb then
            -- Discard non-terminal or non-reward states with
            -- probability (1-nonTermProb).
            valid = false
        end
    end

    return self:get(index)
end


function trans:sample(batch_size)
    local batch_size = batch_size or 1
    assert(batch_size <= self.bufferSize)

    if not self.buf_ind or self.buf_ind + batch_size - 1 > self.bufferSize then
        --print("filling sample buffer")
        self:fill_buffer()
    end

    local index = self.buf_ind

    self.buf_ind = self.buf_ind+batch_size
    local range = {{index, index+batch_size-1}}

    local buf_s, buf_s2, buf_a, buf_r, buf_term = self.buf_s, self.buf_s2,
        self.buf_a, self.buf_r, self.buf_term
    -- buffer extension for object related actoins and states
    local buf_s_for_obj, buf_a_for_obj, buf_a_o, buf_bad_command = self.buf_s_for_obj, self.buf_a_for_obj,
        self.buf_a_o,  self.buf_bad_command
    if self.gpu and self.gpu >=0  then
        buf_s = self.gpu_s
        buf_s2 = self.gpu_s2
        buf_s_for_obj = self.gpu_s_for_obj
    end

    return buf_s[range], buf_a[range], buf_r[range], buf_s2[range], buf_term[range],
        buf_s_for_obj[range], buf_a_for_obj[range], buf_a_o[range],buf_bad_command[range]
end


function trans:concatFrames(index, use_recent)
    local s,t
    if use_recent then
        s, t = self.recent_s, self.recent_t
    else
        s, t = self.s, self.t
    end
    --print("@DEBUG: concatFrames s\n", table.unpack(s))
    --print("@DEBUG: concatFrames t\n", table.unpack(t))
    local fullstate = s[1].new()
    --print("@DEBUG: concatFrames fullstate", fullstate)
    fullstate:resize(self.histLen, unpack(s[1]:size():totable()))

    -- Zero out frames from all but the most recent episode.
    local zero_out = false
    local episode_start = self.histLen

    for i=self.histLen-1,1,-1 do
        if not zero_out then
            for j=index+self.histIndices[i]-1,index+self.histIndices[i+1]-2 do
                if t[j] == 1 then
                    zero_out = true
                    break
                end
            end
        end

        if zero_out then
            fullstate[i]:zero()
        else
            episode_start = i
        end
    end

    if self.zeroFrames == 0 then
        episode_start = 1
    end
    --print("@DEBUG: transition table copy")
    -- Copy frames from the current episode.
    for i=episode_start,self.histLen do
        --print("@DEBUG: transition table self.histIndices",table.unpack(self.histIndices))
        --print("@DEBUG: iteration",i)
        --print("@DEBUG: transition table fullstate\n",fullstate)
        --print("@DEBUG: transition table s[index+self.histIndices[i]-1]\n",s[index+self.histIndices[i]-1])
        fullstate[i]:copy(s[index+self.histIndices[i]-1])
        --print("@DEBUG: transition table fullstate after copy\n",fullstate)

    end

    return fullstate
end


function trans:concatActions(index, use_recent)
    local act_hist = torch.FloatTensor(self.histLen, self.numActions)
    if use_recent then
        a, t = self.recent_a, self.recent_t
    else
        a, t = self.a, self.t
    end

    -- Zero out frames from all but the most recent episode.
    local zero_out = false
    local episode_start = self.histLen

    for i=self.histLen-1,1,-1 do
        if not zero_out then
            for j=index+self.histIndices[i]-1,index+self.histIndices[i+1]-2 do
                if t[j] == 1 then
                    zero_out = true
                    break
                end
            end
        end

        if zero_out then
            act_hist[i]:zero()
        else
            episode_start = i
        end
    end

    if self.zeroFrames == 0 then
        episode_start = 1
    end

    -- Copy frames from the current episode.
    for i=episode_start,self.histLen do
        act_hist[i]:copy(self.action_encodings[a[index+self.histIndices[i]-1]])
    end

    return act_hist
end


function trans:get_recent()
    -- Assumes that the most recent state has been added, but the action has not
    return self:concatFrames(1, true):float():div(255)
end

function trans:isActionIndexSafeToGet(action_ind)
  return action_ind > self.recentMemSize -1 and action_ind < self.numEntries
end
function trans:getByActionIndex(action_index)
  return self:get(action_index-self.recentMemSize+1)
end

function trans:get(index)
    local s = self:concatFrames(index)
    local s2 = self:concatFrames(index+1)
    local ar_index = index+self.recentMemSize-1

    return s, self.a[ar_index], self.r[ar_index], s2, self.t[ar_index+1], self.a_o[ar_index], self.bad_command[ar_index]
end

function trans:add(s, a, r, term, a_o,bad_command)

    assert(s, 'State cannot be nil')
    assert(a, 'Action cannot be nil')
    assert(r, 'Reward cannot be nil')
    assert(a_o, 'an object cannot be nil')
    assert(bad_command, 'Reward cannot be nil')
    --print("@DEBUG: transition table add state in\n", s)
    --print("@DEBUG: transition table add self state\n", self.s:size())
    --print("@DEBUG: transition table: add -" .. a .. r .. a_o .. bad_command)



    -- Always insert at next index, then wrap around
    self.insertIndex = self.insertIndex + 1
    -- Overwrite oldest experience once at capacity
    if self.insertIndex > self.maxSize then
        self.insertIndex = 1
    end
    if self.numEntries == self.maxSize then
        if self.insertIndex  == self.take_action_index[1] then
        --[[once table is full this ensures the action index list doesnt point to tuples which
            have been removed due to the index wrap around]]
            table.remove(self.take_action_index,1)
        end
	    if self.insertIndex  == self.good_take_action_index[1] then
            table.remove(self.good_take_action_index,1)
        end
        --remove retired action from histograms:
        --good action counter decrease
        local evicted_action_index = self.a[self.insertIndex]
        self.action_histogram[1][evicted_action_index] = self.action_histogram[1][evicted_action_index] - (1 - self.bad_command[self.insertIndex])
        --played action counter (in replay memory) decrease
        self.action_histogram[2][evicted_action_index] = self.action_histogram[2][evicted_action_index] - 1
    end
    -- Overwrite (s,a,r,t) at insertIndex
    assert(a>0)
    assert(a~= nil)
    assert(a<self.numActions+1)
    self.s[self.insertIndex] = s:clone():float():mul(255)
    self.a[self.insertIndex] = a
    self.r[self.insertIndex] = r
    self.a_o[self.insertIndex] = a_o
    self.bad_command[self.insertIndex] = bad_command

    self.action_histogram[1][a] = self.action_histogram[1][a] + (1 - bad_command)
    self.action_histogram[2][a] = self.action_histogram[2][a] + 1
    -- Incremenet until at full capacity
    if self.numEntries < self.maxSize then
        self.numEntries = self.numEntries + 1
    end
    if a_o~= 0 then --take action recorded
        table.insert(self.take_action_index,self.insertIndex) --save table index
        if (bad_command == 0) then 	-- if this is a good action save the index in an aux table for later sample balancing
	        table.insert(self.good_take_action_index, self.insertIndex) end
    end

    if term then
        self.t[self.insertIndex] = 1
    else
        self.t[self.insertIndex] = 0
    end
end

function trans:add_recent_state(s, term)
    local s = s:clone():float():mul(255):byte()
    if #self.recent_s == 0 then
        for i=1,self.recentMemSize do
            table.insert(self.recent_s, s:clone():zero())
            table.insert(self.recent_t, 1)
        end
    end

    table.insert(self.recent_s, s)
    if term then
        table.insert(self.recent_t, 1)
    else
        table.insert(self.recent_t, 0)
    end

    -- Keep recentMemSize states.
    if #self.recent_s > self.recentMemSize then
        table.remove(self.recent_s, 1)
        table.remove(self.recent_t, 1)
    end
end


function trans:add_recent_action(a,a_o)
    if #self.recent_a == 0 then
        for i=1,self.recentMemSize do
            table.insert(self.recent_a, 1)
            table.insert(self.recent_a_o, 1)
        end
    end

    table.insert(self.recent_a, a)
    table.insert(self.recent_a_o, a_o)

    -- Keep recentMemSize steps.
    if #self.recent_a > self.recentMemSize then
        table.remove(self.recent_a, 1)
        table.remove(self.recent_a_o, 1)
    end
end


--[[
Override the write function to serialize this class into a file.
We do not want to store anything into the file, just the necessary info
to create an empty transition table.

@param file (FILE object ) @see torch.DiskFile
--]]
function trans:write(file)
    file:writeObject({self.stateDim,
                      self.numActions,
                      self.histLen,
                      self.maxSize,
                      self.bufferSize,
                      self.numEntries,
                      self.insertIndex,
                      self.recentMemSize,
                      self.histIndices})
end


--[[
Override the read function to desearialize this class from file.
Recreates an empty table.

@param file (FILE object ) @see torch.DiskFile
--]]
function trans:read(file)
    local stateDim, numActions, histLen, maxSize, bufferSize, numEntries, insertIndex, recentMemSize, histIndices = unpack(file:readObject())
    self.stateDim = stateDim
    self.numActions = numActions
    self.histLen = histLen
    self.maxSize = maxSize
    self.bufferSize = bufferSize
    self.recentMemSize = recentMemSize
    self.histIndices = histIndices
    self.numEntries = 0
    self.insertIndex = 0

    self.s = torch.ByteTensor(self.maxSize, self.stateDim):fill(0)
    self.a = torch.ShortTensor(self.maxSize):fill(0)
    self.a_o = torch.ShortTensor(self.maxSize):fill(0)
    self.r = torch.zeros(self.maxSize)
    self.t = torch.ByteTensor(self.maxSize):fill(0)
    self.action_encodings = torch.eye(self.numActions)

    -- Tables for storing the last histLen states.  They are used for
    -- constructing the most recent agent state more easily.
    self.recent_s = {}
    self.recent_a = {}
    self.recent_a_o = {}
    self.recent_bad_command = {}
    self.recent_t = {}
    self.take_action_index = {}
    self.good_take_action_index = {}

    self.buf_a      = torch.ShortTensor(self.bufferSize):fill(0)
    self.buf_r      = torch.zeros(self.bufferSize)
    self.buf_term   = torch.ByteTensor(self.bufferSize):fill(0)
    self.buf_s      = torch.ByteTensor(self.bufferSize, self.stateDim * self.histLen):fill(0)
    self.buf_s2     = torch.ByteTensor(self.bufferSize, self.stateDim * self.histLen):fill(0)
    -- set the sample buffer for object related states and actions
    self.buf_a_o     = torch.ShortTensor(self.bufferSize):fill(0)
    self.buf_bad_command = torch.ByteTensor(self.bufferSize):fill(0)
    self.buf_a_for_obj = torch.ShortTensor(self.bufferSize):fill(0)
    self.buf_s_for_obj = torch.ByteTensor(self.bufferSize, self.stateDim * self.histLen):fill(0)

    if self.gpu and self.gpu >= 0 then
        self.gpu_s  = self.buf_s:float():cuda()
        self.gpu_s2 = self.buf_s2:float():cuda()
        self.gpu_s_for_obj = self.buf_s_for_obj:float():cuda()
    end
end
