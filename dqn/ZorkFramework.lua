--- This file defines the class Game environment of a Toy MDP, meant for experimenting with kuz DQN implimentation in lua
local ZorkFramework = torch.class('ZorkFramework')
print ("reading w2v file, wait")
local w2vutils = require 'w2vutils'
symbols = {}
symbol_mapping = {}
tmpTable = {}
local sentence_size = 65 -- 15 inventory + 50 for state description
local reserved_for_inventory = 15
local start_index = 1
--local cons = require 'pl.pretty'

-- source:
--[[this function reads a whole file and returns it as a stringToUpper
  used only on small files genereted by the game for each step]]
function read_file(path)
    local file = io.open(path, "r") -- r read mode and b binary mode
    if file == nil then
      print("error: file not found")
      return nil
    end
    local content = file:read("*all") -- *a or *all reads the whole file
    file:close()
    return content
end

function split(s, pattern)
  local parts = {}
  for i in string.gmatch(s, pattern) do
      table.insert(parts, i)
    end
  return parts
end

--                                Word2Vec
function textEmbedding(game_feedback,inventory_text)
  -- 300 is size of word2vec embbeding
    local matrix = torch.zeros(sentence_size,300)
    local input_text = string.split(game_feedback, " ")

    for i=1 ,#input_text do
  -- check input_text is not longer than sentence_size, line was truncated
    if i > sentence_size - reserved_for_inventory then
  --print('number of words in sentence is larger than ' .. sentence_size - reserved_for_inventory)
  break
    end

    local word = input_text[i]
    local normlized_word = split(word, "%a+")[1] --TODO explain
  --ignore words not in vocab
    local normlized_vec = w2vutils:word2vec(normlized_word)
    if normlized_vec then
      matrix[i] = normlized_vec
    else
      print(normlized_word .. ' not in vocab')
    end
  end


  -- append player inventory
  -- check for empty inventory
  --print (inventory_text)
  if inventory_text ~= nil then
    if inventory_text:match("You are empty handed.\n") then return matrix end
    -- split inventory text and preform embedding
    inventory_text = string.split(inventory_text, " ")
    --print (inventory_text)
    --[[ 3 is the number of words in common to all inventory messages so we ignore them.
    "You are carrying:\n"]]
    for i=1 ,(#inventory_text - 3) do
      -- check input_text is not longer than sentence_size, line was truncated
      --print ("inventory index " .. i)
      if i > reserved_for_inventory then
      --print('number of items in sentence is larger than' .. sentence_size - reserved_for_inventory)
        break
      end

      word = input_text[i + 3]
      if word ~= nil then   -- skip empty spaces
        normlized_word = split(word, "%a+")[1]
        --ignore words not in vocab
        normlized_vec = w2vutils:word2vec(normlized_word)
        if normlized_vec then
          matrix[sentence_size - reserved_for_inventory + i] = normlized_vec
        else
          print('warning ' .. normlized_word .. ' not in vocab')
        end
      end
    end
  end
  return matrix

end

local zork = require ('zork')
-- The GameEnvironment class.
local gameEnv = torch.class('ZorkFramework.GameEnvironment')
--@ screen: word to vec embedding of the current state in some fixed size buffer (zero padded or truncated)
--@ reward: current score - step number
--@ terminal: dead boolean flag
--@ game_env: object from class GameEnvironment
--@ game_actions: array of actions for agent to affect env
--@ agent: dqn player returned by setup function
--@ opt: arguments passed from terminal when session is launched.

function gameEnv:__init(_opt)

    print("@DEBUG Initializing zork framework with scenario ".._opt.env_params.game_scenario)
    --init game and define static actions
    --cons.dump(_opt.env_params)
    assert(_opt.env_params.game_scenario)  --just a sanity check
    local scenario = _opt.env_params.game_scenario or 1
    self.scenario = scenario

    -- for reward shaping
    self.bad_parse_penalty_scale = _opt.env_params.bad_parse_penalty_scale or 0
    --these extrinsic rewards are set in newGame method and given once per episode based on string matching
    self._additional_rewards = {}
    self._inventory_rewards= {}
    self._state={}

    self.tot_steps = 0
    self.tot_inits = 0
    self:newGame()
    --creates object actions from an object list
    function genObjActionTable(obj_list)
      local act = {}
      -- registers entries like this: {action = "take egg",desc = "take item" , object = 1}, ..
      table.foreach(obj_list,function(k,v) table.insert(act,{action="take "..v,desc="take item",object = k}) end)
      return act
    end

    function genActions(action_string_table)
      local act = {}
      --FIXME here we assign all actions with an object index regardless of the actual object,
      -- this is a hack to avoid changing the code to run AEN on all actions
      table.foreach(action_string_table,function(k,v) table.insert(act,{action=v,object = k}) end)
      return act
    end

    function concatTable(src1,src2)
        table.foreach(src2,function(k,v) table.insert(src1,v) end)
        return src1
    end

    local basic_actions = {
        {action = "open egg",desc = "try to open item"},
        {action = "go east"  ,desc = "move east"},
        {action = "go west"  ,desc = "move west"},
        {action = "go north",desc = "move north"},
        {action = "go south",desc = "move south"},
        {action = "go up"  ,desc = "move up"},
        {action = "go down"  ,desc = "move down"},
        {action = "climb tree",desc = "climb up the large tree"},
        --give the agent another chance to observe its environment
        {action = "look",desc = "observe the environment"},
    }
    -- self._actions = self.concatTable(basic_actions,genObjActionTable(self._objects))
    local basic_objects = {"egg","door","tree","leaves","nest"}

    local obj_ext1 = {"bag","bottle","rope", "sword","lantern","knife","mat","mailbox",
    "rug","case","axe", "diamond","leaflet","news","brick"}

    local action_ext1 = {
        {action = "enter",desc = "enter location"},
        {action = "open the window",desc = "open a passage way"},
        {action = "turn lamp on",desc = "turn the light on"},
        {action = "move rug",desc = "move the large rug"},
        {action = "open trap door",desc = "open a celler door"},
        {action = "hit troll with sword", desc = "try to kill deadly monster"}
    }

    --define some default parameters for all scenarios that may be overwritten
    self._step_penalty = -1
    self._step_limit = 100
    self.goal_reward = 100
    self._actions = basic_actions
    self._objects = basic_objects
    if scenario < 5 then
      if scenario < 4 then
        self._terminal_string = "There is no obvious way to open the egg.\0"
        self._inventory_rewards_target=1
      else -- scenario 4
        self._terminal_string = "Your sword is no longer glowing."
      end
      --set objects and action spaces
      if scenario > 1 then
        -- extend to 20 objects
        self._objects = concatTable(self._objects,obj_ext1)
        if scenario >= 3 then
        --extend the number of operations artificially with garbage objects
          for i=#self._objects + 1, 200 do
            table.insert(self._objects,"garbage")
          end
          if scenario == 4 then --extended quest actions and intermidiate rewards
            --enable exteded quest with none object actions
            self._actions = concatTable(self._actions,action_ext1)
          end
        end
      end
      --attach object actions
      self._actions = concatTable(self._actions,genObjActionTable(self._objects))
    else --scenarios 5-6 use external action file (no bloat actions)
      --define actions from file
      local action_strings = read_file('action_strings.txt')
      local tab = split(action_strings,'[^\r\n]+')
      self._actions = genActions(tab)
      self._objects = self._actions
      if scenario == 5 then
        self._terminal_string = "Your sword is no longer glowing."
      elseif scenario == 6  then
        --no step penalty, no goal state
        self._step_penalty = 0
        self._step_limit = 200
        self._terminal_string = nil --run wild
      elseif scenario == 7 then
        self.scenario=3.5
        self._terminal_string = "There is no obvious way to open the egg.\0"
      end
    end

    --print("@DEBUG: action table")
    --cons.dump(self._actions)
    return self
end

--[[ this helper method assigns the state arguments ]]
function gameEnv:_updateState(descriptor, reward, terminal)
  self._state.reward       = reward
  self._state.terminal     = terminal
  self._state.observation  = textEmbedding(unpack(descriptor))
  return self
end


function gameEnv:getState()
  return self._state.observation, self._state.reward, self._state.terminal -- frame,reward,terminal
end

function gameEnv:getCurrentStringState()
  return self.state_string,self.inventory_string
end

function gameEnv:newGame()
  self.tot_inits = self.tot_inits+1
  --print("start new game number",self.tot_inits)
  local result_file_name = zork.zorkInit()
  local result_string,inventory_text = read_file(result_file_name),read_file(zork.zorkInventory())
  self.state_string=result_string
  self.inventory_string=inventory_text
  self:_updateState({result_string},0,false)
  --uncomment to setup intermidiate rewards
  --[[
  if self.scenario > 3 then
    self._additional_rewards['There is no obvious way to open the egg.\0']=50
    self._additional_rewards['The door crashes shut, and you hear someone barring it.\0']=20
  end
  ]]

  if self.scenario < 4 then
    self._inventory_rewards['egg']=0
    self.goal_items_collected=0
  end

  s,r,t =  self:getState()
  return s,r,t, result_string
end

function gameEnv:nextRandomGame()
  return self:newGame()
end

function gameEnv:step(action, training)
  self.tot_steps = self.tot_steps+1
  local current_score, previous_score, previous_lives, reward, terminal
  previous_score ,previous_lives = zork.zorkGetScore(),zork.zorkGetLives()

  local result_file_name,bad_command
  if type(action) == 'string' then
  --allow using free string action for flexibility in testing
  result_file_name,bad_command = zork.zorkGameStep(action)
  else
    --normally expecting NQL to provide a table action with additional info
    result_file_name,bad_command = zork.zorkGameStep(action.action)
  end

  -- read step result string
  local result_string,inventory_text = read_file(result_file_name),read_file(zork.zorkInventory())
  self.state_string=result_string
  self.inventory_string=inventory_text
    --this slows things down a bit but helps to get more information regarding possible
    --movement options
  if bad_command == 0 then
    if result_string:match("You can't go that way.") or
         result_string:match("There is a wall there.") or
         result_string:match("There is no way") or
         result_string:match("I could't find anything.") or
         result_string:match("Wasent he a sailor") then
      bad_command = 1
    end
  end

  current_score = zork.zorkGetScore()
  -- step penalty for bad parse and terminal signal
  if training then
    terminal = previous_lives > zork.zorkGetLives() -- every time we lose life
    bad_parse_penalty = -1*bad_command*self.bad_parse_penalty_scale
  else
    terminal = zork.zorkGetLives() == 0 -- when evaluating agent only when no more lives
    bad_parse_penalty = 0
  end
  reward = current_score - previous_score + self._step_penalty + bad_parse_penalty


  -- check for goal state if defined
  if self._terminal_string and result_string:match(self._terminal_string) then
    if (self.scenario < 4  and self.goal_items_collected == self._inventory_rewards_target)
         or self.scenario >= 4 then
      terminal = true
      reward = reward + self.goal_reward -- give additional reward
      --FIXME use verbose > 2 here
      if --[[self.scenario > 3 and]] not training then
        print("@DEBUG: ####goal state reached in " .. zork.zorkGetNumMoves()
                .. " steps ####", zork.zorkGetLives())
        print(self.goal_items_collected)
      end
    end
  else
  --check for any of the secondary goal states
  table.foreach(self._additional_rewards,
    function(k,v)
      if result_string:match(k) then
        if v>=0 then 
          reward = reward + v
          self._additional_rewards[k]=-1
          if not training then
            print("@DEBUG: intermidiate goal state: " .. k
                    .." reached in",zork.zorkGetNumMoves(),"steps")
          end
        end
      end
    end)
  --check for inventory items
  table.foreach(self._inventory_rewards,
    function(k,v)
      if inventory_text:match(k) then
        if v>=0 then 
          reward = reward + v
          self._inventory_rewards[k]=-1
          if not training then
            print("@DEBUG: inventory goal state: " .. k
            .." reached in",zork.zorkGetNumMoves(),"steps")
          end
          if self.scenario < 4 then 
            self.goal_items_collected=self.goal_items_collected+1 end
          end
      end
    end)
  end
  -- early termination
  if zork.zorkGetNumMoves() > self._step_limit - 1 then
    terminal = true
  end

--[[
    if terminal then print("terminated after",zork.zorkGetNumMoves(),"steps")
       print("total new games started",self.tot_inits,"total steps", self.tot_steps)
    end
]]
  self:_updateState({result_string,inventory_text}, reward, terminal)
  s, r, t = self:getState()
  return s,r,t,result_string,bad_command,command
end

--[[ Function returns the number total number of pixels in one frame/observation
from the current game.]]
function gameEnv:nObsFeature()
  local dim_size = torch.Tensor(2)
  dim_size[1] = sentence_size
  dim_size[2] = 300 --(word representation)
  return dim_size

end

-- Function returns a table with valid actions in the current game.
function gameEnv:getActions()
      return self._actions, self._objects
end


return ZorkFramework
