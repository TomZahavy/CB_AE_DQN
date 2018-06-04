--- This file defines the class Game environment of a Toy MDP, meant for experimenting with kuz DQN implimentation in lua
local toyMdpFramework = torch.class('toyMdpFramework')
local w2vutils = require 'w2vutils'

symbols = {}
symbol_mapping = {}
tmpTable = {}
local sentence_size = 3
local start_index = 1

-- source:

function parseLine( line, start_index)
	-- parse line to update symbols and symbol_mapping
	-- IMP: make sure we're using simple english - ignores punctuation, etc.
	local sindx
  local list_words = string.split(line, " ")
	start_index = start_index or 1
	for i=start_index,#list_words do
		local word = split(list_words[i], "%a+")[1]
		-- word = word:lower()
		if symbol_mapping[word] == nil then
			sindx = #symbols + 1
			symbols[sindx] = word
			symbol_mapping[word] = sindx
		end
	end
end

function embbedingTable()
	tmpTable = torch.zeros(#symbols,300)
	for i=1,#symbols do
		tmpTable[i] = w2vutils:word2vec(symbols[i])
	end
	tmpTable = (128*(tmpTable+1)):floor() --ELA
	--print("tmpTable",tmpTable) --ELA
end


function split(s, pattern)
	local parts = {}
	for i in string.gmatch(s, pattern) do
  	table.insert(parts, i)
	end
	return parts
end

--                                Word2Vec
function textEmbedding(line)
	-- 300 is size of word2vec embbeding
	local matrix = torch.zeros(sentence_size,300)
	input_text = string.split(line, " ")
	for i=1 ,#input_text do
		-- check input_text is not longer than sentence_size, line was truncated
	  if i > sentence_size then
			print('number of words in sentence is larger than' .. sentence_size)
			break
		end
		local word = input_text[i]
		local normlized_word = split(word, "%a+")[1]
		--ignore words not in vocab
  	if symbol_mapping[normlized_word] then
			 matrix[i] = tmpTable[symbol_mapping[normlized_word]]
		else
			print(normlized_word .. ' not in vocab')
		end
	end
	return matrix
end


--[===[                          HOT ONE
function textEmbedding(line)
	local matrix = torch.zeros(sentence_size,#symbols)
	input_text = string.split(line, " ")
	for i=1 ,#input_text do
		-- check input_text is not longer than sentence_size, line was truncated
	  if i > sentence_size then
			print('number of words in sentence is larger than' .. sentence_size)
			break
		end
		local word = input_text[i]
		local normlized_word = split(word, "%a+")[1]
		normlized_word = normlized_word:lower()
		--ignore words not in vocab
  	if symbol_mapping[normlized_word] then
			matrix[i][symbol_mapping[normlized_word]] = 1
		else
			print(normlized_word .. ' not in vocab')
		end
	end
	return matrix
end
--]===]

-- The GameEnvironment class.
local gameEnv = torch.class('toyMdpFramework.GameEnvironment')
local game = {
    {next_stage = {1,2}, descriptor = "Go Left" ,reward= 0,terminal = false },        -- 1
    {next_stage = {8,3}, descriptor = "Dont Go Left" ,reward= 0,terminal = false },   -- 2
    {next_stage = {2,6}, descriptor = "Dont Go Right" ,reward= 0,terminal = false },  -- 3
    {next_stage = {6,5}, descriptor = "Go Right" ,reward= 0,terminal = false },       -- 4
    {next_stage = {1,6}, descriptor = "Go Left" ,reward= 0,terminal = false },        -- 5
    {next_stage = {9,5}, descriptor = "Dont Go Left" ,reward= 0,terminal = false },   -- 6
    {next_stage = {3,10}, descriptor = "Dont Go Right" ,reward= 0,terminal = false }, -- 7
    {next_stage = {7,3}, descriptor = "Go Right" ,reward= 0,terminal = false },       -- 8
    {next_stage = {4,10}, descriptor = "Go Left" ,reward= 0,terminal = false },       -- 9
    {next_stage = {10,10}, descriptor = "Win" ,reward= 10,terminal = true }           --10
}
--@ screen: word to vec embedding of the current state in some fixed size buffer (zero padded or truncated)
--@ reward: current score - step number
--@ terminal: dead boolean flag
--@ game_env: object from class GameEnvironment
--@ game_actions: array of actions for agent to affect env
--@ agent: dqn player returned by setup function
--@ opt: arguments passed from terminal when session is launched.

function gameEnv:__init(_opt)
    print("@DEBUG Initializing toy framework")
		-- build vocab
		for i=1, #game do
			parseLine(game[i].descriptor)
		end
		embbedingTable()
		self._state={}
		self._state.reward = 0
        self._state.terminal = false
		self._state.observation = textEmbedding("Go Left")
		self._step_limit = 100
		local LEFT = 1
		local RIGHT = 2
		self._actions= {LEFT,RIGHT}
		self._current_stage = 1
        self._step_penalty = -1

  return self
end

--[[ this helper method assigns the state arguments ]]
function gameEnv:_updateState(descriptor, reward, terminal)
	self._state.reward       = reward -- @ASK: should this be delta reward or global score ?
  self._state.terminal     = terminal
	self._state.observation  = textEmbedding(descriptor) -- @TODO: in our case frame is the state string descriptor we shold store here the word2vec rep
  return self
end

function gameEnv:getState()
	return self._state.observation, self._state.reward, self._state.terminal -- frame,reward,terminal
end

function gameEnv:newGame()
	self:_updateState(game[1].descriptor ,0,false)
  self._current_stage = 1
  return self:getState()
end

function gameEnv:nextRandomGame()
  return self:newGame()
end

function gameEnv:step(action, training)
	local next_stage, reward, terminal, string
  next_stage = game[self._current_stage].next_stage[action]
	self._current_stage = next_stage
	reward = game[next_stage].reward + self._step_penalty
  terminal = game[next_stage].terminal
  string = game[next_stage].descriptor
  self:_updateState(string, reward, terminal)
  return self:getState()
end

--[[ Function returns the number total number of pixels in one frame/observation
from the current game.
]]
function gameEnv:nObsFeature()
		-- Yamin: we need to redefine the dimentions
    -- return self.api_agent.getStateDims()
		local dim_size = torch.Tensor(2)

		dim_size[1] = sentence_size
		dim_size[2] = 300 --@DEBUG_DIM(word representation)

		return dim_size -- assume matrix size is 3x300

end

-- Function returns a table with valid actions in the current game.
function gameEnv:getActions()
      return self._actions
end

return toyMdpFramework
