require 'torch' 
require 'ZorkFramework'
local i=0
local opt={env_params={game_scenario=3}}
	local gameEnv = ZorkFramework.GameEnvironment(opt)
--stress test
local anomaly_count = 0
for i=1, 1000 do
	s,r,t,ss1 = gameEnv:newGame()
	s,r,t,ss2 = gameEnv:step(gameEnv:getActions()[3],false) -- north
	s,r,t,ss3 = gameEnv:step(gameEnv:getActions()[3],false) -- north
	s,r,t,ss4 = gameEnv:step(gameEnv:getActions()[7],false) -- climb tree
	s,r,t,ss5 = gameEnv:step(gameEnv:getActions()[8],false) -- take egg
	s,r1,t,ss6 = gameEnv:step(gameEnv:getActions()[9],false) -- open egg
	s,r,t,ss7 = gameEnv:step(gameEnv:getActions()[10],false) -- look

	if r1 < 99 then
	print("at iter".. i, ":",r,t)
	print("with the folowing states:", ss1 .. ss2 .. ss3 .. ss4 .. ss5 .. ss6 .. ss7)
	anomaly_count= anomaly_count +1
	end
end
print("iteration ended after num",i)
print("with anomaly rate",anomaly_count/i)
print(gameEnv:getActions())

--story test
local opt={env_params={game_scenario=6}}
local gv = ZorkFramework.GameEnvironment(opt)
print ('init success')
local file = io.open("human_optimal_path.txt", "r") -- r read mode and b binary mode
i=0
while true and file do
	local instruction = file:read("*line")
	--print ('success',instruction .. '\0')
	if instruction == nil then break end
	local s,r,t,ss,bad_p =  gv:step(instruction .. '\0',false)
	i  = i+1
	print ("action" ,instruction ,"\nstate\n",ss,"reward:",r,"parse",bad_p)

	if bad_p == 1 then print("error unexpected bad instruction", i) break end 
end
gv:getGameScore()
file:close()

