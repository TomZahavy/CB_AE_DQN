require('toyMdpFramework')
parseLine("Go Left Right Dont Win")
print("symbols are ",symbols)
print("symbol_mapping is ",symbol_mapping)
opt={}
local gameEnv = toyMdpFramework.GameEnvironment(opt)
print(gameEnv:getState())
local right = 1
local left = 2
print(gameEnv:step(left,false)) --1 > 2
print(gameEnv:step(right,false)) --> 2 > 8
print(gameEnv:step(right,false)) --> 8 > 7
print(gameEnv:step(left,false)) --> 10
print(gameEnv:step(right,false)) --> 10
print(gameEnv:newGame()) --> 1
print(gameEnv:step(left,false)) --1 > 2
print(gameEnv:step(right,false)) --> 2 > 8
print(gameEnv:step(right,false)) --> 8 > 7
print(gameEnv:step(left,false)) --> 10
