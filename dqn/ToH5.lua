require 'hdf5'
require 'initenv'
require 'cunn'
require 'cudnn'
require 'nn'
require 'cutorch'
local myFile = hdf5.open('results.h5', 'w')

for i = 1,5 do
  local agent_name = 'zork_scenario_3_merged_Q-arch_conv_Q_net_seed_' .. i .. i .. i .. '_double_elimination_AEN-arch_conv_AEN_SE_beta_0.05_freq_40000_max_-1_sample_0_drop_prob_0.7_lr0.00043_209a' .. '.t7'
  local agent = torch.load(agent_name)
  local r = torch.Tensor(agent.reward_history)
  myFile:write('/results/'..i, r)

end
myFile:close()
