require 'initenv'
require "Scale"
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'

fs = require 'paths'
plt = require 'gnuplot'
local a_r_table = {}

--plot AEN agent from its original t7 file, a summary file will be created and used when available
function plotAgent(agent_name,limit,title,object_net_info,refresh)
  local r_table = {}
  addAgentToPlotTable(r_table,agent_name,limit,title,object_net_info,refresh)
  plotExpFromTable(r_table,"Average Cumulative Reward")
end

--directly plot AEN agent from summary t7 file
function plotAgentFromSummary(agent_summary,limit,title,object_net_info)
  local r_table = {}
  addAgentFromSummary(r_table,agent_summary,limit,title,object_net_info)
  plotExpFromTable(r_table,"Average Cumulative Reward")
end

--use to extract only performance metrics from the agent file
function summarizeAgent(agent_name,title,agent_o)
  agent = agent_o or torch.load(agent_name..".t7")
  local length = #agent.reward_history
  local DQN_reward = torch.Tensor(agent.reward_history)
  local AEN_loss ,AEN_acc= torch.zeros(length),torch.zeros(length)
  if  #agent.obj_loss_history < 1 then
    print("AEN records were not found")
  else
    for i=1,length do
      AEN_loss[i],AEN_acc[i] = agent.obj_loss_history[i]['AEN_loss'], agent.obj_loss_history[i]['AEN_single_accuracy']
    end
  end
  local agent_summary = {reward = DQN_reward,loss = AEN_loss,acc = AEN_acc,length = length, title = title or string.gsub(agent_name, "_", " " ),arguments = agent.arguments}
  torch.save(agent_name.."_result_summary.t7",agent_summary)
  return agent_summary
end

--use to add multiple agent plots on the same figure
function addAgentToPlotTable(r_table,agent_name,limit,title,object_net_info, refresh)
  local agent_summary
  assert(agent_name and "nil agent file provided")
  if not refresh and fs.filep(agent_name.."_result_summary.t7")  then
    agent_summary = torch.load(agent_name.."_result_summary.t7")
  else
    agent_summary = summarizeAgent(agent_name,title)
  end
  addAgentFromSummary(r_table,agent_summary,limit,title,object_net_info)
end

--same as above but only uses summary (usefull when using multiple machines to avoid copying the entire agent)
function addAgentFromSummary(r_table,agent_summary,limit,title,object_net_info)
  print ("total reward history in file ",agent_summary.length)
  print (agent_summary)
  limit = math.min(limit or agent_summary.length, agent_summary.length)
  title = title or agent_summary.title
  if agent_summary.loss and object_net_info then
    local AEN_stat_table ={{'Binary Cross Entropy loss', agent_summary.loss:narrow(1,1,limit)},{'Accuracy',agent_summary.acc:narrow(1,1,limit)}}
    --plot seperate graph for object network
    plotExpFromTable(AEN_stat_table,'',nil,title or string.gsub(agent_name, "_", " " ) .. " AEN Preformance",nil,nil)
  end

  local agent_reward_for_plot = { title , agent_summary.reward:narrow(1,1,limit)}
  table.insert(r_table,agent_reward_for_plot)
end

--used to create figures from the tables
function plotExpFromTable(table,ylabel,legend_pos,figure_title,fig_num,png)
  --gif should be a string containing png name
  if png ~= nil then
    plt.pngfigure(png)
  else
    plt.figure(fig_num)
  end
  if figure_title then plt.title(figure_title) end
  plt.xlabel('Steps 10k')
  plt.ylabel(ylabel)
  legend_pos = legend_pos or {'right','bottom'}
  plt.movelegend(unpack(legend_pos))
  plt.plot(table)
  gnuplot.plotflush()
end

------legacy------
--insert agents here
--EGG 5  take actions
--[[
local limit= 190
addAgentFromSummary("DQN3_0_1__FULL_Y_test_zork_vanila_1mil_replay",limit,"Vanilla")
addAgentToPlotTable( "DQN3_0_1__FC_restrict_exploration", "AE-Explore", limit,false)
addAgentToPlotTable("DQN3_0_1_zork_FC_greedy_scenario_1_lr1.7e7","AE-Greedy",limit,false)
addAgentToPlotTable( "DQN3_0_1__FC_restrict_exploration_n_action", "AE-DQN", limit,false)
plotExpFromTable(a_r_table,nil,nil,nil,"iclr/egg-5obj-iclr.png")
--plt.pngfigure("iclr/egg-5obj-iclr.png")
--plt.title('DQN Agent Reward - limited action space')
--plt.xlabel('Steps 10k')
--plt.ylabel('Average Cumulative Reward')
--plt.movelegend('right','bottom')
--plt.plot(a_r_table)
--gnuplot.plotflush()
]]
--[[
--EGG 30 take actions
a_r_table = {}
addAgentToPlotTable("DQN3_0_1__zork_FC_vanila_scenario_2_lr_1.9e7","Vanilla",limit,false)
addAgentToPlotTable("DQN3_0_1_zork_FC_greedy_scenario_2_lr1.7e7","Greedy",limit,false)
addAgentToPlotTable("DQN3_0_1__zork_FC_merged_scenario_2_lr_1.9e7","Merged",limit,false)
--plot in main reward graph
plt.figure()
plt.title('DQN Agent Reward - extended action space')
plt.xlabel('Epochs, 10000 steps per epoch')
plt.movelegend('right','bottom')
plt.plot(a_r_table)
]]
--EGG 200 take actions
--[[
a_r_table = {}
addAgentToPlotTable("DQN3_0_1_zork_FC_vanila_scenario_3_lr1.7e7_200a","Vanilla",limit,false)
--addAgentToPlotTable("DQN3_0_1_zork_FC_merged_scenario_3_lr1.7e7","Merged 1.7e7",limit,true)
addAgentToPlotTable("DQN3_0_1_zork_FC_explore_amended_scenario_3_max_3_sample_5_drop_prob_0.9_lr1.7e-06_209a","AE-Explore",limit,false)
addAgentToPlotTable("DQN3_0_1_zork_FC_greedy_scenario_3_lr1.7e-06_209a","AE-Greedy",limit,false)
addAgentToPlotTable("DQN3_0_1_zork_FC_merged_scenario_3_lr1.7e7_200a","AE-DQN",limit,false)

plt.pngfigure("iclr/egg-200obj-iclr.png")
--plt.title('DQN Agent Reward - extreme action space')
plt.xlabel('Steps 10k')
plt.ylabel('Average Cumulative Reward')
plt.movelegend('right','bottom')
plt.plot(a_r_table)
gnuplot.plotflush()

--Troll 200 take actions
a_r_table = {}
limit = 500
addAgentToPlotTable("DQN3_0_1_zork_FC_vanila_scenario_4_lr1.7e-06_215a","Vanilla",limit,false)
addAgentToPlotTable("DQN3_0_1_zork_FC_explore_amended_scenario_4_max_3_sample_5_drop_prob_0.9_lr1.7e-06_215a","AE-Explore",limit,false)
addAgentToPlotTable("DQN3_0_1_zork_FC_greedy_scenario_4_lr1.7e-06_215a","AE-Greedy",limit,false)
--addAgentToPlotTable("DQN3_0_1_zork_FC_merged_amended_scenario_4_max_5_sample_5_drop_prob_0.9_lr1.7e-06_215a","Merged 1.7e-6",limit,true,true)
addAgentToPlotTable("BACK/DQN3_0_1_zork_FC_merged_scenario_4_lr1.7e-06_215a","AE-DQN",limit,false)

plt.pngfigure('iclr/troll-1-200obj-iclr.png')
--plt.title('DQN Agent Reward: Troll quest ,-1 step, extreme action space')
plt.xlabel('Steps 10k')
plt.ylabel('Average Cumulative Reward')
plt.movelegend('left','top')
plt.plot(a_r_table)


--double step penalty Troll quest
a_r_table = {}
limit = 990
addAgentToPlotTable("DQN3_0_1_zork_FC_vanila_scenario_4_step_-2_lr1.7e-06_215a","Vanilla 1.7e-6",limit,false)
--addAgentToPlotTable("DQN3_0_1_a_r_table = {}
zork_FC_merged_amended_scenario_4_step_-2_sample_5_drop_prob_0.8_lr1.7e-06_215a", "Merged s5 1.7e-6",limit,true)
addAgentToPlotTable("DQN3_0_1_zork_FC_merged_amended_scenario_4_step_-2_sample_10_drop_prob_0.99_lr1.7e-06_215a","Merged s10 1.7e-6",limit)
plt.figure(4)
plt.title('DQN Agent Reward: Troll quest, -2 step, extreme action space')
plt.xlabel('Epochs, 10000 steps per epoch')or agent_name
plt.movelegend('right','bottom')
plt.plot(a_r_table)
a_r_table = {}
gnuplot.plotflush()
]]
--experiments
--limit = 95
--a_r_table = {}
--addAgentToPlotTable("zork_scenario_1_merged_Q-arch_conv_q_net_AEN-arch_conv_obj_net_max_2_sample_1_drop_prob_0.9_lr0.0025_14a","2 conv merged lr 0.0025",limit,true)
--addAgentToPlotTable("zork_scenario_1_merged_Q-arch_1HNN100_AEN-arch_linear_max_2_sample_1_drop_prob_0.9_lr0.0017_14a","hnn + lin merged lr 0.0017",limit,true)
--plotExpFromTable(a_r_table,"Average Cumulative Reward",nil,"egg 5",nil,nil)
--[[
limit = 195
a_r_table = {}
addAgentToPlotTable("zork_scenario_3_vanila_Q-arch_conv_q_net_lr0.0025_209a",nil,limit)
--addAgentToPlotTable("zork_scenario_3_merged_Q-arch_1HNN100_AEN-arch_linear_max_2_sample_1_drop_prob_0.9_lr0.0017_209a",nil,limit,true)
--addAgentToPlotTable("zork_scenario_3_vanila_Q-arch_1HNN100_lr0.0017_209a",nil,limit,false,true)
addAgentFromSummary("zork_scenario_3_merged_Q-arch_conv_q_net_AEN-arch_conv_obj_net_max_5_sample_5_drop_prob_0.9_lr0.0025_209a",limit)
addAgentFromSummary("zork_scenario_3_greedy_Q-arch_conv_q_net_AEN-arch_conv_obj_net_max_5_sample_5_lr0.0025_209a",limit)
plotExpFromTable(a_r_table,"Average Cumulative Reward",nil,"egg 200",nil,nil)
]]
