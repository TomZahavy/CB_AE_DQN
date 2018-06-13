require 'plot'

local limit = 500
local exp_4_new_convQ_AEN = {}

addAgentToPlotTable(exp_4_new_convQ_AEN,'zork_scenario_4_vanila_Q-arch_conv_Q_net_seed_111_bp-penalty_1_lr0.00043_215a',limit)
addAgentToPlotTable(exp_4_new_convQ_AEN,'zork_scenario_4_vanila_Q-arch_conv_Q_net_seed_222_bp-penalty_1_lr0.00043_215a',limit)
addAgentToPlotTable(exp_4_new_convQ_AEN,'zork_scenario_4_vanila_Q-arch_conv_Q_net_seed_333_bp-penalty_1_lr0.00043_215a',limit)
addAgentToPlotTable(exp_4_new_convQ_AEN,'zork_scenario_4_vanila_Q-arch_conv_Q_net_seed_444_bp-penalty_1_lr0.00043_215a',limit)
addAgentToPlotTable(exp_4_new_convQ_AEN,'zork_scenario_4_vanila_Q-arch_conv_Q_net_seed_555_bp-penalty_1_lr0.00043_215a',limit)

plotExpFromTable(exp_4_new_convQ_AEN,"Average Cumulative Reward",nil,"Vanilla agents, bp=1")
