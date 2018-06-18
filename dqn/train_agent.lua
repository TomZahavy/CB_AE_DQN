--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then
    require "initenv"
end
require 'plot'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', '', 'name of training framework')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('-eval_samples',3,'number of eval traces of the game to log')

cmd:text()

local opt = cmd:parse(arg)
--- General setup.
EVAL_STEPS = opt.eval_steps

local game_env, game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local bad_action_history = {}
ShallowEliminationBuff = {}
EliminationBuff = {}
ShallowEliminationFPBuff = {}
EliminationFPBuff = {}
LossBuff = {}
AccBuff = {}
local obj_loss_history = {}
local step = 0
time_history[1] = 0

local total_reward
local nrewards
local nepisodes
local episode_reward
local screen, reward, terminal = game_env:getState()
print("Iteration ..", step)
local win = nil
local agent_o, filename
local logfile=io.open(opt.name .. '_log.txt', 'w')
logfile:write('Start')
logfile:close()
while step < opt.steps do
    step = step + 1
    local action_index,a_o = agent:perceive(reward, screen, terminal)
    if not terminal then
        screen, reward, terminal,new_state_string , bad_command = game_env:step(game_actions[action_index], true)
        agent.lastAction_bad = bad_command -- give agent feedback for last command validity
    else
        if opt.random_starts > 0 then
            screen, reward, terminal = game_env:nextRandomGame()
        else
            screen, reward, terminal = game_env:newGame()
        end
    end

    -- display screen
    -- @DEBUG CO: win = image.display({image=screen, win=win})

    if step % opt.prog_freq == 0 and opt.verbose > 0 then
        print("Steps: ", step)
        if opt.verbose > 2 then
            assert(step==agent.numSteps, 'trainer step: ' .. step ..
            ' & agent.numSteps: ' .. agent.numSteps)
            agent:report()
        end
    end

    if step%5000 == 0 then collectgarbage() end

    if step % opt.eval_freq == 0 and step > learn_start then

        screen, reward, terminal = game_env:newGame()

        total_reward = 0
        nrewards = 0
        nepisodes = 0
        episode_reward = 0

        local eval_time = sys.clock()
        local eval_bad_command = 0
        local eval_tot_obj_actions = 0
        eval_step=0
        if agent.shallow_elimination_flag == 1 then agent.val_conf_buf:zero() end

        for estep=1,opt.eval_steps do

            eval_step=eval_step+1
            local action_index,a_o = agent:perceive(reward, screen, terminal, true, 0.05) -- a_o : object for action assume only 1 for now
            -- Play game in test mode (episodes don't end when losing a life)
            local prev_string, prev_inventory = game_env:getCurrentStringState()

            screen, reward, terminal,new_state_string,bad_command = game_env:step(game_actions[action_index])
            episode_reward = episode_reward + reward
            if nepisodes <= opt.eval_samples then
              if estep == 1 then
                logfile=io.open(opt.name .. '_log.txt', 'a')
                logfile:write('@@@@@@@@ eval sample start after '.. (step-agent.learn_start)/opt.eval_freq .. ' learning iterations @@@@@@@@\n')
              end
              logfile:write(prev_string .. prev_inventory)
              logfile:write(game_actions[action_index].action .. '\n')
              logfile:write('reward: ' .. reward ..  ', bad_command: ' .. bad_command .. '\n'..'\n')

              if terminal then
                logfile:write(new_state_string .. '\n')
                logfile:write('$$$$ end of trace  with reward '..episode_reward..' $$$$\n')
                if nepisodes == opt.eval_samples then
                  logfile:write('#### end of eval period ####\n')
                  logfile:close()
                end
              end
            end
            agent.lastAction_bad = bad_command -- update agent feedback on syntax flag for last command
            if a_o ~= 0 then
              eval_bad_command = eval_bad_command + bad_command
              eval_tot_obj_actions = eval_tot_obj_actions + 1
            end
            -- display screen
            -- @DEBUG CO:win = image.display({image=screen, win=win})
            -- record every reward
            if reward > 0 then
               nrewards = nrewards + 1
            end

            if terminal then
                total_reward = total_reward + episode_reward
                episode_reward = 0
                nepisodes = nepisodes + 1
                screen, reward, terminal = game_env:nextRandomGame()
            end
        end
        if step > agent.obj_start and agent.shallow_elimination_flag == 1 then
            --local buff = agent.val_conf_buf
            local buff = torch.FloatTensor(5,4):zero()
            buff[{{},1 }],buff[{{},2 }],buff[{{},3 }],buff[{{},4 }] = agent.val_conf_buf:mean(2):float():squeeze(),
              agent.val_conf_buf:std(2):float():squeeze(),agent.val_conf_buf:min(2):float():squeeze()
              ,agent.val_conf_buf:max(2):float():squeeze()
            print('cols: avg,std,min,max (over min)/ rows: AEN_prediction,shallow_predictions,confidence_values_avg,confidence_min, conf max (over actions)')
            print(buff)
        end

      eval_time = sys.clock() - eval_time
      start_time = start_time + eval_time
	    local ind = #reward_history+1
	    obj_loss_history[ind] = agent:compute_validation_statistics(ind) -- update loss for obj network
      total_reward = total_reward/math.max(1, nepisodes)

      if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
          agent.best_network = agent.network:clone()
      end

      if agent.v_avg then
          v_history[ind] = agent.v_avg
          td_history[ind] = agent.tderr_avg
          qmax_history[ind] = agent.q_max
      end
      print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])
      eval_bad_command = eval_bad_command/eval_tot_obj_actions
      reward_history[ind] = total_reward
	    reward_counts[ind] = nrewards
      episode_counts[ind] = nepisodes
      bad_action_history[ind] = eval_bad_command
      time_history[ind+1] = sys.clock() - start_time

      local time_dif = time_history[ind+1] - time_history[ind]

      local training_rate = opt.actrep*opt.eval_freq/time_dif

      print(string.format(
          '\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' ..
          'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
          'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d, num. bad_actions %.3f',
          step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif,
          training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
          nepisodes, nrewards,eval_bad_command  ))
    end

    if step % opt.save_freq == 0 or step == opt.steps then
        local s, a, r, s2, term,a_o,bad_command = agent.valid_s, agent.valid_a, agent.valid_r,
            agent.valid_s2, agent.valid_term, agent.valid_a_o,agent.valid_bad_command
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term,agent.valid_a_o,agent.valid_bad_command = nil, nil, nil, nil, nil, nil, nil
        local w, dw, g, g2, delta, delta2, deltas, tmp, obj_w, obj_dw = agent.w, agent.dw,
            agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp,agent.obj_w,agent.obj_dw
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp, agent.obj_w,agent.obj_dw= nil, nil, nil, nil, nil, nil, nil, nil, nil, nil

        filename = opt.name
        if opt.save_versions > 0 then
            filename = filename .. "_" .. math.floor(step / opt.save_versions)
        end
        filename = filename  .. "_lr" .. agent.lr .."_" ..agent.n_actions.."a"
        agent_o = {agent = agent,
                            model = agent.network,
                            model_AEN = agent.obj_network,
                            best_model = agent.best_network,
                            reward_history = reward_history,
                            bad_action_history = bad_action_history,
                            ShallowEliminationBuff = ShallowEliminationBuff,
                            EliminationBuff = EliminationBuff,
                            ShallowEliminationFPBuff = ShallowEliminationFPBuff,
                            EliminationFPBuff = EliminationFPBuff,
                            LossBuff = LossBuff,
                            AccBuff = AccBuff,
                            obj_loss_history = obj_loss_history,
                            reward_counts = reward_counts,
                            episode_counts = episode_counts,
                            time_history = time_history,
                            v_history = v_history,
                            td_history = td_history,
                            qmax_history = qmax_history,
                            arguments=opt
                        }
        torch.save(filename ..".t7", agent_o)
        if step > learn_start then
          --plotAgentFromSummary(summarizeAgent(filename,nil,agent_o))
            legend_pos = legend_pos or {'right','bottom'}
            local DQN_reward = torch.Tensor(agent_o.reward_history)
            plt.figure(1)
            plt.title('reward')
            plt.xlabel('Steps 10k')
            plt.movelegend(unpack(legend_pos))
            plt.plot({DQN_reward})
            gnuplot.plotflush()

            --[[plt.figure(2)
            gnuplot.raw('set multiplot layout 3,1')
            local bad_command = torch.Tensor(agent_o.bad_action_history)
            local ShallowEliminationBuff = torch.Tensor(agent_o.ShallowEliminationBuff)
            local EliminationBuff = torch.Tensor(agent_o.EliminationBuff)
            local ShallowEliminationFPBuff = torch.Tensor(agent_o.ShallowEliminationFPBuff)
            local EliminationFPBuff = torch.Tensor(agent_o.EliminationFPBuff)
            local LossBuff = torch.Tensor(agent_o.LossBuff)
            local AccBuff = torch.Tensor(agent_o.AccBuff)

            gnuplot.raw("set title 'bad_command,loss,accuracy'")
            plt.plot({'bad commands',bad_command},{'loss',LossBuff},{'acc',AccBuff})

            gnuplot.raw("set title 'Elimination'")
            plt.plot({'Shallow Elimination',ShallowEliminationBuff},{'Elimination',EliminationBuff})

            gnuplot.raw("set title 'FalsePositives'")
            plt.plot({'Shallow Elimination FP',ShallowEliminationFPBuff},{'Elimination FP',EliminationFPBuff})

            gnuplot.plotflush()--]]
        end

        if opt.saveNetworkParams then
            local nets = {network=w:clone():float()}
            torch.save(filename..'.params.t7', nets, 'ascii')
        end
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2, agent.valid_term,
            agent.valid_a_o,agent.valid_bad_command = s, a, r, s2, term, a_o,bad_command
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp ,agent.obj_w,agent.obj_dw= w, dw, g, g2, delta, delta2, deltas, tmp,obj_w,obj_dw
        print('Saved:', filename .. '.t7')
        io.flush()
        collectgarbage()
    end
  end
plotAgentFromSummary(summarizeAgent(filename,nil,agent_o))
