import numpy as np
import torch
import copy
import datetime
import os
import torch

from tensorboardX import SummaryWriter

from general_func import write_lr, log_print, chg_input_cnn, rc2index
from environment import Env
from replayMemory import ReplayMemory
from brain import Brain_dqn
from model import NeuralNet_cnn
from win_rate import check_win_rate_ai, check_win_rate_put_1st, check_win_rate_random
from init import BANHEN, BANSIZE, WINREN, file_path, lr, lr_filename, model_filename, MEMO, CAPACITY, device, GAMMA, BATCH_SIZE, T, NUM_EPISODES, epoch_num, update_win_rate, flg_fastmode


print("device", device)



if not os.path.exists(file_path):
            os.makedirs(file_path)
        


write_lr(lr) #lrのtextファイルを作成する

now = datetime.datetime.now()
print('{0:%Y%m%d}'.format(now)) 






#tensorboarx
writer_x = SummaryWriter('tfbx2/' + '_' +'{0:%Y%m%d%H%M%S_}'.format(now)+ model_filename +MEMO +'/')



# In[ ]:


#訓練
import os


import pprint
pp = pprint.PrettyPrinter(indent=4)


ban = Env(BANHEN, WINREN)
memory = ReplayMemory(CAPACITY, ban)
brain = Brain_dqn(NeuralNet_cnn, device, ban.size, ban,
                  memory,  GAMMA, BATCH_SIZE,  lr, T, BANHEN, BANSIZE)

match_is_continue = True #試合が継続しているかどうか
train_is_continue = True #訓練を継続するか
reward = 0 #報酬
step = 0#何手目か
step_sum = 0
gen_num = 0  #モデルの初期値
episode_sum = 0#エピソードの累積
search_depth = 3
ep_random_data = 0

log_print("lrはtextファイルから読み取り")


log_print('start : ' + model_filename)
start_time = datetime.datetime.now()
log_print("start time")
log_print(start_time)

print(brain.main_model)
dummy_input = chg_input_cnn(ban, 0)
print(dummy_input.size())
writer_x.add_graph(brain.main_model, dummy_input)

if __name__ == '__main__':
    while train_is_continue:
        for episode in range(NUM_EPISODES):  # 最大試行数分繰り返す
            episode_sum += 1
            ban.ban_reset()
            step = 0  #stepをリセット
            terminal = False #terminalをリセット
            player_side = 0#最初に打つplayer
            player0_train_data = []
            player1_train_data = []
            
            player0_q_data = [] #qデータ?を蓄える
            player1_q_data = [] #qデータ?を蓄える
            
            #print('-'*10)

            
            tmp_data = []
            
            #log_print("episode_sum : " + str(episode_sum))
                
            
            while match_is_continue:
                step += 1
                
                #if step%10==0:
                #    print(step, end="")
                #else:
                #    print('*', end="")
                
                #print("\rstep : {0} ".format(step), end="")

                '''
                print('-'*10)
                print('step')
                print(step)
                print('player 0')
                '''
                #print("-"*15)
                #print("-"*15)
                
                #print('player : {}, step : {}'.format(player_side, step))
                
                
                reward, r, c, state, terminal = brain.decide_action(ban, brain.main_model, player_side, search_depth, step,episode_sum , ep_random_data,fastmode=flg_fastmode)
                
                
                #p_ary = torch.from_numpy(np.array([p_ary])).type(torch.FloatTensor)
                
                state = chg_input_cnn(ban, player_side)
                action = rc2index(r, c)
                action = torch.tensor([[action]], device=device, dtype=torch.long)
                reward = torch.tensor([reward], device=device, dtype=torch.float)
                #action = np.reshape(np.array(action), (1, 1))
                #print("memory.memory",memory.memory)
                #print("state.shape", state.size())
                #print("action.shape",action.shape)
                
                
                ban.ban_applay(player_side, r, c)
                #print([r, c], "reward", reward)
                #ban.ban_print()
                
                
                tmp_data.append([state, action])
                
                if len(tmp_data) >= 3:
                    len_data = len(tmp_data)
                    reward_0 = torch.tensor([0], device=device, dtype=torch.float)
                    #state', 'action', 'next_state', 'reward'
                    memory.push(tmp_data[len_data-3][0], tmp_data[len_data-3][1], state, reward)

                
                    
                if terminal:
                    #print("終了")
                    
                    #終局のときだけ追加
                    memory.push(state, action, None, reward)
                    reward_lose = torch.tensor([-1], device=device, dtype=torch.float)
                    #print("-1*reward", -1*reward)
                    memory.push(tmp_data[len_data-2][0], tmp_data[len_data-2][1], None, -1*reward)
                    
                    break #whileを抜ける

                
                player_side = 1 - player_side#playerを交代する
                
                #print('episode: {}/{}, step : {}, loss : {} '.format(episode, NUM_EPISODES, step, brain.loss_num))

            brain.train(episode_sum, epoch_num, ep_random_data)
            writer_x.add_scalar('Val/Loss', brain.loss_num, episode_sum)
            writer_x.add_scalar('Val/Loss_v', brain.loss_v, episode_sum)
            writer_x.add_scalar('Val/Loss_p', brain.loss_p, episode_sum)
            writer_x.add_scalar('Val/step', step, episode_sum)
            writer_x.add_scalar('Val/lr', brain.lr, episode_sum)
            writer_x.add_scalar('Val/len memory', len(brain.memory), episode_sum)

            #log_print('episode_sum: {:08d}, episode: {:04d}/{:04d}, step:{:04d}, lr : {:03.5f}, loss : {:03.20f}'.format(episode_sum, episode+1, NUM_EPISODES, step, brain.lr, brain.loss_num))


        if True:
            log_print("episode_sum : " + str(episode_sum))
            #ランダムと比較
            win_rate_put_1st = check_win_rate_put_1st(Env, brain, brain.main_model, 1)
            win_rate_random , not_lose_rate_random = check_win_rate_random(Env, brain, brain.main_model, 400)
            
            #log_print("vs put_1st player : "+str(win_rate_put_1st))
            writer_x.add_scalar('Val/win_rate_put_1st player', win_rate_put_1st, episode_sum)
            
            #log_print("vs random player : "+str(win_rate_random))
            log_print("vs random player not lose: "+str(not_lose_rate_random))
            writer_x.add_scalar('Val/win_rate', win_rate_random, episode_sum)
            writer_x.add_scalar('Val/not_lose_rate', not_lose_rate_random, episode_sum)
            #new_modelとnew_modelを比較する
            win_rate_for_check = check_win_rate_ai(Env, brain, brain.main_model, brain.new_model, 200)
            log_print("vs old model : " + str(win_rate_for_check))
            writer_x.add_scalar('Val/vs old model_rate', win_rate_for_check, episode_sum)
            if win_rate_for_check > update_win_rate:#update_win_rate以上だとモデルを更新する
                writer_x.add_scalar('Val/chg_model', 1, episode_sum)
                brain.update_main_network()

                model_filename_update = model_filename + "_" +str(gen_num) +"gen" + ".pht"

                file_path = "models/"+ model_filename
                if not os.path.exists(file_path):
                    os.makedirs(file_path)

                torch.save(brain.main_model.state_dict(), file_path +'/'+ model_filename_update)
                log_print("モデル保存完了:" + file_path +'/'+ model_filename_update)

                #時間を表示
                finish_time_update = datetime.datetime.now()
                elapsed_time_update = finish_time_update - start_time
                log_print(str(gen_num) +"gen"+"finish time")
                log_print(finish_time_update)
                log_print(str(gen_num) +"gen"+"経過時間")
                log_print(elapsed_time_update)


                if False:#gen_num > gen_num_limit:#訓練を終わりにする
                    train_is_continue = False

                gen_num += 1
            else:
                writer_x.add_scalar('Val/chg_model', 0, episode_sum)






    torch.save(brain.main_model.state_dict(), file_path +'/' + model_filename + ".pht")
    log_print("モデル保存完了:" + model_filename)
    finish_time = datetime.datetime.now()

    elapsed_time = finish_time - start_time
    log_print("finish time")
    log_print(finish_time)
    log_print("経過時間")
    log_print(elapsed_time)

