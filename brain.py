import torch
from torch import nn
from torch import optim
import random
import math
import numpy as np
import copy


import torch.nn.functional as F


from general_func import index2rc, rc2index, lr_file_read, chg_input_cnn

class Brain_dqn:
    def __init__(self, network, device, num_actions, ban, ReplayMemory,  GAMMA, BATCH_SIZE, lr, T, BANHEN, BANSIZE):
        self.num_actions = num_actions  

        # 経験を記憶するメモリオブジェクトを生成
        self.memory = ReplayMemory
        self.Transition = self.memory.Transition
        #環境を設定
        self.ban = ban
        self.BANHEN = BANHEN
        self.BANSIZE = BANSIZE
        #定数
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.lr = lr
        self.T = T
        # ニューラルネットワークを設定
        
        #self.main_model = network().to(device)
        #self.new_model = network().to(device)
        
        self.main_model = network(BANHEN, BANSIZE)
        self.new_model = network(BANHEN, BANSIZE)
        
        self.device = device

        #gpu並列処理
        self.main_model = nn.DataParallel(self.main_model).to(self.device)
        self.new_model = nn.DataParallel(self.new_model).to(self.device)
        #print(self.model)  # ネットワークの形を出力

        # 最適化手法の設定
        self.main_optimizer = optim.SGD(self.main_model.parameters(), self.lr, momentum=0.9, weight_decay=0.0001) 
        self.new_optimizer  = optim.SGD(self.new_model.parameters(),  self.lr, momentum=0.9, weight_decay=0.0001) 
        
        #self.main_optimizer = optim.Adam(self.main_model.parameters(), self.lr) 
        #self.new_optimizer = optim.Adam(self.new_model.parameters(), self.lr) 
        #loss
        self.loss_num = 0
        self.loss_v = 0
        self.loss_p = 0
        self.loss_memory = []#lossを貯めてlrの決定に使う
        self.loss_average_tmp =-1#lossの平均値

        #random
        self.eps_threshold = 0
        
        
    #次の手を決める
    def decide_action(self, ban, model, player_side, search_depth, step,episode_sum ,ep_random_data ,fastmode=False):
        reward = 0
        if fastmode:#デバック用　NNを使わない
            #print("fastmode")
            
            action = random.choice(ban.ban_put_available())  # 行動をランダムに返す
            
            r = action[0]
            c = action[1]
            
            v_ary = np.zeros(self.BANSIZE)
            v_output = 0
            reward, r, c = 0, r, c
            
            ban_copy = copy.deepcopy(ban)
            ban_copy.ban_applay(player_side, r, c)#自分が打つ
            
            if ban_copy.ban_fill():
                terminal = True
            else:
                terminal = False
            
            return reward, r, c, None, None, terminal
        
        #print("fastmodeじゃないよ...")
        
        EPS_START = 0.9
        EPS_END = 0.05
        EPS_DECAY = 1000
        
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode_sum / EPS_DECAY)
        
        self.eps_threshold = eps_threshold

        '''
        if step <= 1:
            eps_threshold = max(0.6, eps_threshold)
        elif step == 2:
            eps_threshold = max(0.3, eps_threshold)
        elif step == 3:
            eps_threshold = max(0.1, eps_threshold)
        elif step == 4:
            eps_threshold = max(0.1, eps_threshold)
        elif step >= 5:
            eps_threshold = max(0.05, eps_threshold)
        '''


        if  sample > eps_threshold:# and step >= 1
            
            #手を探索する
            #p_ary= self.searchGameTree(ban, model, player_side, search_depth)
            
            #indexを大きい順に並べる
            
            '''
            p_ary_index = np.argsort(p_ary)[::-1]
            
            
            
            #print("vがmaxのindex = {}".format([int(v_ary_index[0]/14), v_ary_index[0] % 14]))
            ban_put_available = ban.ban_put_available()
            #print(ban_put_available)
            for index in p_ary_index:
                r,c = index2rc(index)
                if [r, c] in ban_put_available:
                    q = p_ary[index]
                    return r, c, p_ary, q
            '''
            reward, r, c, state, terminal = self.rtn_reward(ban, model, player_side)
            
            return reward, r, c, state, terminal
            
            
        else:
            #print("ランダム打ち")
            action = random.choice(ban.ban_put_available())  # 行動をランダムに返す
            r = action[0]
            c = action[1]
            #_, win_flag, p_ary, _, _, _= self.rtn_p_ary(ban, model, player_side, 0)
            
            index = rc2index(r,c)
            #q = p_ary[index]
                
            
            #for g in range(14):
            #    for r in range(14):
            #        print("{:0=+03.3f} ".format(v_ary[14*g + r]), end="")
            #
            #    print('')


            reward, r, c = 0, r, c
            
            ban_copy = copy.deepcopy(ban)
            state = chg_input_cnn(ban_copy, player_side)
            ban_copy.ban_applay(player_side, r, c)#自分が打つ
            
            if ban_copy.ban_fill():
                terminal = True
                reward = 0
                return reward, r, c, state, terminal
            
            elif ban_copy.ban_win(player_side, r, c):
                reward = 1
                terminal = True
                #print("win")
                return reward, r, c, state, terminal
            else:
                reward = 0
                terminal = False
                #print("continue")
                return reward, r, c, state, terminal
                
    
    
    def softmax_numpy(self, x, T):
        #e_x = pow(np.exp(x), 1/T)  #1/T乗をする
        
        c = np.max(x)
        
        exp_a = pow(np.exp(x - c), 1/T)

        y = exp_a / np.sum(exp_a)
        return y
    
        
    def rtn_reward(self, ban, model, player_side):
        reward = 0
        ban_copy = copy.deepcopy(ban)
        
        state = chg_input_cnn(ban_copy, player_side)
        p_ary , _ = model(state.to(self.device))
        p_ary = p_ary.detach().cpu().numpy()[0]
        ban_put_available = ban_copy.ban_put_available()
        
        
        print("--------------")
        print("player_side", player_side)
        ban.ban_print()
        print(p_ary)
        for i in range(10):
            #print(ban_put_available)

            q_ary_for_w = []

            for [r,c] in ban_put_available:
                index = rc2index(r,c)
                q_ary_for_w.append(p_ary[index])

            w = self.softmax_numpy(q_ary_for_w, 1/(1+i))
            print("q_ary_for_w", q_ary_for_w)
            print("weights", w)
            print("ban_put_available", ban_put_available)
            action = random.choices(ban_put_available, weights=w)[0]
            #print(action)
            q = self.rtn_q(ban, model, player_side, action)
            index = rc2index(action[0], action[1])
            print(i, action, q)
            p_ary[index] = (q + p_ary[index])/2
        print(p_ary)
        
        
        p_ary_index = np.argsort(p_ary)[::-1] 
            
        
        #print(ban_put_available)
        
        for index in p_ary_index:
            r,c = index2rc(index)
            if [r, c] in ban_put_available:
                break
        
        ban_copy.ban_applay(player_side, r, c)
        
        #ban_copy.ban_print()
        
        if ban_copy.ban_win(player_side, r, c):
            
            reward = 1
            terminal = True
            #print("win")
            return reward, r, c, state, terminal
        
        elif ban_copy.ban_fill():
            #print("もう打てないよ！！")
            reward = 0
            terminal = True
            #print("fill")
            return reward, r, c, state, terminal

        
        else:
            reward = 0
            terminal = False
            #print("continue")
            return reward, r, c, state, terminal
        
    def rtn_q(self, ban, model, player_side, action):
        q = None
        ban_copy = copy.deepcopy(ban)
        ban_copy.ban_applay(player_side, action[0], action[1])

        if ban_copy.ban_win(player_side, action[0], action[1]):
            print("win")
            q = 1
        elif ban_copy.ban_fill():
            print("fill")
            q = 0
        else:
            state = chg_input_cnn(ban_copy, 1-player_side)
            p_ary , _ = model(state.to(self.device))
            p_ary = p_ary.detach().cpu().numpy()[0]
            p_ary_index = np.argsort(p_ary)[::-1] 
                
            ban_put_available = ban_copy.ban_put_available()
            #print(ban_put_available)
            
            for index in p_ary_index:
                r_op,c_op = index2rc(index)
                if [r_op,c_op] in ban_put_available:
                    break
            print("相手が打つ場所", r_op,c_op)
            ban_copy.ban_applay(1-player_side, r_op,c_op)
            ban_copy.ban_print()
            if ban_copy.ban_win(1-player_side, r_op,c_op):
                print("lose")
                q = -1
            elif ban_copy.ban_fill():
                print("fill op")
                q = 0
            else:
                print("other")
                state = chg_input_cnn(ban_copy, player_side)
                p_ary , _ = model(state.to(self.device))
                p_ary = p_ary.detach().cpu().numpy()[0]
                put_available_position = ban_copy.rtn_put_available_position()
                q = np.max(p_ary + put_available_position)
        
        return q



            
        

            
    
        
    
    def train(self, episode, epoch_num, ep_random_data):#new_modelを訓練する
        
        
        if len(self.memory) < self.BATCH_SIZE or episode < ep_random_data:
                #print('len(self.memory) : {}'.format(len(self.memory)))
                return
        
        self.lr = lr_file_read() #lrをtextファイルから読み取る
        
        
        #print("epoch_num : {}".format(epoch_num))
        
        for g in self.new_optimizer.param_groups:
                    g['lr'] = self.lr
        
        for i in range(epoch_num):
            
            #if i%10==0:
            #        print(i, end="")
            #else:
            #        print('*', end="")
            
            
            
            '''
            モデルはnew_modelとなっている!!
            '''
            
            BATCH_SIZE = self.BATCH_SIZE
            transitions = self.memory.sample(BATCH_SIZE)

            batch = self.Transition(*zip(*transitions))

            #state', 'action', 'next_state', 'reward'
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            
            
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.uint8)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
            put_available_position_batch = torch.cat([s for s in batch.put_available_position
                                                if s is not None])
            
            
            state_action_values = self.new_model(state_batch)[0].gather(1, action_batch)
            
            next_state_values = torch.zeros(BATCH_SIZE, device=self.device)

            #print("-"*20)
            #print(self.new_model(non_final_next_states)[0])
            #print(put_available_position_batch)
            #print(self.new_model(non_final_next_states)[0]*put_available_position_batch)
            next_q_values = self.new_model(non_final_next_states)[0]+put_available_position_batch
            next_state_values[non_final_mask] = next_q_values.max(1)[0].detach()
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
            
            
            #new_modelを訓練する
            self.new_model.train()

            output = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
            
            self.loss_num = output.item()
            self.loss_p = output.item()

            self.new_optimizer.zero_grad()  # 勾配をリセット
            output.backward(retain_graph=True)  # バックプロパゲーションを計算
            self.new_optimizer.step()  # 結合パラメータを更新

        #print("")
        #print("loss_v : {}".format(self.loss_v))
        #print("loss_p : {}".format(self.loss_p))
        #print("loss_num : {}".format(self.loss_num))
        #print("epoch_num : {}".format(epoch_num))
        #self.loss_num = loss_sum/epoch_num
        
    def update_main_network(self):#最新のネットワーク(new)が勝ったらモデルをmainネットワークにアップデートする
        self.main_model.load_state_dict(self.new_model.state_dict())
        
        
