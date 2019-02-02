
# coding: utf-8

# In[1]:


#alpha go zeroを模倣したモデルver1
#を諦めて，gqnっぽいものにしたやつ〜
#を諦めて環境のスケールを調整できるようにしたもの
#


import os
os.environ["http_proxy"] = "http://proxy.uec.ac.jp:8080/"
os.environ["https_proxy"] = "http://proxy.uec.ac.jp:8080/"

import numpy as np
import heapq
import matplotlib.pyplot as plt


BANHEN = 3
BANSIZE = BANHEN**2
WINREN = 3





class Env:
    def __init__(self):
        self.ban = [[-1 for i in range(BANHEN)] for j in range(BANHEN)] #サイズ BANHEN*BANHEN
        self.screen_n_rows = BANHEN #行
        self.screen_n_cols = BANHEN #列
        self.size = self.screen_n_rows * self.screen_n_cols
        #self.enable_actions = np.arange(self.screen_n_rows*self.screen_n_cols)
        self.enable_actions = []
        for i in range(BANHEN):
            for j in range(BANHEN):
                self.enable_actions.append([i,j])
        #print(self.enable_actions)
        #print(self.enable_actions.index([0,0]))
        #self.name = os.path.splitext(os.path.basename(__file__))[0]
        

    def ban_reset(self):
        self.ban = [[-1 for i in range(BANHEN)] for j in range(BANHEN)]

    def ban_applay(self, player, gyo, retu, ban=None):
        if ban is None:
            #print(gyo)
            #print(retu)
            if self.ban[gyo][retu] != -1:
                return False
            else:
                self.ban[gyo][retu] = player
                return True
        else:
            if ban[gyo][retu] != -1:
                return False
            else:
                ban[gyo][retu] = player
                return True
    def ban_win(self, player, gyo, retu, ban=None):
        
        if ban is None:
            if (self.line_cnt(player, gyo, retu, 1, 0) + self.line_cnt(player, gyo, retu, -1, 0) >= WINREN-1) or                 (self.line_cnt(player, gyo, retu, 0, 1) + self.line_cnt(player, gyo, retu, 0, -1) >= WINREN-1) or                 (self.line_cnt(player, gyo, retu, 1, 1) + self.line_cnt(player, gyo, retu, -1, -1) >= WINREN-1) or                 (self.line_cnt(player, gyo, retu, 1, -1) + self.line_cnt(player, gyo, retu, -1, 1) >= WINREN-1):
                return True
            else:
                return False
        else:
            if (self.line_cnt(player, gyo, retu, 1, 0, ban) + self.line_cnt(player, gyo, retu, -1, 0, ban) >= WINREN-1) or                 (self.line_cnt(player, gyo, retu, 0, 1, ban) + self.line_cnt(player, gyo, retu, 0, -1, ban) >= WINREN-1) or                 (self.line_cnt(player, gyo, retu, 1, 1, ban) + self.line_cnt(player, gyo, retu, -1, -1, ban) >= WINREN-1) or                 (self.line_cnt(player, gyo, retu, 1, -1, ban) + self.line_cnt(player, gyo, retu, -1, 1, ban) >= WINREN-1):
                return True
            else:
                return False
    
    
    def ban_ren_cnt(self, player, gyo, retu):#石がどれくらい連続しているか
        ren_num = 0
        ren_num =  max((self.line_cnt(player, gyo, retu, 1,  0) + self.line_cnt(player, gyo, retu, -1,  0)),
                       (self.line_cnt(player, gyo, retu, 0,  1) + self.line_cnt(player, gyo, retu,  0, -1)),
                       (self.line_cnt(player, gyo, retu, 1,  1) + self.line_cnt(player, gyo, retu, -1, -1)),
                       (self.line_cnt(player, gyo, retu, 1, -1) + self.line_cnt(player, gyo, retu, -1,  1)))
       
        return ren_num
    

    def line_cnt(self, player, gyo, retu, dx, dy, ban=None):
        if ban is None:
            cnt = 0
            while True:
                gyo += dy
                retu += dx
                if gyo <0 or gyo > BANHEN-1 or retu < 0 or retu > BANHEN-1:
                    break

                if self.ban[gyo][retu] == player:
                    #print(gyo)
                    #print(retu)
                    cnt += 1
                else:
                    break
            return cnt 
        else:
            cnt = 0
            while True:
                gyo += dy
                retu += dx
                if gyo <0 or gyo > BANHEN-1 or retu < 0 or retu > BANHEN-1:
                    break

                if ban[gyo][retu] == player:
                    #print(gyo)
                    #print(retu)
                    cnt += 1
                else:
                    break
            return cnt 
    
    
    def ban_print(self):#現在の環境を表示
        #print("ban_print")
        #print('   00  01  02  03  04  05  06  07  08  09  10  11  12  13  ')

        
        for i in range(BANHEN):
            print('  {0:02d}'.format(i), end="")
        print('')
        for gyo in range(BANHEN):
            print('{0:02d}'.format(gyo), end="")
            print(' ', end="")
            for retu in range(BANHEN):
                if self.ban[gyo][retu] == -1:
                    print('-   ', end="")
                elif self.ban[gyo][retu] == 0:
                    print('●   ', end="")
                elif self.ban[gyo][retu] == 1:
                    print('○   ', end="")
            print('\n')
    
    def ban_print_p_ary(self, p_ary):#現在の環境を表示
        #print("ban_print_p_ary")
        p_ary_index = heapq.nlargest(10, p_ary)#p_aryの上位 n個を取得
        
        x_p = []
        y_p = []
        v_p = []
        
        x_0 = []
        y_0 = []
        
        x_1 = []
        y_1 = []
        
        
        #print('   00  01  02  03  04  05  06  07  08  09  10  11  12  13  ')
        for gyo in range(BANHEN):
            #print('{0:02d}'.format(gyo), end="")
            #print(' ', end="")
            for retu in range(BANHEN):
                
                y_p.append(BANHEN-1-gyo)
                x_p.append(retu)
                
                
                if self.ban[gyo][retu] == -1:
                    v_p.append(p_ary[BANHEN*gyo + retu])
                    #if p_ary[14*gyo + retu]  in p_ary_index:
                        #print("{:01.1f}|".format(10*p_ary[14*gyo + retu]), end="")
                        #print('#   ', end="")
                    #else:
                    #    print('-   ', end="")
                elif self.ban[gyo][retu] == 0:
                    #print('●   ', end="")
                    v_p.append(0)
                    x_0.append(retu)
                    y_0.append(BANHEN-1-gyo)
        
                elif self.ban[gyo][retu] == 1:
                    #print('○   ', end="")
                    v_p.append(0)
                    x_1.append(retu)
                    y_1.append(BANHEN-1-gyo)
                    
            #print('\n')
        
        plt.scatter(x_p, y_p, s=100, c=v_p, cmap='pink_r')
        plt.colorbar()
        plt.scatter(x_0, y_0, s=100, c="blue", alpha="1", linewidths="1",edgecolors="black")
        plt.scatter(x_1, y_1, s=100, c="white", alpha="1", linewidths="1",edgecolors="black")
        
        '''
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.set_ylim([13,0])
        ax1.set_xlim([0,13])
        
        ax1.scatter(x_p, y_p, s=100, c=v_p, cmap='pink_r')
        fig.colorbar()
        ax1.scatter(x_0, y_0, s=100, c="blue", alpha="0.5", linewidths="1",edgecolors="black")
        ax1.scatter(x_1, y_1, s=100, c="white", alpha="0.5", linewidths="1",edgecolors="black")
        '''
        
        plt.show()   
    
    def ban_fill(self):#盤が埋まっているときtureを返す
        state = True
        for gyo in range(BANHEN):
            for retu in range(BANHEN):
                if self.ban[gyo][retu] == -1:
                    state = False
        return state
    
    def ban_put_available(self):#打てる手を返す
        put_available = []
        for gyo in range(BANHEN):
            for retu in range(BANHEN):
                if self.ban[gyo][retu] == -1:
                    #put_available.append(14*gyo + retu)
                    put_available.append([gyo,retu])
        return put_available
    
    def ban_put_available_state(self, state):#打てる手を返す
        put_available_state = []
        for gyo in range(BANHEN):
            for retu in range(BANHEN):
                if int(state[gyo][retu]) == 0:
                    put_available_state.append([gyo,retu])
        return put_available_state

    def status(self, player_side):  # player_side 0 or 1
        player_status = [
            [0 for i in range(self.screen_n_cols)] for j in range(self.screen_n_rows)]
        
        for r_n in range(self.screen_n_rows):
            for c_n in range(self.screen_n_cols):
                if self.ban[r_n][c_n] == player_side:
                    player_status[r_n][c_n] = 1

        return np.array(player_status)
    
    
    def status_all(self):  # 全てのplayerのstate
        player_status = [
            [0 for i in range(self.screen_n_cols)] for j in range(self.screen_n_rows)]
        
        for r_n in range(self.screen_n_rows):
            for c_n in range(self.screen_n_cols):
                if self.ban[r_n][c_n] != -1:  #空白じゃなかったら
                    player_status[r_n][c_n] = 1

        return np.array(player_status)
    


# In[2]:


from collections import namedtuple
import numpy as np
import random



class ReplayMemory:
    
    def __init__(self, CAPACITY, ban):
        self.ban = ban #環境を設定
        self.capacity = CAPACITY  # メモリの最大長さ
        self.memory = []  # 経験を保存する変数
        self.index = 0  # 保存するindexを示す変数
        self.Transition = namedtuple(
            'Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, state, action, next_state, reward):
        
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す

        # namedtupleのTransitionを使用し、値とフィールド名をペアにして保存します
        self.memory[self.index] = self.Transition(
            state, action, next_state, reward)

        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす

    def sample(self, batch_size):
        '''batch_size分だけ、ランダムに保存内容を取り出す'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)




# In[3]:


def chg_input_cnn(ban, player_side):
    state = ban.status(player_side)
    state_another_player = ban.status(1 - player_side)
    state_all = ban.status_all()
    
    state_out = [state,state_another_player, state_all]
    state_out = torch.from_numpy(np.array([state_out])).type(torch.FloatTensor)
    return state_out

def data_rotation(data, rotation, reverse):  
        # player_side 0 or 1 ,rotation 0,1,2,3回転方向 reverse 0,1反転
        player_status = [
            [0 for i in range(len(data))] for j in range(len(data[0]))]
        screen_n_rows = len(data)
        screen_n_cols = len(data[0])
        
        for r_n in range(len(data)):
            for c_n in range(len(data[0])):
                if reverse == 0:
                    if rotation==0:
                        player_status[r_n][c_n] = data[r_n][c_n]
                    elif rotation==1:
                        player_status[screen_n_cols-1 - c_n][r_n] = data[r_n][c_n]
                    elif rotation==2:
                        player_status[screen_n_rows-1 - r_n][screen_n_cols-1 - c_n] = data[r_n][c_n]
                    elif rotation==3:
                        player_status[c_n][screen_n_rows-1 - r_n] = data[r_n][c_n]
                elif reverse == 1:
                    if rotation==0:
                        player_status[r_n][screen_n_cols-1 - c_n] = data[r_n][c_n]
                    elif rotation==1:
                        player_status[screen_n_cols-1 - c_n][screen_n_rows-1 - r_n] = data[r_n][c_n]
                    elif rotation==2:
                        player_status[screen_n_rows-1 - r_n][c_n] = data[r_n][c_n]
                    elif rotation==3:
                        player_status[c_n][r_n] = data[r_n][c_n]

        return np.array(player_status)
    
def index_rotation(r_n, c_n,rotation, reverse):
    screen_n_rows, screen_n_cols = BANHEN, BANHEN
    if reverse == 0:
        if rotation==0:
            return r_n,c_n
        elif rotation==1:
            return screen_n_cols-1 - c_n,r_n
        elif rotation==2:
            return screen_n_rows-1 - r_n,screen_n_cols-1 - c_n
        elif rotation==3:
            return c_n,screen_n_rows-1 - r_n
    elif reverse == 1:
        if rotation==0:
            return r_n,screen_n_cols-1 - c_n
        elif rotation==1:
            return screen_n_cols-1 - c_n,screen_n_rows-1 - r_n
        elif rotation==2:
            return screen_n_rows-1 - r_n,c_n
        elif rotation==3:
            return c_n,r_n


def chg_input_cnn_to8(ban, player_side):#4方向に回転 and 反転にしてデータ量を8倍にする.
    output = []
    
    for i in range(8):
        rotation = i%4#回転 0,1,2,3
        reverse = int(i/4)#反転 0,1
        
        
        state = data_rotation(ban.status(player_side), rotation, reverse)
        state_another_player = data_rotation(ban.status(1 - player_side), rotation, reverse)
        state_all = ban.status_all()

        state_out = [state,state_another_player, state_all]
        state_out = torch.from_numpy(np.array([state_out])).type(torch.FloatTensor)
        output.append(state_out)
        
    return output


# In[4]:


def index2rc(index):
    return int(index/BANHEN), index % BANHEN

def rc2index(r,c):
    return BANHEN*r + c


# In[12]:


import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
import math
import sys

from multiprocessing import Pool
from multiprocessing import Process





device = torch.device('cuda')
#device = torch.device("cuda:0")
#device = torch.device("cuda:1")

class NeuralNet_cnn(nn.Module):
    def __init__(self):
        super(NeuralNet_cnn, self).__init__()
        ch_num = 10 #50
        self.conv1 = nn.Conv2d(3, ch_num , kernel_size=3 , padding=1 )#入力ch3, 出力ch_num ,カーネルサイズ, 3
        self.conv2 = nn.Conv2d(ch_num, ch_num , kernel_size=3 , padding=1 )
        self.conv3 = nn.Conv2d(ch_num, ch_num , kernel_size=3 , padding=1 )
        self.conv4 = nn.Conv2d(ch_num, ch_num , kernel_size=3 , padding=1 )
        self.conv5 = nn.Conv2d(ch_num, ch_num , kernel_size=3 , padding=1 )
        self.conv6 = nn.Conv2d(ch_num, ch_num , kernel_size=3 , padding=1 )
        self.conv7 = nn.Conv2d(ch_num, ch_num , kernel_size=3 , padding=1 )
        self.conv8 = nn.Conv2d(ch_num, ch_num , kernel_size=3 , padding=1 )
        self.bn1 = nn.BatchNorm2d(ch_num)
        self.bn2 = nn.BatchNorm2d(ch_num)
        self.bn3 = nn.BatchNorm2d(ch_num)
        self.bn4 = nn.BatchNorm2d(ch_num)
        self.bn5 = nn.BatchNorm2d(ch_num)
        self.bn6 = nn.BatchNorm2d(ch_num)
        self.bn7 = nn.BatchNorm2d(ch_num)
        self.bn8 = nn.BatchNorm2d(ch_num)
        self.fc1 = nn.Linear(BANHEN*BANHEN, BANSIZE)
        self.fc2 = nn.Linear(BANSIZE, BANSIZE)
        self.relu1 = nn.ReLU()
        
        #policy
        self.conv_p1 = nn.Conv2d(ch_num, 2 , kernel_size=3 , padding=1 )
        self.bn_p1 = nn.BatchNorm2d(2)
        self.conv_p1 = nn.Conv2d(ch_num, 1 , kernel_size=3 , padding=1 )
        self.bn_p1 = nn.BatchNorm2d(1)
        self.fc_p2 = nn.Linear(BANHEN*BANHEN*2, BANHEN*BANHEN)
        self.softmax_p3 = nn.Softmax()
        self.tanh_p4 = nn.Tanh()
        self.sigmoid_p4 = nn.Sigmoid()
        self.hardtanh_p4 = nn.Hardtanh()
        
        #value
        self.conv_v1 = nn.Conv2d(ch_num, 1 , kernel_size=3 , padding=1 )
        self.bn_v1 = nn.BatchNorm2d(1)
        self.fc_v2 = nn.Linear(BANHEN*BANHEN, BANHEN*BANHEN)
        self.fc_v3 = nn.Linear(BANHEN*BANHEN, 1)
        self.sigmoid_v4 = nn.Sigmoid()
        self.tanh_v4 = nn.Tanh()
        
        
        
    
    def forward(self, x):
        #relu
        
        
        #activation_func = F.leaky_relu
        activation_func = F.relu
        #activation_func = F.hardtanh
        
        
        out1 = activation_func(self.bn1(self.conv1(x)))
        
        out2 = activation_func(self.bn2(self.conv2(out1)))
        
        out3 = activation_func(self.bn3(self.conv3(out2)) + out1)
        
        out4 = activation_func(self.bn4(self.conv4(out3)))
        
        out5 = activation_func(self.bn5(self.conv5(out4)) + out3)
        
        out6 = activation_func(self.bn6(self.conv6(out5)))
        
        out7 = activation_func(self.bn7(self.conv7(out6)) + out5)
        
        
        #policy
        out_p1 = self.bn_p1(self.conv_p1(out7))
        #out_p1 = out_p1.view(-1, BANHEN*BANHEN*1)
        out_p1 = out_p1.view(out_p1.size(0), -1)
        #out_p = F.softmax(self.fc_p2(out_p1), dim=1)
        #out_p1 = self.fc_p2(out_p1)
        #out_p =  self.sigmoid_p4(out_p1)
        #out_p =  self.tanh_p4(out_p1)
        out_p =  self.hardtanh_p4(out_p1)
        #out_p =  F.softmax(out_p1, dim=1)
        #out_p = out_p1
        
        #value
        out_v1 = F.relu(self.bn_v1(self.conv_v1(out7)))
        out_v1 = out_v1.view(-1, BANHEN*BANHEN)
        out_v2 = F.relu(self.fc_v2(out_v1))
        out_v = self.tanh_v4(self.fc_v3(out_v2))
        #out_v = self.sigmoid_v4(self.fc_v3(out_v2))
        
        
        return out_p, out_v




class Brain_dqn:
    def __init__(self, network, num_states, num_actions, ban, ReplayMemory,  GAMMA, BATCH_SIZE, lr, T, pool_num):
        self.num_actions = num_actions  

        # 経験を記憶するメモリオブジェクトを生成
        self.memory = ReplayMemory
        self.Transition = self.memory.Transition
        #環境を設定
        self.ban = ban
        #定数
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.lr = lr
        self.T = T
        # ニューラルネットワークを設定
        
        #self.main_model = network().to(device)
        #self.new_model = network().to(device)
        
        self.main_model = network()
        self.new_model = network()
        
        #gpu並列処理
        self.main_model = nn.DataParallel(self.main_model).to(device)
        self.new_model = nn.DataParallel(self.new_model).to(device)
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
        
        #並列処理　同時に何スレッド使うか
        self.pool_num = pool_num

    #次の手を決める
    def decide_action(self, ban, model, player_side, search_depth, step,episode_sum ,ep_random_data ,fastmode=False):
        reward = 0
        if fastmode:#デバック用　NNを使わない
            #print("fastmode")
            
            action = random.choice(ban.ban_put_available())  # 行動をランダムに返す
            
            r = action[0]
            c = action[1]
            
            v_ary = np.zeros(BANSIZE)
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
        EPS_DECAY = 10000
        
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * episode_sum / EPS_DECAY)
        
        if  sample > eps_threshold:#step > 1 n手目からNNを使う
            
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
                
    
    def update_target_network(self):
        #ネットワークを更新
        #target networkをmain networkと同じにする 
        self.target_model.load_state_dict(self.main_model.state_dict())
    
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
        p_ary , _ = model(state.to(device))
        p_ary = p_ary.detach().cpu().numpy()[0]
        p_ary_index = np.argsort(p_ary)[::-1] 
            
        ban_put_available = ban_copy.ban_put_available()
        #print(ban_put_available)
        
        for index in p_ary_index:
            r,c = index2rc(index)
            if [r, c] in ban_put_available:
                break
        
        ban_copy.ban_applay(player_side, r, c)
        
        #ban_copy.ban_print()
        
        if ban_copy.ban_fill():
            #print("もう打てないよ！！")
            reward = 0
            terminal = True
            #print("fill")
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
        
            
            
        
    
    #ある盤面(env)からあるプレーヤー(player_side_put)が手を打った時のあるプレイヤー(player_side_eval)のv ot qの配列を返す
    def rtn_p_ary(self, ban, model, player_side_put, N):
        
        flg_win = 0#win -> 1 
        flg_lose = 0#lose -> 1 
        
        ban_put_available = ban.ban_put_available() #打てる場所
        state_put_first = chg_input_cnn(ban, player_side_put)
        p_estimale_put, _ = model(state_put_first.to(device))
        
        state_put_first = chg_input_cnn(ban, 1-player_side_put)
        p_estimale_put_opp, _ = model(state_put_first.to(device))
        
        p_ary = np.zeros(BANSIZE)#v array 打つ人のv array
        p_ary[:] = np.nan #全ての要素をnanにする
        
        #numpyに変更
        p_estimale_put = p_estimale_put.detach().cpu().numpy()[0]
        p_estimale_put_opp = p_estimale_put_opp.detach().cpu().numpy()[0]
        
        win_ary = np.zeros(BANSIZE)#win->1, not win->0の配列 勝つ手
        lose_ary = np.zeros(BANSIZE)#lose->1, not lose->0の配列 負ける手
        lose_ary[:] = -1
        
        
        v_ary_win = np.zeros(BANSIZE)#初期値0
        v_ary_lose = np.full(BANSIZE,-1)#初期値-1
        v_ary_tumi = np.zeros(BANSIZE)#初期値0
        
        win_put_num = 0#１で王手，２以上でつみ
        
        #print(ban_put_available)
        #print(len(ban_put_available))
        for [r_n, c_n] in ban_put_available:#打てる手を順に打っていく
            
            index = rc2index(r_n, c_n)
            
            #環境をコピーする 自分用
            ban_copy = copy.deepcopy(ban)
            #自分が打つ 
            ban_copy.ban_applay(player_side, r_n, c_n)
            
            #環境をコピーする 相手用
            ban_copy2 = copy.deepcopy(ban)
            #相手が打つ
            ban_copy2.ban_applay(1-player_side, r_n, c_n)
            
            #どれぐらい石が連結しているのか
            ren_num = ban_copy.ban_ren_cnt(player_side_put, r_n, c_n)
            ren_num_op = ban_copy.ban_ren_cnt(1-player_side_put, r_n, c_n)
            
            ren_num = min(ren_num, WINREN)#上限をWINRENに設定
            ren_num_op = min(ren_num_op, WINREN)#上限をWINRENに設定
            
        
            
            if ban_copy.ban_win(player_side_put, r_n, c_n):#自分が打って自分が勝ったとき
                #print("ban_win")
                p_ary[index] = 1
                win_ary[index] = 1
                v_ary_win[index] = 1
                
                #print("{} {} : {},{}".format(N,player_side,r_n,c_n))
                flg_win = 1
                
                win_put_num += 1
                v_ary_tumi[index] = 1
                
            elif ban_copy2.ban_win(1-player_side_put, r_n, c_n):
                #print("ban_win 1-player")
                lose_ary[index] = 0.5*p_estimale_put[index] + 0.5*(ren_num/WINREN)#打ったら負けるところを-1に
                v_ary_lose[index] = p_estimale_put[index]
                flg_lose = 1#flgを立てる
                
                v_ary_win[index] = p_estimale_put[index]
                v_ary_tumi[index] = 0.9 + 0.1*p_estimale_put[index]
                
            else:
                p_ary[index] = 0.5*p_estimale_put[index] + 0.5*(ren_num/WINREN)
                v_ary_win[index] = p_estimale_put[index]
                v_ary_tumi[index] = 0.9 + 0.1*p_estimale_put[index]
                
        
        p_ary[np.isnan(p_ary)] = 0 
        
        if flg_win == 1:
            p_ary = win_ary
            v_ary = v_ary_win
        elif flg_lose == 1:
            p_ary = lose_ary
            v_ary = v_ary_lose
        else:
            v_ary = p_estimale_put
        
        if win_put_num >= 2:#詰み状態のとき
            #print("詰み！！　これで勝利は確実だ！！")
            v_ary = v_ary_tumi
        
        
        #flg_win, p_ary, win_ary, flg_lose , lose_ary
        return v_ary,flg_win, p_ary, win_ary, flg_lose , lose_ary
                
                
    def searchGameTree(self, ban, model, player_side_put, search_depth):
        ban_copy = copy.deepcopy(ban)
        N_step = 1#探索深さ
        #層１
        #print("層１")
        
        v_ary_1, flg_win_1, p_ary_1, win_ary_1, flg_lose_1 , lose_ary_1 = self.rtn_p_ary(ban_copy, model, player_side_put, N_step)
        
        
        
        v_ary = v_ary_1#初期値
        p_ary = p_ary_1
        
        #print("wight")
        ban.ban_print_p_ary(self.softmax_numpy(p_ary, 0.01))
        #ban.ban_print_p_ary(p_ary)
            
        for i in range(int(BANSIZE/3)):
            ban_copy2 = copy.deepcopy(ban_copy)
            #print("{}回目の探索".format(i))
            #ban.ban_print_p_ary(p_ary)
            
            wight = self.softmax_numpy(p_ary, 0.01)
            index=0 #これから打つindex
            r,c = 0,0
            #print(wight)
            
            counter = 0
            while True:
                #print('*', end="")
                index = random.choices(range(BANSIZE), weights=wight)[0]
                #print(index)
                r,c = index2rc(index)
                if [r, c] in ban_copy2.ban_put_available():
                    #打てる手なら
                    break
                if counter >= 9:
                    #長くかかる場合は強制的にランダムチョイス
                    [r, c] = random.choices(ban_copy2.ban_put_available())[0]
                    index = rc2index(r,c)
                    break
                counter += 1
            #print(index, index2rc(index))
            ban_copy2.ban_applay(player_side_put, r, c)#打つ
            
            if ban_copy2.ban_win(player_side_put, r, c):#打って勝ったら
                p_ary[index] = 1
            else:
                #相手の状況を調べる
                v_ary_op, flg_win_op, p_ary_op, win_ary_op, flg_lose_op , lose_ary_op = self.rtn_p_ary(ban_copy2, model, 1-player_side_put, 2)
                if flg_win_op == 1:#相手が買ったら
                    p_ary[index] = -1
                else:
                
                    r_2, c_2 = index2rc(np.argmax(v_ary_op))
                    
                    ban_copy2.ban_applay(1-player_side_put, r_2, c_2)#相手が打つ
                    
                    #自分の状況を調べる
                    v_ary_2,flg_win_2, p_ary_2, win_ary_2, flg_lose_2 , lose_ary_2 = self.rtn_p_ary(ban_copy2, model, player_side_put, 3)
                    #v_ary_opp2,_,_,_,_,_ = self.rtn_p_ary(ban_copy2, model, 1-player_side_put, 3)
                    
                    v_ary[index] = np.mean(v_ary)
                    p_ary[index] = np.mean(v_ary)
                    
            
        return v_ary
        
            
            
        
    
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
            '''
            if len(self.memory) < self.BATCH_SIZE:
                print('len(self.memory) : {}'.format(len(self.memory)))
                return
            '''
            #BATCH_SIZE = min(len(self.memory), self.BATCH_SIZE)#self.BATCH_SIZEに達するまでは今あるデータで訓練

            BATCH_SIZE = self.BATCH_SIZE
            transitions = self.memory.sample(BATCH_SIZE)

            batch = self.Transition(*zip(*transitions))

            #state', 'action', 'next_state', 'reward'
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            
            
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
            
            state_action_values = self.new_model(state_batch)[0].gather(1, action_batch)
            
            next_state_values = torch.zeros(BATCH_SIZE, device=device)
            next_state_values[non_final_mask] = self.new_model(non_final_next_states)[0].max(1)[0].detach()
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
        
    
    


# In[13]:


#txtファイルにログを書き込む
def log_print(string):
    f = open(log_filename, mode='a')
    f.write(str(string) + '\n')
    f.close()
    print(string)


def write_lr(string):
    f = open(lr_filename, mode='a')
    f.write(str(string) + '\n')
    f.close()
    print("lr書き込み完了")
    
    
def lr_file_read():
    f = open(lr_filename)
    data = f.readlines()
    f.close()
    #print("lr_file_read : {}".format(data[0]))
    return float(data[0])



    


# In[14]:


def decide_action_func(model, ban, state):
        '''現在の状態に応じて、行動を決定する'''
        #state = torch.from_numpy(state).type(torch.FloatTensor)
        # ε-greedy法で徐々に最適行動のみを採用する
        if True:
            model.eval()  # ネットワークを推論モードに切り替える
            with torch.no_grad():
                #action = self.model(state).max(1)[1].view(1, 1)
                out_p, _ = model(state.to(device))
                out_p = out_p.cpu().numpy()[0]
                
                p_index = np.argsort(out_p)[::-1]
                p_index = p_index
                ban_put_available = ban.ban_put_available()
                #print(ban_put_available)
                for index in p_index:
                    r,c = index2rc(index)
                    if [r, c] in ban_put_available:
                        return [r, c] , out_p

                    

def check_win_rate_put_1st(Env, brain, model, max_episode):#indexが小さいところから順に売っていく　負けなかった確率を返す
    
    not_win_0 = 0
    not_win_1 = 0
    ban = Env()
    brain = brain
    
    for episode in range(max_episode):
        ban.ban_reset()
        step = 0
        while True:
            step += 1
            #print('player 0 random')
            
            player_side = 0
            #action = random.choice(ban.ban_put_available())
            action = ban.ban_put_available()[0]

            ban.ban_applay(player_side, action[0], action[1])
            #print(action)
            #ban.ban_print()

            if ban.ban_win(player_side, action[0], action[1]):
                #print('player0 win!!')
                not_win_0 += 1
                break
            if ban.ban_fill():
                not_win_0 += 1
                break

            #print('player 1')
            player_side = 1
            state = chg_input_cnn(ban, player_side)
            action, _ = decide_action_func(model, ban, state)
            
            ban.ban_applay(player_side, action[0], action[1])
            #print(action)
            #ban.ban_print()
            if ban.ban_win(player_side, action[0], action[1]):
                #print('player1 win!!')
                not_win_1 += 1
                break
            if ban.ban_fill():
                not_win_1 += 1
                break
        #print('episode: {}/{}, win_0(AI 0): {}({}%), win_1(AI 1): {}({}%), step: {}'
        #       .format(episode+1, max_episode, win_0, int(100*win_0/(episode+1)),win_1,int(100*win_1/(episode+1)), step))
    win_rate = 100*not_win_1/(max_episode)
    return win_rate

def check_win_rate_random(Env, brain, model, max_episode):#勝率を計算する
    
    win_0 = 0
    win_1 = 0
    hiki = 0
    ban = Env()
    brain = brain
    
    for episode in range(max_episode):
        ban.ban_reset()
        step = 0
        while True:
            step += 1
            #print('player 0 random')
            
            player_side = 0
            action = random.choice(ban.ban_put_available())
            #action = ban.ban_put_available()[0]

            ban.ban_applay(player_side, action[0], action[1])
            #print(action)
            #ban.ban_print()

            if ban.ban_win(player_side, action[0], action[1]):
                #print('player0 win!!')
                win_0 += 1
                break
            if ban.ban_fill():
                hiki += 1
                break

            #print('player 1')
            player_side = 1
            state = chg_input_cnn(ban, player_side)
            action, _ = decide_action_func(model, ban, state)
            
            ban.ban_applay(player_side, action[0], action[1])
            #print(action)
            #ban.ban_print()
            if ban.ban_win(player_side, action[0], action[1]):
                #print('player1 win!!')
                win_1 += 1
                break
            if ban.ban_fill():
                hiki += 1
                break
        #print('episode: {}/{}, win_0(AI 0): {}({}%), win_1(AI 1): {}({}%), step: {}'
        #       .format(episode+1, max_episode, win_0, int(100*win_0/(episode+1)),win_1,int(100*win_1/(episode+1)), step))
    win_rate = 100*win_1/(max_episode)
    not_lose_rate = 100*(win_1 + hiki)/(max_episode)
    return win_rate, not_lose_rate


def check_win_rate_ai(Env, brain, main_model, new_model, max_episode):
    win_main = 0
    win_new = 0
    ban = Env()
    
    
    for episode in range(max_episode):
        ban.ban_reset()
        step = 0
        while True:#main_model先行
            step += 1
            #print('player 0')
            
            player_side = 0
            state = chg_input_cnn(ban, player_side)
            
            if step <= random_put_limit:
                action = random.choice(ban.ban_put_available())
            else:
                action, _= decide_action_func(main_model, ban, state)
            
            ban.ban_applay(player_side, action[0], action[1])
            #print(action)
            #ban.ban_print()

            if ban.ban_win(player_side, action[0], action[1]):
                #print('player0 win!!')
                win_main += 1
                break
            if ban.ban_fill():
                break

            #print('player 1')
            player_side = 1
            state = chg_input_cnn(ban, player_side)
            
            if step <= random_put_limit:
                action = random.choice(ban.ban_put_available())
            else:
                action, _= decide_action_func(new_model, ban, state)
            
            ban.ban_applay(player_side, action[0], action[1])
            #print(action)
            #ban.ban_print()
            if ban.ban_win(player_side, action[0], action[1]):
                #print('player1 win!!')
                win_new += 1
                break
            if ban.ban_fill():
                break

        
        ban.ban_reset()
        step = 0
        while True:#new_model先行
            step += 1
            #print('player 0 random')
            
            player_side = 0
            state = chg_input_cnn(ban, player_side)
            action ,_= decide_action_func(new_model, ban, state)
            
            ban.ban_applay(player_side, action[0], action[1])
            #print(action)
            #ban.ban_print()

            if ban.ban_win(player_side, action[0], action[1]):
                #print('player0 win!!')
                win_new += 1
                break
            if ban.ban_fill():
                break

            #print('player 1')
            player_side = 1
            state = chg_input_cnn(ban, player_side)
            action, _= decide_action_func(main_model, ban, state)
            
            ban.ban_applay(player_side, action[0], action[1])
            #print(action)
            #ban.ban_print()
            if ban.ban_win(player_side, action[0], action[1]):
                #print('player1 win!!')
                win_main += 1
                break
            if ban.ban_fill():
                break

    
    
    win_rate = 100*win_new/(max_episode*2)
    return win_rate



        


# In[15]:


import numpy as np
import torch
import copy
import datetime

from tensorboardX import SummaryWriter



GAMMA = 0.999  # 時間割引率
NUM_EPISODES = 50# 最大試行回数 これを行うごとにネットワークを比較する
BATCH_SIZE = 100
epoch_num = 1 #学習する回数
CAPACITY = 10000
lr = 0.01 #学習係数 初期値
T = 0.1 #逆温度　初期値
pool_num = 1  #並列処理

flg_fastmode = False #探索なしランダム打ち　デバック用

gen_num_limit = 500 #モデルが何世代目になったら訓練をやめるか
random_put_limit = 3#最初はランダムに打つ
update_win_rate = 55 #これ以上の勝率のときネットワークを更新する
model_filename = 'model_cnn_01_31_dqn_var1_part11'
file_path = "models/"+ model_filename
MEMO = "epoch10 lr0.01, dqn スケール可変 GAMMA = 0.999 out_p = tanh EPS_END = 0.05 decay=10000 hardtanh  activation_fun 最後なし bn追加"



if not os.path.exists(file_path):
            os.makedirs(file_path)
        

log_filename = file_path + "/" +  model_filename +'.txt' #ログ用
lr_filename = file_path + "/lr.txt" #ログ用

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




ban = Env()
memory = ReplayMemory(CAPACITY, ban)
brain = Brain_dqn(NeuralNet_cnn, 2*ban.size, ban.size, ban, memory,  GAMMA, BATCH_SIZE,  lr, T, pool_num)

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
                    memory.push(tmp_data[len_data-3][0], tmp_data[len_data-3][1], state, reward_0)

                
                    
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
            #log_print('episode_sum: {:08d}, episode: {:04d}/{:04d}, step:{:04d}, lr : {:03.5f}, loss : {:03.20f}'.format(episode_sum, episode+1, NUM_EPISODES, step, brain.lr, brain.loss_num))


        if episode_sum > ep_random_data:
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
            #log_print("vs old model : " + str(win_rate_for_check))
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

    #メモリデータを多くした．
    #v_outputを追加した．
    #loss関数をgo zeroのようにした．
    #learning rateを可変にした．
    #batch sizeにまでデータがたまるまでは，今あるもので訓練するようにした．
    #Tを可変にした
    
    #やること
    #aiの挙動をみる
    #探索アルゴリズムを修正をする
    
    
    


# In[ ]:


##評価


model_filename_0 = 'model_cnn_8_18_ver4.pht'
model_filename_1 = 'models/model_cnn_01_27_dqn_var1_part5/model_cnn_01_27_dqn_var1_part5_3gen.pht'

'''
model_0 = NeuralNet_cnn().to(device)
param = torch.load(model_filename_0)
model_0.load_state_dict(param)
'''
model_1 = nn.DataParallel(NeuralNet_cnn()).to(device)
param = torch.load(model_filename_1)
model_1.load_state_dict(param)



ban = Env()


# In[ ]:


#ai同士を戦わせて評価





win_0 = 0
win_1 = 0

max_episode = 100
step = 0
for episode in range(max_episode):
    ban.ban_reset()
    step = 0
    while True:
        step += 1
        #print('player 0 random')
        '''
        player_side = 0
        state = ban.status(player_side)
        state_another_player = ban.status(1 - player_side)
        state_all = ban.status_all()
        state = chg_input_cnn(state, state_another_player, state_all)
        action = decide_action_1(state)
        '''
        action = random.choice(ban.ban_put_available())
        #action = ban.ban_put_available()[0]
        
        ban.ban_applay(0, action[0], action[1])
        #print(action)
        #ban.ban_print()
            
        if ban.ban_win(0, action[0], action[1]):
            #print('player0 win!!')
            win_0 += 1
            break
        if ban.ban_fill():
            break
            
        #print('player 1 ai')
        player_side = 1
        state = chg_input_cnn(ban, player_side)
        
        action, _ = decide_action_func(model_1, ban, state)
        
        ban.ban_applay(1, action[0], action[1])
        #print(action)
        #ban.ban_print()
        if ban.ban_win(1, action[0], action[1]):
            #print('player1 win!!')
            win_1 += 1
            break
        if ban.ban_fill():
            break
    print('episode: {}/{}, win_0(AI 0): {}({}%), win_1(AI 1): {}({}%), step: {}'
           .format(episode+1, max_episode, win_0, int(100*win_0/(episode+1)),win_1,int(100*win_1/(episode+1)), step))






# In[ ]:


step = 0
for episode in range(1):
    ban.ban_reset()
    step = 0
    while True:
        step += 1
        print('player 0 you')
        
        
        player_side = 0
        
        state = chg_input_cnn(ban, player_side)
        
        action , out_p= decide_action_func(model_1, ban, state)
        
        ban.ban_print_p_ary(out_p)
        
        
        while True:
            print('何行目?')
            p0_gyo = int(input('>>>  '))
            print('何列目?')
            p0_retu = int(input('>>>  '))
            if ban.ban_applay(0, p0_gyo, p0_retu):
                ban.ban[p0_gyo][p0_retu] = 0
                ban.ban_print()
                
                state = chg_input_cnn(ban, player_side)
                out_p, out_v = model_1(state.to(device))
                #print(out_p.reshape(14,14))
                #print(out_v)
                break
            else:
                print('違う場所をさしてください')
        if ban.ban_win(0, p0_gyo, p0_retu):
            print('player0 win!!')
            break
        
        if ban.ban_fill():
            print("ban fill!")
            break
            
        print('player 1 ai')
        player_side = 1

        state = chg_input_cnn(ban, player_side)
        
        action , out_p= decide_action_func(model_1, ban, state)
        #out_p = out_p.detach().cpu().numpy()
        #print(out_p)
        
        ban.ban_print_p_ary(out_p)
        ban.ban_applay(1, action[0], action[1])
        print(action)
        ban.ban_print()
        out_p, out_v = model_1(state.to(device))
        #out_p = out_p.detach().cpu().numpy()[0]
        #print(out_p.reshape(14,14))
        #print(out_v)
        if ban.ban_win(1, action[0], action[1]):
            #print('player1 win!!')
            win_1 += 1
            break
        if ban.ban_fill():
            break
    


# In[ ]:


# 自己対戦
step = 0
for episode in range(1):
    ban.ban_reset()
    step = 0
    while True:
        step += 1
        print('player 0 you')
        
        player_side = 0
        while True:
            print('何行目?')
            p0_gyo = int(input('>>>  '))
            print('何列目?')
            p0_retu = int(input('>>>  '))
            if ban.ban_applay(player_side, p0_gyo, p0_retu):
                ban.ban_print()
                
                state = chg_input_cnn(ban, player_side)
                out_p, out_v = model_1(state.to(device))
                #print(out_p.reshape(14,14))
                print(out_v)
                break
            else:
                print('違う場所をさしてください')
        if ban.ban_win(player_side, p0_gyo, p0_retu):
            print('player0 win!!')
            break
            
        print('player 1')
        player_side = 1
        while True:
            print('何行目?')
            p0_gyo = int(input('>>>  '))
            print('何列目?')
            p0_retu = int(input('>>>  '))
            if ban.ban_applay(player_side, p0_gyo, p0_retu):
                ban.ban_print()
                
                state = chg_input_cnn(ban, player_side)
                out_p, out_v = model_1(state.to(device))
                #print(out_p.reshape(14,14))
                print(out_v)
                break
            else:
                print('違う場所をさしてください')
        if ban.ban_win(player_side, p0_gyo, p0_retu):
            print('player0 win!!')
            break
    



# In[ ]:


from multiprocessing import Pool


def line_cnt(self, player, gyo, retu, dx, dy):
        cnt = 0
        while True:
            gyo += dy
            retu += dx
            if gyo <0 or gyo > 13 or retu < 0 or retu > 13:
                break

            if self.ban[gyo][retu] == player:
                #print(gyo)
                #print(retu)
                cnt += 1
            else:
                break
        return cnt 
class Y():
    def __init__(self):
        self.a = 111
        self.b = 0
    def t(self, x):
        return x*x

    def f(self, ban_ary):
        if ban_ary == 1:
            return self.t(1) + self.a
        else:
            self.b += 1
            return self.t(self.b)
cls = Y()
index = [ i + 196 for i in range(19600000)]
        
with Pool(8) as p:
    p.map(cls.f, index)
    print("succes!")
print(cls.b)


# In[ ]:


l_2d_ok = [ i + 196 for i in range(196)]

print(l_2d_ok)


# In[ ]:


import torch

from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np

m = nn.Sigmoid()
loss = nn.BCELoss()
loss_c = nn.CrossEntropyLoss()
input = torch.randn(7, 5)
target = torch.LongTensor(3).random_(5)

input = np.random.rand(7, 5)
target = np.argmax(input, axis = 1)

print(input)
print(target)

input = torch.from_numpy(input)
target = torch.from_numpy(target)

output = loss_c(input, target)#torch.LongTensor([0,0,0])

print(output)


# In[ ]:


import pprint
pp = pprint.PrettyPrinter(indent=4)

pp.pprint([1,2,3,4,5,6,7,8])


# In[ ]:


import random
import numpy as np
#wight = [1,2,3,4,5,6,7,8,9,10]
wight = [1,-0.9,0.5]
tmp = [0,0,0]
tmp = np.array(tmp)
for i in range(10000):
    tmp[random.choices(range(3), weights=wight)[0]] += 1
print(tmp/tmp[0])

