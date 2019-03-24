#勝率等を計算する
import numpy as np
import random
import torch

from general_func import index2rc, chg_input_cnn
from init import BANHEN, WINREN, device


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
                ban_put_available = ban.ban_put_available()
                #print(ban_put_available)
                for index in p_index:
                    r,c = index2rc(index)
                    if [r, c] in ban_put_available:
                        return [r, c] , out_p

                    

def check_win_rate_put_1st(Env, brain, model, max_episode):#indexが小さいところから順に売っていく　負けなかった確率を返す
    
    not_win_0 = 0
    not_win_1 = 0
    ban = Env(BANHEN, WINREN)
    brain = brain
    
    for episode in range(1):
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
    ban = Env(BANHEN, WINREN)
    brain = brain
    
    for episode in range(max_episode):
        print("\rstep : {0}/{1} ".format(episode, max_episode), end="")
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
    print()
    win_rate = 100*win_1/(max_episode)
    not_lose_rate = 100*(win_1 + hiki)/(max_episode)
    return win_rate, not_lose_rate

def check_win_rate_random_ai_first(Env, brain, model, max_episode):#勝率を計算する
    
    win_0 = 0
    win_1 = 0
    hiki = 0
    ban = Env(BANHEN, WINREN)
    brain = brain
    
    for episode in range(max_episode):
        print("\rstep : {0}/{1} ".format(episode, max_episode), end="")
        ban.ban_reset()
        step = 0
        while True:
            step += 1
            
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


        
        #print('episode: {}/{}, win_0(AI 0): {}({}%), win_1(AI 1): {}({}%), step: {}'
        #       .format(episode+1, max_episode, win_0, int(100*win_0/(episode+1)),win_1,int(100*win_1/(episode+1)), step))
    win_rate = 100*win_1/(max_episode)
    not_lose_rate = 100*(win_1 + hiki)/(max_episode)
    return win_rate, not_lose_rate


def check_win_rate_ai(Env, brain, main_model, new_model, max_episode):
    win_main = 0
    draw = 0
    win_new = 0
    
    ban = Env(BANHEN, WINREN)
    
    
    for episode in range(max_episode):
        print("\rstep : {0}/{1} ".format(episode, max_episode), end="")
        ban.ban_reset()
        step = 0
        while True:#main_model先行
            step += 1
            #print('player 0')
            
            player_side = 0
            state = chg_input_cnn(ban, player_side)
            
            if step <= 1:#ランダムに打つ
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
                draw += 1
                break

            #print('player 1')
            player_side = 1
            state = chg_input_cnn(ban, player_side)
            
            if step <= 1:
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
                draw += 1
                break

        
        ban.ban_reset()
        step = 0
        while True:#new_model先行
            step += 1
            #print('player 0 random')
            
            player_side = 0
            state = chg_input_cnn(ban, player_side)
            
            if step <= 1:#ランダムに打つ
                action = random.choice(ban.ban_put_available())
            else:
                action, _= decide_action_func(new_model, ban, state)
            
            ban.ban_applay(player_side, action[0], action[1])
            #print(action)
            #ban.ban_print()

            if ban.ban_win(player_side, action[0], action[1]):
                #print('player0 win!!')
                win_new += 1
                break
            if ban.ban_fill():
                draw += 1
                break

            #print('player 1')
            player_side = 1
            state = chg_input_cnn(ban, player_side)
            
            if step <= 1:#ランダムに打つ
                action = random.choice(ban.ban_put_available())
            else:
                action, _= decide_action_func(main_model, ban, state)
            
            ban.ban_applay(player_side, action[0], action[1])
            #print(action)
            #ban.ban_print()
            if ban.ban_win(player_side, action[0], action[1]):
                #print('player1 win!!')
                win_main += 1
                break
            if ban.ban_fill():
                draw += 1
                break

    
    
    win_rate = 100*(win_new)/(win_main + win_new)
    return win_rate
