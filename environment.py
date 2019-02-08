#環境

import numpy as np
import heapq
import matplotlib.pyplot as plt
from general_func import rc2index, index2rc



class Env:
    def __init__(self, BANHEN, WINREN):
        self.BANHEN = BANHEN
        self.BANSIZE = BANHEN**2
        self.WINREN = WINREN

        self.ban = [[-1 for i in range(BANHEN)]
                    for j in range(BANHEN)]  # サイズ BANHEN*BANHEN
        self.screen_n_rows = BANHEN  # 行
        self.screen_n_cols = BANHEN  # 列
        self.size = self.screen_n_rows * self.screen_n_cols
        #self.enable_actions = np.arange(self.screen_n_rows*self.screen_n_cols)
        self.enable_actions = []
        for i in range(BANHEN):
            for j in range(BANHEN):
                self.enable_actions.append([i, j])
        #print(self.enable_actions)
        #print(self.enable_actions.index([0,0]))
        #self.name = os.path.splitext(os.path.basename(__file__))[0]

    def ban_reset(self):
        self.ban = [[-1 for i in range(self.BANHEN)] for j in range(self.BANHEN)]

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
            if (self.line_cnt(player, gyo, retu, 1, 0) + self.line_cnt(player, gyo, retu, -1, 0) >= self.WINREN-1) or (self.line_cnt(player, gyo, retu, 0, 1) + self.line_cnt(player, gyo, retu, 0, -1) >= self.WINREN-1) or (self.line_cnt(player, gyo, retu, 1, 1) + self.line_cnt(player, gyo, retu, -1, -1) >= self.WINREN-1) or (self.line_cnt(player, gyo, retu, 1, -1) + self.line_cnt(player, gyo, retu, -1, 1) >= self.WINREN-1):
                return True
            else:
                return False
        else:
            if (self.line_cnt(player, gyo, retu, 1, 0, ban) + self.line_cnt(player, gyo, retu, -1, 0, ban) >= self.WINREN-1) or (self.line_cnt(player, gyo, retu, 0, 1, ban) + self.line_cnt(player, gyo, retu, 0, -1, ban) >= self.WINREN-1) or (self.line_cnt(player, gyo, retu, 1, 1, ban) + self.line_cnt(player, gyo, retu, -1, -1, ban) >= self.WINREN-1) or (self.line_cnt(player, gyo, retu, 1, -1, ban) + self.line_cnt(player, gyo, retu, -1, 1, ban) >= self.WINREN-1):
                return True
            else:
                return False

    def ban_ren_cnt(self, player, gyo, retu):  # 石がどれくらい連続しているか
        ren_num = 0
        ren_num = max((self.line_cnt(player, gyo, retu, 1,  0) + self.line_cnt(player, gyo, retu, -1,  0)),
                      (self.line_cnt(player, gyo, retu, 0,  1) +
                       self.line_cnt(player, gyo, retu,  0, -1)),
                      (self.line_cnt(player, gyo, retu, 1,  1) +
                       self.line_cnt(player, gyo, retu, -1, -1)),
                      (self.line_cnt(player, gyo, retu, 1, -1) + self.line_cnt(player, gyo, retu, -1,  1)))

        return ren_num

    def line_cnt(self, player, gyo, retu, dx, dy, ban=None):
        if ban is None:
            cnt = 0
            while True:
                gyo += dy
                retu += dx
                if gyo < 0 or gyo > self.BANHEN-1 or retu < 0 or retu > self.BANHEN-1:
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
                if gyo < 0 or gyo > self.BANHEN-1 or retu < 0 or retu > self.BANHEN-1:
                    break

                if ban[gyo][retu] == player:
                    #print(gyo)
                    #print(retu)
                    cnt += 1
                else:
                    break
            return cnt

    def ban_print(self):  # 現在の環境を表示
        #print("ban_print")
        #print('   00  01  02  03  04  05  06  07  08  09  10  11  12  13  ')

        for i in range(self.BANHEN):
            print('  {0:02d}'.format(i), end="")
        print('')
        for gyo in range(self.BANHEN):
            print('{0:02d}'.format(gyo), end="")
            print(' ', end="")
            for retu in range(self.BANHEN):
                if self.ban[gyo][retu] == -1:
                    print('-   ', end="")
                elif self.ban[gyo][retu] == 0:
                    print('●   ', end="")
                elif self.ban[gyo][retu] == 1:
                    print('○   ', end="")
            print('\n')

    def ban_print_p_ary(self, p_ary):  # 現在の環境を表示
        #print("ban_print_p_ary")
        p_ary_index = heapq.nlargest(10, p_ary)  # p_aryの上位 n個を取得

        x_p = []
        y_p = []
        v_p = []

        x_0 = []
        y_0 = []

        x_1 = []
        y_1 = []

        #print('   00  01  02  03  04  05  06  07  08  09  10  11  12  13  ')
        for gyo in range(self.BANHEN):
            #print('{0:02d}'.format(gyo), end="")
            #print(' ', end="")
            for retu in range(self.BANHEN):

                y_p.append(self.BANHEN-1-gyo)
                x_p.append(retu)

                if self.ban[gyo][retu] == -1:
                    v_p.append(p_ary[self.BANHEN*gyo + retu])
                    #if p_ary[14*gyo + retu]  in p_ary_index:
                    #print("{:01.1f}|".format(10*p_ary[14*gyo + retu]), end="")
                    #print('#   ', end="")
                    #else:
                    #    print('-   ', end="")
                elif self.ban[gyo][retu] == 0:
                    #print('●   ', end="")
                    v_p.append(0)
                    x_0.append(retu)
                    y_0.append(self.BANHEN-1-gyo)

                elif self.ban[gyo][retu] == 1:
                    #print('○   ', end="")
                    v_p.append(0)
                    x_1.append(retu)
                    y_1.append(self.BANHEN-1-gyo)

            #print('\n')

        plt.scatter(x_p, y_p, s=100, c=v_p, cmap='pink_r')
        plt.colorbar()
        plt.scatter(x_0, y_0, s=100, c="blue", alpha="1",
                    linewidths="1", edgecolors="black")
        plt.scatter(x_1, y_1, s=100, c="white", alpha="1",
                    linewidths="1", edgecolors="black")

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

    def ban_fill(self):  # 盤が埋まっているときtureを返す
        state = True
        for gyo in range(self.BANHEN):
            for retu in range(self.BANHEN):
                if self.ban[gyo][retu] == -1:
                    state = False
        return state

    def ban_put_available(self):  # 打てる手を返す
        put_available = []
        for gyo in range(self.BANHEN):
            for retu in range(self.BANHEN):
                if self.ban[gyo][retu] == -1:
                    #put_available.append(14*gyo + retu)
                    put_available.append([gyo, retu])
        return put_available

    def ban_put_available_state(self, state):  # 打てる手を返す
        put_available_state = []
        for gyo in range(self.BANHEN):
            for retu in range(self.BANHEN):
                if int(state[gyo][retu]) == 0:
                    put_available_state.append([gyo, retu])
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
                if self.ban[r_n][c_n] != -1:  # 空白じゃなかったら
                    player_status[r_n][c_n] = 1

        return np.array(player_status)

    def rtn_put_available_position(self):
        #空いている場所-> 1,埋まっている場所->0
        put_available_position = np.zeros(self.BANSIZE)

        for r_n in range(self.screen_n_rows):
            for c_n in range(self.screen_n_cols):
                if self.ban[r_n][c_n] == -1:  #空いていたら
                    index = rc2index(r_n,c_n)
                    put_available_position[index] = 1
        
        return put_available_position
