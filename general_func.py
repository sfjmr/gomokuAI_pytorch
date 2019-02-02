import torch
import numpy as np

from train import BANHEN,BANSIZE,log_filename,lr_filename

def chg_input_cnn(ban, player_side):
    state = ban.status(player_side)
    state_another_player = ban.status(1 - player_side)
    state_all = ban.status_all()

    state_out = [state, state_another_player, state_all]
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
                    if rotation == 0:
                        player_status[r_n][c_n] = data[r_n][c_n]
                    elif rotation == 1:
                        player_status[screen_n_cols-1 -
                                      c_n][r_n] = data[r_n][c_n]
                    elif rotation == 2:
                        player_status[screen_n_rows-1 -
                                      r_n][screen_n_cols-1 - c_n] = data[r_n][c_n]
                    elif rotation == 3:
                        player_status[c_n][screen_n_rows -
                                           1 - r_n] = data[r_n][c_n]
                elif reverse == 1:
                    if rotation == 0:
                        player_status[r_n][screen_n_cols -
                                           1 - c_n] = data[r_n][c_n]
                    elif rotation == 1:
                        player_status[screen_n_cols-1 -
                                      c_n][screen_n_rows-1 - r_n] = data[r_n][c_n]
                    elif rotation == 2:
                        player_status[screen_n_rows-1 -
                                      r_n][c_n] = data[r_n][c_n]
                    elif rotation == 3:
                        player_status[c_n][r_n] = data[r_n][c_n]

        return np.array(player_status)


def index_rotation(r_n, c_n, rotation, reverse):
    screen_n_rows, screen_n_cols = BANHEN, BANHEN
    if reverse == 0:
        if rotation == 0:
            return r_n, c_n
        elif rotation == 1:
            return screen_n_cols-1 - c_n, r_n
        elif rotation == 2:
            return screen_n_rows-1 - r_n, screen_n_cols-1 - c_n
        elif rotation == 3:
            return c_n, screen_n_rows-1 - r_n
    elif reverse == 1:
        if rotation == 0:
            return r_n, screen_n_cols-1 - c_n
        elif rotation == 1:
            return screen_n_cols-1 - c_n, screen_n_rows-1 - r_n
        elif rotation == 2:
            return screen_n_rows-1 - r_n, c_n
        elif rotation == 3:
            return c_n, r_n


def chg_input_cnn_to8(ban, player_side):  # 4方向に回転 and 反転にしてデータ量を8倍にする.
    output = []

    for i in range(8):
        rotation = i % 4  # 回転 0,1,2,3
        reverse = int(i/4)  # 反転 0,1

        state = data_rotation(ban.status(player_side), rotation, reverse)
        state_another_player = data_rotation(
            ban.status(1 - player_side), rotation, reverse)
        state_all = ban.status_all()

        state_out = [state, state_another_player, state_all]
        state_out = torch.from_numpy(
            np.array([state_out])).type(torch.FloatTensor)
        output.append(state_out)

    return output


# In[4]:


def index2rc(index):
    return int(index/BANHEN), index % BANHEN


def rc2index(r, c):
    return BANHEN*r + c


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
