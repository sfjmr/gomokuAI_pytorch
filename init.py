import torch
import datetime


BANHEN = 14
BANSIZE = BANHEN**2
WINREN = 5

device = ("cuda" if torch.cuda.is_available() else "cpu")


GAMMA = 0.7
 # 時間割引率
NUM_EPISODES = 50  # 最大試行回数 これを行うごとにネットワークを比較する
BATCH_SIZE = 100
epoch_num = 1  # 学習する回数
CAPACITY = 10000
lr = 0.01  # 学習係数 初期値
T = 0.1  # 逆温度　初期値
pool_num = 1  # 並列処理

flg_fastmode = False  # 探索なしランダム打ち　デバック用

gen_num_limit = 500  # モデルが何世代目になったら訓練をやめるか
random_put_limit = 3  # 最初はランダムに打つ
update_win_rate = 55  # これ以上の勝率のときネットワークを更新する

EPS_START = 0.9
EPS_END = 0.2
EPS_DECAY = 1000

random_search_value = 100
        


now = now = datetime.datetime.now()
model_filename = 'model_cnn_02_25_dqn_var1_part1_{0:%Y%m%d%H%M%S_}'.format(now)
file_path = "models/" + model_filename
MEMO = " {} lr={},  GAMMA={} ".format(model_filename, lr,GAMMA)

log_filename = file_path + "/" + model_filename + '.txt'  # ログ用
lr_filename = file_path + "/lr.txt"  # ログ用
