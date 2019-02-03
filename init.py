import torch

BANHEN = 3
BANSIZE = BANHEN**2
WINREN = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


GAMMA = 0.999  # 時間割引率
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
model_filename = 'model_cnn_01_31_dqn_var1_part11'
file_path = "models/" + model_filename
MEMO = "epoch10 lr0.01, dqn スケール可変 GAMMA = 0.999 out_p = tanh EPS_END = 0.05 decay=10000 hardtanh  activation_fun 最後なし bn追加"

log_filename = file_path + "/" + model_filename + '.txt'  # ログ用
lr_filename = file_path + "/lr.txt"  # ログ用
