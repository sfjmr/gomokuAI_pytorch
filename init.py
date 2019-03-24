#各種パラメーターを設定
import torch
import datetime

#盤の大きさ
BANHEN = 14
BANSIZE = BANHEN**2
#何個石が並べば勝つか
WINREN = 5

device = ("cuda" if torch.cuda.is_available() else "cpu")


GAMMA = 0.7 # 時間割引率
NUM_EPISODES = 50  # 最大試行回数 これを行うごとにネットワークを比較する
BATCH_SIZE = 100 #学習するバッチサイズ
epoch_num = 1  # エポック数
CAPACITY = 10000 #蓄えるデータ数
lr = 0.01  # 学習係数 初期値
T = 0.1  # 逆温度　初期値
pool_num = 1  # 並列処理

flg_fastmode = False  # 探索なしランダム打ち　デバック用

gen_num_limit = 500  # モデルが何世代目になったら訓練をやめるか(使わない)
random_put_limit = 3  # 最初はランダムに打つ
update_win_rate = 55  # これ以上の勝率のときネットワークを更新する

#訓練でaiが石をどのくらいの確率でランダムに打つか
EPS_START = 0.9 # 初期値
EPS_END = 0.2 #最終的にこの確率になる
EPS_DECAY = 1000 # どれぐらいの速度でゆっくり収束するか　値が大きいほど遅く収束する

random_search_value = 100
        

#現在時刻
now = datetime.datetime.now()

#モデルのファイルネーム
model_filename = 'model_5moku_{0:%Y%m%d%H%M%S_}'.format(now)
file_path = "models/" + model_filename
MEMO = " {} lr={},  GAMMA={} ".format(model_filename, lr,GAMMA)

log_filename = file_path + "/" + model_filename + '.txt'  # ログ用
lr_filename = file_path + "/lr.txt"  # ログ用
