# gomokuAI_pytorch

This is a Python3 code of reinforcement-learning to play Noughts & Crosses.

これは強化学習で五目並べをプレイするpython3のコードです．

使用したアルゴリズムはDQNです．

## 環境

pytorch  
tensorboardX

## 使い方

init.pyで各種パラメーターを設定します． 

学習を始めるにはtrain.pyを動かします．  
```python trian.py```  

初期設定では50ゲームごとにモデルを比較し，新モデルが古いモデルよりも強かったら（勝率が55%を超えたら）モデルを更新するようにします．  
モデルの場所はmodels/以下にあります．

## 結果

三目並べでランダムプレイヤーとaiとの勝負でaiが後手のとき大体99%ぐらい

https://github.com/sfjmr/gomokuAI_pytorch/commit/8ed3bacdd1e70ef272c1a8a1b4c86a88e46a2636#commitcomment-32269474
