#試合データを蓄積

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
            'Transition', ('state', 'action', 'next_state', 'put_available_position','reward'))

    def push(self, state, action, next_state, put_available_position,reward):
        
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)  # メモリが満タンでないときは足す

        # namedtupleのTransitionを使用し、値とフィールド名をペアにして保存します
        self.memory[self.index] = self.Transition(
            state, action, next_state, put_available_position, reward)

        self.index = (self.index + 1) % self.capacity  # 保存するindexを1つずらす

    def sample(self, batch_size):
        '''batch_size分だけ、ランダムに保存内容を取り出す'''
        return random.sample(self.memory, batch_size)

    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)
