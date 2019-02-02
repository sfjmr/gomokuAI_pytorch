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
    def __init__(self, BANHEN, BANSIZE):
        super(NeuralNet_cnn, self).__init__()
        ch_num = 10  # 50
        
        self.BANHEN = BANHEN
        self.BANSIZE = BANSIZE
        # 入力ch3, 出力ch_num ,カーネルサイズ, 3
        self.conv1 = nn.Conv2d(3, ch_num, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ch_num, ch_num, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ch_num, ch_num, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(ch_num, ch_num, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(ch_num, ch_num, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(ch_num, ch_num, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(ch_num, ch_num, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(ch_num, ch_num, kernel_size=3, padding=1)
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
        self.conv_p1 = nn.Conv2d(ch_num, 2, kernel_size=3, padding=1)
        self.bn_p1 = nn.BatchNorm2d(2)
        self.conv_p1 = nn.Conv2d(ch_num, 1, kernel_size=3, padding=1)
        self.bn_p1 = nn.BatchNorm2d(1)
        self.fc_p2 = nn.Linear(BANHEN*BANHEN*2, BANHEN*BANHEN)
        self.softmax_p3 = nn.Softmax()
        self.tanh_p4 = nn.Tanh()
        self.sigmoid_p4 = nn.Sigmoid()
        self.hardtanh_p4 = nn.Hardtanh()

        #value
        self.conv_v1 = nn.Conv2d(ch_num, 1, kernel_size=3, padding=1)
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
        out_p = self.hardtanh_p4(out_p1)
        #out_p =  F.softmax(out_p1, dim=1)
        #out_p = out_p1

        #value
        out_v1 = F.relu(self.bn_v1(self.conv_v1(out7)))
        out_v1 = out_v1.view(-1, self.BANHEN**2)
        out_v2 = F.relu(self.fc_v2(out_v1))
        out_v = self.tanh_v4(self.fc_v3(out_v2))
        #out_v = self.sigmoid_v4(self.fc_v3(out_v2))

        return out_p, out_v
