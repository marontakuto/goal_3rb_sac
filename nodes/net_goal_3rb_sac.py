# -*- coding: utf-8 -*-

"""
このファイルではネットワークの構造を決めています
"""

from __future__ import division
import pfrl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SAC_Policy(nn.Module):
    def __init__(
            self, 
            conv_num, mid_layer_num, mid_units1, mid_units2, mid_units3, cnv_act, ful_act, 
            n_actions, n_input_channels, n_added_input, img_width, img_height
        ):

        self.conv_num = conv_num
        self.mid_layer_num = mid_layer_num
        self.cnv_act = cnv_act
        self.ful_act = ful_act
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.n_added_input = n_added_input
        self.img_width = img_width
        self.img_height = img_height
        super(SAC_Policy, self).__init__()
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        # convolution
        if self.conv_num == 1:
            channels = [16, 64] #各畳込みでのカーネル枚数
            kernels = [5, 5] #各畳込みでのカーネルサイズ
            pool_info = [2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)
        
        elif self.conv_num == 2:
            channels = [8, 16, 64, 128] #各畳込みでのカーネル枚数
            kernels = [5, 5, 5, 5] #各畳込みでのカーネルサイズ
            pool_info = [2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.conv2_1 = nn.Conv2d(channels[1], channels[2], kernels[2])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv2_2 = nn.Conv2d(channels[2], channels[3], kernels[3])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)

        elif self.conv_num == 3:
            channels = [8, 16, 32, 64, 128, 256] #各畳込みでのカーネル枚数
            kernels = [3, 3, 3, 3, 2, 2] #各畳込みでのカーネルサイズ
            pool_info = [2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.conv2_1 = nn.Conv2d(channels[1], channels[2], kernels[2])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv2_2 = nn.Conv2d(channels[2], channels[3], kernels[3])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.conv3_1 = nn.Conv2d(channels[3], channels[4], kernels[4])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv3_2 = nn.Conv2d(channels[4], channels[5], kernels[5])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)

        # Advantage
        if self.mid_layer_num == 1:
            self.fc1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.fc1.weight)

            self.fc5 = nn.Linear(mid_units1, n_actions)
            nn.init.kaiming_normal_(self.fc5.weight)

        elif self.mid_layer_num == 2:
            self.fc1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.fc1.weight)

            self.fc2 = nn.Linear(mid_units1, mid_units2)
            nn.init.kaiming_normal_(self.fc2.weight)

            self.fc5 = nn.Linear(mid_units2, n_actions)
            nn.init.kaiming_normal_(self.fc5.weight)

        elif self.mid_layer_num == 3:
            self.fc1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.fc1.weight)

            self.fc2 = nn.Linear(mid_units1, mid_units3)
            nn.init.kaiming_normal_(self.fc2.weight)

            self.fc3 = nn.Linear(mid_units3, mid_units2)
            nn.init.kaiming_normal_(self.fc3.weight)

            self.fc5 = nn.Linear(mid_units2, n_actions)
            nn.init.kaiming_normal_(self.fc5.weight)
    
    def forward(self, state):
        if self.n_added_input:
            img = state[:, :-self.n_added_input]
            lidar = state[:, -self.n_added_input:]
        else:
            img = state
        
        img = torch.reshape(img, (-1, self.n_input_channels, self.img_width, self.img_height))

        #convolution
        if self.conv_num == 1:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = h.view(-1, self.img_input)
        elif self.conv_num == 2:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv2_1(h))
            h = self.cnv_act(self.conv2_2(h))
            h = self.pool(h)
            h = h.view(-1, self.img_input)
        elif self.conv_num == 3:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv2_1(h))
            h = self.cnv_act(self.conv2_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv3_1(h))
            h = self.cnv_act(self.conv3_2(h))
            h = self.pool(h)
            h = h.view(-1, self.img_input)
        
        if self.n_added_input:
            h = torch.cat((h, lidar), axis=1)

        #全結合層の構成
        if self.mid_layer_num == 1:
            h = self.ful_act(self.fc1(h))
            h = self.fc(h)
        elif self.mid_layer_num == 2:
            h = self.ful_act(self.fc1(h))
            h = self.ful_act(self.fc2(h))
            h = self.fc5(h)
        elif self.mid_layer_num == 3:
            h = self.ful_act(self.fc1(h))
            h = self.ful_act(self.fc2(h))
            h = self.ful_act(self.fc3(h))
            h = self.fc5(h)
        
        return torch.distributions.Normal(h, 1.0)

class SAC_QFunc(nn.Module):
    def __init__(
            self, 
            conv_num, mid_layer_num, mid_units1, mid_units2, mid_units3, cnv_act, ful_act, 
            n_actions, n_input_channels, n_added_input, img_width, img_height
        ):

        self.conv_num = conv_num
        self.mid_layer_num = mid_layer_num
        self.cnv_act = cnv_act
        self.ful_act = ful_act
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.n_added_input = n_added_input
        self.img_width = img_width
        self.img_height = img_height
        super(SAC_QFunc, self).__init__()
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        # convolution
        if self.conv_num == 1:
            channels = [16, 64] #各畳込みでのカーネル枚数
            kernels = [5, 5] #各畳込みでのカーネルサイズ
            pool_info = [2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)
        
        elif self.conv_num == 2:
            channels = [8, 16, 64, 128] #各畳込みでのカーネル枚数
            kernels = [5, 5, 5, 5] #各畳込みでのカーネルサイズ
            pool_info = [2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.conv2_1 = nn.Conv2d(channels[1], channels[2], kernels[2])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv2_2 = nn.Conv2d(channels[2], channels[3], kernels[3])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)

        elif self.conv_num == 3:
            channels = [8, 16, 32, 64, 128, 256] #各畳込みでのカーネル枚数
            kernels = [3, 3, 3, 3, 2, 2] #各畳込みでのカーネルサイズ
            pool_info = [2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.conv2_1 = nn.Conv2d(channels[1], channels[2], kernels[2])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv2_2 = nn.Conv2d(channels[2], channels[3], kernels[3])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.conv3_1 = nn.Conv2d(channels[3], channels[4], kernels[4])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv3_2 = nn.Conv2d(channels[4], channels[5], kernels[5])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)

        # Advantage
        if self.mid_layer_num == 1:
            self.fc1 = nn.Linear(self.img_input + n_added_input + n_actions, mid_units1)
            nn.init.kaiming_normal_(self.fc1.weight)
            self.fc5 = nn.Linear(mid_units1, 1)
            nn.init.kaiming_normal_(self.fc5.weight)

        elif self.mid_layer_num == 2:
            self.fc1 = nn.Linear(self.img_input + n_added_input + n_actions, mid_units1)
            nn.init.kaiming_normal_(self.fc1.weight)
            self.fc2 = nn.Linear(mid_units1, mid_units2)
            nn.init.kaiming_normal_(self.fc2.weight)
            self.fc5 = nn.Linear(mid_units2, 1)
            nn.init.kaiming_normal_(self.fc5.weight)

        elif self.mid_layer_num == 3:
            self.fc1 = nn.Linear(self.img_input + n_added_input + n_actions, mid_units1)
            nn.init.kaiming_normal_(self.fc1.weight)
            self.fc2 = nn.Linear(mid_units1, mid_units3)
            nn.init.kaiming_normal_(self.fc2.weight)
            self.fc3 = nn.Linear(mid_units3, mid_units2)
            nn.init.kaiming_normal_(self.fc3.weight)
            self.fc5 = nn.Linear(mid_units2, 1)
            nn.init.kaiming_normal_(self.fc5.weight)
    
    def forward(self, stat_act):

        state, action = stat_act

        if self.n_added_input:
            img = state[:, :-self.n_added_input]
            lidar = state[:, -self.n_added_input:]
            lidar_act = torch.cat([lidar, action], dim=-1)
        else:
            img = state
        
        img = torch.reshape(img, (-1, self.n_input_channels, self.img_width, self.img_height))

        #convolution
        if self.conv_num == 1:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = h.view(-1, self.img_input)
        elif self.conv_num == 2:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv2_1(h))
            h = self.cnv_act(self.conv2_2(h))
            h = self.pool(h)
            h = h.view(-1, self.img_input)
        elif self.conv_num == 3:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv2_1(h))
            h = self.cnv_act(self.conv2_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv3_1(h))
            h = self.cnv_act(self.conv3_2(h))
            h = self.pool(h)
            h = h.view(-1, self.img_input)
        
        if self.n_added_input:
            h = torch.cat((h, lidar_act), axis=1)
        else:
            h = torch.cat((h, action), axis=1)

        #全結合層の構成
        if self.mid_layer_num == 1:
            h = self.ful_act(self.fc1(h))
            q = self.fc5(h)
        elif self.mid_layer_num == 2:
            h = self.ful_act(self.fc1(h))
            h = self.ful_act(self.fc2(h))
            q = self.fc5(h)
        elif self.mid_layer_num == 3:
            h = self.ful_act(self.fc1(h))
            h = self.ful_act(self.fc2(h))
            h = self.ful_act(self.fc3(h))
            q = self.fc5(h)
        
        return q

class Dueling_Q_Func(nn.Module):
    def __init__(self, n_actions, n_input_channels, n_added_input=0, img_width=48, img_height=27):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.n_added_input = n_added_input
        self.img_width = img_width
        self.img_height = img_height
        super(Dueling_Q_Func, self).__init__()
        self.pool = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv_num = 2
        self.mid_layer_num = 2
        self.cnv_act = F.relu
        self.ful_act = F.relu

        # convolution
        if self.conv_num == 1:
            channels=[16, 64] #各畳込みでのカーネル枚数
            kernels=[5, 5] #各畳込みでのカーネルサイズ
            pool_info=[2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)
        
        elif self.conv_num == 2:
            channels=[8, 16, 64, 128] #各畳込みでのカーネル枚数
            kernels=[5, 5, 5, 5] #各畳込みでのカーネルサイズ
            pool_info=[2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.conv2_1 = nn.Conv2d(channels[1], channels[2], kernels[2])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv2_2 = nn.Conv2d(channels[2], channels[3], kernels[3])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)

        elif self.conv_num == 3:
            channels=[8, 16, 32, 64, 128, 256] #各畳込みでのカーネル枚数
            kernels=[3, 3, 3, 3, 2, 2] #各畳込みでのカーネルサイズ
            pool_info=[2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.conv2_1 = nn.Conv2d(channels[1], channels[2], kernels[2])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv2_2 = nn.Conv2d(channels[2], channels[3], kernels[3])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.conv3_1 = nn.Conv2d(channels[3], channels[4], kernels[4])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv3_2 = nn.Conv2d(channels[4], channels[5], kernels[5])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)
        
        # Advantage
        if self.mid_layer_num == 1:
            units=[580] #各中間層のユニット数
            self.al1 = nn.Linear(self.img_input+self.n_added_input, units[0])
            nn.init.kaiming_normal_(self.al1.weight)

            self.al5 = nn.Linear(units[0], n_actions)
            nn.init.kaiming_normal_(self.al5.weight)
        
        elif self.mid_layer_num == 2:
            units=[512, 512] #各中間層のユニット数
            self.al1 = nn.Linear(self.img_input+self.n_added_input, units[0])
            nn.init.kaiming_normal_(self.al1.weight)

            self.al2 = nn.Linear(units[0], units[1])
            nn.init.kaiming_normal_(self.al2.weight)

            self.al5 = nn.Linear(units[1], n_actions)
            nn.init.kaiming_normal_(self.al5.weight)
        
        elif self.mid_layer_num == 3:
            units=[512, 512, 512] #各中間層のユニット数
            self.al1 = nn.Linear(self.img_input+self.n_added_input, units[0])
            nn.init.kaiming_normal_(self.al1.weight)

            self.al2 = nn.Linear(units[0], units[1])
            nn.init.kaiming_normal_(self.al2.weight)

            self.al3 = nn.Linear(units[1], units[2])
            nn.init.kaiming_normal_(self.al3.weight)

            self.al5 = nn.Linear(units[2], n_actions)
            nn.init.kaiming_normal_(self.al5.weight)
        
        # State Value
        if self.mid_layer_num == 1:
            self.vl1 = nn.Linear(self.img_input+self.n_added_input, units[0])
            nn.init.kaiming_normal_(self.vl1.weight)

            self.vl5 = nn.Linear(units[0], 1)
            nn.init.kaiming_normal_(self.vl5.weight)
        
        elif self.mid_layer_num == 2:
            self.vl1 = nn.Linear(self.img_input+self.n_added_input, units[0])
            nn.init.kaiming_normal_(self.vl1.weight)

            self.vl2 = nn.Linear(units[0], units[1])
            nn.init.kaiming_normal_(self.vl2.weight)

            self.vl5 = nn.Linear(units[1], 1)
            nn.init.kaiming_normal_(self.vl5.weight)
        
        elif self.mid_layer_num == 3:
            self.vl1 = nn.Linear(self.img_input+self.n_added_input, units[0])
            nn.init.kaiming_normal_(self.vl1.weight)

            self.vl2 = nn.Linear(units[0], units[1])
            nn.init.kaiming_normal_(self.vl2.weight)

            self.vl3 = nn.Linear(units[1], units[2])
            nn.init.kaiming_normal_(self.vl3.weight)

            self.vl5 = nn.Linear(units[2], 1)
            nn.init.kaiming_normal_(self.vl5.weight)

    def forward(self, state):
        if self.n_added_input:
            img = state[:,:-self.n_added_input]
            sen = state[:,-self.n_added_input:]
            # img = F.reshape(img,(-1,self.n_input_channels, self.img_width, self.img_height))
        else:
            img = state
        
        img = torch.reshape(img,(-1,self.n_input_channels, self.img_width, self.img_height))

        #convolution
        if self.conv_num == 1:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = h.view(-1,self.img_input)
        elif self.conv_num == 2:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv2_1(h))
            h = self.cnv_act(self.conv2_2(h))
            h = self.pool(h)
            h = h.view(-1,self.img_input)
        elif self.conv_num == 3:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv2_1(h))
            h = self.cnv_act(self.conv2_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv3_1(h))
            h = self.cnv_act(self.conv3_2(h))
            h = self.pool(h)
            h = h.view(-1,self.img_input)
        
        if self.n_added_input:
            h = torch.cat((h,sen), axis=1)

        #全結合層の構成
        if self.mid_layer_num == 1:
            ha = self.ful_act(self.al1(h))
            ya = self.al5(ha)
            hs = self.ful_act(self.vl1(h))
            ys = self.vl5(hs)
        elif self.mid_layer_num == 2:
            ha = self.ful_act(self.al1(h))
            ha = self.ful_act(self.al2(ha))
            ya = self.al5(ha)
            hs = self.ful_act(self.vl1(h))
            hs = self.ful_act(self.vl2(hs))
            ys = self.vl5(hs)
        elif self.mid_layer_num == 3:
            ha = self.ful_act(self.al1(h))
            ha = self.ful_act(self.al2(ha))
            ha = self.ful_act(self.al3(ha))
            ya = self.al5(ha)
            hs = self.ful_act(self.vl1(h))
            hs = self.ful_act(self.vl2(hs))
            hs = self.ful_act(self.vl3(hs))
            ys = self.vl5(hs)

        batch_size = img.shape[0]#=1
        mean = torch.reshape(torch.sum(ya, dim=1) / self.n_actions, (batch_size, 1))
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        ya, ys = torch.broadcast_tensors(ya, ys)
        q = pfrl.action_value.DiscreteActionValue(ya + ys)

        return q

class Dueling_Q_Func_Optuna(nn.Module):
    def __init__(self, 
    conv_num, mid_layer_num, mid_units1, mid_units2, mid_units3, cnv_act, ful_act, 
    n_actions, n_input_channels, n_added_input, img_width, img_height):
        self.conv_num = conv_num
        self.mid_layer_num = mid_layer_num
        self.cnv_act = cnv_act
        self.ful_act = ful_act
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.n_added_input = n_added_input
        self.img_width = img_width
        self.img_height = img_height
        super(Dueling_Q_Func_Optuna, self).__init__()
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)

        # convolution
        if self.conv_num == 1:
            channels = [16, 64] #各畳込みでのカーネル枚数
            kernels = [5, 5] #各畳込みでのカーネルサイズ
            pool_info = [2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)
        
        elif self.conv_num == 2:
            channels = [8, 16, 64, 128] #各畳込みでのカーネル枚数
            kernels = [5, 5, 5, 5] #各畳込みでのカーネルサイズ
            pool_info = [2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.conv2_1 = nn.Conv2d(channels[1], channels[2], kernels[2])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv2_2 = nn.Conv2d(channels[2], channels[3], kernels[3])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)

        elif self.conv_num == 3:
            channels = [8, 16, 32, 64, 128, 256] #各畳込みでのカーネル枚数
            kernels = [3, 3, 3, 3, 2, 2] #各畳込みでのカーネルサイズ
            pool_info = [2] #[何回おきの畳み込みでプーリングするか]
            self.conv1_1 = nn.Conv2d(n_input_channels, channels[0], kernels[0])
            nn.init.kaiming_normal_(self.conv1_1.weight)
            self.conv1_2 = nn.Conv2d(channels[0], channels[1], kernels[1])
            nn.init.kaiming_normal_(self.conv1_2.weight)
            self.conv2_1 = nn.Conv2d(channels[1], channels[2], kernels[2])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv2_2 = nn.Conv2d(channels[2], channels[3], kernels[3])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.conv3_1 = nn.Conv2d(channels[3], channels[4], kernels[4])
            nn.init.kaiming_normal_(self.conv2_1.weight)
            self.conv3_2 = nn.Conv2d(channels[4], channels[5], kernels[5])
            nn.init.kaiming_normal_(self.conv2_2.weight)
            self.img_input = calculate(img_width, img_height, channels, kernels, pool_info)

        # Advantage
        if self.mid_layer_num == 1:
            self.al1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.al5 = nn.Linear(mid_units1, n_actions)
            nn.init.kaiming_normal_(self.al5.weight)

        elif self.mid_layer_num == 2:
            self.al1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.al2 = nn.Linear(mid_units1, mid_units2)
            nn.init.kaiming_normal_(self.al2.weight)

            self.al5 = nn.Linear(mid_units2, n_actions)
            nn.init.kaiming_normal_(self.al5.weight)

        elif self.mid_layer_num == 3:
            self.al1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.al2 = nn.Linear(mid_units1, mid_units3)
            nn.init.kaiming_normal_(self.al2.weight)

            self.al3 = nn.Linear(mid_units3, mid_units2)
            nn.init.kaiming_normal_(self.al3.weight)

            self.al5 = nn.Linear(mid_units2, n_actions)
            nn.init.kaiming_normal_(self.al5.weight)

        # State Value
        if self.mid_layer_num == 1:
            self.vl1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.vl5 = nn.Linear(mid_units1, 1)
            nn.init.kaiming_normal_(self.al5.weight)

        elif self.mid_layer_num == 2:
            self.vl1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.vl2 = nn.Linear(mid_units1, mid_units2)
            nn.init.kaiming_normal_(self.al2.weight)

            self.vl5 = nn.Linear(mid_units2, 1)
            nn.init.kaiming_normal_(self.al5.weight)

        elif self.mid_layer_num == 3:
            self.vl1 = nn.Linear(self.img_input + n_added_input, mid_units1)
            nn.init.kaiming_normal_(self.al1.weight)

            self.vl2 = nn.Linear(mid_units1, mid_units3)
            nn.init.kaiming_normal_(self.al2.weight)

            self.vl3 = nn.Linear(mid_units3, mid_units2)
            nn.init.kaiming_normal_(self.al3.weight)

            self.vl5 = nn.Linear(mid_units2, 1)
            nn.init.kaiming_normal_(self.al5.weight)
    
    def forward(self, state):
        if self.n_added_input:
            img = state[:, :-self.n_added_input]
            sen = state[:, -self.n_added_input:]
        else:
            img = state
        
        img = torch.reshape(img, (-1, self.n_input_channels, self.img_width, self.img_height))

        #convolution
        if self.conv_num == 1:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = h.view(-1, self.img_input)
        elif self.conv_num == 2:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv2_1(h))
            h = self.cnv_act(self.conv2_2(h))
            h = self.pool(h)
            h = h.view(-1, self.img_input)
        elif self.conv_num == 3:
            h = self.cnv_act(self.conv1_1(img))
            h = self.cnv_act(self.conv1_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv2_1(h))
            h = self.cnv_act(self.conv2_2(h))
            h = self.pool(h)
            h = self.cnv_act(self.conv3_1(h))
            h = self.cnv_act(self.conv3_2(h))
            h = self.pool(h)
            h = h.view(-1, self.img_input)
        
        if self.n_added_input:
            h = torch.cat((h, sen), axis=1)

        #全結合層の構成
        if self.mid_layer_num == 1:
            ha = self.ful_act(self.al1(h))
            ya = self.al5(ha)
            hs = self.ful_act(self.vl1(h))
            ys = self.vl5(hs)
        elif self.mid_layer_num == 2:
            ha = self.ful_act(self.al1(h))
            ha = self.ful_act(self.al2(ha))
            ya = self.al5(ha)
            hs = self.ful_act(self.vl1(h))
            hs = self.ful_act(self.vl2(hs))
            ys = self.vl5(hs)
        elif self.mid_layer_num == 3:
            ha = self.ful_act(self.al1(h))
            ha = self.ful_act(self.al2(ha))
            ha = self.ful_act(self.al3(ha))
            ya = self.al5(ha)
            hs = self.ful_act(self.vl1(h))
            hs = self.ful_act(self.vl2(hs))
            hs = self.ful_act(self.vl3(hs))
            ys = self.vl5(hs)

        batch_size = img.shape[0]
        mean = torch.reshape(torch.sum(ya, dim=1) / self.n_actions, (batch_size, 1))
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        ya, ys = torch.broadcast_tensors(ya, ys)
        q = pfrl.action_value.DiscreteActionValue(ya + ys)

        return q

# 畳み込み・プーリングを終えた画像の1次元入力数の計算
def calculate(img_width, img_height, channels, kernels, pool_info):
    cnv_num = len(channels) # 畳み込み回数
    pool_interval = pool_info[0] # プーリングする間隔(何回の畳み込みごとか)
    for i in range(cnv_num):
        img_width = img_width - (kernels[i] - 1)
        img_height = img_height - (kernels[i] - 1)
        if (i + 1) % pool_interval == 0:
            img_width = math.ceil(img_width / 2)
            img_height = math.ceil(img_height / 2)
    img_input = img_width * img_height * channels[-1]
    # print(img_width, img_height, img_input)
    return img_input