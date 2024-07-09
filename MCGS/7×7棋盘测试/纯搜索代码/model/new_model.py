# -*- coding: utf-8 -*-
import sys
import os
import random

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np


def conv3x3(in_channels, out_channels, stride=1):
    # 3x3 convolution
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    # Residual block
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = False
        if in_channels != out_channels or stride != 1:
            self.downsample = True
            self.downsample_conv = conv3x3(in_channels, out_channels, stride=stride)
            self.downsample_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample_conv(residual)
            residual = self.downsample_bn(residual)

        out += residual
        out = self.relu(out)
        return out


class NeuralNetWork(nn.Module):
    """Policy and Value Network
    """

    def __init__(self, num_layers, num_channels, n, action_size):
        super(NeuralNetWork, self).__init__()

        # residual block
        res_list = [ResidualBlock(3, num_channels)] + [ResidualBlock(num_channels, num_channels) for _ in range(num_layers - 1)]
        self.res_layers = nn.Sequential(*res_list)

        # policy head
        self.p_conv = nn.Conv2d(num_channels, 4, kernel_size=1, padding=0, bias=False)
        self.p_bn = nn.BatchNorm2d(num_features=4)
        self.relu = nn.ReLU(inplace=True)

        self.p_fc = nn.Linear(4 * n ** 2, action_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

        # value head
        self.v_conv = nn.Conv2d(num_channels, 2, kernel_size=1, padding=0, bias=False)
        self.v_bn = nn.BatchNorm2d(num_features=2)

        self.v_fc1 = nn.Linear(2 * n ** 2, 256)
        self.v_fc2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        # residual block
        out = self.res_layers(inputs)

        # policy head
        p = self.p_conv(out)
        p = self.p_bn(p)
        p = self.relu(p)

        p = self.p_fc(p.view(p.size(0), -1))
        p = self.log_softmax(p)

        # value head
        v = self.v_conv(out)
        v = self.v_bn(v)
        v = self.relu(v)

        v = self.v_fc1(v.view(v.size(0), -1))
        v = self.relu(v)
        v = self.v_fc2(v)
        v = self.tanh(v)

        return p, v


class AlphaLoss(nn.Module):
    """
    Custom loss as defined in the paper :
    (z - v) ** 2 --> MSE Loss
    (-pi * logp) --> Cross Entropy Loss
    z : self_play_winner
    v : winner
    pi : self_play_probas
    p : probas

    The loss is then averaged over the entire batch
    """

    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, log_ps, vs, target_ps, target_vs):
        value_loss = torch.mean(torch.pow(vs - target_vs, 2)) # 使用均方误差
        policy_loss = -torch.mean(torch.sum(target_ps * log_ps, 1)) # 使用交叉熵损失

        return value_loss + policy_loss


class NeuralNetWorkWrapper():
    """train and predict
    """

    def __init__(self, lr, l2, num_layers, num_channels, n, action_size, train_use_gpu=True, libtorch_use_gpu=True):
        '''
        :param lr:学习率
        :param l2:L2正则化项
        :param num_layers:网络层数
        :param num_channels:每个卷积层的通道数
        :param n:神经网络输入数据的大小
        :param action_size:动作空间大小
        '''
        self.lr = lr # 学习率
        self.l2 = l2 # L2正则化
        self.num_channels = num_channels # 每个卷积层的通道数
        self.n = n # 神经网络的输入数据的大小

        self.libtorch_use_gpu = libtorch_use_gpu # 指示LibTorch（PyTorch的C++前端）是否使用GPU加速
        self.train_use_gpu = train_use_gpu # 指示是否使用GPU加速

        self.neural_network = NeuralNetWork(num_layers, num_channels, n, action_size)
        if self.train_use_gpu:
            self.neural_network.cuda()

        self.optim = AdamW(self.neural_network.parameters(), lr=self.lr, weight_decay=self.l2)
        self.alpha_loss = AlphaLoss() # 损失函数定义

    def train(self, example_buffer, batch_size, epochs):

        for epo in range(1, epochs + 1):
            self.neural_network.train() # 启用训练模式

            # 从example_buffer中随机取batch_size条数据
            train_data = random.sample(example_buffer, batch_size)

            # extract train data
            board_batch, last_action_batch, cur_player_batch, p_batch, v_batch = list(zip(*train_data))

            state_batch = self._data_convert(board_batch, last_action_batch, cur_player_batch)
            p_batch = torch.Tensor(p_batch).cuda() if self.train_use_gpu else torch.Tensor(p_batch)
            v_batch = torch.Tensor(v_batch).unsqueeze(1).cuda() if self.train_use_gpu else torch.Tensor(v_batch).unsqueeze(1)

            # 清空优化器的梯度
            self.optim.zero_grad()

            # 前向传播计算策略和值估计
            log_ps, vs = self.neural_network(state_batch)
            # 计算损失
            loss = self.alpha_loss(log_ps, vs, p_batch, v_batch)
            # 反向传播更新模型参数
            loss.backward()

            self.optim.step()

            # calculate entropy
            new_p, _ = self._infer(state_batch)

            entropy = -np.mean(
                np.sum(new_p * np.log(new_p + 1e-10), axis=1)
            )

            print("EPOCH: {}, LOSS: {}, ENTROPY: {}".format(epo, loss.item(), entropy))

    def infer(self, feature_batch):
        """predict p and v by raw input
           return numpy
        """
        board_batch, last_action_batch, cur_player_batch = list(zip(*feature_batch))
        states = self._data_convert(board_batch, last_action_batch, cur_player_batch)

        self.neural_network.eval()
        log_ps, vs = self.neural_network(states)

        return np.exp(log_ps.cpu().detach().numpy()), vs.cpu().detach().numpy()

    def _infer(self, state_batch):
        """predict p and v by state
           return numpy object
        """

        self.neural_network.eval()
        log_ps, vs = self.neural_network(state_batch)

        return np.exp(log_ps.cpu().detach().numpy()), vs.cpu().detach().numpy()

    def _data_convert(self, board_batch, last_action_batch, cur_player_batch):
        """
        :param board_batch:
        :param last_action_batch:
        :param cur_player_batch:
        return tensor
        """
        n = self.n

        board_batch = torch.Tensor(board_batch).unsqueeze(1)
        state0 = (board_batch > 0).float()
        state1 = (board_batch < 0).float()

        state2 = torch.zeros((len(last_action_batch), 1, n, n)).float()

        for i in range(len(board_batch)):
            if cur_player_batch[i] == -1:
                temp = state0[i].clone()
                state0[i].copy_(state1[i])
                state1[i].copy_(temp)

            last_action = last_action_batch[i]
            if last_action != -1:
                x, y = last_action // self.n, last_action % self.n
                state2[i][0][x][y] = 1

        res =  torch.cat((state0, state1, state2), dim=1)
        # res = torch.cat((state0, state1), dim=1)
        return res.cuda() if self.train_use_gpu else res

    def set_learning_rate(self, lr):
        """set learning rate
        """

        for param_group in self.optim.param_groups:
            param_group['lr'] = lr

    def load_model(self, folder="models", filename="checkpoint"):
        """load model from file
        """

        filepath = os.path.join(folder, filename)
        state = torch.load(filepath)
        self.neural_network.load_state_dict(state['network'])
        self.optim.load_state_dict(state['optim'])

    def save_model(self, folder="models", filename="checkpoint"):
        """save model to file
        """

        if not os.path.exists(folder):
            os.mkdir(folder)

        filepath = os.path.join(folder, filename)
        state = {'network':self.neural_network.state_dict(), 'optim':self.optim.state_dict()}
        torch.save(state, filepath)

        # save torchscript
        filepath += '.pt'
        self.neural_network.eval()

        if self.libtorch_use_gpu:
            self.neural_network.cuda()
            example = torch.rand(1, 3, self.n, self.n).cuda()
        else:
            self.neural_network.cpu()
            example = torch.rand(1, 3, self.n, self.n).cpu()

        traced_script_module = torch.jit.trace(self.neural_network, example)
        traced_script_module.save(filepath)

        if self.train_use_gpu:
            self.neural_network.cuda()
        else:
            self.neural_network.cpu()