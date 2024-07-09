#!/usr/bin/env python
import torch
import threading
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib
matplotlib.use("Agg")

import numpy as np

BOARD_LEN = 7

class board_data(Dataset):
    def __init__(self, dataset): # dataset = np.array of (s, p, v)
        self.X = [item[0] for item in dataset]
        self.y_p = [item[1] for item in dataset]
        self.y_v = [item[2] for item in dataset]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx=0):
        # return np.int64(self.X[idx].transpose(2,0,1)), self.y_p[idx], self.y_v[idx]
        # return np.int64(self.X.transpose(2,0,1)), self.y_p, self.y_v
        return np.int64(self.X), self.y_p, self.y_v

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

    def forward(self, s):
        s = s.view(-1, 3, BOARD_LEN, BOARD_LEN)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s


class ResBlock(nn.Module):
    def __init__(self, inplanes=128, planes=128, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(128, 4, kernel_size=1) # value head
        self.bn = nn.BatchNorm2d(4)
        self.fc1 = nn.Linear(4*BOARD_LEN*BOARD_LEN, 32)
        self.fc2 = nn.Linear(32, 1)

        self.conv1 = nn.Conv2d(128, 32, kernel_size=1) # policy head
        self.bn1 = nn.BatchNorm2d(32)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(BOARD_LEN*BOARD_LEN*32, BOARD_LEN*BOARD_LEN)

    def forward(self,s):

        v = F.relu(self.bn(self.conv(s))) # value head
        v = v.view(-1, 4*BOARD_LEN*BOARD_LEN)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s))) # policy head
        p = p.view(-1, BOARD_LEN*BOARD_LEN*32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return p[0], v


class ConnectNet(nn.Module):
    def __init__(self):
        super(ConnectNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(6):
            setattr(self, "res_%i" % block,ResBlock())
        self.outblock = OutBlock()

    def forward(self,s):
        s = self.conv(s)
        for block in range(6):
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        torch.cuda.empty_cache()
        return s

class AlphaLoss(torch.nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    # def forward(self, y_value, value, y_policy, policy):
    #     value_error = (value - y_value) ** 2
    #     policy_error = torch.sum((-policy*
    #                             (1e-8 + y_policy.float()).float().log()), 1)
    #     total_error = (value_error.view(-1).float() + policy_error).mean()
    #     return total_error
    def forward(self, log_ps, vs, target_ps, target_vs):


        value_loss = torch.mean(torch.pow(vs - target_vs, 2))
        policy_loss = -torch.mean(torch.sum(target_ps * log_ps, 1))

        return value_loss + policy_loss