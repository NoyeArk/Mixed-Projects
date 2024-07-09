import os
import sys
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
import torch.nn.functional as F
from torch.autograd import Variable

from model import ConnectNet, AlphaLoss

import logging
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)


class NeuralNetWorkWrapper():
    """train and predict
    """

    def __init__(self, args):
        """ init
        """
        self.lr = args.lr
        # self.l2 = l2
        self.batch_size = args.batch_size
        self.epochs = args.num_epochs
        self.n = 7

        self.libtorch_use_gpu = False
        self.train_use_gpu = True

        self.neural_net_name = args.neural_net_name
        self.neural_network = ConnectNet()

        if self.train_use_gpu:
            self.neural_network.cuda()

        self.optim = AdamW(self.neural_network.parameters(), lr=self.lr, betas=(0.8, 0.999))
        self.alpha_loss = AlphaLoss()

    def train_net(self, iteration):
        # 收集数据
        logger.info("加载训练数据中...")
        data_path = "./datasets/iter_%d/" % iteration
        datasets = []
        for idx, file in enumerate(os.listdir(data_path)):
            filename = os.path.join(data_path, file)
            with open(filename, 'rb') as fo:
                datasets.extend(pickle.load(fo, encoding='bytes'))

        logger.info("成功加载数据从 %s." % data_path)

        cuda = torch.cuda.is_available()
        if cuda:
            self.neural_network.cuda()

        scheduler = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[50, 100, 150, 200, 250, 300, 400], gamma=0.77)

        self.train(datasets, scheduler, iteration)

    def train(self, example_buffer, scheduler, iteration):
        losses_per_epoch = self.load_results(iteration + 1) # 加载之前保存的结果
        for epoch in range(1, self.epochs + 1):
            self.neural_network.train()

            # sample
            train_data = random.sample(example_buffer, self.batch_size)

            # extract train data
            board_batch, last_action_batch, cur_player_batch, p_batch, v_batch = list(zip(*train_data))

            state_batch = self._data_convert(board_batch, last_action_batch, cur_player_batch)
            p_batch = torch.Tensor(p_batch).cuda() if self.train_use_gpu else torch.Tensor(p_batch)
            v_batch = torch.Tensor(v_batch).unsqueeze(
                1).cuda() if self.train_use_gpu else torch.Tensor(v_batch).unsqueeze(1)

            # zero the parameter gradients
            self.optim.zero_grad()

            # forward + backward + optimize
            log_ps, vs = self.neural_network(state_batch)
            loss = self.alpha_loss(log_ps, vs, p_batch, v_batch)
            loss.backward()

            self.optim.step() # 优化参数
            self.optim.zero_grad() # 清空梯度

            # calculate entropy
            new_p, _ = self._infer(state_batch)

            entropy = -np.mean(
                np.sum(new_p * np.log(new_p + 1e-10), axis=0)
            )

            print("EPOCH: {}, LOSS: {}, ENTROPY: {}".format(epoch, loss.item(), entropy))

            if (epoch % 2) == 0:  # 保存模型
                self.save_as_pickle("losses_per_epoch_iter%d.pkl" % (iteration + 1), losses_per_epoch)
                torch.save({
                    'epoch': epoch + 1, \
                    'state_dict': self.neural_network.state_dict(), \
                    'optimizer': self.optim.state_dict(), \
                    'scheduler': scheduler.state_dict(), \
                    }, os.path.join("./model_data/", \
                                    "%s_iter%d.pth.tar" % (self.neural_net_name, (iteration + 1))))

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
        """convert data format
           return tensor
        """
        n = self.n
        '''state0存储当前玩家的棋子位置，state1存储对方玩家的棋子位置，state2存储当前动作的位置'''
        board_batch = torch.Tensor(board_batch).unsqueeze(1)
        state0 = (board_batch > 0).float()
        state1 = (board_batch < 0).float()

        state2 = torch.zeros((len(last_action_batch), 1, n, n)).float()

        for i in range(len(board_batch)):
            if cur_player_batch[i] == 0:
                temp = state0[i].clone()
                state0[i].copy_(state1[i])
                state1[i].copy_(temp)

            last_action = last_action_batch[i]
            if last_action != -1:
                # 得到x，y坐标
                x, y = last_action // self.n, last_action % self.n
                state2[i][0][x][y] = 1

        res =  torch.cat((state0, state1, state2), dim=1)
        # res = torch.cat((state0, state1), dim=1)
        return res.cuda() if self.train_use_gpu else res

    def _data_convert1(self, board_batch):
        """convert data format
           return tensor
        """
        n = self.n
        '''state0存储当前玩家的棋子位置，state1存储对方玩家的棋子位置，state2存储当前动作的位置'''
        board_batch = torch.Tensor((len(board_batch), n, n, 1))
        state_1 = (board_batch < 0).float()
        state0 = (board_batch == 0).float()
        state1 = (board_batch > 0).float()

        state2 = torch.zeros((len(board_batch), 1, n, n)).float()

        for i in range(len(board_batch)):
            if board_batch[i][0][0][3] == 1:
                state2[i][0] = torch.ones((n ,n)).float()

        print(state_1.shape, state0.shape, state1.shape, state2.shape)

        res =  torch.cat((state_1, state0, state1, state2), dim=1)
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

    def save_as_pickle(self, filename, data):
        completeName = os.path.join("./model_data/", \
                                    filename)
        with open(completeName, 'wb') as output:
            pickle.dump(data, output)

    def load_pickle(self, filename):
        completeName = os.path.join("./model_data/", \
                                    filename)
        with open(completeName, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        return data

    def load_results(self, iteration):
        """ Loads saved results if exists """
        losses_path = "./model_data/losses_per_epoch_iter%d.pkl" % iteration
        if os.path.isfile(losses_path):
            losses_per_epoch = self.load_pickle("losses_per_epoch_iter%d.pkl" % iteration)
            logger.info("Loaded results buffer")
        else:
            losses_per_epoch = []
        return losses_per_epoch

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