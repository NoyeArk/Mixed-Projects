#!/usr/bin/env python

from model import ConnectNet, AlphaLoss, board_data
import os
import random
import pickle
import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import matplotlib
matplotlib.use('TkAgg')  # 切换到TkAgg后端，如果有其他可用的后端也可以尝试
import matplotlib.pyplot as plt

import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def save_as_pickle(filename, data):
    completeName = os.path.join("./model_data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)
        
def load_pickle(filename):
    completeName = os.path.join("./model_data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def load_state(net, optimizer, scheduler, args, iteration, new_optim_state=True):
    """ Loads saved model and optimizer states if exists """
    base_path = "./model_data/"
    checkpoint_path = os.path.join(base_path, "%s_iter%d.pth.tar" % (args.neural_net_name, iteration))
    start_epoch, checkpoint = 0, None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
    if checkpoint != None:
        if (len(checkpoint) == 1) or (new_optim_state == True):
            net.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded checkpoint model %s." % checkpoint_path)
        else:
            start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("Loaded checkpoint model %s, and optimizer, scheduler." % checkpoint_path)    
    return start_epoch

def load_results(iteration):
    """ Loads saved results if exists """
    losses_path = "./model_data/losses_per_epoch_iter%d.pkl" % iteration
    if os.path.isfile(losses_path):
        losses_per_epoch = load_pickle("losses_per_epoch_iter%d.pkl" % iteration)
        logger.info("Loaded results buffer")
    else:
        losses_per_epoch = []
    return losses_per_epoch

def train(net, dataset, optimizer, scheduler, start_epoch, cpu, args, iteration):
    torch.manual_seed(cpu)
    cuda = torch.cuda.is_available()
    net.train() # 训练模式
    criterion = AlphaLoss()
    
    train_set = board_data(dataset)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
    losses_per_epoch = load_results(iteration + 1) # 加载之前保存的结果
    
    logger.info("开启训练进程...")
    update_size = len(train_loader)//10

    update_size = 1
    print("Update step size: %d" % update_size)

    for epoch in range(start_epoch, args.num_epochs):
        total_loss = 0.0
        losses_per_batch = []

        # 采样数据
        train_data = random.sample(dataset, args.batch_size)
        # 打包训练数据
        state_batch, policy_batch, v_batch = list(zip(*train_data))
        # 转换数据类型
        print(len(policy_batch))
        policy_batch = torch.Tensor(policy_batch).cuda()
        v_batch = torch.Tensor(v_batch).unsqueeze(1).cuda()

        if cuda:
            state, policy, value = state.cuda(), policy.cuda(), value.cuda()

        policy_predict, value_predict = net(state) # policy_predict = torch.Size([batch, 4672]) value_predict = torch.Size([batch, 1])
        loss = criterion(value_predict[:,0], value, policy_predict, policy) # 计算该条数据的损失
        loss = loss / args.gradient_acc_steps
        loss.backward() # 执行反向传播，将梯度存储在模型的参数张量中，以便优化器来使用它们更新参数
        clip_grad_norm_(net.parameters(), args.max_norm) # 梯度裁剪：限制梯度大小，防止梯度爆炸

        if (epoch % args.gradient_acc_steps) == 0: #
            optimizer.step() # 根据梯度和优化算法来更新模型的参数，以最小化损失函数
            optimizer.zero_grad() # 清空梯度，准备下一轮的梯度计算

        total_loss += loss.item() # item()返回损失值得标量值
        if i % update_size == (update_size - 1):    # print every update_size-d mini-batches of size = batch_size
            losses_per_batch.append(args.gradient_acc_steps * total_loss / update_size)
            print('[迭代 %d] 线程 ID: %d [Epoch: %d, %5d/ %d points] total loss per batch: %.3f' %
                  (iteration, os.getpid(), epoch + 1, (i + 1)*args.batch_size, len(train_set), losses_per_batch[-1]))
            print("Policy (actual, predicted):",policy[0].argmax().item(),policy_predict[0].argmax().item())
            print("Policy 标签:", policy[0])
            print("Policy 预测:", policy_predict[0])
            print("Value (actual, predicted):", value[0].item(), value_predict[0,0].item())
            #print("Conv grad: %.7f" % net.conv.conv1.weight.grad.mean().item())
            #print("Res18 grad %.7f:" % net.res_18.conv1.weight.grad.mean().item())
            print(" ")
            total_loss = 0.0 # 重新设置为0，以准备下一组批次的损失值计算
        
        scheduler.step() # 自动调整学习率
        if len(losses_per_batch) >= 1:
            losses_per_epoch.append(sum(losses_per_batch)/len(losses_per_batch))
        if (epoch % 2) == 0: # 保存模型
            save_as_pickle("losses_per_epoch_iter%d.pkl" % (iteration + 1), losses_per_epoch)
            torch.save({
                    'epoch': epoch + 1,\
                    'state_dict': net.state_dict(),\
                    'optimizer' : optimizer.state_dict(),\
                    'scheduler' : scheduler.state_dict(),\
                }, os.path.join("./model_data/",\
                    "%s_iter%d.pth.tar" % (args.neural_net_name, (iteration + 1))))

        '''
        # Early stopping
        if len(losses_per_epoch) > 50:
            if abs(sum(losses_per_epoch[-4:-1])/3-sum(losses_per_epoch[-16:-13])/3) <= 0.00017:
                break
        '''

    logger.info("训练完成!")
    # 下面是开始绘制训练相关曲线
    fig = plt.figure()
    ax = fig.add_subplot(222)
    # ax.scatter([e for e in range(start_epoch, (len(losses_per_epoch) + start_epoch))], losses_per_epoch)
    # 绘制折线图
    ax.plot([e for e in range(start_epoch, (len(losses_per_epoch) + start_epoch))], losses_per_epoch, marker='o')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per batch")
    ax.set_title("Loss vs Epoch")
    plt.savefig(os.path.join("./model_data/", "Loss_vs_Epoch_iter%d_%s.png" % ((iteration + 1), datetime.datetime.today().strftime("%Y-%m-%d"))))
    plt.show()
    
def train_net(args, iteration, new_optim_state):
    # 收集数据
    logger.info("加载训练数据中...")
    data_path="./datasets/iter_%d/" % iteration
    datasets = []
    for idx,file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path,file)
        with open(filename, 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes'))

    # debug_error(datasets)
    # print(datasets.shape)
    # datasets = datasets[0]
    # datasets = np.array(datasets)
    logger.info("成功加载数据从 %s." % data_path)
    
    # 训练网络
    net = ConnectNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.8, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150,200,250,300,400], gamma=0.77)
    start_epoch = load_state(net, optimizer, scheduler, args, iteration, new_optim_state)
    
    train(net, datasets, optimizer, scheduler, start_epoch, 0, args, iteration)


def _data_convert(self, board_batch):
    # 创建一个和board_batch大小相同的tensor
    board_batch = torch.Tensor(board_batch).unsqueeze(1)

    for i in range(len(board_batch)):
        if board_batch[i][3] == -1:
            temp = state0[i].clone()
            state0[i].copy_(state1[i])
            state1[i].copy_(temp)

        last_action = last_action_batch[i]
        if last_action != -1:
            x, y = last_action // self.n, last_action % self.n
            state2[i][0][x][y] = 1

    res = torch.cat((state0, state1, state2), dim=1)
    return res.cuda()