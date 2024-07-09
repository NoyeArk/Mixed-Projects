#!/usr/bin/env python
import pickle
import os
import collections
import numpy as np
import math
import encoder_decoder_c4 as ed
import copy
import torch
import torch.multiprocessing as mp
import datetime
import logging
from tqdm import tqdm

from model import ConnectNet
from board import Board
from mcgs_search import AlphaZeroMCGS

WHITE = 0
BLACK = 1

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

def save_as_pickle(filename, data):
    completeName = os.path.join("./datasets/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)

def load_pickle(filename):
    completeName = os.path.join("./datasets/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

def UCT_search(game_state, num_reads,net,temp):
    root = UCTNode(game_state, move=None, parent=DummyNode())
    for i in range(num_reads):
        leaf = root.select_leaf()
        encoded_s = ed.encode_board(leaf.game)
        encoded_s = encoded_s.transpose(2,0,1)
        encoded_s = torch.from_numpy(encoded_s).float().cuda()
        child_priors, value_estimate = net(encoded_s)
        child_priors = child_priors.detach().cpu().numpy().reshape(-1); value_estimate = value_estimate.item()
        if leaf.game.check_winner() == True or leaf.game.actions() == []: # if somebody won or draw
            leaf.backup(value_estimate); continue
        leaf.expand(child_priors) # need to make sure valid moves
        leaf.backup(value_estimate)
    return root

def do_decode_n_move_pieces(board,move):
    board.drop_piece(move)
    return board

def get_policy(root, temp=1):
    #policy = np.zeros([7], dtype=np.float32)
    #for idx in np.where(root.child_number_visits!=0)[0]:
    #    policy[idx] = ((root.child_number_visits[idx])**(1/temp))/sum(root.child_number_visits**(1/temp))
    return ((root.child_number_visits)**(1/temp))/sum(root.child_number_visits**(1/temp))

def mcgs_self_play(net, num_games, start_idx, cpu, args, iteration):
    logger.info("[CPU: %d]: 开始 MCGS 自博弈..." % cpu)

    if not os.path.isdir("./datasets/iter_%d" % iteration):
        if not os.path.isdir("datasets"):
            os.mkdir("datasets")
        os.mkdir("datasets/iter_%d" % iteration)

    for idx in tqdm(range(start_idx, num_games + start_idx)):
        mcgs = AlphaZeroMCGS(net=net, c_puct=1, n_iters=args.search_node, board_len=7)

        logger.info("[CPU: %d]: Game %d" % (cpu, idx))
        current_board = Board()
        checkmate = False # 用于跟踪游戏是否结束
        dataset = [] # 用于存储游戏中的状态、策略和价值等信息，以便用于神经网络的训练数据。
        value = 0 # 用于记录价值标签
        move_count = 0 # 用于记录游戏中的移动次数

        while checkmate == False and current_board.available_action != []:
            if move_count < 11:
                t = args.temperature_MCTS # 默认是1.1
                mcgs.set_temp(t)
            else:
                t = 0.1
                mcgs.set_temp(t)

            # 将当前游戏状态编码为board_state，为了后续处理
            # board_state = copy.deepcopy(ed.encode_board(current_board))
            board_state = current_board
            action, policy = mcgs.get_action(current_board)

            print("[CPU: %d]: Game %d POLICY:\n " % (cpu, idx), policy)
            current_board.do_action(action) # 更新棋盘

            dataset.append([current_board.board, action, current_board.current_player, policy])
            # dataset.append([board_state, policy])
            print("[Iteration: %d CPU: %d]: Game %d 当前棋盘:\n" % (iteration, cpu, idx), current_board.board); print(" ")

            if current_board.is_game_over()[0] == True: # 如果游戏结束
                if current_board.current_player == WHITE: # 当前玩家是白方，说明上一次移动是黑方，即黑方获胜
                    value = 1
                elif current_board.current_player == 1: # 当前是黑方，说明上一次移动是白方，即白方获胜
                    value = -1
                checkmate = True # 标志游戏结束

            move_count += 1
            # checkmate = True # 测试用

        dataset_p = [] # 用于存储处理后的数据，包括状态、策略和价值
        for idx,data in enumerate(dataset):
            state, action, player, policy = data
            """默认先手是黑方，idx是偶数则为黑方，是奇数则为白方"""
            if idx == 0:
                dataset_p.append([state, action, player, policy, 0])
            elif idx % 2 == 0: # 如果黑方获胜，则value为1
                dataset_p.append([state, action, player, policy, value])
            elif idx % 2 == 1: # 如果黑方获胜，则value标记为-1
                dataset_p.append([state, action, player, policy, -value])
        print()
        del dataset

        save_as_pickle("iter_%d/" % iteration +\
                       "dataset_iter%d_cpu%i_%i_%s" % (iteration, cpu, idx, datetime.datetime.today().strftime("%Y-%m-%d")), dataset_p)

def run_mcgs(args, start_idx=0, iteration=0):
    net_to_play="%s_iter%d.pth.tar" % (args.neural_net_name, iteration) # cc4_current_net__iter0.pth.tar
    path = "./model_data/"

    net = ConnectNet() # 构建一个神经网络模型实例
    if torch.cuda.is_available():
        net.cuda()
    
    if args.MCGS_num_processes > 1: # 如果设置线程数大于1
        logger.info("多进程蒙特卡洛图自博弈准备模型...")

        mp.set_start_method("spawn",force=True) # 与主进程独立的子进程
        net.share_memory() # 模型参数共享给多个进程
        net.eval() # 评估模式
    
        current_net_filename = os.path.join(path, net_to_play)
        if os.path.isfile(current_net_filename): # 检查文件是否存在
            checkpoint = torch.load(current_net_filename)
            net.load_state_dict(checkpoint['state_dict'])
            logger.info("正在加载 %s 模型." % current_net_filename)
        else: # 创建一个新的文件保存模型
            torch.save({'state_dict': net.state_dict()}, os.path.join(path,\
                        net_to_play))
            logger.info("初始化模型.")
        
        processes = []
        if args.MCGS_num_processes > mp.cpu_count(): # 用于获取计算机上可用的CPU核心数量
            num_processes = mp.cpu_count()
            logger.info("所需的进程数量超过了可用的CPU核心数量！ 将 MCGS_num_processes 设置为 %d" % num_processes)
        else:
            num_processes = args.MCGS_num_processes
        
        logger.info("正在启动 %d 进程..." % num_processes)
        with torch.no_grad():
            for i in range(num_processes): # 开启线程
                p = mp.Process(target=mcgs_self_play, args=(net, args.num_games_per_MCGS_process, start_idx, i, args, iteration))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
        logger.info("完成多线程 MCGS 自博弈!")
    
    elif args.MCGS_num_processes == 1: # 如果设置线程数为1
        logger.info("正在准备模型为 MCGS...")
        net.eval()
        
        current_net_filename = os.path.join(path, net_to_play)
        if os.path.isfile(current_net_filename):
            checkpoint = torch.load(current_net_filename)
            net.load_state_dict(checkpoint['state_dict'])
            logger.info("Loaded %s model." % current_net_filename)
        else:
            torch.save({'state_dict': net.state_dict()}, os.path.join(path,\
                        net_to_play))
            logger.info("初始化模型.")
        
        with torch.no_grad():
            mcgs_self_play(net, args.num_games_per_MCGS_process, start_idx, 0, args, iteration)
        logger.info("完成 MCGS!自博弈")
