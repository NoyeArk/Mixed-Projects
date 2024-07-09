# 导入必要的库
import os
import sys
import torch
import random
import threading
import numpy as np
import multiprocessing

from tqdm import tqdm
import time

from model import ConnectNet
from board import Board
from mcts import AlphaZeroMCTS
from mcgs import AlphaZeroMCGS
from logger import logger_config

logger = logger_config(log_path='outcome.log', logging_name='比赛结果记录')

NODE = -2
TIME = 2

SEARCH_TIME = 5   # 搜索时间
SEARCH_NODE = 200 # 搜索节点数

TREE_WIN = -1     # 树搜索胜
GRAPH_WIN = 1     # 图搜索胜

TREE_PLAYER = -2  # 树搜索玩家定义
GRAPH_PLAYER = 2  # 图搜索玩家定义

model = ConnectNet()
model_file = os.path.join(os.path.abspath("."), 'model/hex_net__iter2.pth.tar')
checkpoint = torch.load(model_file)
state_dict = checkpoint['state_dict']

# 将权重移到GPU
for key in state_dict:
    state_dict[key] = state_dict[key].to('cuda')

# 加载权重到模型
model.load_state_dict(state_dict)
model.eval()

step = 0

def print_game_info(board):
    global step
    print('当前线程：', threading.current_thread().ident, '   当前游戏步数：', step)
    print(board.board)
    step += 1

def init_game():
    global step
    step = 0

def simulate_game(head_start):
    mcts = AlphaZeroMCTS(policy_value_net=model, c_puct=1, type=NODE, n_iters=SEARCH_NODE)
    mcgs = AlphaZeroMCGS(net=model, c_puct=1, type=NODE, n_iters=SEARCH_NODE)

    board = Board()  # 每次模拟一局游戏，都重新初始化一个board

    if head_start == TREE_PLAYER:
        while not board.is_game_over()[0]: # 没有结束
            board.do_action(mcts.get_action(board))
            print_game_info(board) # 打印步数信息

            if board.is_game_over()[0]: # 如果游戏结束
                return TREE_WIN

            board.do_action(mcgs.get_action(board))
            print_game_info(board)

        return GRAPH_WIN

    elif head_start == GRAPH_PLAYER:
        while not board.is_game_over()[0]:  # 游戏是否结束
            board.do_action(mcgs.get_action(board))
            print_game_info(board)

            if board.is_game_over()[0]:
                return GRAPH_WIN

            board.do_action(mcts.get_action(board))
            print_game_info(board)

        return TREE_WIN

def run_games(num_games_per_process, result_queue):
    tree_wins = 0
    graph_wins = 0
    tree_wins_step = 0
    graph_wins_step = 0

    with tqdm(total=num_games_per_process, unit="game") as pbar:
        for loop in range(num_games_per_process):
            init_game()
            # 从0开始，所以先是图搜索，再是树搜索
            head_start = GRAPH_PLAYER if loop % 2 == 0 else TREE_PLAYER
            if head_start == GRAPH_PLAYER:
                print('--------先手为图搜索--------')
            else:
                print('--------先手为树搜索--------')

            result = simulate_game(head_start)
            global step
            if result == TREE_WIN:
                print('#########树搜索胜#########')
                tree_wins += 1
                tree_wins_step += step
            elif result == GRAPH_WIN:
                print('#########图搜索胜#########')
                graph_wins += 1
                graph_wins_step += step
            pbar.update(1)

    result_queue.put((tree_wins, graph_wins, tree_wins_step, graph_wins_step))


if __name__ == "__main__":

    num_games = 2
    num_processes = 1  # 选择要使用的进程数量

    # 创建一个队列来收集每个进程的结果
    result_queue = multiprocessing.Queue()
    processes = []
    games_per_process = num_games // num_processes
    for _ in range(num_processes):
        process = multiprocessing.Process(target=run_games, args=(games_per_process, result_queue))
        processes.append(process)
        process.start()

    # 等待所有进程完成
    for process in processes:
        process.join()

    total_tree_wins = 0
    total_graph_wins = 0
    total_tree_wins_step = 0
    total_graph_wins_step = 0

    for _ in range(num_processes):
        tree_wins, graph_wins, tree_wins_step, graph_wins_step = result_queue.get()
        total_tree_wins += tree_wins
        total_graph_wins += graph_wins
        total_tree_wins_step += tree_wins_step
        total_graph_wins_step += graph_wins_step

    print("tree wins:", total_tree_wins)
    print("graph wins:", total_graph_wins)
    print('tree wins/step:',total_tree_wins_step / total_tree_wins)
    print('graph wins/step:',total_graph_wins_step / total_graph_wins)

    logger.info("tree wins:%s", total_tree_wins)
    logger.info("graph wins:%s", total_graph_wins)
    logger.info("tree wins/step:", total_tree_wins_step / total_tree_wins)
    logger.info("graph wins/step:", total_graph_wins_step / total_graph_wins)

