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

from policy_value_net import PolicyValueNet
from board import Board

from search.minimax import Minimax
from search.mcgs import AlphaZeroMCGS
from logger import logger_config

logger = logger_config(log_path='outcome.log', logging_name='比赛结果记录')

NODE = -2
TIME = 2

SEARCH_TIME = 5   # 搜索时间
SEARCH_NODE = 500 # 搜索节点数

TREE_WIN = -1     # 树搜索胜
GRAPH_WIN = 1     # 图搜索胜

TREE_PLAYER = -2  # 树搜索玩家定义
GRAPH_PLAYER = 2  # 图搜索玩家定义

model = PolicyValueNet(11, 2, 20, 128, False)
model_file = os.path.join(os.path.abspath("."), 'model/20b128c_hex11.pth')
model_dict = torch.load(model_file, map_location='cpu').state_dict()
model.load_state_dict(model_dict)
model.eval()

step = 0

def print_game_info():
    global step
    print('当前线程：', threading.current_thread().ident, '   当前游戏步数：', step)
    step += 1

def init_game():
    global step
    step = 0

def simulate_game(head_start, game_round, thread_id):
    mcgs = AlphaZeroMCGS(net=model, c_puct=1, type=NODE, n_iters=SEARCH_NODE, game_round=game_round, thread_id=thread_id)
    minimax = Minimax(net=model, max_node=SEARCH_NODE)

    board = Board()  # 每次模拟一局游戏，都重新初始化一个board

    if head_start == TREE_PLAYER:
        while not board.is_game_over()[0]: # 没有结束
            board.do_action(minimax.get_action(board))
            print_game_info() # 打印步数信息

            if board.is_game_over()[0]: # 如果游戏结束
                return TREE_WIN

            board.do_action(mcgs.get_action(board))
            print_game_info()

        return GRAPH_WIN

    elif head_start == GRAPH_PLAYER:
        while not board.is_game_over()[0]:  # 游戏是否结束
            board.do_action(mcgs.get_action(board))
            print_game_info()

            if board.is_game_over()[0]:
                return GRAPH_WIN

            board.do_action(minimax.get_action(board))
            print_game_info()

        return TREE_WIN

def run_games(num_games_per_process, result_queue, thread_id):
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

            result = simulate_game(head_start, loop, thread_id)

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

            if loop % 100 == 0:
                logger.info("minimax wins:%s", tree_wins)
                logger.info("graph wins:%s", graph_wins)

    result_queue.put((tree_wins, graph_wins, tree_wins_step, graph_wins_step))


if __name__ == "__main__":

    num_games = 1200
    num_processes = 1  # 选择要使用的进程数量

    # 创建一个队列来收集每个进程的结果
    result_queue = multiprocessing.Queue()
    processes = []
    games_per_process = num_games // num_processes
    for idx in range(num_processes):
        process = multiprocessing.Process(target=run_games, args=(games_per_process, result_queue, idx))
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

    print("minimax wins:", total_tree_wins)
    print("graph wins:", total_graph_wins)
    print('minimax wins/step:',total_tree_wins_step / total_tree_wins)
    print('graph wins/step:',total_graph_wins_step / total_graph_wins)

    logger.info("minimax wins:%s", total_tree_wins)
    logger.info("graph wins:%s", total_graph_wins)
    logger.info("minimax wins/step:", total_tree_wins_step / total_tree_wins)
    logger.info("graph wins/step:", total_graph_wins_step / total_graph_wins)

