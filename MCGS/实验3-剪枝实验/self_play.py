# 导入必要的库
import os
import sys
import torch
import pickle
import random
import threading
import numpy as np
import multiprocessing
import onnxruntime

from tqdm import tqdm
import time

from policy_value_net import neuralnetwork as net
from board import Board
from mcts import AlphaZeroMCTS
from mcgs import AlphaZeroMCGS
from logger import logger_config
from draw import draw_graph

logger = logger_config(log_path='outcome.log', logging_name='比赛结果记录')

NODE = -2
TIME = 2

SEARCH_TIME = 5   # 搜索时间
SEARCH_NODE = 200 # 搜索节点数

TREE_WIN = -1     # 树搜索胜
GRAPH_WIN = 1     # 图搜索胜

TREE_PLAYER = -2  # 树搜索玩家定义
GRAPH_PLAYER = 2  # 图搜索玩家定义

model = net(input_layers=3, board_size=8, learning_rate=0.1)
with open('model_5400.pkl', 'rb') as file:
    model = pickle.load(file)

step = 0

def print_game_info():
    global step
    print('当前线程：', threading.current_thread().ident, '   当前游戏步数：', step)
    step += 1

def init_game():
    global step
    step = 0

def simulate_game(head_start, game_round, thread_id):
    mcts = AlphaZeroMCTS(policy_value_net=model, c_puct=1, type=NODE, n_iters=SEARCH_NODE)
    mcgs = AlphaZeroMCGS(net=model, c_puct=1, type=NODE, n_iters=SEARCH_NODE, game_round=game_round, thread_id=thread_id)

    board = Board()  # 每次模拟一局游戏，都重新初始化一个board

    if head_start == TREE_PLAYER:
        while not board.is_game_over()[0]: # 没有结束
            board.do_action(mcts.get_action(board))
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

            board.do_action(mcts.get_action(board))
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
            head_start = GRAPH_PLAYER if loop % 2 == 1 else TREE_PLAYER
            if head_start == GRAPH_PLAYER:
                print('--------先手为图搜索--------')
            else:
                print('--------先手为树搜索--------')

            result = simulate_game(head_start, loop, thread_id)
            draw_graph(loop, thread_id) # 画图
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
                logger.info("tree wins:%s", total_tree_wins)
                logger.info("graph wins:%s", total_graph_wins)

    result_queue.put((tree_wins, graph_wins, tree_wins_step, graph_wins_step))


if __name__ == "__main__":

    num_games = 1
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

    print("tree wins:", total_tree_wins)
    print("graph wins:", total_graph_wins)
    print('tree wins/step:',total_tree_wins_step / total_tree_wins)
    print('graph wins/step:',total_graph_wins_step / total_graph_wins)

    logger.info("tree wins:%s", total_tree_wins)
    logger.info("graph wins:%s", total_graph_wins)
    logger.info("tree wins/step:", total_tree_wins_step / total_tree_wins)
    logger.info("graph wins/step:", total_graph_wins_step / total_graph_wins)

