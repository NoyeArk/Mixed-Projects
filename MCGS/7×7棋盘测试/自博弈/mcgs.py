import os
import sys
import copy
import time
import math
import queue
import threading
import numpy as np
import networkx as nx
from typing import Tuple, Union
from typing import Tuple, Iterable
from datetime import datetime

import matplotlib
matplotlib.use("agg")  # 选择一个非交互式的后端
import matplotlib.pyplot as plt

import torch
from node import Node
from board import Board
from hash import compute_zobrist_hash

UPDATE_EDGE = 0
UPDATE_NODE = 1
SEARCH_TIME = 0

EMPTY = -1
WHITE = 0
BLACK = 1
BOARD_LEN = 7

NODE = -2
TIME = 2

class AlphaZeroMCGS:
    """
    基于策略-价值网络的蒙特卡洛搜索树
    """

    def __init__(self, net, c_puct: float = 1, type=NODE, n_iters=100.0) -> None:
        self.use_mcgs = True
        self.c_puct = c_puct
        self.n_iters = n_iters # 如果是NODE，则为节点数，如果是TIME，则为搜索时间
        self.net = net.cuda()
        self.type = type

        self.board_len = 7

        self.G = nx.DiGraph()
        self.hash_table = dict()
        self.closure = []
        self.pre_action = -1

        '''以下为图搜索中用到的超参数'''
        self.v_min = -1
        self.v_max = 1
        self.q_epsilon = 0.01

        '''搜索记录变量'''
        self.tn_times_hash = 0 # 哈希查到TN节点的次数
        self.prune_node_num = 0 # 剪枝的节点总和
        self.prune_edge_num = 0 # 剪枝的边总和
        self.node_count = 0 #
        self.search_rate = 0 # 胜率 -暂时没用
        self.iter_num = 0 # 当前迭代次数

        """初始化操作"""
        self.root_node_id = 0  # 根节点为编号0
        self.hash_table[0] = 0
        self.G.add_node(0, pi=1, parent=0, N_s=0, Q_s=0, hash=0, chess_num=0)

    def print_closure(self):
        if len(self.closure) == True:
            print("列表为空")
        else:
            print('列表不为空')

    def get_action(self, board0: Board) -> Union[Tuple[int, np.ndarray], int]:
        self.iter_num = 0
        start_time = time.time()
        trajectory = []

        key, _ = compute_zobrist_hash(board0.board)
        if key in self.hash_table.keys() and self.use_mcgs:
            self.root_node_id = self.hash_table[key]

        # 判断是否更换根节点
        if self.iter_num != 0:
            assert self.root_node_id != pre_node_id

        # 初始化self.closure列表
        if self.root_node_id == 0:
            self.closure.append(0)

        if self.type == TIME:  # 控制搜索时间
            while time.time() - start_time <= self.n_iters:
                self.iter_num += 1
                trajectory, value, leaf_node_id = self.select_and_expand(self.root_node_id, board0)
                self.backup(trajectory, value, leaf_node_id)

        elif self.type == NODE:  # 控制搜索节点数
            while self.iter_num <= self.n_iters:
                self.iter_num += 1
                trajectory, value, leaf_node_id = self.select_and_expand(self.root_node_id, board0)
                self.backup(trajectory, value, leaf_node_id)

        visits = np.array([self.G.nodes[child]['N_s'] for child in self.G.successors(self.root_node_id)])
        pi_ = self.__getPi(visits)
        actions = [self.G.edges[(self.root_node_id, child)]['action'] for child in self.G.successors(self.root_node_id)]
        children = list(self.G.successors(self.root_node_id))
        action = int(np.random.choice(actions, p=pi_))

        # 删除无用节点
        pre_node_num = self.G.number_of_nodes()
        pre_edge_num = self.G.number_of_edges()
        self.delete_node_pruned(children[actions.index(action)])
        self.prune_node_num += pre_node_num - self.G.number_of_nodes()
        self.prune_edge_num += pre_edge_num - self.G.number_of_edges()

        self.pre_action = action
        return action

    def select_and_expand(self, root_node_id: int, board0: Board) -> Union[Tuple[int, np.ndarray], int]:
        board = board0.copy()
        current_node_id = root_node_id
        next_node_id = 0
        trajectory = [current_node_id]

        while self.G.out_degree(current_node_id) != 0: # 要么选到叶子节点 要么是TN结点
            next_node_id = self.selection(current_node_id)
            trajectory.append(next_node_id)
            edge = self.G.edges[(current_node_id, next_node_id)]
            board.do_action(edge['action'])

            if self.G.in_degree(next_node_id) > 1 and self.use_mcgs:  # 说明是一个TN结点
                """如果下一节点为对方，则Q_s为负，边值也为负"""
                V = self.G.nodes[next_node_id]['Q_s']
                q_delta = V - edge['Q_s_a']
                if abs(q_delta) > self.q_epsilon:
                    q_phi = edge['N_s_a'] * q_delta + V
                    q_0_phi = max(self.v_min, min(q_phi, self.v_max))
                    value = q_0_phi
                    return trajectory, value, next_node_id

            is_over, winner = board.is_game_over()
            if is_over and winner is not None:
                if winner != board.current_player:
                    value = -1
                else:
                    value = 1
                return trajectory, value, next_node_id

            current_node_id = next_node_id

        p, value = self.predict(board, self.pre_action)
        self.expansion(current_node_id, board.board, board.available_action, p, board.current_player)
        return trajectory, value, current_node_id

    def predict(self, board, action):
        # encoded_s = ed.encode_board(board)
        encoded_s = self._data_convert(board, action)
        # encoded_s = encoded_s.transpose(2, 0, 1)
        with torch.no_grad():
            p, value = self.net(encoded_s)
            p.cpu()
            value.cpu()
        return p, value

    def _data_convert(self, board, last_action):
        n = BOARD_LEN
        _board = torch.Tensor(board.board).unsqueeze(0)
        state0 = (_board > 0).float()
        state1 = (_board < 0).float()
        state2 = torch.zeros((1, n, n)).float()
        if board.current_player == WHITE:
            temp = state0.clone()
            state0.copy_(state1)
            state1.copy_(temp)
        if last_action != -1:
            x, y = last_action // n, last_action % n
            state2[0][x][y] = 1
        res = torch.cat((state0, state1, state2), dim=1)
        return res.cuda()

    def selection(self, parent_id):
        children_id = list(self.G.neighbors(parent_id))

        if len(children_id) == 0:
            return None
        else:
            best_node_id = max(children_id, key=lambda id: self.uct_value(id, parent_id))
            assert nx.has_path(self.G, parent_id, best_node_id)
            return best_node_id

    def expansion(self, parent_id, board0, action:list, p, player):
        for idx in range(len(action)):
            board0[int(action[idx] // self.board_len)][int(action[idx] % self.board_len)] = player
            _hash, chess_num = compute_zobrist_hash(board0)

            if _hash in self.hash_table.keys() and self.use_mcgs: # 该节点已经存在
                self.tn_times_hash += 1
                self.G.add_edge(parent_id, self.hash_table[_hash], action=action[idx], N_s_a=0, Q_s_a=0)
            else: # 创建新节点
                self.node_count += 1
                new_node_id = self.node_count
                # 创建新结点
                self.G.add_node(new_node_id, pi=p[idx], N_s=0, Q_s=0, hash=_hash, chess_num=chess_num)
                # 创建一条边
                self.G.add_edge(parent_id, new_node_id, action=action[idx], N_s_a=0, Q_s_a=0)
                # 将其添加到在线维护的哈希表中
                self.hash_table[_hash] = new_node_id

            board0[int(action[idx] // self.board_len)][int(action[idx] % self.board_len)] = -1  # -1代表EMPTY

    def backup(self, trajectory: list, value, leaf_node_id):
        _value = value
        target = float('nan')

        if self.G.in_degree(leaf_node_id) == 1: # 是新拓展的叶子节点
            self.G.nodes[leaf_node_id]['Q_s'] = value
            self.G.nodes[leaf_node_id]['N_s'] = 1
        elif self.G.nodes[leaf_node_id]['Q_s'] == 0: # 是叶子TN节点，进行值更新
            self.update_value(UPDATE_NODE, _value, leaf_node_id)

        if len(trajectory) == 1: # 为根节点
            return

        for i in range(len(trajectory) - 1, 0, -1): # 逆序遍历
            parent_node_id = trajectory[i-1]
            child_node_id = trajectory[i]
            edge = self.G.edges[(parent_node_id, child_node_id)]

            _value = -_value
            if math.isnan(target):
                self.update_value(UPDATE_NODE, _value, parent_node_id)
                self.update_value(UPDATE_EDGE, _value, parent_node_id, child_node_id)
            else:
                assert (target != None)
                q_delta = target - edge['Q_s_a']
                q_phi = edge['N_s_a'] * q_delta + target
                q_0_phi = max(self.v_min, min(q_phi, self.v_max))
                v = -q_0_phi
                self.update_value(UPDATE_NODE, v, parent_node_id)
                self.update_value(UPDATE_EDGE, v, parent_node_id, child_node_id)

            if self.G.in_degree(parent_node_id) > 1 and self.use_mcgs:
                target = self.G.nodes[parent_node_id]['Q_s']
            else:
                target = float('nan')

    def delete_node_pruned(self, real_node_id):
        for node_id in self.closure:
            if self.G.nodes[node_id]['chess_num'] <= self.G.nodes[real_node_id]['chess_num'] and node_id!= real_node_id:
                child_list = list(self.G.neighbors(node_id)) # 求出该节点的所有孩子节点
                for child_id in child_list:
                    if self.G.in_degree(child_id) > 1: # 还有其他节点指向它
                        if self.G.has_edge(real_node_id, child_id): # 从真正的父节点可以指向它
                            continue
                        else:
                            if child_id not in self.closure:
                                self.closure.append(child_id)
                    elif self.G.in_degree(child_id) == 1:
                        if child_id not in self.closure:
                            self.closure.append(child_id)

                    self.G.remove_edge(node_id, child_id)
                self.hash_table.pop(self.G.nodes[node_id]['hash']) # 从哈希表中删除
                self.G.remove_node(node_id) # 从图中删除
                self.closure.remove(node_id) # 从列表中删除


    def __getPi(self, visits, T=0.5) -> np.ndarray:
        """ 根据节点的访问次数计算 π """
        x = 1/T * np.log(visits + 1e-10)
        x = np.exp(x - x.max())
        pi = x/x.sum()
        return pi

    def reset_root(self):
        """ 重置根节点 """
        self.G.clear()
        self.hash_table.clear()
        self.G.add_node(0, pi=1, parent=0, N_s=0, Q_s=0, hash=0)
        self.root_node_id = 0
        self.node_count = 0
        self.tn_times_hash = 0

    def set_self_play(self, is_self_play: bool):
        """ 设置蒙特卡洛树的自我博弈状态 """
        self.is_self_play = is_self_play

    def set_searchTime(self, searchTime):
        self.n_iters = searchTime

    def uct_value(self, child_id, parent_id):
        c = 1
        ucb = c * self.G.nodes[child_id]['pi'] * math.sqrt(self.G.nodes[parent_id]['N_s'] + 1) / (self.G.nodes[child_id]['N_s'] + 1)  # c * sqrt(log(self.parent.visits + 1) / self.visits)
        return -self.G.nodes[child_id]['Q_s'] + ucb

    def draw(self):
        print('正在进行绘图')
        plt.figure(figsize=(12, 10))  # 设置图像尺寸为宽10，高8

        # 绘制图形
        pos = nx.spring_layout(self.G)  # 定义节点位置
        nx.draw(self.G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=8, font_color="green")
        nx.draw_networkx_edges(self.G, pos, width=1.0, alpha=0.5, edge_color="gray")

        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # 保存图像到指定路径
        save_folder = "D:\\Code\\game\\HexU2\\HexUI1\\graph"
        os.makedirs(save_folder, exist_ok=True)  # 创建文件夹（如果不存在）
        plt.savefig(os.path.join(save_folder, f"{current_time}.png"))

        print('绘图完成')


    def update_value(self, type, value, parent_node_id, child_node_id=None):
        if type == UPDATE_NODE:
            """更新结点值"""
            current_q = self.G.nodes[parent_node_id]['Q_s']
            current_n = self.G.nodes[parent_node_id]['N_s']
            # 更新结点Q值
            new_q = (current_n * current_q + value) / (current_n + 1)
            self.G.nodes[parent_node_id]['Q_s'] = new_q
            # 更新结点N值
            new_n = current_n + 1
            self.G.nodes[parent_node_id]['N_s'] = new_n

        elif type == UPDATE_EDGE:
            """更新边值"""
            current_q = self.G.edges[(parent_node_id, child_node_id)]['Q_s_a']
            current_n = self.G.edges[(parent_node_id, child_node_id)]['N_s_a']
            # 更新结点Q值
            new_q = (current_n * current_q - value) / (current_n + 1)
            self.G.edges[(parent_node_id, child_node_id)]['Q_s_a'] = new_q
            # 更新结点N值
            new_n = current_n + 1
            self.G.edges[(parent_node_id, child_node_id)]['N_s_a'] = new_n

    """以下为调试用的函数"""

    def iter_debug_info(self):
        print('当前迭代次数：', self.iter_num)
        print('当前根节点的访问次数：', self.G.nodes[self.root_node_id]['N_s'])
        if self.G.nodes[self.root_node_id]['N_s'] == 60:
            print('此时根节点访问次数已经达到了60')
            self.child_debug_info()
            input()

    def child_debug_info(self):
        children = list(self.G.neighbors(self.root_node_id))
        for child in children:
            ucb = self.G.nodes[child]['pi'] * math.sqrt(math.log(self.G.nodes[self.root_node_id]['N_s'] + 1) / (
                        self.G.nodes[child]['N_s'] + 1))
            print('id:', child, end='')
            print('   ucb:', ucb, end='')
            print('   q:', self.G.nodes[child]['Q_s'], end='')
            print('   n:', self.G.nodes[child]['N_s'], end='')
            print('   value:', -self.G.nodes[child]['Q_s'] + ucb)