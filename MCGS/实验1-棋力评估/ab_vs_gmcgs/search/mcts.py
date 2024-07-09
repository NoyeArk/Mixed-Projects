from typing import Tuple, Union
import numpy as np
import time
import math
from node import Node
from board import Board

NODE = -2
TIME = 2

class AlphaZeroMCTS:

    def __init__(self, policy_value_net, c_puct: float = 1, type=NODE, n_iters=100.0, is_self_play=False) -> None:

        self.type = type

        self.c_puct = c_puct
        self.n_iters = n_iters
        self.is_self_play = is_self_play
        self.policy_value_net = policy_value_net
        self.root = Node(prior_prob=1, parent=None)
        # 没别的用处
        self.search_count = 0
        self.search_rate = 0

        self.node_count = 0

    def get_action(self, board0: Board) -> Union[Tuple[int, np.ndarray], int]:
        """
        根据当前局面返回下一步动作。
        :param board: Board,棋盘。
        :return:
        """
        start_time = time.time()
        count = 0
        # while time.time() - start_time <= self.n_iters:
        while self.root.N <= self.n_iters:
        # while count <= 500:

            # 拷贝棋盘
            board = board0.copy()

            # 如果没有遇到叶节点，就一直向下搜索并更新棋盘
            node = self.root
            #d=0
            while not node.is_leaf_node():
                #d=d+1
                action, node = node.select()
                board.do_action(action)
                #print(i,d)

            # 判断游戏是否结束，如果没结束就拓展叶节点
            is_over, winner = board.is_game_over()
            if not is_over:
                count += 1
                p, value = self.policy_value_net.predict(board)
                #print(p)
                #print(value if board.current_player==1 else -value)
                # 添加狄利克雷噪声
                if self.is_self_play:
                    p = 0.75*p + 0.25 * \
                        np.random.dirichlet(0.03*np.ones(len(p)))
                node.expand(zip(board.available_action, p))
                self.node_count += len(board.available_action)
            elif winner is not None:
                if winner != board.current_player:
                    value = -1
                else:
                    value = 1
            else:
                # assert(False)
                value = 0
            # 反向传播
            node.backup(value)

        for action, child in self.root.children.items():
            U = child.P * math.sqrt(self.root.N) / (1 + child.N)
            score = U - child.Q  # 注意self.Q是对手胜率

        # 计算 π，在自我博弈状态下：游戏的前三十步，温度系数为 1，后面的温度系数趋于无穷小
        T = 0.5 if len(board0.state) <= 30 else 1e-3
        visits = np.array([i.N for i in self.root.children.values()])
        pi_ = self.__getPi(visits, T)

        # 根据 π 选出动作及其对应节点
        actions = list(self.root.children.keys())
        action = int(np.random.choice(actions, p=pi_))

        rate = (-self.root.children[action].Q / 2) * 100 + 50
        # print(f'迭代次数:{count}')
        # print('搜索的节点数：', self.node_count)
        # print(f'INFO: Win  Rate:{float(rate): < 10.2f}%')
        self.search_count = count
        self.search_rate = rate

        if self.is_self_play:
            # 创建维度为 board_len^2 的 π
            pi = np.zeros(board0.board_len**2)
            pi[actions] = pi_
            # 更新根节点
            self.root = self.root.children[action]
            self.root.parent = None
            return action, pi
        else:
            self.reset_root()
            return action

    def __getPi(self, visits, T) -> np.ndarray:
        """ 根据节点的访问次数计算 π """
        # pi = visits**(1/T) / np.sum(visits**(1/T)) 会出现标量溢出问题，所以使用对数压缩
        x = 1/T * np.log(visits + 1e-10)
        x = np.exp(x - x.max())
        pi = x/x.sum()
        return pi

    def reset_root(self):
        """ 重置根节点 """
        self.root = Node(prior_prob=1, c_puct=self.c_puct, parent=None)
        self.node_count = 0

    def set_self_play(self, is_self_play: bool):
        """ 设置蒙特卡洛树的自我博弈状态 """
        self.is_self_play = is_self_play