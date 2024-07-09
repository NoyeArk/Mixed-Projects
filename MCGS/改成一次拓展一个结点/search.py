from typing import Tuple, Union
import numpy as np
import time
from node import Node
from board import Board


class AlphaZeroMCTS:
    """
    基于策略-价值网络的蒙特卡洛搜索树
    """

    def __init__(self, policy_value_net, c_puct: float = 1, n_iters=100.0, is_self_play=False) -> None:
        """
        初始化。
        :param policy_value_net: PolicyValueNet，策略价值网络
        :param c_puct: 搜索常数。
        :param n_iters: 搜索迭代次数。
        :param is_self_play: bool,是否使用自博弈状态。
        """
        self.c_puct = c_puct
        self.n_iters = n_iters
        self.is_self_play = is_self_play
        self.policy_value_net = policy_value_net
        self.root = Node(prior_prob=1, parent=None)

        self.node_count = 0
        self.search_rate = 0


    def get_action(self, board0: Board) -> Union[Tuple[int, np.ndarray], int]:
        """
        根据当前局面返回下一步动作。
        :param board: Board,棋盘。
        :return:
        """

        start_time = time.time()
        count = 0
        while time.time() - start_time <= self.n_iters:
        # for i in range(self.n_iters):
            count += 1
            # 拷贝棋盘
            board = board0.copy()

            # 如果没有遇到叶节点，就一直向下搜索并更新棋盘
            node = self.root
            while not node.is_leaf_node():
                #d=d+1
                print('_____________select________________')
                action, node = node.select()
                board.do_action(action)
                #print(i,d)
                #print(board.board)

            # 判断游戏是否结束，如果没结束就拓展叶节点
            is_over, winner = board.is_game_over()
            if not is_over:
                p, value = self.policy_value_net.predict(board)
                #print(p)

                #print(value if board.current_player==1 else -value)

                node.expand(zip(board.available_action, p))
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

        visits = np.array([i.N for i in self.root.children.values()])
        pi_ = self.__getPi(visits)

        # 根据 π 选出动作及其对应节点
        actions = list(self.root.children.keys())
        action = int(np.random.choice(actions, p=pi_))

        rate = (-self.root.children[action].Q / 2) * 100 + 50
        print(f'INFO: Iter Count:{count}')
        print(f'INFO: Win  Rate:{float(rate): < 10.2f}%')
        self.node_count = count
        self.search_rate = rate

        self.reset_root()
        return action


    def __getPi(self, visits, T=0.5) -> np.ndarray:
        """ 根据节点的访问次数计算 π """
        # pi = visits**(1/T) / np.sum(visits**(1/T)) 会出现标量溢出问题，所以使用对数压缩
        x = 1/T * np.log(visits + 1e-10)
        x = np.exp(x - x.max())
        pi = x/x.sum()
        return pi


    def reset_root(self):
        """ 重置根节点 """
        self.root = Node(prior_prob=1, c_puct=self.c_puct, parent=None)


    def set_self_play(self, is_self_play: bool):
        """ 设置蒙特卡洛树的自我博弈状态 """
        self.is_self_play = is_self_play


    def set_searchTime(self, searchTime):
        self.n_iters = searchTime


