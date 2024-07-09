import numpy as np
import sys
from copy import copy
from board import Board

rows = 3
cols = 3

board = np.zeros((rows, cols))
# 0 ->blank
# 1 --> 'x'
# 2-> 'o'

class AlphaBeta:

    def __init__(self, net):
        self.net = net

        self.inf = 9999999999
        self.neg_inf = -9999999999
        self.depth = 2

    def get_action(self, board):
        _, nextState = self.minimax(board, self.neg_inf, self.inf, True, self.depth)
        for action in range(board.board_len**2):
            if board.board[action//11][action%11] != nextState.board[action//11][action%11]:
                return action
        return None

    #  value,nextState=minimax(state,neg_inf,inf,True,2,2,1)
    def minimax(self, board, alpha, beta, maximizing, depth):
        if depth == 0:  # 递归的终止条件之一。如果搜索达到指定深度，函数将返回当前状态的效用值
            _, value = self.net.predict(board)
            return value, board # 改为神经网络预测

        # 这一行代码用于找到游戏状态中还未被占据的位置，即哪些位置可以进行下一步的决策。
        available_action = board.get_available()
        returnState = board.copy()

        # 没有可选动作，直接返回
        if len(available_action) == 0:
            _, value = self.net.predict(board)
            return value, returnState

        if maximizing:  # 处理最大化部分
            utility = self.neg_inf
            # for i in range(0, len(available_action)-1): # 遍历每个可选动作
            for action in available_action:
                nextState = board.copy()
                nextState.do_action(action)  # 将当前动作下到对应棋盘中
                # 调用 minimax 递归函数以获取对手的最佳响应
                Nutility, Nstate = self.minimax(nextState, alpha, beta, False, depth - 1)
                if Nutility > utility:
                    utility = Nutility
                    returnState = nextState.copy()
                if utility > alpha:
                    alpha = utility
                if alpha >= beta:
                    break

            return utility, returnState

        else:  # 处理最小化部分
            utility = self.inf
            for action in available_action:
                nextState = board.copy()
                nextState.do_action(action)
                Nutility, Nstate = self.minimax(nextState, alpha, beta, True, depth - 1)
                if Nutility < utility:
                    utility = Nutility
                    returnState = nextState.copy()
                if utility < beta:
                    beta = utility
                if alpha >= beta:
                    break
            return utility, returnState

