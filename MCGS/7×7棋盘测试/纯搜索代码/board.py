from copy import deepcopy
from collections import OrderedDict
import torch
import numpy as np


class Board:
    """
    棋盘类
    """

    EMPTY = -1
    WHITE = 0
    BLACK = 1
    CX = [-1, 0, 1, 0, -1, 1]
    CY = [0, -1, 0, 1, 1, -1]

    def __init__(self, board_len=11, n_feature_planes=2):
        """
        初始化棋盘。
        :param board_len: int 棋盘边长。
        :param n_feature_planes: int 特征平面的个数。
        """
        self.board_len = board_len
        self.current_player = self.BLACK
        self.n_feature_planes = n_feature_planes
        self.available_action = list(range(self.board_len**2))
        # 棋盘状态字典，key为action，value为current_play。
        self.state = OrderedDict()
        # 上一个落点
        self.pre_action = None
        # 历史棋局的矩阵形式。
        self.board = np.full(shape=(11, 11), fill_value=self.EMPTY)
        self.board_transpose = np.full(shape=(11, 11), fill_value=self.EMPTY)

    def copy(self):
        """
        复制棋盘。
        :return: 棋盘类。
        """
        return deepcopy(self)

    def clear_board(self):
        """
        清空棋盘。
        :return:
        """
        self.state.clear()
        self.board_len = 11
        self.current_player = self.BLACK
        self.state = OrderedDict()
        self.n_feature_planes = 2
        self.pre_action = None
        self.current_player = self.BLACK
        self.available_action = list(range(self.board_len**2))
        self.board = np.full(shape=(11, 11), fill_value=self.EMPTY)

    def in_board(self, x: int, y: int) -> bool:
        """
        判断棋子是否在棋盘中。
        :param x: 棋子横坐标。
        :param y: 棋子纵坐标。
        :return: 是否在棋盘中，是->True 否则->False
        """
        return 0 <= x < self.board_len and 0 <= y < self.board_len

    def get_available(self):
        """
        获取动作。
        :return:返回动作。
        """
        available = []
        for i in range(self.board_len):
            for j in range(self.board_len):
                if self.board[i][j] == self.EMPTY:
                    available.append(i * self.board_len + j)
        return list(available)

    def dfs(self, board, start: int, end: int):
        """
        深度优先算法，判断是否联通。
        :param start: 起点坐标。
        :param end: 终点坐标。
        :return: 是否联通。
        """
        q = []
        dfs_air_visit = np.full(shape=(11, 11), fill_value=False)
        q.append(start)
        dfs_air_visit[start//self.board_len][start%self.board_len] = True
        while len(q) > 0:
            Vn = q.pop(0)
            for dir in range(6):
                x = Vn // self.board_len + self.CX[dir]
                y = Vn % self.board_len + self.CY[dir]
                if self.in_board(x, y):
                    if board[x][y] == board[start//self.board_len][start%self.board_len]:
                        Vw = x * self.board_len + y
                        if Vw == end:
                            return True
                        if not dfs_air_visit[x][y]:
                            q.append(Vw)
                            dfs_air_visit[x][y] = True
        return False

    def set_empty(self, action:int):
        self.board[int(action//self.board_len)][int(action % self.board_len)] = self.EMPTY

    def do_action(self, action: int):
        """
        落子更新棋盘。
        :param action: 落子位置，范围[0, board_len^2-1]
        :return:
        """
        self.pre_action = action
        self.state[action] = self.current_player
        self.board[int(action//self.board_len)][int(action % self.board_len)] = self.current_player
        self.board_transpose = np.transpose(self.board)
        self.current_player = self.BLACK + self.WHITE - self.current_player
        self.available_action = self.get_available()


    def undo(self):
        """
        悔两步棋
        :return:
        """
        # 1
        undo_action = self.pre_action
        self.state.pop(undo_action)
        self.board[undo_action // self.board_len][undo_action % self.board_len] = self.EMPTY
        self.board_transpose = np.transpose(self.board)
        self.pre_action = next(reversed(self.state))

        # 2
        undo_action = self.pre_action
        self.state.pop(undo_action)
        self.board[undo_action // self.board_len][undo_action % self.board_len] = self.EMPTY
        self.board_transpose = np.transpose(self.board)
        self.pre_action = next(reversed(self.state))

        self.available_action = self.get_available()


    def is_over(self, board, color):
        """
        判断是否红方胜利。
        :return: 是否红方胜利。
        """
        if color not in board[0] or color not in board[10]:
            return False
        start = []
        end = []
        for i in range(self.board_len):
            if board[0][i] == color:
                start.append(i)
            if board[10][i] == color:
                end.append(10 * self.board_len + i)
        for i in start:
            for j in end:
                if self.dfs(board, i, j):
                    return True
        return False


    def is_game_over(self):
        """
        判断游戏是否结束。
        :return: Tuple[bool, int]
        *bool->是否结束（分出胜负或者平局）True 否则 False
        *int ->游戏的赢家。
                **如果游戏分出胜负，Board.BLACK 或者 Board.WHITE
                **如果游戏未分出胜负，None
        """
        if len(self.state) < 21:
            return False, None
        if self.is_over(self.board, self.BLACK):
            return True, self.BLACK
        if self.is_over(self.board_transpose, self.WHITE):
            return True, self.WHITE
        return False, None


    def get_feature_planes(self) -> torch.Tensor:
        """
        棋盘状态特征张量，维度：（n_feature_planes, board_len, board_len）
        :return: torch.Tensor,棋盘状态特征张量.
        """
        n = self.board_len
        feature_planes = torch.zeros((self.n_feature_planes, n, n))

        # 添加历史信息
        for i in range(self.board_len):
            for j in range(self.board_len):
                if self.board[i][j] == self.current_player:
                    feature_planes[0][i][j] = 1
                if self.board[i][j] == self.BLACK + self.WHITE - self.current_player:
                    feature_planes[1][i][j] = 1
        if self.current_player == self.WHITE:
            feature_planes = feature_planes.transpose(1, 2)
        return feature_planes
