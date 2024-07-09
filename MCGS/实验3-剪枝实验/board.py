from copy import deepcopy
from collections import OrderedDict
import torch
import numpy as np

class Board:
    """
    棋盘类
    """

    EMPTY = 0
    WHITE = -1
    BLACK = 1
    CX = [-1, 0, 1, 0, -1, 1]
    CY = [0, -1, 0, 1, 1, -1]

    def __init__(self, board_len=8, n_feature_planes=2):
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
        self.board = np.full(shape=(self.board_len, self.board_len), fill_value=self.EMPTY)
        self.board_transpose = np.full(shape=(self.board_len, self.board_len), fill_value=self.EMPTY)

    def copy(self):
        """
        复制棋盘。
        :return: 棋盘类。
        """
        return deepcopy(self)

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

    def is_over(self):
        x_tmp, y_tmp = self.pre_action//self.board_len, self.pre_action%self.board_len
        for i in range(5):
            if sum(self.board[x_tmp - 4 + i:x_tmp + 1 + i, y_tmp]) == 5:
                return True
            elif sum(self.board[x_tmp, y_tmp - 4 + i:y_tmp + 1 + i]) == 5:
                return True
            elif self.board[x_tmp + i - 4, y_tmp + i - 4] + self.board[x_tmp + i - 3, y_tmp + i - 3] + self.board[
                x_tmp + i - 2, y_tmp + i - 2] + \
                    self.board[x_tmp + i - 1, y_tmp + i - 1] + self.board[x_tmp + i, y_tmp + i] == 5:
                return True
            elif self.board[x_tmp + i - 4, y_tmp - i + 4] + self.board[x_tmp + i - 3, y_tmp - i + 3] + self.board[
                x_tmp + i - 2, y_tmp - i + 2] + \
                    self.board[x_tmp + i - 1, y_tmp - i + 1] + self.board[x_tmp + i, y_tmp - i] == 5:
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
        if len(self.state) < 10:
            return False, None
        if self.is_over():
            if self.board[self.pre_action//self.board_len, self.pre_action%self.board_len] == self.BLACK:
                return True, self.BLACK
            else:
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
