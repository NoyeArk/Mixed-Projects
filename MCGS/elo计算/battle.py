from ast import Tuple
from collections import OrderedDict
import copy
import json
import math
import random
import sys
import time
import numpy as np
import torch
from net import ReversiNet
import onnx
import onnxruntime
import numpy as np


class Board:
    """棋盘类"""

    EMPTY = -1
    WHITE = 0
    BLACK = 1

    def __init__(self, board_len=15):
        """
        Parameters
        ----------
        board_len: int
            棋盘边长
        n_feature_planes: int
            特征平面的个数，必须为偶数
        """
        self.board_len = board_len
        self.current_player = self.BLACK
        self.available_action = list(range(self.board_len**2))
        # 棋盘状态字典，key 为 action，value 为 current_player
        self.state = OrderedDict()
        # 上一个落点
        self.pre_action = None
        self.board = np.full(shape=(15, 15), fill_value=self.EMPTY)

    def copy(self):
        """复制棋盘"""
        return copy.deepcopy(self)

    def clear_board(self):
        """清空棋盘"""
        self.state.clear()
        self.pre_action = None
        self.current_player = self.BLACK
        self.available_action = list(range(self.board_len**2))
        self.board = np.full(shape=(15, 15), fill_value=self.EMPTY)

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

    def do_action(self, action: int):
        """落子并更新棋盘
        Parameters
        ----------
        action: int
            落子位置，范围为 `[0, board_len^2 -1]`
        """
        self.pre_action = action
        self.state[action] = self.current_player
        self.board[action // self.board_len][
            action % self.board_len
        ] = self.current_player
        self.current_player = self.WHITE + self.BLACK - self.current_player
        self.available_action = self.get_available()

    def is_game_over(self):
        """判断游戏是否结束
        Returns
        -------
        is_over: bool
            游戏是否结束，分出胜负或者平局则为 `True`, 否则为 `False`
        winner: int
            游戏赢家，有以下几种:
            * 如果游戏分出胜负，则为 `ChessBoard.BLACK` 或 `ChessBoard.WHITE`
            * 如果还有分出胜负或者平局，则为 `None`
        """
        # 如果下的棋子不到 9 个，就直接判断游戏还没结束
        if len(self.state) < 9:
            return False, None

        n = self.board_len
        act = self.pre_action
        player = self.state[act]
        row, col = act // n, act % n

        # 搜索方向
        directions = [
            [(0, -1), (0, 1)],  # 水平搜索
            [(-1, 0), (1, 0)],  # 竖直搜索
            [(-1, -1), (1, 1)],  # 主对角线搜索
            [(1, -1), (-1, 1)],
        ]  # 副对角线搜索

        for i in range(4):
            count = 1
            for j in range(2):
                flag = True
                row_t, col_t = row, col
                while flag:
                    row_t = row_t + directions[i][j][0]
                    col_t = col_t + directions[i][j][1]
                    if (
                        0 <= row_t < n
                        and 0 <= col_t < n
                        and self.state.get(row_t * n + col_t, self.EMPTY) == player
                    ):
                        # 遇到相同颜色时 count+1
                        count += 1
                    else:
                        flag = False
            # 分出胜负
            if count >= 5:
                return True, player

        # 平局
        if not self.available_action:
            return True, None

        return False, None


# model = torch.jit.load("data/gomoku_230201.pth")
TIME_LIMIT = 0.2
MAX_EVAL = 25
ACTION_SIZE = 15 * 15

DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

random.seed(2023, 2)


class Game:
    board_hash = [
        [random.getrandbits(48) for _ in range(15**2)],
        [random.getrandbits(48) for _ in range(15**2)],
    ]
    init_hash = random.getrandbits(48)

    def __init__(self):
        self.board = torch.zeros((2, 15, 15), dtype=torch.bool)
        self.board_hash = Game.init_hash
        self.stones = 0

    def place(self, x, y, color):
        # assert color == self.stones % 2
        self.step(x * 15 + y)

    def step(self, a):
        next_player = self.stones % 2
        self.board_hash ^= Game.board_hash[next_player][a]
        # assert self.board[next_player, a // 15, a % 15] == False
        self.board[next_player, a // 15, a % 15] = True
        self.stones += 1

    def __hash__(self) -> int:
        return self.board_hash

    def to_nnet_input(self):
        next_player = self.stones % 2
        if next_player == 0:
            return self.board[None].float()
        else:
            return self.board[None, [1, 0]].float()

    def invalid_mask(self):
        return self.board.any(0)


class MCTS:
    def __init__(self, nnet):
        self.nnet = nnet

        self.Qs = {}  # stores Q values for s,a (as defined in the paper)
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Ns = {}  # stores #times board s was visited

    def best_move(self, game, timeout=TIME_LIMIT):
        tic = time.time()
        s = hash(game)
        while time.time() - tic < timeout:
            v = self._search(copy.deepcopy(game))
            # print(v)
            if self.Ns[s] >= MAX_EVAL:
                break
        # sys.stderr.write(f"Searched {self.Ns[s]} moves in {time.time() - tic} sec\n")
        # policy = self.Qsa[s]
        # policy = torch.softmax(policy, dim=0).detach().numpy()
        # action = np.random.choice(range(225), p=policy)
        # print(action)
        return self.Qsa[s].argmax().item()
        # return action

    def _search(self, game: Game):
        s = hash(game)

        if s not in self.Qsa:  # leaf node
            # Qsa = self.nnet(game.to_nnet_input()).flatten()
            Qsa = self.nnet.run(None, {"input": game.to_nnet_input().numpy()})[0]
            Qsa = torch.from_numpy(Qsa.flatten())
            invalid_mask = game.invalid_mask().flatten()
            Qsa = Qsa.masked_fill_(invalid_mask, float("-inf"))
            Qs = torch.sum(torch.softmax(Qsa / 0.01, 0) * Qsa.sigmoid())
            # Qs = Qsa.max()

            self.Qsa[s] = Qsa
            self.Qs[s] = Qs
            self.Ns[s] = 1
            return Qs

        Qs = self.Qs[s]
        if Qs < 0.01 or Qs > 0.99:
            self.Ns[s] = math.inf
            return Qs
        if game.stones == 225:
            self.Ns[s] = math.inf
            return torch.tensor(0.5)

        Qsa = self.Qsa[s]
        a = torch.distributions.Categorical(logits=Qsa / 1).sample().item()
        game.step(a)
        v = 1 - self._search(game)
        Qsa[a] = torch.log(v.clamp_min(1e-5) / (1 - v).clamp_min(1e-5))
        Qs = torch.sum(torch.softmax(Qsa / 1, 0) * Qsa.sigmoid())
        self.Qs[s] = Qs
        self.Ns[s] += 1
        return Qs


def battle_one_game(model_1, model_2):
    game = Game()
    mcts_1 = MCTS(model_1)
    mcts_2 = MCTS(model_2)
    board = Board(15)
    who = random.randint(0, 1)
    model_1_color = 0
    model_2_color = 0
    if who == 1:
        model_2_color = 1
    else:
        model_1_color = 1
    if model_1_color == 0:
        while True:
            move = mcts_1.best_move(game)
            game.place(move // 15, move % 15, model_1_color)
            board.do_action(move)
            flag, winner = board.is_game_over()
            if flag:
                if winner != None:
                    return 1
                else:
                    return 0
            move = mcts_2.best_move(game)
            game.place(move // 15, move % 15, model_2_color)
            board.do_action(move)
            flag, winner = board.is_game_over()
            if flag:
                if winner != None:
                    return -1
                else:
                    return 0
    else:
        while True:
            move = mcts_2.best_move(game)
            game.place(move // 15, move % 15, model_1_color)
            board.do_action(move)
            flag, winner = board.is_game_over()
            if flag:
                if winner != None:
                    return -1
                else:
                    return 0
            move = mcts_1.best_move(game)
            game.place(move // 15, move % 15, model_2_color)
            board.do_action(move)
            flag, winner = board.is_game_over()
            if flag:
                if winner != None:
                    return 1
                else:
                    return 0


from net import ReversiNet


def battle_n_games(path_1: str, path_2: str, num_battle):
    # model_0 = torch.jit.load("6b128f_export/6b128f_c13.pt")
    # model_0 = ReversiNet()
    # m0Dict = torch.load("./6b128f/checkpoint0.pth")
    # model_0.load_state_dict(m0Dict)
    # # model_1 = torch.jit.load("6b128f_export/6b128f_c14.pt")
    # model_1 = ReversiNet()
    # m1Dict = torch.load("./6b128f/checkpoint1.pth")
    # model_1.load_state_dict(m1Dict)
    # model_0.eval()
    # model_1.eval()
    model_0 = onnxruntime.InferenceSession(path_1)
    model_1 = onnxruntime.InferenceSession(path_2)
    model_one_times = 0
    model_two_times = 0
    draw = 0
    for i in range(num_battle):
        time_1 = time.time()
        who = battle_one_game(model_0, model_1)
        if who == 1:
            model_one_times += 1
        elif who == -1:
            model_two_times += 1
        else:
            draw += 1
        time_2 = time.time()
        print(f"Round {i + 1} Using Time: {time_2 - time_1}")
    print("________________________________________________________________")
    print("Weights battle")
    print(model_one_times)
    print(model_two_times)
    print(draw)
    return model_one_times, model_two_times, draw


if __name__ == "__main__":
    # import multiprocessing

    # pool = multiprocessing.Pool()
    # for i in range(15):
    t1 = time.time()
    battle_n_games(
        f"./6b128f_onnx/6b128f_15.onnx", f"./10b128f_onnx/10b128f_15.onnx", 10000
    )
    t2 = time.time()
    print(t2 - t1)
