#!/usr/bin/env python
import torch
import numpy as np
from board import Board

EMPTY = -1
WHITE = 0
BLACK = 1

BOARD_LEN = 7


def _data_convert(board, last_action):
    """convert data format
       return tensor
    """
    n = BOARD_LEN
    '''state0存储当前玩家的棋子位置，state1存储对方玩家的棋子位置，state2存储当前动作的位置'''
    _board = torch.Tensor(board.board).unsqueeze(0)
    state0 = (_board > 0).float()
    state1 = (_board < 0).float()

    # state2 = torch.zeros((len(_board), 1, n, n)).float()
    state2 = torch.zeros((1, n, n)).float()

    if board.current_player == WHITE:
        temp = state0.clone()
        state0.copy_(state1)
        state1.copy_(temp)

    if last_action != -1:
        # 得到x，y坐标
        x, y = last_action // n, last_action % n
        state2[0][x][y] = 1

    # print(state0.shape, state1.shape, state2.shape)

    res = torch.cat((state0, state1, state2), dim=1)
    # res = torch.cat((state0, state1), dim=1)
    return res.cuda()

def encode_board(board):
    board_state = board.board
    encoded = np.zeros([board.board_len, board.board_len, 3]).astype(int)
    # encoder_dict = {"O":0, "X":1}
    encoder_dict = {EMPTY:0, WHITE:1, BLACK:2}
    for row in range(board.board_len):
        for col in range(board.board_len):
            # if board_state[row,col] != EMPTY:
            encoded[row, col, encoder_dict[board_state[row,col]]] = 1
    if board.current_player == BLACK:
        encoded[:,:,3] = 1 # 黑方移动，将最后一个维度置1
    return encoded

def decode_board(encoded):
    decoded = np.zeros([6,7]).astype(str)
    decoded[decoded == "0.0"] = " "
    decoder_dict = {0:"O", 1:"X"}
    for row in range(6):
        for col in range(7):
            for k in range(2):
                if encoded[row, col, k] == 1:
                    decoded[row, col] = decoder_dict[k]
    cboard = board()
    cboard.current_board = decoded
    cboard.player = encoded[0,0,2]
    return cboard