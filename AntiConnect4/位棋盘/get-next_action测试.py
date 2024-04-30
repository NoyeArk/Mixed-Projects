import json
import random
import itertools
import numpy as np

board_size = 11
BLACK = 1
WHITE = -1
EMPTY = 0
BLACK_WIN = 10
WHITE_WIN = 11

NOT_END = -40
END = 40
search_num = 1

CX = [-1, 1, 0, 0, 1, -1, -1, 1]
CY = [0, 0, -1, 1, 1, -1, 1, -1]


class Board():
    def __init__(self):
        self.b_0_3 = np.uint64(0)
        self.b_4_7 = np.uint64(0)
        self.b_8_10 = np.uint64(0)

'''
判断下一步可以落子的位置
黑棋盘和白棋盘继续或，所有为0的是空位置，可以下
'''

def get_next_action(black_board, white_board):
    action_list = []
    action_board = Board()

    action_board.b_0_3 = black_board.b_0_3 | white_board.b_0_3
    action_board.b_4_7 = black_board.b_4_7 | white_board.b_4_7
    action_board.b_8_10 = black_board.b_8_10 | white_board.b_8_10

    # neg_board = ~action_board.b_0_3
    bin_str = bin(neg_board).replace('0b', '')
    for i in range(len(bin_str)):
        # 1代表为空
        if i <= 43 and (np.uint64(1) << np.uint64(i)) & (neg_board):
            x = i // 11
            y = i - x * 11
            action_list.append([x, y])

    neg_board = ~action_board.b_4_7
    bin_str = bin(neg_board).replace('0b', '')
    for i in range(len(bin_str)):
        if i <= 43 and (np.uint64(1) << np.uint64(i)) & neg_board:
            x = i // 11 + 4
            y = i - (x - 4) * 11
            action_list.append([x, y])

    neg_board = ~action_board.b_8_10
    bin_str = bin(neg_board).replace('0b', '')
    for i in range(len(bin_str)):
        if i <= 32 and (np.uint64(1) << np.uint64(i)) & neg_board:
            x = i // 11 + 8
            y = i - (x - 8) * 11
            action_list.append([x, y])
    return action_list

def update_board(board, action):
    # 例如为0,3，其中3实际上是4
    if action["y"] <= 3:
        shift_num = action["y"] * 11 + action["x"]
        print((np.uint64(1) << np.uint(shift_num)))
        print(bin(np.uint64(1) << np.uint(shift_num)))
        board.b_0_3 = board.b_0_3 | (np.uint64(1) << np.uint(shift_num))
    elif action["y"] <= 7:
        action["y"] -= 4
        shift_num = action["y"] * 11 + action["x"]
        board.b_4_7 = (board.b_4_7) | ((np.uint64(1) << np.uint(shift_num)))
    else:
        action["y"] -= 8
        shift_num = action["y"] * 11 + action["x"]
        board.b_8_10 = board.b_8_10 | (np.uint64(1) << np.uint(shift_num))
    return board

if __name__ == '__main__':
    black_board = Board()
    white_board = Board()
    # 解析读入的JSON
    full_input = {"requests":[{"x":-1,"y":-1},{"x":0,"y":10},{"x":10,"y":10},{"x":0,"y":1},{"x":1,"y":0},{"x":9,"y":0},{"x":10,"y":1},{"x":0,"y":9},{"x":10,"y":9},{"x":2,"y":10},{"x":8,"y":10},{"x":0,"y":3},{"x":7,"y":0},{"x":7,"y":10},{"x":3,"y":0},{"x":10,"y":3},{"x":10,"y":4},{"x":4,"y":10},{"x":1,"y":1},{"x":0,"y":7},{"x":1,"y":9},{"x":9,"y":1},{"x":1,"y":8},{"x":8,"y":9},{"x":9,"y":6},{"x":3,"y":9},{"x":5,"y":1},{"x":3,"y":1},{"x":1,"y":3},{"x":1,"y":4},{"x":4,"y":9},{"x":9,"y":7},{"x":9,"y":5},{"x":9,"y":3},{"x":6,"y":9},{"x":8,"y":2},{"x":1,"y":5},{"x":2,"y":2},{"x":3,"y":8},{"x":7,"y":2},{"x":4,"y":2},{"x":6,"y":2},{"x":2,"y":6},{"x":8,"y":5},{"x":3,"y":4},{"x":4,"y":3},{"x":4,"y":7},{"x":7,"y":8}],"responses":[{"x":0,"y":0},{"x":10,"y":0},{"x":1,"y":10},{"x":9,"y":10},{"x":2,"y":0},{"x":8,"y":0},{"x":0,"y":2},{"x":10,"y":2},{"x":5,"y":10},{"x":0,"y":6},{"x":10,"y":6},{"x":3,"y":10},{"x":6,"y":0},{"x":0,"y":4},{"x":4,"y":0},{"x":5,"y":0},{"x":10,"y":5},{"x":6,"y":10},{"x":0,"y":8},{"x":0,"y":5},{"x":10,"y":8},{"x":9,"y":9},{"x":9,"y":4},{"x":2,"y":9},{"x":7,"y":9},{"x":4,"y":1},{"x":2,"y":1},{"x":1,"y":2},{"x":6,"y":1},{"x":7,"y":1},{"x":8,"y":1},{"x":9,"y":8},{"x":9,"y":2},{"x":5,"y":9},{"x":1,"y":7},{"x":1,"y":6},{"x":8,"y":4},{"x":2,"y":4},{"x":5,"y":2},{"x":8,"y":8},{"x":6,"y":8},{"x":5,"y":8},{"x":4,"y":8},{"x":2,"y":8},{"x":8,"y":7},{"x":7,"y":7},{"x":6,"y":7}]}


    # 分析自己收到的输入和自己过往的输出，并恢复状态
    rival_trajectory = full_input["requests"]
    my_trajectory = full_input["responses"]

    if (int(rival_trajectory[0]['x']) == -1):
        # L[1:] 从第二个元素开始截取列表
        rival_trajectory = rival_trajectory[1:]

    # print(my_trajectory)

    for i in range(len(rival_trajectory)):
        update_board(black_board, rival_trajectory[i])
    for i in range(len(my_trajectory)):
        update_board(white_board, my_trajectory[i])

    action_list = get_next_action(black_board, white_board)
    print(black_board.b_0_3)
    print(action_list)
    # if game_end(black_board) == END:
    #     print('黑棋连4')
    if game_end(white_board) == END:
        print('白棋连4')
    # my_action = random.randint(0, len(best_move) - 1)  # 产生 0 到 len(best_move) - 1 的一个整数型随机数
    #
    # x = best_move[my_action][1]
    # y = best_move[my_action][0]

    # # TODO: 作出决策并输出
    # my_action = { "x": x, "y": y }

    # print(json.dumps({
    #     "response": my_action,
    # }))
