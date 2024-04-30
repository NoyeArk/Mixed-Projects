import json
import copy
import math
import time
import random
import itertools
import numpy as np
import sys  # 导入sys模块

sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000

BLACK = 1
WHITE = -1
EMPTY = 0
BLACK_WIN = 10
WHITE_WIN = 11

boardH = 11
boardW = 11

NOT_END = -40
GAME_END = 40


class Board():
    def __init__(self):
        self.b_0_3 = np.uint64(0)
        self.b_4_7 = np.uint64(0)
        self.b_8_10 = np.uint64(0)


def get_next_action(black_board, white_board):
    action_list = []
    action_board = Board()

    action_board.b_0_3 = black_board.b_0_3 | white_board.b_0_3
    action_board.b_4_7 = black_board.b_4_7 | white_board.b_4_7
    action_board.b_8_10 = black_board.b_8_10 | white_board.b_8_10

    neg_board = ~action_board.b_0_3
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


def init_board(board, action):
    # 例如为0,3，其中3实际上是4
    if action["y"] <= 3:
        shift_num = action["y"] * 11 + action["x"]
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


def update_board(board, action):
    new_board = Board()
    new_board.b_0_3 = board.b_0_3
    new_board.b_4_7 = board.b_4_7
    new_board.b_8_10 = board.b_8_10
    x = action["x"]
    y = action["y"]
    # 例如为0,3，其中3实际上是4
    if x <= 3:
        shift_num = x * 11 + y
        new_board.b_0_3 = board.b_0_3 | (np.uint64(1) << np.uint(shift_num))
    elif x <= 7:
        x -= 4
        shift_num = x * 11 + y
        new_board.b_4_7 = (board.b_4_7) | ((np.uint64(1) << np.uint(shift_num)))
    else:
        x -= 8
        shift_num = x * 11 + y
        new_board.b_8_10 = board.b_8_10 | (np.uint64(1) << np.uint(shift_num))
    return new_board


def game_end(board):
    # 一共要判断三种，横向、纵向、斜方向
    # 先判断右下
    row_list = []

    # 应该是4个4个判断
    row_list = []
    row_list.append(board.b_0_3 & np.uint64(2047))
    row_list.append((board.b_0_3 & np.uint64(4192256)) >> np.uint64(11 * 1))
    row_list.append(
        (board.b_0_3 & np.uint64(8585740288)) >> np.uint64(11 * 2))  # 8585740288 11111111111 00000000000 00000000000
    row_list.append((board.b_0_3 & np.uint64(17583596109824)) >> np.uint64(
        11 * 3))  # 17583596109824 11111111111000000000000000000000000000000000

    row_list.append(board.b_4_7 & np.uint64(2047))
    row_list.append((board.b_4_7 & np.uint64(4192256)) >> np.uint64(11 * 1))
    row_list.append((board.b_4_7 & np.uint64(8585740288)) >> np.uint64(11 * 2))
    row_list.append((board.b_4_7 & np.uint64(17583596109824)) >> np.uint64(11 * 3))

    row_list.append(board.b_8_10 & np.uint64(2047))
    row_list.append((board.b_8_10 & np.uint64(4192256)) >> np.uint64(11 * 1))
    row_list.append((board.b_8_10 & np.uint64(8585740288)) >> np.uint64(11 * 2))

    for i in range(8):
        # 判断八个四行, 第一行不用移位
        first = row_list[i]
        second = row_list[i + 1] >> np.uint64(1)
        third = row_list[i + 2] >> np.uint64(2)
        forth = row_list[i + 3] >> np.uint64(3)

        outcome_and = first & second & third & forth
        if (bin(outcome_and).count('1')) >= 1:
            return GAME_END

    # 判断左下
    for i in range(8):
        # 判断八个四行, 第一行不用移位
        first = row_list[i]
        second = row_list[i + 1] << np.uint64(1)
        third = row_list[i + 2] << np.uint64(2)
        forth = row_list[i + 3] << np.uint64(3)

        outcome_and = first & second & third & forth
        if (bin(outcome_and).count('1')) >= 1:
            return GAME_END

    # 竖向
    for loop in range(8):
        outcome_and = row_list[loop] & row_list[loop + 1] & row_list[loop + 2] & row_list[loop + 3]
        if (bin(outcome_and).count('1')) >= 1:
            return GAME_END

    # 横向
    '''
    00000000000
    11110000000
    00000000000
    00000000000
    00000000000
    00000000000
    00 00000000000 00000000000 00000000000 00010000000 00000000000 0000000
    00 00000000000 00000000000 00000000000 00010000000 00000000000 0001111
    00 0000 00000000000 00000000000 01000000000 00000000000 00000000000  000
    '''
    board_list = [board.b_0_3, board.b_4_7, board.b_8_10]
    for loop in range(3):  # 0 1 2
        refer_num = np.uint64(15)  # 1111
        if loop <= 1: # 0 1 2
            board_num = 4
        else:
            board_num = 3
        for j in range(board_num): # j 是要判断的行数
            for i in range(8): # 判断一行
                # print(refer_num)
                if i != 0:
                    refer_num = refer_num << np.uint64(1)
                outcome = board_list[loop] & refer_num
                if outcome == refer_num:
                    return GAME_END
            # j=0,4
            # refer_num = np.uint64(15) << np.uint64(4 * (j + 1))
            refer_num = refer_num << np.uint64(4)
    return NOT_END

def in_board(x, y):
    if x < 0 or x > 10 or y < 0 or y > 10:
        return False
    return True


class TreeNode(object):

    def __init__(self, board=None, parent=None, player=None):
        self.board = board  # board：[black_board, white_board]
        self.idx = 0

        self.action = get_next_action(board[0], board[1])

        self.quality = 0
        self.visit_times = 1
        self.UCB_value = 0
        self.player = player
        self.is_expand = True

        self.parent = parent
        self.child = []

    def set_parent(self, parent):
        self.parent = parent

    def set_child(self, child):
        self.child.append(child)

    def child_Num(self):
        return self.information.childNum

def Selection(node):
    # 没有全部拓展
    choice = random.randint(0, len(node.action) - 1)
    for child in node.child:
        if child.idx == choice:
            return Selection(child)
    return choice

def Simulation(node):
    value = 0
    # 一开始如果是白方 变成 黑方
    player = node.player * -1
    black_board = copy.deepcopy(node.board[0])
    white_board = copy.deepcopy(node.board[1])
    while True:
        # 在这里又变成了白方
        player = -1 * player
        action_list = get_next_action(black_board, white_board)
        num = len(action_list)
        # print(num)
        # print(action_list)
        if len(action_list) == 0:
            value = 0
            break
        # 从所有动作中随机抽取一个
        action = random.randint(0, len(action_list) - 1)
        # 更新棋盘
        if player == WHITE:
            white_board = update_board(white_board, {"x": action_list[action][0], "y": action_list[action][1]})
        if player == BLACK:
            black_board = update_board(black_board, {"x": action_list[action][0], "y": action_list[action][1]})
        # 判断游戏是否结束
        if game_end(black_board) == GAME_END:
            value = 1
            break
        elif game_end(white_board) == GAME_END:
            value = 0
            break
    node.quality += value

def max_value(node):
    Max_value = TreeNode(node.board)
    for Child in node.child:
        # print(Child.quality)
        if Max_value.quality < Child.quality:
            Max_value = Child
    return Max_value

def MCTS(black_board, white_board):
    root = TreeNode([black_board, white_board], None, player=BLACK)

    for i in range(len(root.action)):
        action = {"x": root.action[i][0], "y": root.action[i][1]}
        # if action == {'x': 2, 'y': 6}:
        #     print()
        new_board = [root.board[0], update_board(root.board[1], action)]
        leaf_node = TreeNode(new_board, root, BLACK if root.player == WHITE else WHITE)
        leaf_node.idx = i
        root.child.append(leaf_node)
        if game_end(leaf_node.board[1]) == GAME_END:
            leaf_node.is_expand = False
            '''
            00111001010
            11101110101
            01010100100
            01001101010 
            '''

    begin_time = time.time()

    choice = 0
    for loop in range(4000):
        # print(loop)
        end_time = time.time()
        if end_time - begin_time >= 5:
            break

        if root.child[choice].is_expand == True:
            Simulation(root.child[choice])

        choice += 1
        if choice == len(root.action):
            choice = 0

    best = max_value(root)
    # best = best_node(root.child[0])
    # print('max_value:', best.UCB_value)
    return root.action[best.idx]


if __name__ == '__main__':

    black_board = Board()
    white_board = Board()
    # 解析读入的JSON
    full_input = {"requests":[{"x":-1,"y":-1},{"x":0,"y":10},{"x":10,"y":10},{"x":0,"y":1},{"x":1,"y":0},{"x":9,"y":0},{"x":10,"y":1},{"x":0,"y":9},{"x":10,"y":9},{"x":2,"y":10},{"x":8,"y":10},{"x":0,"y":3},{"x":7,"y":0},{"x":7,"y":10},{"x":3,"y":0},{"x":10,"y":3},{"x":10,"y":4},{"x":4,"y":10},{"x":1,"y":1},{"x":0,"y":7},{"x":1,"y":9},{"x":9,"y":1},{"x":1,"y":8},{"x":8,"y":9},{"x":9,"y":6},{"x":3,"y":9},{"x":5,"y":1},{"x":3,"y":1},{"x":1,"y":3},{"x":1,"y":4},{"x":4,"y":9},{"x":9,"y":7},{"x":9,"y":5},{"x":9,"y":3},{"x":6,"y":9},{"x":8,"y":2},{"x":1,"y":5},{"x":2,"y":2},{"x":3,"y":2},{"x":2,"y":3},{"x":7,"y":2},{"x":6,"y":8},{"x":4,"y":2},{"x":3,"y":8},{"x":8,"y":4},{"x":3,"y":4},{"x":4,"y":3},{"x":7,"y":4}],"responses":[{"x":0,"y":0},{"x":10,"y":0},{"x":1,"y":10},{"x":9,"y":10},{"x":2,"y":0},{"x":8,"y":0},{"x":0,"y":2},{"x":10,"y":2},{"x":5,"y":10},{"x":0,"y":6},{"x":10,"y":6},{"x":3,"y":10},{"x":6,"y":0},{"x":0,"y":4},{"x":4,"y":0},{"x":5,"y":0},{"x":10,"y":5},{"x":6,"y":10},{"x":0,"y":8},{"x":0,"y":5},{"x":10,"y":8},{"x":9,"y":9},{"x":9,"y":4},{"x":2,"y":9},{"x":7,"y":9},{"x":4,"y":1},{"x":2,"y":1},{"x":1,"y":2},{"x":6,"y":1},{"x":7,"y":1},{"x":8,"y":1},{"x":9,"y":8},{"x":9,"y":2},{"x":5,"y":9},{"x":1,"y":7},{"x":1,"y":6},{"x":3,"y":3},{"x":5,"y":2},{"x":4,"y":4},{"x":2,"y":4},{"x":2,"y":6},{"x":7,"y":6},{"x":2,"y":5},{"x":4,"y":6},{"x":8,"y":5},{"x":6,"y":2},{"x":8,"y":6},{"x":3,"y":6}]}



    # 分析自己收到的输入和自己过往的输出，并恢复状态
    rival_trajectory = full_input["requests"]
    my_trajectory = full_input["responses"]

    if (int(rival_trajectory[0]['x']) == -1):
        # L[1:] 从第二个元素开始截取列表
        rival_trajectory = rival_trajectory[1:]

    for i in range(len(rival_trajectory)):
        init_board(black_board, rival_trajectory[i])
    for i in range(len(my_trajectory)):
        init_board(white_board, my_trajectory[i])

    print(game_end(white_board))

    action_list = get_next_action(black_board, white_board)

    # print(action_list)
    # print(len(action_list))

    best_move = MCTS(black_board, white_board)

    print(game_end(white_board))

    x = best_move[1]
    y = best_move[0]

    # TODO: 作出决策并输出
    my_action = {"x": x, "y": y}

    print(json.dumps({
        "response": my_action,
    }))
