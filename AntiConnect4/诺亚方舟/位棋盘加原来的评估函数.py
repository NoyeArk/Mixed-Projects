import json
import random
import itertools
import numpy as np

BLACK = 1
WHITE = -1
EMPTY = 0
BLACK_WIN = 10
WHITE_WIN = 11

NOT_END = -40
GAME_END = 40
DEPTH = 2

class Board():
    def __init__(self):
        self.b_0_3 = np.uint64(0)
        self.b_4_7 = np.uint64(0)
        self.b_8_10 = np.uint64(0)

def get_next_action(black_board, white_board):
    action_list = []
    action_board_list = []

    action_board_list.append(black_board.b_0_3 | white_board.b_0_3)
    action_board_list.append(black_board.b_4_7 | white_board.b_4_7)
    action_board_list.append(black_board.b_8_10 | white_board.b_8_10)

    for idx in range(3):
        neg_board = ~action_board_list[idx]
        bin_str = bin(neg_board).replace('0b', '')
        if idx <= 2:
            limit_i = 43
        else:
            limit_i = 32
        for i in range(len(bin_str)):
            # 1代表为空
            if i <= limit_i and (np.uint64(1) << np.uint64(i)) & (neg_board):
                x = i // 11 + idx * 4
                y = i - (x - 4 * idx) * 11
                action_list.append([x, y])
    return action_list

def update_board(board, action):
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

def game_end(board):
    # 一共要判断三种，横向、纵向、斜方向
    # 判断左下
    row_list = []
    row_list.append(board.b_0_3 & np.uint64(2047))
    row_list.append((board.b_0_3 & np.uint64(4192256)) >> np.uint64(11 * 1 + 1))
    row_list.append((board.b_0_3 & np.uint64(8585740288)) >> np.uint64(11 * 2 + 2))
    row_list.append((board.b_0_3 & np.uint64(17583596109824)) >> np.uint64(11 * 3 + 3))

    row_list.append((board.b_4_7 & np.uint64(2047)) >> np.uint64(4))
    row_list.append((board.b_4_7 & np.uint64(4192256)) >> np.uint64(11 * 1 + 5))
    row_list.append((board.b_4_7 & np.uint64(8585740288)) >> np.uint64(11 * 2 + 6))
    row_list.append((board.b_4_7 & np.uint64(17583596109824)) >> np.uint64(11 * 3 + 7))

    row_list.append((board.b_8_10 & np.uint64(2047)) >> np.uint64(8))
    row_list.append((board.b_8_10 & np.uint64(4192256)) >> np.uint64(11 * 1 + 9))
    row_list.append((board.b_8_10 & np.uint64(8585740288)) >> np.uint64(11 * 2 + 10))

    for loop in range(8):
        outcome_and = row_list[loop] & row_list[loop + 1] & row_list[loop + 2] & row_list[loop + 3]
        if (bin(outcome_and).count('1')) >= 1:
            return GAME_END

    # 纵向,相邻四行之间进行与运算，如果有1，则连成4个
    row_list = []

    row_list.append(board.b_0_3 & np.uint64(2047))
    row_list.append((board.b_0_3 & np.uint64(4192256)) >> np.uint64(11 * 1))
    row_list.append((board.b_0_3 & np.uint64(8585740288)) >> np.uint64(11 * 2))  # 8585740288 11111111111 00000000000 00000000000
    row_list.append((board.b_0_3 & np.uint64(17583596109824)) >> np.uint64( 11 * 3))  # 17583596109824 11111111111000000000000000000000000000000000

    row_list.append(board.b_4_7 & np.uint64(2047))
    row_list.append((board.b_4_7 & np.uint64(4192256)) >> np.uint64(11 * 1))
    row_list.append((board.b_4_7 & np.uint64(8585740288)) >> np.uint64(11 * 2))
    row_list.append((board.b_4_7 & np.uint64(17583596109824)) >> np.uint64(11 * 3))

    row_list.append(board.b_8_10 & np.uint64(2047))
    row_list.append((board.b_8_10 & np.uint64(4192256)) >> np.uint64(11 * 1))
    row_list.append((board.b_8_10 & np.uint64(8585740288)) >> np.uint64(11 * 2))

    for loop in range(8):
        outcome_and = row_list[loop] & row_list[loop + 1] & row_list[loop + 2] & row_list[loop + 3]
        if (bin(outcome_and).count('1')) >= 1:
            return GAME_END

    # 横向
    board_list = [board.b_0_3, board.b_4_7, board.b_8_10]
    refer_num = np.uint64(15)  # 1111
    for loop in range(3):
        if loop <= 1:
            board_num = 4
        else:
            board_num = 3
        for j in range(board_num):
            for i in range(8):
                refer_num = refer_num << np.uint64(i)
                outcome = board_list[loop] & refer_num
                if outcome == refer_num:
                    return GAME_END
            refer_num = np.uint64(15) << np.uint64(4 * (j + 1))
    return NOT_END

def in_board(x, y):
    if x < 0 or x > 10 or y < 0 or y > 10:
        return False
    return True

def get_index(int_num, loop):
    index = []
    int_num = int(int_num)
    bin_str = bin(int_num).replace('0b', '')
    for i in range(len(bin_str)):
        if (1 << i) & int_num:
            x = i // 11 + 4 * loop
            y = i - (x - 4 * loop) * 11
            if x != 11 and y != 11:
                index.append([x, y])
    return index

def grade(four_list):
    # 每个位置的value
    black_num = 0
    white_num = 0
    empty_num = 0
    for i in range(len(four_list)):
        if four_list[i] == BLACK:
            black_num += 1
        elif four_list[i] == WHITE:
            white_num += 1
        else:
            empty_num += 1

    # 四个白子
    if white_num == 4:
        return -10000000
    elif white_num == 3 and empty_num == 1:
        return -100000
    elif white_num == 2 and empty_num == 2:
        return -1000
    elif white_num == 1 and empty_num == 3:
        return -10
    elif white_num != 0 and black_num != 0:
        return 0
    elif black_num == 1 and empty_num == 3:
        return 10
    elif black_num == 2 and empty_num == 2:
        return 1000
    elif black_num == 3 and empty_num == 1:
        return 100000
    elif black_num == 4:
        return 10000000
    else:
        return 0

def Value(black_board, white_board):
    # 得到汇总后的棋盘
    board = np.zeros((11, 11), dtype=int)
    action_idx_list = [black_board.b_0_3, black_board.b_4_7, black_board.b_8_10, # 4 4 3
                         white_board.b_0_3, white_board.b_4_7, white_board.b_8_10]
    for loop in range(len(action_idx_list)): # 0 1 2 3 4 5
        if loop <= 2:
            player = BLACK
        else:
            player = WHITE
        action_index = get_index(action_idx_list[loop], loop % 3)
        for i in range(len(action_index)):
            board[action_index[i][0], action_index[i][1]] = player

    value = 0
    for x in range(11):
        for y in range(11):
            sequence_of_four = [board[x, y]]
            location_value = 0
            # 方向向下
            for i in range(1, 4):
                if in_board(x + i, y):
                    sequence_of_four.append(board[x + i, y])
            location_value += grade(sequence_of_four)
            sequence_of_four = [board[x, y]]
            # 方向向左 x不变 y变大
            for i in range(1, 4):
                if in_board(x, y + i):
                    sequence_of_four.append(board[x, y + i])
            location_value += grade(sequence_of_four)
            sequence_of_four = [board[x, y]]
            # 方向左下
            for i in range(1, 4):
                if in_board(x + i, y + i):
                    sequence_of_four.append(board[x + i, y + i])
            location_value += grade(sequence_of_four)
            sequence_of_four = [board[x, y]]
            # 方向右下
            for i in range(1, 4):
                if in_board(x + i, y - i):
                    sequence_of_four.append(board[x + i, y - i])
            location_value += grade(sequence_of_four)
            value += location_value
    return value

def getAction(black_board, white_board):
    # 从根节点开始展开，求MAX值，注意：返回值中下标为1的项才是行动内容
    return _getMax(black_board, white_board)[1]

def _getMax(black_board, white_board, depth=0, alpha=-float('inf'), beta=float('inf')):
    # 如果深度超限，或者无法继续展开，则返回当前状态的评价值
    legalActions = get_next_action(black_board, white_board)
    if depth == DEPTH or len(legalActions) == 0:
        return Value(black_board, white_board), None
    # 否则，就继续往下遍历吃豆人可能的下一步
    maxVal = None
    bestAction = None
    for action in legalActions:
        dict_action = {"x":action[0], "y":action[1]}
        value = _getMin(black_board, update_board(white_board, dict_action), depth+1, alpha, beta)[0]
        if value is not None and (maxVal == None or value > maxVal):
            maxVal = value
            bestAction = action
        # 按照α-β剪枝算法，如果v>β，则直接返回v
        if value is not None and value > beta:
            return value, action
        # 按照α-β剪枝算法，这里还需要更新α的值
        if value is not None and value > alpha:
            alpha = value
    return maxVal, bestAction

def _getMin(black_board, white_board, depth=0, alpha=-float('inf'), beta=float('inf')):
    # 如果深度超限，或者无法继续展开，则返回当前状态的评价值
    legalActions = get_next_action(black_board, white_board)
    if depth == DEPTH or len(legalActions) == 0:
        return Value(black_board, white_board), None
    # 否则，就继续往下遍历当前鬼怪可能的下一步
    minVal = None
    bestAction = None
    for action in legalActions:
        dict_action = {"x": action[0], "y": action[1]}
        value = _getMax(update_board(black_board, dict_action), white_board, depth+1, alpha, beta)[0]
        if value is not None and (minVal == None or value < minVal):
            minVal = value
            bestAction = action
        # 按照α-β剪枝算法，如果v>β，则直接返回v
        if value is not None and value < alpha:
            return value, action
        # 按照α-β剪枝算法，这里还需要更新α的值
        if value is not None and value < beta:
            beta = value
    return minVal, bestAction

if __name__ == '__main__':

    black_board = Board()
    white_board = Board()
    # 解析读入的JSON
    full_input = {"requests":[{"x":0,"y":0},{"x":1,"y":0},{"x":10,"y":0},{"x":0,"y":10},{"x":10,"y":10}],"responses":[{"x":2,"y":0},{"x":4,"y":0},{"x":5,"y":0},{"x":6,"y":0}]}
    # 分析自己收到的输入和自己过往的输出，并恢复状态
    rival_trajectory = full_input["requests"]
    my_trajectory = full_input["responses"]

    if (int(rival_trajectory[0]['x']) == -1):
        # L[1:] 从第二个元素开始截取列表
        rival_trajectory = rival_trajectory[1:]

    for i in range(len(rival_trajectory)):
        update_board(black_board, rival_trajectory[i])
    for i in range(len(my_trajectory)):
        update_board(white_board, my_trajectory[i])

    best_move = getAction(black_board, white_board)

    x = best_move[1]
    y = best_move[0]

    # TODO: 作出决策并输出
    my_action = { "x": x, "y": y }

    print(json.dumps({
        "response": my_action,
    }))
