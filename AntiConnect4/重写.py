import json
import random
import itertools
import numpy as np

DIRECTION[4][4] = {{0, -1, 0, 1},
                   {-1, 0, 1, 0},
                   {-1, -1, 1, 1},
                   {-1, 1, 1, -1}};

PosValue = [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
            [5, 4, 3, 3, 3, 3, 3, 3, 3, 4, 5],
            [5, 4, 3, 2, 2, 2, 2, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1, 1, 1, 2, 3, 4, 5],
            [5, 4, 3, 2, 2, 2, 2, 2, 3, 4, 5],
            [5, 4, 3, 3, 3, 3, 3, 3, 3, 4, 5],
            [5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5],
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]]
BLACK = 1
WHITE = -1
EMPTY = 0
BLACK_WIN = 10
WHITE_WIN = 11

NOT_END = -40
GAME_END = 40
search_num = 1

CHENG_4_SCORE = -5000000 # 成4
HUO_3_SCORE = -100000 # 活3
CHONG_3_SCORE = -10000 # 冲3
DAN_HUO_2_SCORE = -8000 # 单活2
TIAO_HUO_3_SCORE = -7000 # 跳活2

CHENG_4_STRING = "1111"
HUO_3_STRING = "01110"
CHONG_3_STRING_1_1 = "0111-"
CHONG_3_STRING_1_2 = "-1110"
CHONG_3_STRING_2_1 = "1011"
CHONG_3_STRING_2_2 = "1101"
DAN_HUO_2_STRING = "0110"
TIAO_HUO_2_STRING = "101"

MIAN_3_SCORE = 500;
HUO_2_SCORE = 50;
MIAN_2_SCORE = 10;
CHENG_5_STRING = "11111";
HUO_4_STRING = "011110";
CHONG_4_STRING_1_1 = "01111-";
CHONG_4_STRING_1_2 = "-11110";
CHONG_4_STRING_2_1 = "10111";
CHONG_4_STRING_2_2 = "11101";
CHONG_4_STRING_3 = "11011";
DAN_HUO_3_STRING = "01110";
TIAO_HUO_3_STRING_1_1 = "1011";
TIAO_HUO_3_STRING_1_2 = "1101";
MIAN_3_1_1 = "00111-";
MIAN_3_1_2 = "-11100";
MIAN_3_2_1 = "01011-";
MIAN_3_2_2 = "-11010";
MIAN_3_3_1 = "01101-";
MIAN_3_3_2 = "-10110";
MIAN_3_4_1 = "10011";
MIAN_3_4_2 = "11001";
MIAN_3_5 = "10101";
MIAN_3_6 = "-01110-";
HUO_2_STRING_1 = "001100";
HUO_2_STRING_2 = "01010";
HUO_2_STRING_3 = "1001";
MIAN_2_1_1 = "00011-";
MIAN_2_1_2 = "-11000";
MIAN_2_2_1 = "00101-";
MIAN_2_2_2 = "-10100";
MIAN_2_3_1 = "01001-";
MIAN_2_3_2 = "-10010";
MIAN_2_4 = "10001";

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
            index.append([x, y])
    return index

def f(loc, color):
    s = ''
    x = loc[0]
    y = loc[1]
    score = 0
    weight = PosValue[x][y] # 权重
    # 分别构造四个方向的局面的字符串表示
    for dir in range(4):
        # 计算该方向上的起始点坐标
        rBegin = x + DIRECTION[dir][0] * 4
        cBegin = y + DIRECTION[dir][1] * 4
        # 坐标递增的方向
        rDir = DIRECTION[dir][2]
        cDir = DIRECTION[dir][3]
        # 计算该方向上的终止点坐标
        rEnd = x + rDir * 4
        cEnd = y + cDir * 4
        # 当行列没到终点的时候（表示没有收集齐9个点），循环
        r = rBegin
        c = cBegin
        while (r != rEnd | c != cEnd):
            # 如果这个点没有超过棋盘范围，是自己颜色就记为1，是空记为0，是对手记为-，超过棋盘的点记为 #
            if in_board(r, c):
                if (chessBoard[r][c] == color):
                    s += "1"
                elif (chessBoard[r][c] == EMPTY):
                    s += "0"
                else:
                    s += "-"
            else:
                s += "#"
            r += rDir
            c += cDir



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
        print(action_index)
        for i in range(len(action_index)):
            board[action_index[i][0], action_index[i][1]] = player

    alley = 0
    enemy = 0
    for x in range(11):
        for y in range(11):
            if board[x][y] == WHITE:
                alley += f([x, y], WHITE)
            else:
                enemy += f([x, y], WHITE)
    heuristic = 10 * alley - enemy
    return heuristic

def getAction(black_board, white_board):
    # 从根节点开始展开，求MAX值，注意：返回值中下标为1的项才是行动内容
    return _getMax(board)[1]

def _getMax(black_board, white_board, depth=0, alpha=-float('inf'), beta=float('inf')):
    # 如果深度超限，或者无法继续展开，则返回当前状态的评价值
    legalActions = get_next_action(black_board, white_board)
    if depth == DEPTH or len(legalActions) == 0:
        return Value(black_board, white_board), None
    # 否则，就继续往下遍历吃豆人可能的下一步
    maxVal = None
    bestAction = None
    for action in legalActions:
        # 考虑只有一个吃豆人的情况，直接求其MIN分支的评价值，agentIndex从1开始遍历所有鬼怪
        value = _getMin(get_next_board(board, action, WHITE), depth+1, alpha, beta)[0]
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

def _getMin(board, depth=0, alpha=-float('inf'), beta=float('inf')):
    # 如果深度超限，或者无法继续展开，则返回当前状态的评价值
    if policy == "Around":
        legalActions = get_next_action0(board, BLACK)
    elif policy == "Global":
        legalActions = get_next_action1(board, BLACK)
    if depth == DEPTH or len(legalActions) == 0:
        return Value(board, WHITE), None
    # 否则，就继续往下遍历当前鬼怪可能的下一步
    minVal = None
    bestAction = None
    for action in legalActions:
        # 考虑只有一个吃豆人的情况，直接求其MIN分支的评价值，agentIndex从1开始遍历所有鬼怪
        value = _getMax(get_next_board(board, action, BLACK), depth+1, alpha, beta)[0]
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
    full_input = {
        "requests": [{"x": -1, "y": -1},  {"x": 7, "y": 9}, {"x": 2, "y": 4}, {"x": 0, "y": 7},
                     {"x": 9, "y": 2}, {"x": 4, "y": 8}, {"x": 5, "y": 10}, {"x": 2, "y": 1}, {"x": 1, "y": 5},
                     {"x": 10, "y": 5}, {"x": 3, "y": 2}, {"x": 8, "y": 4}, {"x": 3, "y": 1}, {"x": 2, "y": 6},
                     {"x": 6, "y": 2}, {"x": 6, "y": 9}, {"x": 4, "y": 7}, {"x": 7, "y": 6}, {"x": 0, "y": 4},
                     {"x": 10, "y": 9}, {"x": 5, "y": 8}, {"x": 5, "y": 0}, {"x": 9, "y": 7}, {"x": 9, "y": 10},
                     {"x": 6, "y": 7}, {"x": 8, "y": 7}, {"x": 8, "y": 9}, {"x": 9, "y": 9}, {"x": 4, "y": 5},
                     {"x": 2, "y": 2}],
        "responses": [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 0, "y": 10}, {"x": 10, "y": 10}, {"x": 0, "y": 9},
                      {"x": 0, "y": 8}, {"x": 0, "y": 6}, {"x": 4, "y": 0}, {"x": 4, "y": 10}, {"x": 6, "y": 10},
                      {"x": 10, "y": 4}, {"x": 10, "y": 6}, {"x": 9, "y": 0}, {"x": 0, "y": 1}, {"x": 1, "y": 0},
                      {"x": 1, "y": 10}, {"x": 8, "y": 10}, {"x": 10, "y": 2}, {"x": 0, "y": 5}, {"x": 10, "y": 8},
                      {"x": 10, "y": 7}, {"x": 6, "y": 0}, {"x": 0, "y": 2}, {"x": 7, "y": 10}, {"x": 2, "y": 0},
                      {"x": 8, "y": 0}, {"x": 10, "y": 3}, {"x": 2, "y": 10}, {"x": 9, "y": 1}]}
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

    Value(black_board, white_board)

    action_list = get_next_action(black_board, white_board)

    best_move = random.randint(0, len(action_list) - 1)  # 产生 0 到 len(best_move) - 1 的一个整数型随机数

    x = action_list[best_move][1]
    y = action_list[best_move][0]

    # TODO: 作出决策并输出
    my_action = { "x": x, "y": y }

    print(json.dumps({
        "response": my_action,
    }))
