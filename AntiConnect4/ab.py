import json
import random
import itertools
import numpy as np

board_size = 11
BLACK = 1
WHITE = -1
EMPTY = 0

DEPTH = 1
policy = "Around"

BLACK_WIN = 10
WHITE_WIN = 11

CX = [-1, 1, 0, 0, 1, -1, -1, 1]
CY = [0, 0, -1, 1, 1, -1, 1, -1]

def connect_len(board, color, loc_x, loc_y, dx, dy):
    # 一个加、一个减
    x = loc_x
    y = loc_y
    connect_num = 1

    while True:
        x += dx
        y += dy
        if x > 10 or x < 0 or y > 10 or y < 0:
            break # 代表这个方向超出范围
        if board[x][y] == color:
            connect_num += 1
        else:
            break

    x = loc_x
    y = loc_y
    while True:
        x -= dx
        y -= dy
        if x > 10 or x < 0 or y > 10 or y < 0:
            break # 代表这个方向超出范围
        if board[x][y] == color:
            connect_num += 1
        else:
            break
    return connect_num

def game_end(board):
    # 得到黑白棋的序列
    black_list = []
    white_list = []

    for i in range(11):
        for j in range(11):
            if board[i][j] == BLACK:
                black_list.append([i, j])
            elif board[i][j] == WHITE:
                white_list.append([i, j])

    for i in range(len(black_list)):
        # 判断黑棋有没有连成四子
        color = BLACK
        for j in range(4):
            dx = CX[j * 2]
            dy = CY[j * 2]
            connect_num = connect_len(board, color, black_list[i][0], black_list[i][1], dx, dy)
            if connect_num >= 4: # 连成
                return WHITE_WIN

    for i in range(len(white_list)):
        # 判断白棋有没有连成四子
        color = WHITE
        for j in range(4):
            dx = CX[j * 2]
            dy = CY[j * 2]
            connect_num = connect_len(board, color, white_list[i][0], white_list[i][1], dx, dy)
            if connect_num >= 4: # 连成
                return BLACK_WIN
    return False # 没有结束

class Board():
    def __init__(self, rival_trajectory, my_trajectory):
        self.search_num = 1
        self.count = 0

        # 二维的，没有reshape过
        self.board = np.zeros((board_size, board_size), dtype=int)

        for i in range(len(rival_trajectory)):
            x = int(rival_trajectory[i]['y'])
            y = int(rival_trajectory[i]['x'])
            self.board[x][y] = BLACK
        self.count += len(rival_trajectory)

        for i in range(len(my_trajectory)):
            x = int(my_trajectory[i]['y'])
            y = int(my_trajectory[i]['x'])
            self.board[x][y] = WHITE
        self.count += len(my_trajectory)

def in_board(x, y):
        # 判断[x, y]是否在棋盘内
        if x < 0 or x > 10 or y < 0 or y > 10:
            return False
        return True

def grade(four_list, player):
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
        return 10000000 * player
    elif white_num == 3 and empty_num == 1:
        return 100000 * player
    elif white_num == 2 and empty_num == 2:
        return 1000 * player
    elif white_num == 1 and empty_num == 3:
        return 10 * player
    elif white_num != 0 and black_num != 0:
        return 0
    elif black_num == 1 and empty_num == 3:
        return -10 * player
    elif black_num == 2 and empty_num == 2:
        return -1000 * player
    elif black_num == 3 and empty_num == 1:
        return -100000 * player
    elif black_num == 4:
        return -10000000 * player
    else:
        return 0

def Value(board, player):
    value = 0
    for x in range(board_size):
        for y in range(board_size):
            sequence_of_four = [board[x, y]]
            location_value = 0
            # 方向向上
            for i in range(1, 4):
                if in_board(x - i, y):
                    sequence_of_four.append(board[x - i, y])
            location_value += grade(sequence_of_four, player)
            sequence_of_four = [board[x, y]]
            # 方向向下
            for i in range(1, 4):
                if in_board(x + i, y):
                    sequence_of_four.append(board[x + i, y])
            location_value += grade(sequence_of_four, player)
            sequence_of_four = [board[x, y]]
            # 方向向右 x不变 y变小
            for i in range(1, 4):
                if in_board(x, y - i):
                    sequence_of_four.append(board[x, y - i])
            location_value += grade(sequence_of_four, player)
            sequence_of_four = [board[x, y]]
            # 方向向左 x不变 y变大
            for i in range(1, 4):
                if in_board(x, y + i):
                    sequence_of_four.append(board[x, y + i])
            location_value += grade(sequence_of_four, player)
            sequence_of_four = [board[x, y]]
            # 方向右上
            for i in range(1, 4):
                if in_board(x - i, y + i):
                    sequence_of_four.append(board[x - i, y + i])
            location_value += grade(sequence_of_four, player)
            sequence_of_four = [board[x, y]]
            # 方向右下
            for i in range(1, 4):
                if in_board(x + i, y + i):
                    sequence_of_four.append(board[x + i, y + i])
            location_value += grade(sequence_of_four, player)
            sequence_of_four = [board[x, y]]
            # 方向左上
            for i in range(1, 4):
                if in_board(x - i, y - i):
                    sequence_of_four.append(board[x - i, y - i])
            location_value += grade(sequence_of_four, player)
            # 方向左下
            for i in range(1, 4):
                if in_board(x + i, y - i):
                    sequence_of_four.append(board[x + i, y - i])
            location_value += grade(sequence_of_four, player)

            value += location_value
    return value

def get_next_board(board, action, player):
    # 将子节点的棋局拓展
    new_board = board.copy()
    new_board[action[0]][action[1]] = player
    return new_board

def get_next_action0(board, player):
    # action_list是当前局面可以的落子位置，默认自己是白棋
    action_list = []
    # 0 到 4
    layer = 0

    while layer <= 4:
        # print(layer)
        # 第一行 x不变，y增大
        for y in range(layer, board_size-layer):
            if board[layer][y] == EMPTY:
                new_board = board.copy()
                new_board[layer][y] = player
                if game_end(new_board) == False:
                    action_list.append([layer, y])

        # 最后一行 x不变，y增大
        for y in range(layer, board_size-layer):
            if board[board_size-layer-1][y] == EMPTY:
                new_board = board.copy()
                new_board[board_size-layer-1][y] = player
                if game_end(new_board) == False:
                    action_list.append([board_size-layer-1, y])

        # 第一列 y不变，x增大
        for x in range(layer+1, board_size-layer-1):
            if board[x][layer] == EMPTY:
                new_board = board.copy()
                new_board[x][layer] = player
                if game_end(new_board) == False:
                    action_list.append([x, layer])

        # 最后一列 y不变，x增大
        for x in range(layer+1, board_size-layer-1):
            if board[x][board_size-layer-1] == EMPTY:
                new_board = board.copy()
                new_board[x][board_size-layer-1] = player
                if game_end(new_board) == False:
                    action_list.append([x, board_size-layer-1])

        layer += 1
        if len(action_list) != 0:
            break
    return action_list

def get_next_action1(board, player):
    # action_list是当前局面可以的落子位置，默认自己是白棋
    action_list = []
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == EMPTY:
                new_board = board.copy()
                new_board[i][j] = player
                if game_end(new_board) == False:
                    action_list.append([i, j])
    return action_list

def getAction(board):
    # 从根节点开始展开，求MAX值，注意：返回值中下标为1的项才是行动内容
    return _getMax(board)[1]

def _getMax(board, depth=0, alpha=-float('inf'), beta=float('inf')):
    # 如果深度超限，或者无法继续展开，则返回当前状态的评价值
    if policy == "Around":
        legalActions = get_next_action0(board, WHITE)
    elif policy == "Global":
        legalActions = get_next_action1(board, WHITE)
    if depth == DEPTH or len(legalActions) == 0:
        return Value(board, BLACK), None
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

    full_input = json.loads(input())
    rival_trajectory = full_input["requests"]
    my_trajectory = full_input["responses"]

    if (int(rival_trajectory[0]['x']) == -1):
        # L[1:] 从第二个元素开始截取列表
        rival_trajectory = rival_trajectory[1:]

    board = Board(rival_trajectory, my_trajectory)

    # 三个阶段 0-40  41-80  81-120
    if board.count >= 90:
        policy = "Global"

    if board.count <= 100:
        DEPTH = 1
    elif board.count <=110:
        DEPTH = 2
    elif board.count <= 115:
        DEPTH = 7

    best_move = getAction(board.board)
    x = best_move[1]
    y = best_move[0]

    # TODO: 作出决策并输出
    my_action = { "x": x, "y": y }

    print(json.dumps({
        "response": my_action
    }))
    exit(0)