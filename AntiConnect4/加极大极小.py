import json
import random
import itertools
import numpy as np

board_size = 11
BLACK = 1
WHITE = -1
EMPTY = 0

DEPTH = 10

BLACK_WIN = 10
WHITE_WIN = 11

CX = [-1, 1, 0, 0, 1, -1, -1, 1]
CY = [0, 0, -1, 1, 1, -1, 1, -1]

def connect_len(board, color, loc_x, loc_y, dx, dy):
    '''
    :param color: 所要判断的棋子颜色
    :param loc_x: 所要判断的棋子x坐标
    :param loc_y: 所要判断的棋子y坐标
    :param dx: 所要判断的棋子连接方向x
    :param dy: 所要判断的棋子连接方向y
    :return: 返回的是连子的数量
    '''
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

class Node():
    def __init__(self, board, action, parent, value):
        self.value = value
        self.board = board

        self.child = []
        self.action = action
        self.parent = parent

        self.max_child_value = -10000000

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

def value(board):
    value = 0
    for x in range(board_size):
        for y in range(board_size):
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

def get_next_board(board, action):
    # 将子节点的棋局拓展
    new_board = board.copy()
    new_board[action[0]][action[1]] = WHITE
    return new_board

def get_next_action(board):
    # action_list是当前局面可以的落子位置，默认自己是白棋
    action_list = []
    for i in range(board_size):
        for j in range(board_size):
            if board[i][j] == EMPTY:
                action_list.append([i, j])
    return action_list

def expand_child(board, action, parent):
    new_board = board.copy()
    new_board[action[0], action[1]] = WHITE
    Value = value(new_board)
    leaf_node = Node(board=new_board, action=action, parent=parent, value=Value)

    return leaf_node


def tree_search(board):
    # 得到当前的根节点
    root = Node(board=board, action=[], parent=None, value=0)

    current = root
    # 下一步所有的动作序列
    action_list = get_next_action(current.board)
    for i in range(len(action_list)):
        leaf_node = expand_child(board=current.board, action=action_list[i], parent=current)
        current.child.append(leaf_node)

    # 目标检测-先把那些会导致自己输的action给删除
    out_index = []
    for i in range(len(current.child)):
        # 黑棋获胜，AI自己下错，删除该节点
        flag = game_end(current.child[i].board)
        if flag == BLACK_WIN:
            out_index.append(i)
        # 白棋获胜
        elif flag == WHITE_WIN:
            pass
        else:  # 游戏继续
            pass
    for i in range(len(out_index)):
        del current.child[out_index[i] - i]

    # 现在返回所有优先级最好的动作序列列表
    value_list = []
    for i in range(len(current.child)):
        value_list.append(current.child[i].value)
    max_value = max(value_list)  # 求列表最大值
    best_move = []
    for i in range(len(current.child)):
        if max_value == current.child[i].value:
            best_move.append(current.child[i].action)

    # 产生 0 到 len(best_move) - 1 的一个整数型随机数
    my_action = random.randint(0, len(best_move) - 1)

    return best_move[my_action]

def Max_value(board, layers, action):
    if game_end(board) == WHITE_WIN or layers >= DEPTH:
        return value(board), action
    v = -100000000
    action = []
    temp = []
    action_list = get_next_action(board)
    for i in range(len(action_list)):
        new_board = get_next_board(board, action_list[i])
        (new_value, temp) = Min_value(new_board, layers + 1, action_list[i])
        if new_value > v:
            v = new_value
            action = temp
    return (v, action)

def Min_value(board, layers, action):
    if game_end(board) == BLACK_WIN or layers >= DEPTH:
        return value(board), action
    v = 100000000
    action = []
    temp = []
    action_list = get_next_action(board)
    for i in range(len(action_list)):
        new_board = get_next_board(board, action_list[i])
        (new_value, temp) = Max_value(new_board, layers + 1, action_list[i])
        if new_value < v:
            v = new_value
            action = temp
    return (v, action)

if __name__ == '__main__':

    full_input = json.loads(input())
    rival_trajectory = full_input["requests"]
    my_trajectory = full_input["responses"]

    if (int(rival_trajectory[0]['x']) == -1):
        # L[1:] 从第二个元素开始截取列表
        rival_trajectory = rival_trajectory[1:]

    board = Board(rival_trajectory, my_trajectory)

    # tree search返回的是马上要下的动作序列
    if board.count >= 110:
        v, best_move = Max_value(board.board, 0, [])
    else:
        best_move = tree_search(board.board)


    x = best_move[1]
    y = best_move[0]

    # TODO: 作出决策并输出
    my_action = { "x": x, "y": y }

    print(json.dumps({
        "response": my_action
    }))
    exit(0)