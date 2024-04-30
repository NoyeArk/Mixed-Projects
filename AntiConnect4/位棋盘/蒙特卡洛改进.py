import json
import copy
import math
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
    # 判断左下
    row_list = []
    # 0010
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

class TreeNode(object):

    def __init__(self, board=None, parent=None, player=None):
        self.board = board #board：[black_board, white_board]
        self.idx = 0

        self.action = get_next_action(board[0], board[1])


        self.quality = 0
        self.visit_times = 1
        self.UCB_value = 0
        self.player = player
        self.is_expand = False

        self.parent = parent
        self.child = []

    def set_parent(self,parent):
        self.parent = parent

    def set_child(self,child):
        self.child.append(child)

    def child_Num(self):
        return self.information.childNum

    def Calc_UCB(self):
        # c = 0.000001
        c = 1 / math.sqrt(2)
        quality = self.quality
        visit_times = self.visit_times
        father_visit_value = self.parent.visit_times
        # print('quality:', quality)
        # print('visit_times:', visit_times)
        # print('father_visit_value:', father_visit_value)
        self.UCB_value = quality / visit_times + c * math.sqrt(2 * math.log(father_visit_value) / visit_times)
        # print(self.UCB_value)

def Selection(node):
    # 没有全部拓展
    choice = random.randint(0, len(node.action) - 1)
    for child in node.child:
        if child.idx == choice:
            return Selection(child)
    return choice

def Expansion(parent, action_idx, player):
    action = {"x": parent.action[action_idx][0], "y": parent.action[action_idx][1]}
    if player == WHITE:
        new_board = [parent.board[0], update_board(parent.board[1], action)]
    if player == BLACK:
        new_board = [update_board(parent.board[0], action), parent.board[1]]
    leaf_node = TreeNode(new_board, parent, BLACK if parent.player == WHITE else WHITE)
    leaf_node.idx = action_idx
    leaf_node.is_expand = True
    parent.child.append(leaf_node)
    return leaf_node

def selection(parent):
    '''
        我觉得在这里选择是有问题的，因为如果你的第一层，如果进行过了best_node，那么你后来的模仿其实都是
        没有意义的，因为它后来只会模拟你选择过的结点。
    '''
    # 全部拓展
    if len(parent.action) == len(parent.child):
        return parent
    # 没有全部拓展
    choice = random.randint(0, len(parent.action) - 1)
    for child in parent.child:
        if child.idx == choice:
            return selection(child)
    action = {"x": parent.action[choice][0], "y": parent.action[choice][1]}
    if parent.player == WHITE: # 如果父节点是白方，那子节点就是黑方，更新黑方棋盘
        new_board = [update_board(parent.board[0], action), parent.board[1]]
    if parent.player == BLACK: # 如果父节点是黑方，那子节点就是白方
        new_board = [parent.board[0], update_board(parent.board[1], action)]
    leaf_node = TreeNode(new_board, parent, BLACK if parent.player == WHITE else WHITE)
    leaf_node.idx = choice
    leaf_node.is_expand = True
    parent.child.append(leaf_node)
    return leaf_node

def Simulation(node):
    value = 0
    # 一开始如果是白方 变成 黑方
    real_player = node.player
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
            if real_player == WHITE: # 如果黑棋连成4个，也就是白棋赢了，我方是白棋视角
                value = 1
            elif real_player == BLACK:
                value = 0
            break
        elif game_end(white_board) == GAME_END:
            if real_player == WHITE:  # 如果白棋连成4个，也就是黑棋赢了，我方是白棋视角
                value = 0
            elif real_player == BLACK:
                value = 1
            break
    node.quality += value

def Backup(node):
    current = node
    value = current.quality
    flag = 1
    while current.parent != None:
        flag = flag * -1
        current.parent.visit_times += 1
        if flag == 1: # 代表爷爷和孙子的关系,相同的value
            current.parent.quality += value
        elif flag == -1: # 代表父亲和儿子的关系,不同的value
            if value == 1:
                current.parent.quality += 0
            elif value == 0:
                current.parent.quality += 1
        current.Calc_UCB()
        current = current.parent

def best_node(root):
    Max_UCB_Node = TreeNode(root.board)
    for Child in root.child:
        # print(Child.UCB_value)
        if Max_UCB_Node.UCB_value < Child.UCB_value:
            Max_UCB_Node = Child
    return Max_UCB_Node

def max_visit(node):
    Max_visit = TreeNode(node.board)
    for Child in node.child:
        print(Child.visit_times)
        if Max_visit.visit_times < Child.visit_times:
            Max_UCB_Node = Child
    return Max_UCB_Node

def MCTS(black_board, white_board):
    root = TreeNode([black_board, white_board], None, player=BLACK)
    node = TreeNode([black_board, white_board], root, player=BLACK)
    root.child.append(node)

    for loop in range(5000):
        # print(loop) # 从0开始，0 2 4 6都是我方视角，白棋
        if loop == 3700:
            print()

        leaf_node = selection(parent=node)
        if leaf_node == node:
            node = best_node(node)
        else:
            Simulation(leaf_node)
            Backup(leaf_node)
        # Simulation(leaf_node)
        # Backup(leaf_node)


    best = max_visit(root.child[0])
    # best = best_node(root.child[0])
    # print('max_visit:', best.visit_times)
    # print('max_value:', best.UCB_value)
    return root.action[best.idx]

if __name__ == '__main__':

    black_board = Board()
    white_board = Board()
    # 解析读入的JSON
    full_input = {"requests":[{"x":-1,"y":-1},{"x":0,"y":10},{"x":10,"y":10},{"x":0,"y":1},{"x":1,"y":0},{"x":9,"y":0},{"x":10,"y":1},{"x":0,"y":9},{"x":10,"y":9},{"x":2,"y":10},{"x":8,"y":10},{"x":0,"y":3},{"x":7,"y":0},{"x":7,"y":10},{"x":3,"y":0},{"x":10,"y":3},{"x":10,"y":4},{"x":4,"y":10},{"x":1,"y":1},{"x":0,"y":7},{"x":1,"y":9},{"x":9,"y":1},{"x":1,"y":8},{"x":8,"y":9},{"x":9,"y":6},{"x":3,"y":9},{"x":5,"y":1},{"x":3,"y":1},{"x":1,"y":3},{"x":1,"y":4},{"x":4,"y":9},{"x":9,"y":7},{"x":9,"y":5},{"x":9,"y":3},{"x":6,"y":9},{"x":8,"y":2},{"x":1,"y":5},{"x":2,"y":2},{"x":3,"y":8},{"x":7,"y":2},{"x":4,"y":2},{"x":7,"y":6}],"responses":[{"x":0,"y":0},{"x":10,"y":0},{"x":1,"y":10},{"x":9,"y":10},{"x":2,"y":0},{"x":8,"y":0},{"x":0,"y":2},{"x":10,"y":2},{"x":5,"y":10},{"x":0,"y":6},{"x":10,"y":6},{"x":3,"y":10},{"x":6,"y":0},{"x":0,"y":4},{"x":4,"y":0},{"x":5,"y":0},{"x":10,"y":5},{"x":6,"y":10},{"x":0,"y":8},{"x":0,"y":5},{"x":10,"y":8},{"x":9,"y":9},{"x":9,"y":4},{"x":2,"y":9},{"x":7,"y":9},{"x":4,"y":1},{"x":2,"y":1},{"x":1,"y":2},{"x":6,"y":1},{"x":7,"y":1},{"x":8,"y":1},{"x":9,"y":8},{"x":9,"y":2},{"x":5,"y":9},{"x":1,"y":7},{"x":1,"y":6},{"x":8,"y":4},{"x":2,"y":4},{"x":5,"y":2},{"x":2,"y":5},{"x":8,"y":6}]}

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

    action_list = get_next_action(black_board, white_board)
    # print(action_list)
    # print(len(action_list))

    best_move = MCTS(black_board, white_board)

    x = best_move[1]
    y = best_move[0]

    # TODO: 作出决策并输出
    my_action = { "x": x, "y": y }

    print(json.dumps({
        "response": my_action,
    }))
