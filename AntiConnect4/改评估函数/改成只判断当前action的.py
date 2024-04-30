import json
import random
import itertools
import numpy as np

CX = [-1, 1, 0, 0, 1, -1, -1, 1]
CY = [0, 0, -1, 1, 1, -1, 1, -1]

class Node():
    def __init__(self, board, action, parent):
        self.value = 0
        self.board = board

        self.action = action
        self.child = []
        self.parent = parent

        self.max_child_value = -10000000

class Game():
    def __init__(self):
        self.board_size = 11
        self.BLACK = 1
        self.WHITE = -1
        self.EMPTY = 0

        self.BLACK_WIN = 10
        self.WHITE_WIN = 11

        self.search_num = 1

        self.root = None

        # 二维的，没有reshape过
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)

    # 默认把AI当成后手 对手：黑棋  自己：白棋
    def init_board(self, rival_trajectory, my_trajectory):
        for i in range(len(rival_trajectory)):
            y = int(rival_trajectory[i]['x'])
            x = int(rival_trajectory[i]['y'])
            self.board[x][y] = self.BLACK

        for i in range(len(my_trajectory)):
            y = int(my_trajectory[i]['x'])
            x = int(my_trajectory[i]['y'])
            self.board[x][y] = self.WHITE

    def connect_len(self, color, loc_x, loc_y, dx, dy):
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
            if self.board[x][y] == color:
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
            if self.board[x][y] == color:
                connect_num += 1
            else:
                break
        return connect_num

    def game_end(self, board):
        # 得到黑白棋的序列
        black_list = []
        white_list = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == self.BLACK:
                    black_list.append([i, j])
                elif board[i][j] == self.WHITE:
                    white_list.append([i, j])

        for i in range(len(black_list)):
            # 判断黑棋有没有连成四子
            color = self.BLACK
            for j in range(4):
                dx = CX[j * 2]
                dy = CY[j * 2]
                connect_num = self.connect_len(color, black_list[i][0], black_list[i][1], dx, dy)
                if connect_num >= 4: # 连成
                    return self.WHITE_WIN

        for i in range(len(white_list)):
            # 判断白棋有没有连成四子
            color = self.WHITE
            for j in range(4):
                dx = CX[j * 2]
                dy = CY[j * 2]
                connect_num = self.connect_len(color, white_list[i][0], white_list[i][1], dx, dy)
                if connect_num >= 4: # 连成
                    return self.BLACK_WIN
        return False # 没有结束

    def in_board(self, x, y):
        # 判断[x, y]是否在棋盘内
        if x < 0 or x > 10 or y < 0 or y > 10:
            return False
        return True

    def grade(self, four_list):
        # 每个位置的value
        black_num = 0
        white_num = 0
        empty_num = 0

        for i in range(len(four_list)):
            if four_list[i] == self.BLACK:
                black_num += 1
            elif four_list[i] == self.WHITE:
                white_num += 1
            else:
                empty_num += 1

        # 四个白子
        if white_num == 4:
            return -10000000
        elif white_num == 1 and black_num == 3:
            return 1000
        elif white_num == 1 and black_num == 2:
            return 800
        elif white_num == 1 and black_num == 1:
            return 400
        elif white_num == 1 and black_num == 0:
            return 100

        elif white_num == 2 and black_num == 2:
            return 600
        elif white_num == 2 and black_num == 1:
            return 300
        elif white_num == 2 and black_num == 0:
            return 50

        elif white_num == 3 and black_num == 1:
            return 0
        elif white_num == 3 and black_num == 0:
            return -50
        else:
            return 0

    def value(self, node):
        # 判断当前结点的价值
        value = 0
        x = node.action[0]
        y = node.action[1]

        # 方向向下
        sequence_of_four = [node.board[x, y]]
        for i in range(1, 4):
            if self.in_board(x + i, y):
                sequence_of_four.append(node.board[x + i, y])
        value += self.grade(sequence_of_four)

        # 方向向上
        sequence_of_four = [node.board[x, y]]
        for i in range(1, 4):
            if self.in_board(x - i, y):
                sequence_of_four.append(node.board[x - i, y])
        value += self.grade(sequence_of_four)

        # 方向向左 x不变 y变大
        sequence_of_four = [node.board[x, y]]
        for i in range(1, 4):
            if self.in_board(x, y + i):
                sequence_of_four.append(node.board[x, y + i])
        value += self.grade(sequence_of_four)

        # 方向向右 x不变 y变小
        sequence_of_four = [node.board[x, y]]
        for i in range(1, 4):
            if self.in_board(x, y - i):
                sequence_of_four.append(node.board[x, y - i])
        value += self.grade(sequence_of_four)

        # # 方向左下
        # sequence_of_four = [node.board[x, y]]
        # for i in range(1, 4):
        #     if self.in_board(x + i, y + i):
        #         sequence_of_four.append(node.board[x + i, y + i])
        # value += self.grade(sequence_of_four)
        #
        # # 方向右下
        # sequence_of_four = [node.board[x, y]]
        # for i in range(1, 4):
        #     if self.in_board(x + i, y - i):
        #         sequence_of_four.append(node.board[x + i, y - i])
        # value += self.grade(sequence_of_four)

        return value

    def get_next_state(self, board, x, y):
        # 将子节点的棋局拓展
        new_board = board.copy()
        new_board[x][y] = self.WHITE
        return new_board

    def expand_child(self, parent):
        # 扩展该结点，就是要生成他的子节点，首先要选择相应的方向
        child = []
        # action_list是当前局面可以的落子位置，默认自己是白棋
        action_list = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if parent.board[i][j] == self.EMPTY:
                    action_list.append([i, j])
        return action_list

    def tree_search(self):
        self.root = Node(board=self.board, action=[], parent=None)
        # 得到当前局面的评估值

        for loop in range(self.search_num):
            current = self.root  # 1代表棋盘
            # 所有的动作序列
            action_list = self.expand_child(current)

            # 将action_list中的动作真正下在棋盘
            for i in range(len(action_list)):
                new_board = self.get_next_state(current.board, x=action_list[i][0], y=action_list[i][1])
                # 扩展该结点
                leaf_node = Node(board=new_board, action=action_list[i], parent=current)
                value = self.value(leaf_node)
                leaf_node.value = value

                if value > current.max_child_value:
                    current.max_child_value = value
                current.child.append(leaf_node)

            # 目标检测
            out_index = []
            for i in range(len(current.child)):
                # 黑棋获胜，AI自己下错，删除该节点
                if self.game_end(current.child[i].board) == self.BLACK_WIN:
                    out_index.append(i)
                # 白棋获胜
                elif self.game_end(current.child[i].board) == self.WHITE_WIN:
                    pass
                else:  # 游戏继续
                    pass
            for i in range(len(out_index)):
                del current.child[out_index[i] - i]

        # 现在返回所有优先级最好的动作序列列表
        best_move = []
        for i in range(len(self.root.child)):
            if self.root.child[i].value == self.root.max_child_value:
                best_move.append(current.child[i].action)
        return best_move

if __name__ == '__main__':
    game = Game()

    # 解析读入的JSON
    full_input = json.loads(input())

    if "data" in full_input:
        my_data = full_input["data"];  # 该对局中，上回合该Bot运行时存储的信息
    else:
        my_data = None

    # 分析自己收到的输入和自己过往的输出，并恢复状态
    rival_trajectory = full_input["requests"]
    my_trajectory = full_input["responses"]

    if (int(rival_trajectory[0]['x']) == -1):
        # L[1:] 从第二个元素开始截取列表
        rival_trajectory = rival_trajectory[1:]

    game.init_board(rival_trajectory, my_trajectory)
    best_move = game.tree_search()

    if len(best_move) == 0:
        print(json.dumps({
            "response": { "x": '', "y": '' }
        }))
        exit(0)

    # 产生 0 到 len(best_move) - 1 的一个整数型随机数
    my_action = random.randint(0, len(best_move) - 1)

    x = best_move[my_action][1]
    y = best_move[my_action][0]

    # TODO: 作出决策并输出
    my_action = { "x": x, "y": y }

    print(json.dumps({
        "response": my_action
    }))
    exit(0)