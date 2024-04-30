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

search_num = 1

CX = [-1, 1, 0, 0, 1, -1, -1, 1]
CY = [0, 0, -1, 1, 1, -1, 1, -1]

class My_Queue():
    def __init__(self):
        self.priority_list = []
        self.action_list = []
        self.length = 0
        self.min_priority = 10000

    def swap(self, object1, object2):
        temp = object1
        object1 = object2
        object2 = temp

    def sort(self):
        for i in range(0, self.length - 1):
            for j in range(i + 1, self.length):
                if self.priority_list[i] > self.priority_list[j]:
                    temp = self.priority_list[i]
                    self.priority_list[i] = self.priority_list[j]
                    self.priority_list[j] = temp

                    temp = self.action_list[i]
                    self.action_list[i] = self.action_list[j]
                    self.action_list[j] = temp

    def put(self, priority, board):
        self.priority_list.append(priority)
        self.action_list.append(board)
        self.length += 1

        self.sort()
        self.min_priority = self.priority_list[0]

    def get(self):
        list = [self.priority_list[0], self.action_list[0]]
        del self.priority_list[0]
        del self.action_list[0]
        self.length -= 1
        return list

    def print(self):
        list = []
        for i in range(self.length):
            list.append([self.priority_list[i], self.action_list[i]])

        print(list)

class Game():
    def __init__(self):
        # 二维的，没有reshape过
        self.board = np.zeros((board_size, board_size), dtype=int)

    # 默认把AI当成后手 对手：黑棋  自己：白棋
    def init_board(self, rival_trajectory, my_trajectory):
        for i in range(len(rival_trajectory)):
            y = int(rival_trajectory[i]['x'])
            x = int(rival_trajectory[i]['y'])
            self.board[x][y] = BLACK

        for i in range(len(my_trajectory)):
            y = int(my_trajectory[i]['x'])
            x = int(my_trajectory[i]['y'])
            self.board[x][y] = WHITE

    def print_board(self, board):
        print()
        pretty_print_map = {
            1: '\x1b[0;31;40mB',  # b代表黑色
            0: '\x1b[0;31;43m.',  # .代表黄色
            -1: '\x1b[0;31;47mW',  # w代表白色
        }
        board = np.copy(board)
        # 原始棋盘内容
        raw_board_contents = []

        for i in range(board_size):
            # 🐍:row是行级的
            row = []
            for j in range(board_size):
                row.append(pretty_print_map[board[i, j]])
                row.append('\x1b[0m')

            raw_board_contents.append(' '.join(row))
        # 行标签 N~1
        row_labels = ['%2d' % i for i in range(1, board_size + 1, 1)]

        header_footer_rows = ['   ' + '  '.join('ABCDEFGHIJKLMNOPQRS'[:board_size]) + '    ']
        header_footer_rows_num = ['   ' + '1  2  3  4  5  6  7  8  9  10 11']

        row_labels_alpha = []
        Alpha = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
        for i in range(board_size):
            row_labels_alpha.append(Alpha[i])
        # 带标注的每一行的内容
        # 加了一个空格
        annotated_board_contents = [' '.join(r) for r in zip(row_labels, raw_board_contents, row_labels_alpha)]
        # 带标注的棋盘
        # itertools.chain将不同容器中的元素连接起来，便于遍历
        annotated_board = '\n'.join(
            itertools.chain(header_footer_rows_num, annotated_board_contents, header_footer_rows))
        print(annotated_board)

    def connect_len(self, color, loc_x, loc_y, dx, dy):
        # 一个加、一个减
        x = loc_x
        y = loc_y
        connect_num = 1
        # 这一块应该可以不用while把
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
        # 应该是要判断每个黑白棋各自是否连成4子
        # 先得到黑棋的序列
        black_list = []
        white_list = []
        for i in range(board_size):
            for j in range(board_size):
                if board[i][j] == BLACK:
                    black_list.append([i, j])
                elif board[i][j] == WHITE:
                    white_list.append([i, j])

        for i in range(len(black_list)):
            # 现在判断黑棋有没有连成四子
            color = BLACK
            for j in range(4):
                dx = CX[j * 2]
                dy = CY[j * 2]
                connect_num = self.connect_len(color, black_list[i][0], black_list[i][1], dx, dy)
                if connect_num >= 4: # 如果连成
                    return WHITE_WIN

        for i in range(len(white_list)):
            # 现在判断白棋有没有连成四子
            color = WHITE
            for j in range(4):
                dx = CX[j * 2]
                dy = CY[j * 2]
                connect_num = self.connect_len(color, white_list[i][0], white_list[i][1], dx, dy)
                if connect_num >= 4: # 如果连成
                    return BLACK_WIN
        return False # 没有结束

    def in_board(self, x, y):
        if x < 0 or x > 10 or y < 0 or y > 10:
            return False
        return True

    def f(self, action):
        # 我想的是用当前位置所处的3×3方格中同类棋子的数目
        # 应该是越少越好，同时也满足优先队列的条件
        x = action[0]
        y = action[1]

        color = WHITE
        same_color_num = 0
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                if self.in_board(i, j) and self.board[i][j] == color:
                    same_color_num += 1
        return same_color_num

    def get_next_state(self, board, x, y):
        new_board = board.copy()
        new_board[x][y] = WHITE
        return new_board

    def expand_child(self, root):
        # 扩展该结点，就是要生成他的子节点，首先要选择相应的方向
        child = []
        # action_list是当前局面可以的落子位置，默认自己是白棋
        action_list = []
        for i in range(board_size):
            for j in range(board_size):
                if root[i][j] == EMPTY:
                    action_list.append([i, j])

        return action_list

    def tree_search(self):
        # frontier是一个优先队列
        closed = []
        frontier = My_Queue()
        frontier.put(10000, self.board)

        for loop in range(search_num):
            # 先注释掉
            # if not frontier.empty():
            #     return False

            # 选择一个最好的叶子结点，并把该节点移出frontier
            current = frontier.get()[1]  # 1代表棋盘
            # print('此轮父节点：')
            self.print_board(current)

            # 肯定没有结束，扩展该节点
            # child_list是生成的下一步棋的所有可能的子棋盘
            action_list = self.expand_child(current)

            # child就只是为了自己看
            child = []
            for i in range(len(action_list)):
                new_board = self.get_next_state(current, x=action_list[i][0], y=action_list[i][1])
                child.append(new_board)

            print('新拓展的儿子结点：')
            for i in range(len(child)):
                game.print_board(child[i])

            # 目标检测
            # 不对，判断游戏结束应该是在自己模拟完一步之后在进行判断
            out_index = []
            for i in range(len(child)):
                # 黑棋获胜，AI自己下错了
                if game.game_end(child[i]) == BLACK_WIN:
                    out_index.append(i)
                # 白棋获胜，删除该结点
                elif game.game_end(child[i]) == WHITE_WIN:
                    pass
                else:  # 游戏继续
                    pass

            for i in range(len(out_index)):
                '''
                比如原来有40个，其中要把第38个给删了
                删了3个之后，就剩下了37个，下标最大为36，原本最大的40->36，相当于
                i = 0: 删除1个，
                '''
                del action_list[out_index[i]- i]

            # 将拓展的所有合法的child添加到frontier里
            for i in range(len(action_list)):
                # 用f()错位数来当作优先级，f越大，越不接近终局，越不优先
                priority = self.f(action_list[i])
                frontier.put(priority, action_list[i])
                # frontier.print()

        # 现在返回所有优先级最好的动作序列列表
        best_move = []
        for i in range(len(frontier.action_list)):
            if frontier.priority_list[i] == frontier.min_priority:
                best_move.append(frontier.action_list[i])

        return best_move

if __name__ == '__main__':
    game = Game()

    # 解析读入的JSON
    full_input = {"requests":[{"x":-1,"y":-1},{"x":0,"y":10},{"x":10,"y":10},{"x":0,"y":1},{"x":1,"y":0},{"x":9,"y":0},{"x":10,"y":1},{"x":0,"y":9},{"x":10,"y":9},{"x":2,"y":10},{"x":8,"y":10},{"x":0,"y":3},{"x":7,"y":0},{"x":7,"y":10},{"x":3,"y":0},{"x":10,"y":3},{"x":10,"y":4},{"x":4,"y":10},{"x":1,"y":1},{"x":0,"y":7},{"x":1,"y":9},{"x":9,"y":1},{"x":1,"y":8},{"x":8,"y":9},{"x":9,"y":6},{"x":3,"y":9},{"x":5,"y":1},{"x":3,"y":1},{"x":1,"y":3},{"x":1,"y":4},{"x":4,"y":9},{"x":9,"y":7},{"x":9,"y":5},{"x":9,"y":3},{"x":6,"y":9},{"x":8,"y":2},{"x":1,"y":5},{"x":8,"y":8},{"x":10,"y":7},{"x":2,"y":2},{"x":6,"y":8},{"x":8,"y":5},{"x":3,"y":8},{"x":4,"y":2},{"x":2,"y":6},{"x":4,"y":3},{"x":3,"y":5},{"x":6,"y":3},{"x":7,"y":2},{"x":7,"y":7},{"x":6,"y":2},{"x":6,"y":6},{"x":8,"y":3},{"x":2,"y":3},{"x":5,"y":3},{"x":6,"y":5},{"x":3,"y":7},{"x":5,"y":7}],"responses":[{"x":0,"y":0},{"x":10,"y":0},{"x":1,"y":10},{"x":9,"y":10},{"x":2,"y":0},{"x":8,"y":0},{"x":0,"y":2},{"x":10,"y":2},{"x":5,"y":10},{"x":0,"y":6},{"x":10,"y":6},{"x":3,"y":10},{"x":6,"y":0},{"x":0,"y":4},{"x":4,"y":0},{"x":5,"y":0},{"x":10,"y":5},{"x":6,"y":10},{"x":0,"y":8},{"x":0,"y":5},{"x":10,"y":8},{"x":9,"y":9},{"x":9,"y":4},{"x":2,"y":9},{"x":7,"y":9},{"x":4,"y":1},{"x":2,"y":1},{"x":1,"y":2},{"x":6,"y":1},{"x":7,"y":1},{"x":8,"y":1},{"x":9,"y":8},{"x":9,"y":2},{"x":5,"y":9},{"x":1,"y":7},{"x":1,"y":6},{"x":8,"y":7},{"x":7,"y":4},{"x":7,"y":8},{"x":7,"y":5},{"x":2,"y":4},{"x":6,"y":7},{"x":3,"y":4},{"x":5,"y":5},{"x":4,"y":4},{"x":5,"y":6},{"x":2,"y":7},{"x":5,"y":2},{"x":8,"y":6},{"x":8,"y":4},{"x":3,"y":6},{"x":7,"y":3},{"x":3,"y":3},{"x":3,"y":2},{"x":2,"y":5},{"x":2,"y":8},{"x":4,"y":6},{"x":5,"y":4}]}



    # 分析自己收到的输入和自己过往的输出，并恢复状态
    rival_trajectory = full_input["requests"]
    my_trajectory = full_input["responses"]

    if (int(rival_trajectory[0]['x']) == -1):
        # L[1:] 从第二个元素开始截取列表
        rival_trajectory = rival_trajectory[1:]

    game.init_board(rival_trajectory, my_trajectory)
    game.print_board(game.board)
