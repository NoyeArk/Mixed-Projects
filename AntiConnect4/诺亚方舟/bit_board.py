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
    '''
    0000000000000000000000000000000 00100100000 00000000000 00000010000    1207959568
    0000000000000000000000000000000001001000000000000000000000010000    1207959568
    1111111111111111111111111111111110110111111111111111111111101111    18446744072501592000
    1111111111111111111111111111111110110111111111111111111111101111
    '''
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

def game_end(board):
    # 一共要判断三种，横向、纵向、斜方向
    print('board.b_0_3:', board.b_0_3)
    print('1的个数:', bin(board.b_0_3).count('1'))
    print('1的位置:')

    '''
                               3,7        2,6,10       1,5,9       0,4,8
     00000000000000000000  00100000100 01001001100 00000101000 00000100000
     00000000000000000000  00000000000 00000000000 00000000000 00000000000
       0000000000000000000000000000000 00000000000 00000000000 00000000000
    '''
    # 斜方向，先变成纵向
    print('******这里是左下******')
    # 先判断右下
    row_list = []
    print(board.b_0_3)   #           00100000100 01001001100 00000101000 00000100000
    print(board.b_4_7)   #           01101010001 00010000000 10000010010 00100000001
    print(board.b_8_10)  #           00000000000 01000100000 11101011000 00000110000
    '''
              00000100000 0
             000001010000 1
            0100100110000 2
           00100000100000 3
          001000000010000 4
         1000001001000000 5
        00010000000000000 6
       011010100010000000 7
      0000011000000000000 8
     11101011000000000000 9
    010001000000000000000 10
    '''
    row_list.append(board.b_0_3 & np.uint64(2047))  # 2047 11111111111
    row_list.append((board.b_0_3 & np.uint64(4192256)) >> np.uint64(11-1))  # 4192256 11111111111 00000000000
    row_list.append((board.b_0_3 & np.uint64(8585740288)) >> np.uint64(11*2-2))  # 8585740288 11111111111 00000000000 00000000000
    row_list.append((board.b_0_3 & np.uint64(17583596109824)) >> np.uint64(11*3-3))  # 17583596109824 11111111111000000000000000000000000000000000

    row_list.append((board.b_4_7 & np.uint64(2047)) << np.uint64(4))
    row_list.append((board.b_4_7 & np.uint64(4192256)) >> np.uint64(11*1-5))
    row_list.append((board.b_4_7 & np.uint64(8585740288)) >> np.uint64(11*2-6))
    row_list.append((board.b_4_7 & np.uint64(17583596109824)) >> np.uint64(11*3-7))

    row_list.append((board.b_8_10 & np.uint64(2047)) << np.uint64(8))
    row_list.append((board.b_8_10 & np.uint64(4192256)) >> np.uint64(11*1-9))
    row_list.append((board.b_8_10 & np.uint64(8585740288)) >> np.uint64(11*2-10))

    for loop in range(8):
        if loop == 6:
            print()
        """
        0: 0 1 2 3
        1: 1 2 3 4
        ...
        7: 7 8 9 10
        """
        outcome_and = row_list[loop] & row_list[loop+1] & row_list[loop+2] & row_list[loop+3]
        if (bin(outcome_and).count('1')) >= 1:
            return END

    print('******这里是右下******')
    # 判断左下
    row_list = []
    row_list.append(board.b_0_3 & np.uint64(2047))  # 2047 11111111111
    row_list.append((board.b_0_3 & np.uint64(4192256)) >> np.uint64(11*1+1))  # 4192256 11111111111 00000000000
    row_list.append((board.b_0_3 & np.uint64(8585740288)) >> np.uint64(11*2+2))  # 8585740288 11111111111 00000000000 00000000000
    row_list.append((board.b_0_3 & np.uint64(17583596109824)) >> np.uint64(11*3+3))  # 17583596109824 11111111111000000000000000000000000000000000

    row_list.append((board.b_4_7 & np.uint64(2047)) >> np.uint64(4))
    row_list.append((board.b_4_7 & np.uint64(4192256)) >> np.uint64(11*1+5))
    row_list.append((board.b_4_7 & np.uint64(8585740288)) >> np.uint64(11*2+6))
    row_list.append((board.b_4_7 & np.uint64(17583596109824)) >> np.uint64(11*3+7))
    # 10000010001001001100 00000101000 00000100000
    row_list.append((board.b_8_10 & np.uint64(2047)) >> np.uint64(8))
    row_list.append((board.b_8_10 & np.uint64(4192256)) >> np.uint64(11*1+9))
    row_list.append((board.b_8_10 & np.uint64(8585740288)) >> np.uint64(11*2+10))
    '''
    
    '''
    print(board.b_0_3)
    print(board.b_4_7)
    print(board.b_8_10)
    for loop in range(8):
        outcome_and = row_list[loop] & row_list[loop + 1] & row_list[loop + 2] & row_list[loop + 3]
        if (bin(outcome_and).count('1')) >= 1:
            return END

    print('******这里是纵向******')
    # 纵向,相邻四行之间进行与运算，如果有1，则连成4个
    row_list = []
    # print(board.b_0_3)   #           00100000100 01001001100 00000001100 00000100000
    # print(board.b_4_7)   #           01101010001 00010000000 10000010010 00100000001
    # print(board.b_8_10)  #           00000000000 01000100000 11101011000 00000110000
    '''
    00000100000
    00000001100
    01001001100
    '''
    row_list.append(board.b_0_3 & np.uint64(2047)) # 2047 11111111111
    row_list.append((board.b_0_3 & np.uint64(4192256)) >> np.uint64(11*1)) # 4192256 11111111111 00000000000
    row_list.append((board.b_0_3 & np.uint64(8585740288)) >> np.uint64(11*2)) # 8585740288 11111111111 00000000000 00000000000
    row_list.append((board.b_0_3 & np.uint64(17583596109824)) >> np.uint64(11*3)) # 17583596109824 11111111111000000000000000000000000000000000

    row_list.append(board.b_4_7 & np.uint64(2047))
    row_list.append((board.b_4_7 & np.uint64(4192256)) >> np.uint64(11*1))
    row_list.append((board.b_4_7 & np.uint64(8585740288)) >> np.uint64(11*2))
    row_list.append((board.b_4_7 & np.uint64(17583596109824)) >> np.uint64(11*3))

    row_list.append(board.b_8_10 & np.uint64(2047))
    row_list.append((board.b_8_10 & np.uint64(4192256)) >> np.uint64(11*1))
    row_list.append((board.b_8_10 & np.uint64(8585740288)) >> np.uint64(11*2))
    for i in range(len(row_list)):
        print(f'row_list{i}:', bin(row_list[i]))
    for loop in range(8):

        outcome_and = row_list[loop] & row_list[loop+1] & row_list[loop+2] & row_list[loop+3]
        if (bin(outcome_and).count('1')) >= 1:
            return END

    print('******这里是横向******')
    # 横向
    board_list = [board.b_0_3, board.b_4_7, board.b_8_10]
    for loop in range(3): # 0 1 2
        refer_num = np.uint64(15) # 1111
        if loop <= 1:
            board_num = 4
        else:
            board_num = 3
        for j in range(board_num):
            for i in range(8):
                refer_num = refer_num << np.uint64(i)
                outcome = board_list[loop] & refer_num
                if outcome == refer_num:
                    print('outcome:', outcome)
                    print('i:', i, 'j:', j)
                    print('board_num:', board_num)
                    return END
            refer_num = np.uint64(15) << np.uint64(4*(j+1))
    return NOT_END

if __name__ == '__main__':
    black_board = Board()
    white_board = Board()
    # 解析读入的JSON
    full_input = {"requests":[{"x":-1,"y":-1},{"x":0,"y":10},{"x":10,"y":10},{"x":0,"y":1},{"x":1,"y":0},{"x":9,"y":0},{"x":10,"y":1},{"x":0,"y":9},{"x":10,"y":9},{"x":2,"y":10},{"x":8,"y":10},{"x":0,"y":3},{"x":7,"y":0},{"x":7,"y":10},{"x":3,"y":0},{"x":10,"y":3},{"x":10,"y":4},{"x":4,"y":10},{"x":1,"y":1},{"x":0,"y":7},{"x":1,"y":9},{"x":9,"y":1},{"x":1,"y":8},{"x":8,"y":9},{"x":9,"y":6},{"x":3,"y":9},{"x":5,"y":1},{"x":3,"y":1},{"x":1,"y":3},{"x":1,"y":4},{"x":4,"y":9},{"x":9,"y":7},{"x":9,"y":5},{"x":9,"y":3},{"x":6,"y":9},{"x":8,"y":2},{"x":1,"y":5},{"x":2,"y":2},{"x":3,"y":8},{"x":7,"y":2},{"x":4,"y":2}],"responses":[{"x":0,"y":0},{"x":10,"y":0},{"x":1,"y":10},{"x":9,"y":10},{"x":2,"y":0},{"x":8,"y":0},{"x":0,"y":2},{"x":10,"y":2},{"x":5,"y":10},{"x":0,"y":6},{"x":10,"y":6},{"x":3,"y":10},{"x":6,"y":0},{"x":0,"y":4},{"x":4,"y":0},{"x":5,"y":0},{"x":10,"y":5},{"x":6,"y":10},{"x":0,"y":8},{"x":0,"y":5},{"x":10,"y":8},{"x":9,"y":9},{"x":9,"y":4},{"x":2,"y":9},{"x":7,"y":9},{"x":4,"y":1},{"x":2,"y":1},{"x":1,"y":2},{"x":6,"y":1},{"x":7,"y":1},{"x":8,"y":1},{"x":9,"y":8},{"x":9,"y":2},{"x":5,"y":9},{"x":1,"y":7},{"x":1,"y":6},{"x":8,"y":4},{"x":2,"y":4},{"x":5,"y":2},{"x":6,"y":7},{"x":10,"y":7}]}
    '''
                              3,7        2,6,10       1,5,9       0,4,8
    00000000000000000000  00000000000 00000000000 00000000000 00000000000      
    00000000000000000000  00000000000 00000000000 00000000000 00000000000   
      0000000000000000000000000000000 00000000000 00000000000 00000000000  
    '''
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
    print(game_end(black_board))
    # my_action = random.randint(0, len(best_move) - 1)  # 产生 0 到 len(best_move) - 1 的一个整数型随机数
    #
    # x = best_move[my_action][1]
    # y = best_move[my_action][0]

    # # TODO: 作出决策并输出
    # my_action = { "x": x, "y": y }

    # print(json.dumps({
    #     "response": my_action,
    # }))
