import numpy as np

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
    # 从第一位开始
    paff = np.int64(4611686018427388000)
    # 例如为0,3，其中3实际上是4
    shift_num = action[0] * 11 + action[1] + 1
    if shift_num <= 62:
        # 先计算出右移的位数
        board.b_0_63 = board.b_0_63 | (paff >> shift_num)
    else:
        # 例如为5,7，其中7实际上是8 对应55+8=63
        shift_num -= 62
        board.b_64_120 = board.b_64_120 | (paff >> shift_num)
    return board

class Board():
    def __init__(self):
        self.b_0_3 = np.uint64(0)
        self.b_4_7 = np.uint64(0)
        self.b_8_10 = np.uint64(0)

def get_next_action0(black_board, white_board):
    action_list = []
    action_board = Board()

    action_board.b_0_3 = black_board.b_0_3 | white_board.b_0_3
    action_board.b_4_7 = black_board.b_4_7 | white_board.b_4_7
    action_board.b_8_10 = black_board.b_8_10 | white_board.b_8_10

    '''
    黑： 8_10
    00100000000 00000110000 00000000000
    白： 0_3
    00000000001 00000000001

    0000000000000000000000000000000000000000000000000000100000000001
    1111111111111111111111111111111111111111111111111111011111111110    
    18446744073709550000
    18446744073709549566
    '''

    # 不进行取反操作
    # neg_board = ~action_board.b_0_3
    bin_str = bin(action_board.b_0_3).replace('0b', '')
    for i in range(len(bin_str)):
        # 0001 0001
        # 1代表为空
        if i <= 43 and (((np.uint64(1) << np.uint64(i)) & (action_board.b_0_3)) == False):
            x = i // 11
            y = i - x * 11
            action_list.append([x, y])

    # neg_board = ~action_board.b_4_7
    bin_str = bin(action_board.b_4_7).replace('0b', '')
    for i in range(len(bin_str)):
        if i <= 43 and (((np.uint64(1) << np.uint64(i)) & (action_board.b_4_7)) == False):
            x = i // 11 + 4
            y = i - (x - 4) * 11
            action_list.append([x, y])

    # neg_board = ~action_board.b_8_10
    bin_str = bin(action_board.b_8_10).replace('0b', '')
    for i in range(len(bin_str)):
        if i <= 32 and (((np.uint64(1) << np.uint64(i)) & (action_board.b_8_10)) == False):
            x = i // 11 + 8
            y = i - (x - 8) * 11
            action_list.append([x, y])
    return action_list

def get_next_action1(black_board, white_board):
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

if __name__ == '__main__':

    black_board = Board()
    white_board = Board()
    # 解析读入的JSON
    full_input = {"requests":[{"x":5,"y":9},{"x":8,"y":10},{"x":4,"y":9}],"responses":[{"x":0,"y":0},{"x":0,"y":1}]}

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

    print('不取反：')
    action_list = get_next_action0(black_board, white_board)
    print('取反：')
    action_list = get_next_action1(black_board, white_board)

    print(action_list)
    print(len(action_list))

