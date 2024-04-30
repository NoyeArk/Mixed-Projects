import numpy as np
#
'''
128 - 121 = 7

   12345678912 345689
00 00000000000 00000000000 00000000000 00010000000 00000000000 0000000
00 0000 00000000000 00000000000 01000000000 00000000000 00000000000  000

第一个数的前62表示0-61,因为有两位不用
第二个数的后59表示62-120

10000000000000000000000000000000000
10000000000000000000000000000000000

action       十六进制                  十进制
        0x4000000000000000     4611686018427388000
0,0     0x2000000000000000     2305843009213694000
0,1     0x1000000000000000
0,2     0x800000000000000
0,3     0x400000000000000
0,4     0x200000000000000
8,1

&	与	两个位都为1时，结果才为1
|	或	两个位都为0时，结果才为0
^	异或	两个位相同为0，相异为1
~	取反	0变1，1变0
<<	左移	各二进位全部左移若干位，高位丢弃，低位补0
>>	右移	各二进位全部右移若干位，对无符号数，高位补0，有符号数，各编译器处理方法不一样，有的补符号位（算术右移），有的补0（逻辑右移）
'''

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
        self.b_0_63 = np.uint64(0)
        self.b_64_120 = np.uint64(0)

def test(int_num):
    int_num = int(int_num)
    bin_str = bin(int_num).replace('0b','')
    for i in range(len(bin_str)):
        if (1 << i) & int_num:
             print(" the number '1' position is "+ str(i))
             # x = (i - 1) // 11 + 3
             # y = i - (x - 3) * 11
             # x = i // 11 + 4 * 1
             # y = i - (x - 4) * 11

             x = (i) // 11 + 4 * 2
             y = i - (x - 4 * 2) * 11
             print('x:', x)
             print('y:', y)

def bin_2_str(bin):
    """
    二进制转换为字符串
    """
    return ''.join([chr(i) for i in [int(b, 2) for b in bin.split(' ')]])

if __name__ == '__main__':
    test(np.uint64(12)) # 1 1 0 0
    # print(bin_2_str(20))
    a = np.uint(3) # 0011
    b = np.uint(15) # 1111
    ans = a & b # 3
    print(ans)