import math
import random

BLACK = 1
WHITE = -1

def paff(num):
    if len(num) == 1:
        return num[0], num[0]
    else:
        min0, max0 = paff(num[:len(num)//2])
        min1, max1 = paff(num[len(num)//2:])
        if min0 < min1:
            min1 = min0
        if max0 > max1:
            max1 = max0
        return min1, max1

if __name__ == '__main__':
    parent = WHITE
    player = BLACK if parent == WHITE else WHITE
    if player == BLACK:
        print('BLACK')
    else:
        print('WHITE')

    for i in range(37):
        choice = random.randint(0, 37 - 1)
        print(choice)

    S = [8, 2, 6, 3, 9, 13, 7, 5, 40, 2, 8]
    min = S[0]
    max = S[0]
    # neko = S[:len(S)//2]
    # print(neko)
    # neko = S[len(S)//2:]
    # print(neko)
    print(paff(S))