action = [1, 3, 4]
pi = [0.1, 0.2, 0.3]

# 使用zip()函数将两个列表对应元素组合成键值对
result = dict(zip(action, pi))

print(result[4])
