import numpy as np

# 创建一个长度为121的policy数组和一个长度小于121的pi_数组
policy = np.zeros(121)  # 初始化policy为0
pi_ = np.array([11, 0.1, 0.2, 0.3])  # 示例pi_数组

# 示例actions数组，包含了pi_中的下标
actions = np.array([0, 10, 20, 30])

# 将pi_中的值赋给policy中对应位置
policy[actions] = pi_

# 打印更新后的policy数组
print(policy)
