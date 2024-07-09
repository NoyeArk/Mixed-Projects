import matplotlib.pyplot as plt
import numpy as np

# 横坐标数据
x_values = np.arange(1, 31)
# 剪枝百分比数据
pruning_percentages = np.random.randint(0, 100, size=30)

# 创建面积图
plt.fill_between(x_values, pruning_percentages, color='skyblue', alpha=0.4)

# 设置横纵坐标标签
plt.xlabel('横坐标', fontsize=12)
plt.ylabel('剪枝百分比', fontsize=12)

# 设置标题
plt.title('剪枝百分比随横坐标变化面积图', fontsize=14)

# 显示图表
plt.show()