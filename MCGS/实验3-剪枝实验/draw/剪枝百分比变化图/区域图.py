import matplotlib.pyplot as plt
import numpy as np
import os
import re
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='C:\Windows\Fonts\simkai.ttf')

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

# 初始化存储数据的列表
x_values = []
current_node_counts = []
cumulative_pruned_node_counts = []
current_edge_counts = []
cumulative_pruned_edge_counts = []

# 打开日志文件并逐行读取数据，使用UTF-8编码
with open('log_game0.log', 'r', encoding='utf-8') as file:
    for line in file:
        if '横坐标' in line:
            x_values.append(float(line.split(':')[-1].strip()))
        if '当前节点数' in line:
            current_node_counts.append(int(re.search(r'当前节点数：(\d+)', line).group(1)))
        if '累计剪枝节点数' in line:
            cumulative_pruned_node_counts.append(int(re.search(r'累计剪枝节点数：(\d+)', line).group(1)))
        if '当前边数' in line:
            current_edge_counts.append(int(re.search(r'当前边数：(\d+)', line).group(1)))
        if '累计剪枝边数' in line:
            cumulative_pruned_edge_counts.append(int(re.search(r'累计剪枝边数：(\d+)', line).group(1)))

sum_node = [x + y for x, y in zip(current_node_counts, cumulative_pruned_node_counts)]
sum_edge = [x + y for x, y in zip(current_edge_counts, cumulative_pruned_edge_counts)]
sum_prune = [x + y for x, y in zip(cumulative_pruned_node_counts, cumulative_pruned_edge_counts)]
sum = [x + y for x, y in zip(sum_node, sum_edge)]
pruning_percentages = [x / y for x, y in zip(sum_prune, sum)]


plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内`

# 创建区域图
plt.plot(x_values, pruning_percentages, color='skyblue')
plt.fill_between(x_values, pruning_percentages, color='skyblue', alpha=0.4)

# 设置横纵坐标标签
plt.xlabel('Max方游戏步数/步', fontsize=12, fontproperties=font)
plt.ylabel('删除百分比dp/%', fontsize=12, fontproperties=font)

plt.ylim(0, 1)


# 设置标题
# plt.title('剪枝百分比随横坐标变化图', fontsize=14)

plt.savefig('test2.svg', dpi=300, format="svg", transparent=True)