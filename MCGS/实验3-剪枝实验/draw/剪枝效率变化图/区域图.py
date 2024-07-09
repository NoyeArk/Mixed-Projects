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

prune_node = [cumulative_pruned_node_counts[i] - cumulative_pruned_node_counts[i-1] for i in range(1, len(cumulative_pruned_node_counts))]
prune_node.insert(0, cumulative_pruned_node_counts[0])
sum_node = [prune_node[0]+current_node_counts[0]]
for i in range(1, len(prune_node)):
    sum_node.append(sum_node[i-1]+prune_node[i])
addition_node = [current_node_counts[i] + prune_node[i] - current_node_counts[i-1] for i in range(1, len(prune_node))]
addition_node.insert(0, current_node_counts[0]+prune_node[0])

print(cumulative_pruned_node_counts)
print(current_node_counts)
print(prune_node)
print(addition_node)

pruning_percentages = [x / y for x, y in zip(prune_node, addition_node)]

# 创建区域图
plt.plot(x_values, pruning_percentages, color='skyblue')
plt.fill_between(x_values, pruning_percentages, color='skyblue', alpha=0.4)

# 设置横纵坐标标签
plt.xlabel('我方游戏步数/步', fontsize=12, fontproperties=font)
plt.ylabel('删除效率df/%', fontsize=12, fontproperties=font)


# 设置标题
# plt.title('剪枝效率变化图', fontsize=14)

plt.savefig('test.svg', dpi=300, format="svg", transparent=True)