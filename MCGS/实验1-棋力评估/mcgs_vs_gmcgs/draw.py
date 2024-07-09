import matplotlib.pyplot as plt
import numpy as np
import re
import os
from scipy.interpolate import make_interp_spline

plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号

def draw_graph(game_round, thread_id=1):
    # 保存图像到 graph/thread 文件夹下
    save_folder = os.path.join('graph', 'thread' + str(thread_id))  # 包括文件夹名称
    os.makedirs(save_folder, exist_ok=True)  # 创建文件夹（如果不存在）

    # 初始化存储数据的列表
    x_values = []
    current_node_counts = []
    cumulative_pruned_node_counts = []
    current_edge_counts = []
    cumulative_pruned_edge_counts = []

    file_name = 'log/thread' + str(thread_id) + '/log_game' + str(game_round) + '.log'

    # 打开日志文件并逐行读取数据，使用UTF-8编码
    with open(file_name, 'r', encoding='utf-8') as file:
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


    # 创建一个美观的图表风格
    plt.style.use('ggplot')

    # 平滑的x轴数据
    x_smooth = np.linspace(min(x_values), max(x_values), 300)

    # 使用样条插值平滑曲线
    spline = make_interp_spline(x_values, current_node_counts)
    current_node_counts_smooth = spline(x_smooth)

    spline = make_interp_spline(x_values, cumulative_pruned_node_counts)
    cumulative_pruned_node_counts_smooth = spline(x_smooth)

    spline = make_interp_spline(x_values, current_edge_counts)
    current_edge_counts_smooth = spline(x_smooth)

    spline = make_interp_spline(x_values, cumulative_pruned_edge_counts)
    cumulative_pruned_edge_counts_smooth = spline(x_smooth)

    # 绘制节点数和累计剪枝节点数的平滑曲线
    plt.figure(figsize=(12, 6))
    plt.fill_between(x_smooth, current_node_counts_smooth, cumulative_pruned_node_counts_smooth, color='green')
    plt.plot(x_smooth, current_node_counts_smooth, label='图中当前节点数', color='b', linewidth=2)
    plt.plot(x_smooth, cumulative_pruned_node_counts_smooth, label='累计剪枝节点数', color='r', linewidth=2)
    plt.xlabel('搜索次数/次', fontsize=12)
    plt.ylabel('节点数/个', fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.title('节点数变化图', fontsize=16)
    plt.grid(True)

    # 保存图像
    save_path = os.path.join(save_folder, 'node_counts_plot' + str(game_round) + '.png')  # 设置完整的保存路径
    plt.savefig(save_path)  # 指定文件名和格式

    plt.figure(figsize=(12, 6))
    plt.fill_between(x_smooth, current_edge_counts_smooth, cumulative_pruned_edge_counts_smooth, color='green')
    plt.plot(x_smooth, current_edge_counts_smooth, label='图中当前节点数', color='b', linewidth=2)
    plt.plot(x_smooth, cumulative_pruned_edge_counts_smooth, label='累计剪枝节点数', color='r', linewidth=2)
    plt.xlabel('搜索次数/次', fontsize=12)
    plt.ylabel('边数/条', fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.title('边数变化图', fontsize=16)
    plt.grid(True)

    # 保存图像
    save_path = os.path.join(save_folder, 'edge_counts_plot' + str(game_round) + '.png')  # 设置完整的保存路径
    plt.savefig(save_path)  # 指定文件名和格式

    # 绘制当前节点数、当前边数、累计节点数和累计边数的平滑曲线
    plt.figure(figsize=(12, 6))
    plt.plot(x_smooth, current_edge_counts_smooth, label='图中当前边数', color='g', linewidth=2)
    plt.plot(x_smooth, cumulative_pruned_edge_counts_smooth, label='累计剪枝边数', color='y', linewidth=2)
    plt.plot(x_smooth, current_node_counts_smooth, label='图中当前节点数', color='b', linewidth=2)
    plt.plot(x_smooth, cumulative_pruned_node_counts_smooth, label='累计剪枝节点数', color='r', linewidth=2)
    plt.xlabel('搜索次数/次', fontsize=12)
    plt.ylabel('数量', fontsize=12)
    plt.legend(loc='upper left', fontsize=12)
    plt.title('节点数和边数变化图', fontsize=16)
    plt.grid(True)

    # 保存图像
    save_path = os.path.join(save_folder, 'node_edge_plot' + str(game_round) + '.png')  # 设置完整的保存路径
    plt.savefig(save_path)  # 指定文件名和格式


if __name__ == '__main__':
    draw_graph(0)