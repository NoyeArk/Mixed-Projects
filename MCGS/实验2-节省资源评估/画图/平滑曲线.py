import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

import seaborn as sns  # 导入Seaborn库

from matplotlib.font_manager import FontProperties
font = FontProperties(fname='C:\Windows\Fonts\simkai.ttf')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 指定存储日志文件的文件夹路径
log_folder = 'thread4'

# 初始化字典来存储平均值
averages = {}

cnt = 0

# filename = 'log_game11.log'
# 遍历文件夹中的所有文件
for filename in os.listdir(log_folder):
    if filename.endswith('.log'):
        cnt += 1
        file_path = os.path.join(log_folder, filename)

        # 打开日志文件
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 初始化字典来存储横坐标和对应的有效节点数总和以及计数
        data = {}
        for line in lines:
            if '横坐标' in line:
                x_value = int(re.search(r'横坐标: (\d+)', line).group(1))
            elif '有效结点数' in line:
                active_node_count = int(re.search(r'有效结点数：(\d+)', line).group(1))
                if x_value in data:
                    data[x_value]['sum'] += active_node_count
                    data[x_value]['count'] += 1
                else:
                    data[x_value] = {'sum': active_node_count, 'count': 1}

        # 计算平均值
        for x, info in data.items():
            average = info['sum'] / info['count']
            if x in averages:
                averages[x].append(average)
            else:
                averages[x] = [average]

x_values = []
active_node_counts = []
# 打印平均值
for x, average_list in averages.items():
    avg = sum(average_list) / len(average_list)
    x_values.append(x)
    active_node_counts.append(avg)
    # print(f'横坐标 {x}: 平均有效结点数 = {avg}')

x = np.array(x_values)
y = np.array(active_node_counts)
# 使用Seaborn库中的颜色方案
sns.set_palette("husl")

# 进行多项式回归拟合
degree = 5  # 多项式的次数
coefficients = np.polyfit(x, y, degree)  # 拟合多项式系数

plt.rcParams['xtick.direction'] = 'in'  #将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'  #将y轴的刻度方向设置向内

# 生成拟合后的数据点
x_fit = np.linspace(x.min(), x.max(), 100)
y_fit = np.polyval(coefficients, x_fit)

# 计算标准差
y_std = np.std(y - np.polyval(coefficients, x))

# 绘制原始数据点和拟合曲线
plt.plot(x, y, 'o', label='该次搜索的活跃结点数')
plt.plot(x_fit, y_fit, label=f'活跃结点数均值曲线')

plt.xlabel('Max方游戏步数/步', fontsize=12, fontproperties=font)
plt.ylabel('活跃结点个数/个', fontsize=12, fontproperties=font)

# 添加阴影表示波动范围
plt.fill_between(x_fit, y_fit - y_std, y_fit + y_std, alpha=0.2, color='gray', label='一个标准差的置信区间')

# 添加图例
plt.legend(prop=font)

# 保存图表为SVG格式
plt.savefig('my_plot1.svg', dpi=300, format='svg', transparent=True)

# 显示图表（可选）
plt.show()