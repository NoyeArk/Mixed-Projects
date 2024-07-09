import os
import re
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 指定存储日志文件的文件夹路径
log_folder = 'thread4'

# 初始化字典来存储平均值
averages = {}

cnt = 0

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

# 创建散点图
plt.figure(figsize=(10, 6))  # 调整图像大小
plt.scatter(x_values, active_node_counts, c='skyblue', label='数据点', marker='o', edgecolors='black', linewidths=1, s=50)

# 设置横纵坐标标签
plt.xlabel('搜索次数', fontsize=12)
plt.ylabel('活跃结点数', fontsize=12)

# 设置标题
plt.title('活跃结点数变化散点图', fontsize=14)

# 删除所有网格线
plt.grid(False)

# 添加图例
plt.legend()

# 保存图像为SVG格式
plt.savefig('thread3.svg', dpi=300, format="svg", transparent=True)

# 显示图像
plt.show()