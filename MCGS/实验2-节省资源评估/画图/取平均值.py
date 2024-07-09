import re

# 读取日志文件
with open('test.log', 'r', encoding='utf-8') as file:
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
averages = {x: info['sum'] / info['count'] for x, info in data.items()}

# 打印平均值
for x, average in averages.items():
    print(f'横坐标 {x}: 平均有效结点数 = {average}')
