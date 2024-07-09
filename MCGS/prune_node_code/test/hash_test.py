import re

line = "2023-09-26 10:07:05,874 - 比赛结果记录 - INFO - 当前节点数：119565 当前边数：106537"

# 使用正则表达式查找 "当前边数" 后面的数值
match = re.search(r'当前边数：(\d+)', line)

# 如果找到匹配项，提取数值
if match:
    current_edge_count = int(match.group(1))
    print(current_edge_count)
else:
    print("未找到 '当前边数：' 字符串")
