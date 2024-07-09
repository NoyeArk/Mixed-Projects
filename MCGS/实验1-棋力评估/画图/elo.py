import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='C:\Windows\Fonts\simkai.ttf')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签SimHei
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 指定横坐标的值
x = [500, 1000, 2000, 5000]

# 基准值
base_value = 1255

# 数据
minimax = [-1255, -1107, -1089, -1000]
alpha_beta = [-470, -337, -260, -200]
mcts = [514, 531, 536, 550]
mcgs = [514, 515, 517, 537]
gmcgs = [525, 550, 570, 601]

# 将数据与基准值相加
minimax = [val + base_value for val in minimax]
alpha_beta = [val + base_value for val in alpha_beta]
mcts = [val + base_value for val in mcts]
mcgs = [val + base_value for val in mcgs]
gmcgs = [val + base_value for val in gmcgs]

# 使用Seaborn颜色方案
# sns.set_palette("husl")

# 创建曲线
plt.figure(figsize=(10, 6))  # 调整图像大小

plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内

# 画曲线
plt.plot(x, minimax, label='Minimax', marker='o', linewidth=2)
plt.plot(x, alpha_beta, label='Alpha-Beta', marker='s', linewidth=2)
plt.plot(x, mcts, label='MCTS', marker='^', linewidth=2)
plt.plot(x, mcgs, label='MCGS', marker='D', linewidth=2)
plt.plot(x, gmcgs, label='MCGSG', marker='P', linewidth=2)

# 设置横纵坐标标签和标题
plt.xlabel('搜索结点数/个', fontsize=12, fontproperties=font)
plt.ylabel('Elo分数/分', fontsize=12, fontproperties=font)
# plt.title('Elo曲线图', fontsize=14)

# 设置坐标轴刻度字体大小
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# 设置图例位置和字体大小
plt.legend(bbox_to_anchor=(1.0, 0.85), fontsize=10)

# 保存图像为SVG格式
plt.savefig('curves.svg', dpi=300, format="svg", transparent=True)

# 显示图像
plt.show()