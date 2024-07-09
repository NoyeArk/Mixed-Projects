import networkx as nx

# 创建一个有向图
G = nx.DiGraph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (6, 9)])

# 选择一个起始节点
start_node = 1

# 使用深度优先搜索遍历图，并记录访问的节点
visited_nodes = set()
stack = [start_node]

while stack:
    node = stack.pop()
    if node not in visited_nodes:
        visited_nodes.add(node)
        neighbors = list(G.neighbors(node))
        stack.extend(neighbors)

# 输出从起始节点出发可以遍历到的节点数
print(f"从节点 {start_node} 出发可以遍历到的节点数: {len(visited_nodes)}")
