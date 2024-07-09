
def print_child_info(G, node_id):
    print('当前根节点的出度：', G.out_degree(node_id))
    print('当前进行选择的父节点为：:', node_id)
    print('该父节点的所有孩子结点为：:')
    print(list(G.neighbors(node_id)))

def print_ucb_info(G, child_id, ucb):
    print('id:', child_id, end='')
    print('   ucb:', ucb, end='')
    print('   q:', G.nodes[child_id]['Q_s'], end='')
    print('   value:', -G.nodes[child_id]['Q_s'] + ucb)

