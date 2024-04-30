
# relative package
from random import choice
import random
import math
import time
import numpy as np

namelist = 10000
action_num = 400
ActionList = random.sample(range(1,namelist),action_num)
# ActionList = [random.randint(0,1000)for i in range(random.randint(20,100))]
print(ActionList)

class information(object):
    def __init__(self):
        self.action = random.randint(1,len(ActionList))
        self.quality = 0
        self.visit_times = 1
        self.name = None
        self.childNum = 0
        self.UCB_value = 0

class Deliver(object):
    def __init__(self):
        self.parent = None
        self.leaf_name = None
        self.flag = False

class TreeNode(object):
    def __init__(self):

        self.information = information()
        self.information.name = self.policy_network()

        self.parent = None
        self.child = []

    def set_parent(self,parent):
        self.parent = parent

    def set_child(self,child):
        self.child += [child]
        self.information.childNum += 1

    def child_Num(self):
        return self.information.childNum

    def Calc_UCB(self):
        c = 0.000001
        quality = self.information.quality
        visit_times = self.information.visit_times
        father_visit_value = self.parent.information.visit_times
        self.information.UCB_value = quality / visit_times + c * math.sqrt(2 * math.log(father_visit_value) / visit_times)

    def policy_network(self):
        return choice(ActionList)

    def value_network(Quality):
        Quality = random.uniform(0, 1)

def best_node(Node):
    Max_UCB_Node = TreeNode()
    for Child in Node.child:
        if Max_UCB_Node.information.UCB_value < Child.information.UCB_value:
            Max_UCB_Node = Child
    return Max_UCB_Node

# MCTS
def Selection(Node):
    # print('**********这是Selection************')
    # print(Node.information.action,'   ',Node.information.childNum)
    deliver = Deliver()

    if Node.information.action == Node.information.childNum: # all expended
        deliver.flag = False
        return deliver
    else:
        flag = True
        select_node = TreeNode()
        for Child in Node.child:
            # print(select_node.information.name, '   ', Child.information.name)
            if select_node.information.name == Child.information.name: # 如果选的节点已经存在
                select_node = Child
                flag = False # False代表该节点已经存在
                break

        if flag == False: # False代表该节点已经存在
            return Selection(select_node)
        else: # True代表该节点还未被拓展，则返回该节点
            deliver.parent = Node
            deliver.leaf_name = select_node.information.name
            deliver.flag = True
            return deliver

def Expansion(Node,node_name):
    # print('**********这是Expansion************')
    new_node = TreeNode()
    new_node.information.name = node_name
    new_node.set_parent(Node)
    Node.set_child(new_node)

    return new_node

def Simulation(node):
    node.information.quality = random.random()

def Backup(Node):
    if Node.parent != None:
        Node.parent.information.visit_times += 1
        Node.parent.information.quality += Node.information.quality
        Node.Calc_UCB()
        Backup(Node.parent)
    else:
        pass

def printNode(Node):
    print('name:', Node.information.name,end='')
    print('  action:', Node.information.action,end='')
    print('  quality:', Node.information.quality,end='')
    print('  childNum:', Node.information.childNum,end='')
    print('  UCB_value:', Node.information.UCB_value,end='')
    print('  visit_times:', Node.information.visit_times,end='')
    print('  parent:', Node.parent.information.name)

def Init_Root():
    root = TreeNode()
    root.action = root.policy_network()
    root.information.name = 'root'
    root.information.quality = random.random()
    root.information.UCB_value = 666666

    return root

if __name__ == '__main__':
    # Init_MCTS
    root = Init_Root()
    Real_Root = TreeNode()
    Real_Root.information.name = '我是真正的根节点'
    Real_Root.parent = TreeNode()
    root.parent = Real_Root
    Real_Root.child = root

    deliver = Deliver()
    time_start = time.time()

    # print('****************第一轮搜索****************')
    for search_times in range(10000):
        if search_times == 3000:
            print()
        print('**************************正在进行第',search_times+1,'次搜索中哦，请耐心等待呀~~~')

        deliver = Selection(root) # first selection

        flag = True

        if deliver.flag == True: # is not all expended
            print('当前所处的老节点:',end='')
            printNode(root)

            node = Expansion(deliver.parent,deliver.leaf_name)
            Simulation(node)
            Backup(node)
            # printNode(root)
            print('刚刚诞生的新节点:',end='')
            printNode(node)

            # for Child in root.child:
            #     printNode(Child)


        elif deliver.flag == False:   # is all expended
            # print('\n这次搜索的根节点为',end='')
            # printNode(root)

            root = best_node(root)
            print('好孩子是',end='')
            printNode(root)

    printNode(Real_Root)
    time_end = time.time()
    print('**************************我亲耐滴使用者，搜索完成啦，历时:',time_end-time_start,'s')


