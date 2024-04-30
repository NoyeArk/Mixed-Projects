'''
1.博弈
2.时间序列
'''
'''
机器学习分类:
    1.监督学习
    2.非监督学习
    3.强化学习
神经网络可以做很多次空间变换，多次非线性变换，“语义鸿沟"
'''

from torchvision import datasets    # 数据集
from torchvision import transforms  # 数据变换
from torch.utils.data import DataLoader # 数据载入

# 模型开发
from torch import nn        # 构建神经网络用
from torch.nn import functional as F  # F.relu()没有内部状态

# 可视化
import matplotlib.pyplot as plt
# %matplotlib inline

# 超参数
batch_size = 64
lr = 1e-3
epochs = 2
INPUT_SIZE = 28*28
N1 = 64
N2 = 64
N_CLASSES = 10

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(INPUT_SIZE, N1)
        self.layer2 = nn.Linear(N1, N2)
        self.layer3 = nn.Linear(N2, N_CLASSES)
        self.linnear_relu_stacking = nn.Sequential()
        self.linnear_relu_stacking.append(self.layer1)
        self.linnear_relu_stacking.append(nn.ReLU())
        self.linnear_relu_stacking.append(self.layer2)
        self.linnear_relu_stacking.append(nn.ReLU())
        self.linnear_relu_stacking.append(self.layer3)

    def forward(self, x):
        x = self.Flatten(x)
        x = self.linnear_relu_stacking(x)
        return x

def training_loop(training_loader, model, loss_fn, optimizer):
    for batch,(x, y) in enumerate(training_loader):
        y_hat = model(x)

        loss = loss_fn(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f'loss:{loss:>7f}'

                  )
    pass

if __name__ == '__main__':
    # 1.数据准备
    # 1.1 获取数据
    training_data = datasets.FashionMNIST(root='data', train=True, download=True)
    test_data = datasets.FashionMNIST(root='data', train=False, download=True)

    print(type(training_data.data))
    # print(training_data)
    # print(training_data.data)
    #
    # print(training_data.data.shape)

    fig_idx = 9
    # plt.imshow(training_data.data[fig_idx])
    # plt.show()

    print(training_data.class_to_idx)

    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # 2.模型开发
    # 2.1 定义神经网络
    model = MLP()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.sgd(model.parameters(), lr=lr)
    # 2.2 定义损失函数

    # 2.3 定义优化算法

    # 2.4 训练神经网络

    # 3.模型部署

    pass