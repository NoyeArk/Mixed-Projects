import torch
import matplotlib.pyplot as plt

import utils
from model import Net
from utils import get_data


if __name__ == '__main__':
    model = Net(n_feature=11, n_hidden=10, n_output=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
    loss_func = torch.nn.CrossEntropyLoss()

    x_train, x_test, y_train, y_test = get_data()
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    epochs = 1000
    train_losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i in range(len(x_train)):
            # 将输入和标签转换为 PyTorch 张量
            inputs = torch.tensor(x_train[i], dtype=torch.float32)
            labels = torch.tensor(y_train[i], dtype=torch.long)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # 计算平均损失
        average_loss = running_loss / len(x_train)
        train_losses.append(average_loss)

        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {average_loss}')

    # 绘制损失曲线
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()