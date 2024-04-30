import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import plotting

if __name__ == '__main__':
    print("加载模型中...")
    # 加载模型
    with open('forest.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    print("加载模型完成！")

    # 读取数据
    data = pd.read_csv(r'D:\workSpace\比赛\美赛\美赛校赛2024\Problem\cardio_train.csv', header=0)
    print(type(data))

    # 划分训练集和测试集
    x = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    feat_labels = data.columns[1:-1]

    plotting(loaded_model, x_test, y_test)
    plt.show()