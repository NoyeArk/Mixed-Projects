import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split
# 可视化
import matplotlib.pyplot as plt
from utils import load_data, get_data


if __name__ == '__main__':

    data = load_data()
    # data['age'] = (data['age'] / 365)
    # print(data['age'])

    # 划分训练集和测试集
    # x = data.iloc[:, 1:-1].values
    # y = data.iloc[:, -1].values

    x_train, x_test, y_train, y_test = get_data()
    feat_labels = data.columns[1:-1]

    # 构建随机森林
    forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1, max_depth=3, verbose=1)
    forest.fit(x_train, y_train)

    importances = forest.feature_importances_
    print("重要性：", importances)

    x_columns = data.columns[1:-1]
    indices = np.argsort(importances)[::-1]

    x_columns_indices = []
    for f in range(x_train.shape[1]):
        # 对于最后需要逆序排序，我认为是做了类似决策树回溯的取值，从叶子收敛到根，根部重要程度高于叶子。
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
        x_columns_indices.append(feat_labels[indices[f]])

    # 筛选变量（选择重要性比较高的变量）
    threshold = 0.15
    x_selected = x_train[:, importances > threshold]

    plt.figure(figsize=(10, 6))
    plt.title("数据集中各个特征的重要程度", fontsize=18)
    plt.ylabel("import level", fontsize=15, rotation=90)
    plt.rcParams['font.sans-serif'] = ["SimHei"]
    plt.rcParams['axes.unicode_minus'] = False

    for i in range(x_columns.shape[0]):
        plt.bar(i, importances[indices[i]], color='orange', align='center')
        plt.xticks(np.arange(x_columns.shape[0]), x_columns_indices, rotation=90, fontsize=15)

    plt.show()

    print("构建森林完成，开始保存结果...")
    # 保存模型
    with open('forest.pkl', 'wb') as model_file:
        pickle.dump(forest, model_file)

    print("保存特征选择完成！")