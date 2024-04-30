# 决策树相关库
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'  # 使用微软雅黑字体
from sklearn.metrics import accuracy_score, recall_score

from plotting import plotting
from utils import get_data


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_data()

    tree = DecisionTreeClassifier(max_depth=10)
    tree.fit(x_train, y_train)
    plotting(tree, x_test, y_test)
    plt.show()

    # 在测试集上进行预测
    y_pred = tree.predict(x_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"决策树模型的准确率为: {accuracy}")

    # 计算召回率
    recall = recall_score(y_test, y_pred)
    print(f"决策树模型的召回率为: {recall}")