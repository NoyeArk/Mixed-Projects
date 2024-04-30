import pandas as pd
# K近邻算法相关库
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import precision_recall_curve,roc_curve,average_precision_score,auc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score


from plotting import plotting
from utils import load_data, split_data_blog_2


if __name__ == '__main__':
    data = load_data()
    x_train, x_test, y_train, y_test = split_data_blog_2(data)

    # 构建K近邻模型
    knn = KNeighborsClassifier(n_neighbors=5)

    # scores = cross_val_score(knn, standard_features, labels, cv=5)
    # print("准确率：", scores.mean())

    # 训练模型
    knn.fit(x_train, y_train)
    y_predict = knn.predict(x_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_predict)
    print(f"准确率为: {accuracy}")
    # 精准率
    print("精准率：", precision_score(y_test, y_predict))
    # 召回率
    print("召回率：", recall_score(y_test, y_predict))
    # F1-Score
    print("F1得分：", f1_score(y_test, y_predict))

    plotting(knn, x_test, y_test)
    plt.show()
