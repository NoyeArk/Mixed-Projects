# SGD分类相关库
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'  # 使用微软雅黑字体
from sklearn.metrics import accuracy_score, recall_score

from plotting import plotting
from utils import get_data


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_data()

    sgd = SGDClassifier(loss="log_loss")
    sgd.fit(x_train, y_train)
    plotting(sgd, x_test, y_test)

    y_pred = sgd.predict(x_test)

    print(f"决策树模型的准确率为: {accuracy_score(y_test, y_pred)}")
    print(f"决策树模型的召回率为: {recall_score(y_test, y_pred)}")

    plt.show()
