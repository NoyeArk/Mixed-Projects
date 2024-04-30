import lightgbm
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

plt.rcParams['font.family'] = 'SimHei'  # 使用微软雅黑字体
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score

from plotting import plotting
from utils import get_data


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_data()

    lgb_clf = lightgbm.LGBMClassifier(boosting_type='gbdt', random_state=1)

    parameters = {'max_depth': np.arange(2, 20, 1)}
    GS = GridSearchCV(lgb_clf, param_grid=parameters, cv=10, scoring='f1', n_jobs=-1)
    GS.fit(x_train, y_train)

    # 测试集
    test_pred = GS.best_estimator_.predict(x_test)
    print(f"准确率为: {accuracy_score(y_test, test_pred)}")
    print(f"精确率为: {precision_score(y_test, test_pred)}")
    print(f"召回率为: {recall_score(y_test, test_pred)}")
    print(f"AUC为: {roc_auc_score(y_test, test_pred)}")

    # F1-score
    print("F1_score of LGBMClassifier is : ", round(f1_score(y_true=y_test, y_pred=test_pred), 2))