import xgboost as xgb
from sklearn.metrics import accuracy_score

from utils import get_data


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_data()

    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(x_train, y_train)

    y_pred = xgb_model.predict(x_test)

    print("Accuracy:")
    print(accuracy_score(y_test, y_pred))