import pandas as pd
# 数据集预处理相关库
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data():
    # 读取数据
    data = pd.read_csv(r'D:\Code\python\test-tfen\美赛校赛\processed_data.csv', header=0)
    print(type(data))
    return data


def split_data_blog_2(data):
    data['age'] = (data['age'] / 365)
    print(data['age'])

    # 去掉标签列和ID列
    columns_to_drop = ['id', 'cardio']
    features = data.drop(columns=columns_to_drop)
    labels = data['cardio'].values

    # 数据预处理
    features = pd.get_dummies(features)
    standard_features = StandardScaler().fit_transform(features)
    # print(standard_features)
    # print(labels.shape)
    x_train, x_test, y_train, y_test = train_test_split(standard_features, labels, test_size=0.2)

    return x_train, x_test, y_train, y_test


def get_data():
    data = load_data()
    x_train, x_test, y_train, y_test = split_data_blog_2(data)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    df = load_data()

    # 清理数据
    df.drop(df[(df['height'] > df['height'].quantile(0.975)) | (df['height'] < df['height'].quantile
    (0.025))].index, inplace=True)

    df.drop(df[(df['weight'] > df['weight'].quantile(0.975)) | (df['weight'] < df['weight'].quantile
    (0.025))].index, inplace=True)

    print("Diastilic pressure is higher than systolic one in {0} cases".format(df[df['ap_lo'] > df
    ['ap_hi']].shape[0]))

    df.drop(df[(df['ap_hi'] > df['ap_hi'].quantile(0.975)) | (df['ap_hi'] < df['ap_hi'].quantile
    (0.025))].index, inplace=True)

    df.drop(df[(df['ap_lo'] > df['ap_lo'].quantile(0.975)) | (df['ap_lo'] < df['ap_lo'].quantile
    (0.025))].index, inplace=True)

    print("Diastilic pressure is higher than systolic one in {0} cases".format(df[df['ap_lo'] > df
    ['ap_hi']].shape[0]))

    print("数据处理前的统计信息")
    print(df.describe())

    df.to_csv('processed_data.csv', index=True)