import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd
import dataGenerator
from normalizer import *


def get_gbm(x_train, x_test, y_train, y_test, params):
    # 训练并返回预测结果
    train_data = lgb.Dataset(x_train, label=y_train)
    validation_data = lgb.Dataset(x_test, label=y_test)
    gbm = lgb.train(params, train_data, valid_sets=[validation_data])
    return gbm


def binary_rolling(X, Y, train_inter, test_inter):
    params = {
        'learning_rate': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'objective': 'binary',
    }

    y_pred = []
    i = train_inter+test_inter
    while i < len(X):
        x = minmax(X[i-train_inter-test_inter: i])
        y = minmax(Y[i-train_inter-test_inter: i])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_inter, shuffle=False)
        model = get_gbm(x_train, x_test, y_train, y_test, params)
        y_pred.extend(model.predict(x_test))
        i += test_inter
    else:
        x = minmax(X[i-train_inter-test_inter:])
        y = minmax(Y[i-train_inter-test_inter:])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=len(y)-train_inter, shuffle=False)
        model = get_gbm(x_train, x_test, y_train, y_test, params)
        y_pred.extend(model.predict(x_test))
    return np.array(y_pred)


def regression_rolling(X, Y, train_inter, test_inter):
    params = {
        'learning_rate': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'max_depth': 5,
        'objective': 'regression',
    }

    y_pred = []
    i = train_inter+test_inter
    while i < len(X):
        x = norm(X[i-train_inter-test_inter: i])
        y = norm(Y[i-train_inter-test_inter: i])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_inter, shuffle=False)
        model = get_gbm(x_train, x_test, y_train, y_test, params)
        y_pred.extend(model.predict(x_test))
        i += test_inter
    else:
        x = norm(X[i-train_inter-test_inter:])
        y = norm(Y[i-train_inter-test_inter:])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=len(y)-train_inter, shuffle=False)
        model = get_gbm(x_train, x_test, y_train, y_test, params)
        y_pred.extend(model.predict(x_test))
    return np.array(y_pred)


def binary_predict(df, window=1, rolling=0):
    # df index:time columns:close, high, low, open, volume
    params = {
        'learning_rate': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'max_depth': 5,
        'objective': 'binary',
    }
    x_train, y_train, x_test = dataGenerator.binary_latest_xy(df, window=window, rolling=rolling)
    train_data = lgb.Dataset(x_train, label=y_train)
    return lgb.train(params, train_data).predict(x_test)[0]


if __name__ == '__main__':
    time = 1
    length = 10000
    row_df = pd.read_csv('BTCUSDT-1h.csv', index_col=0).iloc[:length]
    X, Y = dataGenerator.binary_xy(row_df, time, False)
    # Y = np.append(Y[1:], 1)

    # 365 * 24
    train_size = len(Y) - 5000
    # 24 * 5
    test_size = 1000

    none_len = length-len(Y)
    start_index = train_size+none_len


    predict = binary_rolling(X, Y, train_size, test_size)
    # 返回0到1之间的一个概率， >0.5就更可能涨， 2(y_pred-0.5)得到持仓
    position = 2 * (predict - 0.5)
    # 真实涨跌， 1涨 0跌 ，2（Y-0.5)得到 1或-1
    label = 2 * (Y[train_size:] - 0.5)
    # 两者相乘得到不考虑价格的盈亏
    result = position * label
    # print(result)
    print(np.mean(result))
    # result乘以return   c[t]/c[t-1] - 1

    print(start_index)
    price_diff = np.array(row_df['close'])[start_index:] / np.array(row_df['close'])[start_index-time:-time] - 1
    output = result * price_diff
    print(np.mean(output))
