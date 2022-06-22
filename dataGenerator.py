import pandas as pd
import numpy as np
import bottleneck as bn
import talib


def indicators(df, append=False):
    if append:
        new_df = df.copy()
    else:
        new_df = pd.DataFrame(data=None, index=df.index)
    o = np.array(df['open'])
    h = np.array(df['high'])
    l = np.array(df['low'])
    c = np.array(df['close'])
    v = np.array(df['volume'])
    #
    ts_w = 14
    new_df['ts_sum'] = bn.move_sum(c, ts_w, 1)
    new_df['ts_mean'] = bn.move_mean(c, ts_w, 1)
    new_df['ts_median'] = bn.move_median(c, ts_w, 1)
    new_df['ts_max'] = bn.move_max(c, ts_w, 1)
    new_df['ts_min'] = bn.move_min(c, ts_w, 1)
    new_df['ts_argmax'] = bn.move_argmax(c, ts_w, 1)
    new_df['ts_argmin'] = bn.move_argmin(c, ts_w, 1)
    new_df['ts_std'] = bn.move_std(c, ts_w, 1)
    new_df['ts_var'] = bn.move_var(c, ts_w, 1)
    new_df['ts_rank'] = bn.move_rank(c, ts_w, 1)
    # # talib-momentum
    # new_df['ta_ADX'] = talib.ADX(h, l, c, ts_w)
    # new_df['ta_ADXR'] = talib.ADXR(h, l, c, ts_w)
    # new_df['ta_APO'] = talib.APO(c, ts_w-2, ts_w*2-2)
    # new_df['ta_AROONOSC '] = talib.AROONOSC(h, l, ts_w)
    # new_df['ta_BOP'] = talib.BOP(o, h, l, c)
    # new_df['ta_CCI '] = talib.CCI(h, l, c, ts_w)
    # new_df['ta_CMO'] = talib.CMO(c, ts_w)
    # new_df['ta_DX'] = talib.DX(h, l, c, ts_w)
    # new_df['ta_MFI'] = talib.MFI(h, l, c, v, ts_w)
    # new_df['ta_MINUS_DI'] = talib.MINUS_DI(h, l, c, ts_w)
    # new_df['ta_MINUS_DM'] = talib.MINUS_DM(h, l, ts_w)
    # new_df['ta_MOM'] = talib.MOM(c, ts_w)
    # new_df['ta_PLUS_DI'] = talib.PLUS_DI(h, l, c, ts_w)
    # new_df['ta_PLUS_DM'] = talib.PLUS_DM(h, l, ts_w)
    # new_df['ta_PPO'] = talib.PPO(c, ts_w-2, ts_w*2-2)
    # new_df['ta_ROC'] = talib.ROC(c, ts_w)
    # new_df['ta_ROCP'] = talib.ROCP(c, ts_w)
    # new_df['ta_ROCR'] = talib.ROCR(c, ts_w)
    # new_df['ta_ROCR100'] = talib.ROCR100(c, ts_w)
    # new_df['ta_RSI'] = talib.RSI(c, ts_w)
    # new_df['ta_TRIX'] = talib.TRIX(c, ts_w)
    # new_df['ta_WILLR'] = talib.WILLR(h, l, c, ts_w)
    # # talib-volatility
    # new_df['ta_ATR'] = talib.ATR(h, l, c, ts_w)
    # new_df['ta_NATR'] = talib.NATR(h, l, c, ts_w)
    # new_df['ta_TRANGE'] = talib.NATR(h, l, c)
    return new_df


def minmax_norm(df):
    return (df - df.min()) / (df.max() - df.min())


def regression_xy(df, window=1, norm=False):
    # for regression
    close = np.array(df['close'])
    x = indicators(df, False)
    shift = max([x.reset_index()[col].first_valid_index() for col in x.columns])
    if shift < window:
        shift = window
    x = x.iloc[shift - window:-window]
    y = close[shift:] / close[shift-window:-window] - 1
    if norm:
        x = minmax_norm(x)
    return np.array(x), np.array(y)


def binary_xy(df, window=1, norm=False):
    # for binary classification

    x = indicators(df, False)
    shift = max([x.reset_index()[col].first_valid_index() for col in x.columns])
    if shift < window:
        shift = window
    close = np.array(df['close'])
    x = x.iloc[shift - window:-window]
    y = np.where(close[shift:] - close[shift - window:-window] >= 0, 1, 0)
    if norm:
        x = minmax_norm(x)
    return np.array(x), np.array(y)


def binary_latest_xy(df, window=1, rolling=0):
    close = np.array(df['close'])[-rolling:]
    x = indicators(df, False).iloc[-rolling:-window]
    y = np.where(close[window:] - close[:-window] >= 0, 1, 0)
    x_predict = indicators(df, False).iloc[-window]
    return np.array(x), np.array(y), np.array([x_predict])


if __name__ == '__main__':
    row_df = pd.read_csv('BTCUSDT-1h.csv', index_col=0)
    ind_df = indicators(row_df, True)
    print(ind_df)
    # std_ind = minmax_norm(ind_df)
    # print(std_ind)
    # print(len(Label(row_df, 1, 3).binary()))
    # std_ind.to_csv('X.csv')
    x, y = binary_xy(row_df, 24)

    print(x.shape)
    print(y.shape)

