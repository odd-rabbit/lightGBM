import numpy as np


def minmax(arr, axis=0):
    return (arr - arr.min(axis)) / (arr.max(axis) - arr.min(axis))


def norm(arr, axis=0, order=2):
    # 范数
    l2 = np.atleast_1d(np.linalg.norm(arr, order, axis))
    l2[l2 == 0] = 1
    return arr / l2


def zscore(arr):
    # mean=0, std=1 with normal distribution data
    return (arr - np.mean(arr, 0)) / np.std(arr, 0)


def robust(arr):
    q75, q25 = np.percentile(arr, [75, 25])
    return (arr - np.median(arr, 0)) / (q75 - q25)


# if __name__ == '__main__':

