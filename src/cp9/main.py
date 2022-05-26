from typing import Callable

import numpy as np

np.random.seed(0)

XY = (np.ndarray, np.ndarray, Callable[[float, float], float])


def E(xy: XY, g):
    x_set, y_set, f_XY = xy
    return np.sum([g(x_i, y_i) * f_XY(x_i, y_i) for x_i in x_set for y_i in y_set])


def Cov(xy: XY):
    x_set, y_set, f_XY = XY
    mean_X = E(XY, lambda x, y: x)
    mean_Y = E(XY, lambda x, y: y)
    return np.sum([(x_i - mean_X) * (y_i - mean_Y) * f_XY(x_i, y_i) for x_i in x_set for y_i in y_set])


def f_X(x, XY):
    _, y_set, f_XY = XY
    return np.sum([f_XY(x, y_k) for y_k in y_set])


def f_Y(y, XY):
    x_set, _, f_XY = XY
    return np.sum([f_XY(x_k, y) for x_k in x_set])


if __name__ == '__main__':
    x_set = np.arange(1, 7)
    y_set = np.arange(1, 7)

    def f_xy(x, y):
        if x in x_set and y in y_set:
            return x * y / 441
        return 0

    XY = [x_set, y_set, f_xy]

    print('独立な確率変数は無相関になる')
    print(f'Cov = {Cov(XY)}')
    print(f'fx(1) * fy(1) = {f_X(1, XY) * f_Y(1, XY)}, fxy(1, 1) = {f_xy(1, 1)}')

    x_set = np.arange(0, 2)
    y_set = np.arange(-1, 3)

    def f_xy(x, y):
        if (x, y) in [(0, 0), (1, 1), (1, -1)]:
            return 1 / 3
        return 0

    XY = [x_set, y_set, f_xy]
    print('無相関でも独立とは限らない')
    print(f'Cov = {Cov(XY)}')
    print(f'fx(0) * fy(0) = {f_X(0, XY) * f_Y(0, XY)}, fxy(0, 0) = {f_xy(0, 0)}')
