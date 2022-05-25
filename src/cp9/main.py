from typing import Callable

import numpy as np

np.random.seed(0)

XY = (np.ndarray, np.ndarray, Callable[[float, float], float])


def E(xy: XY, g):
    x_set, y_set, f_XY = xy
    return np.sum([g(x_i, y_i) * f_XY(x_i, y_i) for x_i in x_set for y_i in y_set])


def Cov(xy: XY, g):
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

