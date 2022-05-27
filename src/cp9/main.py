from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

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


    def show_sum_norm():
        rv1 = stats.norm(1, np.sqrt(2))
        rv2 = stats.norm(2, np.sqrt(3))

        sample_size = int(1e6)
        x_sample = rv1.rvs(sample_size)
        y_sample = rv2.rvs(sample_size)
        sum_sample = x_sample + y_sample

        print('X-N(1,2), Y-N(2,3)のときのX+Yの期待値、分散')
        print(np.mean(sum_sample), np.var(sum_sample))

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        rv = stats.norm(3, np.sqrt(5))
        xs = np.linspace(rv.isf(0.995), rv.isf(0.005), 100)

        ax.hist(sum_sample, bins=100, density=True, alpha=0.5, label='N(1,2) + N(2,3)', color='gray')
        ax.plot(xs, rv.pdf(xs), label='N(3,5)', color='gray')
        ax.plot(xs, rv1.pdf(xs), label='N(1,2)', color='gray')
        ax.plot(xs, rv2.pdf(xs), label='N(2,3)', color='gray')

        ax.legend()
        ax.set_xlim(rv.isf(0.995), rv.isf(0.005))
        plt.show()


    def show_sum_poisson():
        rv1 = stats.poisson(3)
        rv2 = stats.poisson(4)

        sample_size = int(1e6)
        x_sample = rv1.rvs(sample_size)
        y_sample = rv2.rvs(sample_size)
        sum_sample = x_sample + y_sample

        print('X-Poi(3), Y-Poi(4)のときのX+Yの期待値、分散')
        print(np.mean(sum_sample), np.var(sum_sample))

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        rv = stats.poisson(7)
        xs = np.arange(20)
        hist, _ = np.histogram(sum_sample, bins=20, range=(0, 20), normed=True)

        ax.bar(xs, hist, alpha=0.5, label='Poi(3) + Poi(4)')
        ax.plot(xs, rv.pmf(xs), label='Poi(7)', color='gray')
        ax.plot(xs, rv1.pmf(xs), label='Poi(3)', ls='--', color='gray')
        ax.plot(xs, rv2.pmf(xs), label='Poi(4)', ls=':', color='gray')

        ax.legend()
        ax.set_xlim(-0.5, 20)
        ax.set_xticks(np.arange(20))
        plt.show()

