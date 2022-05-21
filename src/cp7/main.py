import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.optimize import minimize_scalar


def show_pdf(x_range, f):
    xs = np.linspace(x_range[0], x_range[1], 100)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    ax.plot(xs, [f(x) for x in xs], label='f(x)', color='gray')
    ax.hlines(0, -0.2, 1.2, alpha=0.3)
    ax.vlines(0, -0.2, 2.2, alpha=0.3)
    ax.vlines(xs.max(), 0, 2.2, linestyles=':', color='gray')

    xs = np.linspace(0.4, 0.6, 100)
    ax.fill_between(xs, [f(x) for x in xs], label='prop')
    ax.set_xticks(np.arange(-0.2, 1.3, 0.1))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.2, 2.1)
    ax.legend()

    plt.show()


def show_cdf(x_range, f):
    xs = np.linspace(x_range[0], x_range[1], 100)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    ax.plot(xs, [f(x) for x in xs], label='F(x)', color='gray')
    ax.hlines(0, -0.1, 1.1, alpha=0.3)
    ax.vlines(0, -0.1, 1.1, alpha=0.3)
    ax.vlines(xs.max(), 0, 1, linestyles=':', color='gray')

    ax.set_xticks(np.arange(-0.1, 1.2, 0.1))
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.legend()

    plt.show()


if __name__ == '__main__':
    x_range = np.array([0, 1])


    def f(x):
        if x_range[0] <= x <= x_range[1]:
            return 2 * x
        else:
            return 0


    X = [x_range, f]

    print(f'0.6 - 0.4の積分結果 = {integrate.quad(f, 0.4, 0.6)}')

    res = minimize_scalar(f)
    print('確率の性質を確認')
    print(f'f(x)の最小値 = {res.fun}')
    print(f'inf - (-inf)の積分結果 = {integrate.quad(f, -np.inf, np.inf)[0]}')


    def F(x):
        return integrate.quad(f, -np.inf, x)[0]


    y_range = [3, 5]


    def g(y):
        if y_range[0] <= y <= y_range[1]:
            return (y - 3) / 2
        else:
            return 0


    def G(y):
        return integrate.quad(g, -np.inf, y)[0]


    def integrand(x):
        return x * f(x)


    print(f'期待値 = {integrate.quad(integrand, -np.inf, np.inf)[0]}')
