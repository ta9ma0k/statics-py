import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
from scipy.optimize import minimize_scalar

LINE_STYLES = ['-', '--', ':']


def E(X, g=lambda x: x):
    x_range, f = X

    def integrand(x):
        return g(x) * f(x)

    return integrate.quad(integrand, -np.inf, np.inf)[0]


def V(X, g=lambda x: x):
    x_range, f = X
    mean = E(X, g)

    def integrand(x):
        return (g(x) - mean) ** 2 * f(x)

    return integrate.quad(integrand, -np.inf, np.inf)[0]


def check_prob(X):
    x_range, f = X
    f_min = minimize_scalar(f).fun
    assert f_min >= 0, '密度関数が負の値になる'

    prob_sum = np.round(integrate.quad(f, -np.inf, np.inf)[0], 6)
    assert prob_sum == 1, f'確率の和が{prob_sum}になります'

    print(f'mean = {E(X):.3f}')
    print(f'ver  = {V(X):.3f}')


def plot_prob(X, x_min, x_max):
    x_range, f = X

    def F(x):
        return integrate.quad(f, -np.inf, x)[0]

    xs = np.linspace(x_min, x_max, 100)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(xs, [f(x) for x in xs], label='f(x)', color='gray')
    ax.plot(xs, [F(x) for x in xs], label='F(x)', ls='--', color='gray')

    ax.legend()
    plt.show()


def N(mu, sigma):
    x_range = [-np.inf, np.inf]

    def f(x):
        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    return x_range, f


if __name__ == '__main__':
    mu, sigma = 2, 0.5
    X = N(mu, sigma)
    check_prob(X)
    plot_prob(X, 0, 4)
