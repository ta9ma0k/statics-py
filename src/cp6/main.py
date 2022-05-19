from typing import Callable

import numpy as np
from scipy.special import comb, factorial

from common.prob_util import check_prob, plot_prob


class Probability:
    def __init__(self, p: float):
        if p < 0 or 1 < p:
            raise ValueError()
        self.value = p


ProbabilityDistribution = (np.ndarray, Callable[[int], float])


def Bern(p: Probability) -> ProbabilityDistribution:
    x_set = np.array([0, 1])

    def f(x):
        if x in x_set:
            return p.value ** x * (1 - p.value) ** (1 - x)
        else:
            return 0

    return x_set, f


def Bin(n: int, p: Probability) -> ProbabilityDistribution:
    x_set = np.arange(n + 1)

    def f(x):
        if x in x_set:
            return comb(n, x) * p.value ** x * (1 - p.value) ** (n - x)
        else:
            return 0

    return x_set, f


def Ge(p: Probability, n: int = 30) -> ProbabilityDistribution:
    if n < 0 or 30 < n:
        raise ValueError()
    x_set = np.arange(1, n)

    def f(x):
        if x in x_set:
            return p.value * (1 - p.value) ** (x - 1)
        else:
            return 0

    return x_set, f


def Poi(lam: float, n: int = 20) -> ProbabilityDistribution:
    if n < 0 or 20 < n:
        raise ValueError()
    x_set = np.arange(n)

    def f(x):
        if x in x_set:
            return np.power(lam, x) / factorial(x) * np.exp(-lam)
        else:
            return 0

    return x_set, f


if __name__ == '__main__':

    X = Poi(3)

    check_prob(X)
    plot_prob(X)
