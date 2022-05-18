from typing import Callable

import numpy as np

from common.prob_util import check_prob, plot_prob


def bern(p: float) -> (np.ndarray, Callable[[int], float]):
    x_set = np.array([0, 1])

    def f(x):
        if x in x_set:
            return p ** x * (1 - p) ** (1 - x)
        else:
            return 0

    return x_set, f


if __name__ == '__main__':
    p = 0.3
    X = bern(p)

    check_prob(X)
    plot_prob(X)
