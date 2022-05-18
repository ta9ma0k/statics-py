from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

LINE_STYLES = ['-', '--', ':']

ProbSet = (list[int], Callable[[int], float])


def E(x: ProbSet, g=lambda x: x):
    x_set, f = x
    return np.sum([g(x_k) * f(x_k) for x_k in x_set])


def V(x: ProbSet, g=lambda x: x):
    x_set, f = x
    mean = E(x, g)
    return np.sum([(g(x_k) - mean) ** 2 * f(x_k) for x_k in x_set])


def check_prob(x: ProbSet):
    x_set, f = x
    prob = np.array([f(x_k) for x_k in x_set])
    assert np.all(prob >= 0), '負の確率があります'
    prob_sum = np.round(np.sum(prob), 6)
    assert prob_sum == 1, f'確率の和が{prob_sum}になりました'
    print(f'期待値は{E(x):.4}')
    print(f'分散は{V(x):.4}')


def plot_prob(x: ProbSet):
    x_set, f = x
    prob = np.array([f(x_k) for x_k in x_set])
    mean = E(x)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(x_set, prob, label='prob')
    ax.vlines(mean, 0, 1, label='mean')
    ax.set_xticks(np.append(x_set, mean))
    ax.set_ylim(0, prob.max() * 1.2)
    ax.legend()

    plt.show()
