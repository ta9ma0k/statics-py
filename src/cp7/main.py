import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    x_range = np.array([0, 1])

    def f(x):
        if x_range[0] <= x <= x_range[1]:
            return 2 * x
        else:
            return 0

    X = [x_range, f]

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
