import matplotlib.pyplot as plt
import numpy as np


class ProbabilityDistribution:
    def __init__(self):
        self.x_set = np.arange(1, 7)
        self.prob = np.array([self.f(x) for x in self.x_set])

    def f(self, x: int) -> float:
        if x in self.x_set:
            return x / 21
        else:
            return 0

    def cdf(self, x: int) -> float:
        return np.sum([self.f(x_k) for x_k in self.x_set if x_k <= x])

    def distribution(self) -> dict:
        return dict(zip(self.x_set, self.prob))


class JointProbabilityDistribution:
    def __init__(self):
        self.x_set = np.arange(2, 13)
        self.y_set = np.arange(1, 7)
        self.prob = np.array([[self.f(x_i, y_j) for y_j in self.y_set] for x_i in self.x_set])

    def f(self, x: int, y: int) -> float:
        if 1 <= y <= 6 and 1 <= x - y <= 6:
            return y * (x - y) / 441
        return 0

    def f_x(self, x: int) -> float:
        return np.sum([self.f(x, y_k) for y_k in self.y_set])

    def f_y(self, y: int) -> float:
        return np.sum([self.f(x_k, y) for x_k in self.x_set])


def e(x: np.ndarray, pd: ProbabilityDistribution, g=lambda x: x) -> float:
    return np.sum([g(x_k) * pd.f(x_k) for x_k in x])


def v(x: np.ndarray, pd: ProbabilityDistribution, g=lambda x: x) -> float:
    mean = e(x, pd, g)
    return np.sum([(g(x_k) - mean) ** 2 * pd.f(x_k) for x_k in x])


def show_pd(pd: ProbabilityDistribution):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(pd.x_set, pd.prob)
    ax.set_xlabel('phenomenon')
    ax.set_ylabel('prob')
    plt.show()


def show_jpd(jpd: JointProbabilityDistribution):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    c = ax.pcolor(jpd.prob)
    ax.set_xticks(np.arange(jpd.prob.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(jpd.prob.shape[0]) + 0.5, minor=False)
    ax.set_xticklabels(np.arange(1, 7), minor=False)
    ax.set_yticklabels(np.arange(2, 13), minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    fig.colorbar(c, ax=ax)
    plt.show()


if __name__ == '__main__':
    pd = ProbabilityDistribution()

    print('???????????????')
    print(f'f(x) > 0 : {np.all(pd.prob >= 0)}')
    print(f'sum f(x) = 1 : {np.sum(pd.prob)}')

    print('??????????????????')
    print(f'F(3) = {pd.cdf(3)}')

    print('?????????')
    print(e(pd.x_set, pd))
    print(e(pd.x_set, pd, g=lambda x: 2 * x + 3))
    print(2 * e(pd.x_set, pd) + 3)

    mean = e(pd.x_set, pd)
    print('??????')
    print(v(pd.x_set, pd))
    print(v(pd.x_set, pd, lambda x: x * 2 + 3))
    print(2 ** 2 * v(pd.x_set, pd))

    jpd = JointProbabilityDistribution()
    print('??????????????????')
    print(f'fx(x) > 0 : {np.all(jpd.prob >= 0)}')
    print(f'sum f(x) = 1 : {np.sum(jpd.prob)}')

    prob_x = np.array([jpd.f_x(x_k) for x_k in jpd.x_set])
    prob_y = np.array([jpd.f_y(y_k) for y_k in jpd.y_set])
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.bar(jpd.x_set, prob_x)
    ax1.set_title('x')
    ax1.set_xlabel('x value')
    ax1.set_ylabel('prob')
    ax1.set_xticks(jpd.x_set)

    ax2.bar(jpd.y_set, prob_y)
    ax2.set_title('y')
    ax2.set_xlabel('y value')
    ax2.set_ylabel('prob')
    ax2.set_xticks(jpd.y_set)

    plt.show()
