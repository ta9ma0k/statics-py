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


def show_hist(pd: ProbabilityDistribution):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(pd.x_set, pd.prob)
    ax.set_xlabel('phenomenon')
    ax.set_ylabel('prob')
    plt.show()


if __name__ == '__main__':
    pd = ProbabilityDistribution()

    print('確率の性質')
    print(f'f(x) > 0 : {np.all(pd.prob >= 0)}')
    print(f'sum f(x) = 1 : {np.sum(pd.prob)}')

    print('累積分布関数')
    print(f'F(3) = {pd.cdf(3)}')

