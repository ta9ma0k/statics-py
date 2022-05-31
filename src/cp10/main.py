import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

pd.options.display.precision = 3

if __name__ == '__main__':
    df = pd.read_csv("../../data/ch4_scores400.csv")
    scores = np.array(df['点数'])

    p_mean = np.mean(scores)
    p_var = np.var(scores)

    print(f'母平均 = {p_mean}')
    print(f'母分散 = {p_var}')


    def show(p_mean, p_var):
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        xs = np.arange(101)
        rv = stats.norm(p_mean, np.sqrt(p_var))
        ax.plot(xs, rv.pdf(xs), color='gray')
        ax.hist(scores, bins=100, range=(0, 100), density=True)


    plt.show()

    np.random.seed(0)
    n = 20
    sample = np.random.choice(scores, n)

    np.random.seed(1111)
    n_samples = 10000
    samples = np.random.choice(scores, (n_samples, n))

    for i in range(5):
        print(f'{i+1}回目の標本平均: {np.mean(samples[i]):.3f}')
    sample_means = np.mean(samples, axis=1)
    print('標本平均の平均　不偏性')
    print(np.mean(sample_means))
    print('サンプルサイズを１万にしたときの標本平均　一致性')
    print(np.mean(np.random.choice(scores, int(1e6))))

    for i in range(5):
        print(f'{i + 1}回目の標本分散: {np.var(samples[i]):.3f}')
    sample_vars = np.var(samples, axis=1)
    print('標本分散の平均')
    print(np.mean(sample_vars))

    sample_u_vars = np.var(samples, axis=1, ddof=1)
    print('不偏分散の平均 不偏性')
    print(np.mean(sample_u_vars))
    print('サンプルサイズを１万にしたときの不偏分散 一致性')
    print(np.var(np.random.choice(scores, int(1e6)), ddof=1))
