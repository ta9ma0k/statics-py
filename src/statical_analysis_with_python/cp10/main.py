import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

pd.options.display.precision = 3

if __name__ == '__main__':
    df = pd.read_csv("../../../data/ch4_scores400.csv")
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
    s_mean = np.mean(sample)
    u_var = np.var(sample, ddof=1)

    np.random.seed(1111)
    n_samples = 10000
    samples = np.random.choice(scores, (n_samples, n))

    for i in range(5):
        print(f'{i + 1}回目の標本平均: {np.mean(samples[i]):.3f}')
    sample_means = np.mean(samples, axis=1)
    print('標本平均の平均　不偏性')
    print(np.mean(sample_means))
    print('サンプルサイズを１万にしたときの標本平均　一致性')
    # print(np.mean(np.random.choice(scores, int(1e6))))

    for i in range(5):
        print(f'{i + 1}回目の標本分散: {np.var(samples[i]):.3f}')
    sample_vars = np.var(samples, axis=1)
    print('標本分散の平均')
    print(np.mean(sample_vars))

    sample_u_vars = np.var(samples, axis=1, ddof=1)
    print('不偏分散の平均 不偏性')
    print(np.mean(sample_u_vars))
    print('サンプルサイズを１万にしたときの不偏分散 一致性')
    # print(np.var(np.random.choice(scores, int(1e6)), ddof=1))

    rv = stats.norm()
    lcl = s_mean - rv.isf(0.025) * np.sqrt(p_var / n)
    ucl = s_mean - rv.isf(0.975) * np.sqrt(p_var / n)
    print('95% confidence interval')
    print(lcl, ucl)


    def show_ci():
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)

        rv = stats.norm()
        n_samples = 20
        ax.vlines(p_mean, 0, 21)
        for i in range(n_samples):
            sample_ = samples[i]
            s_mean_ = np.mean(sample_)
            lcl = s_mean_ - rv.isf(0.025) * np.sqrt(p_var / n)
            ucl = s_mean_ - rv.isf(0.975) * np.sqrt(p_var / n)

            if lcl <= p_mean <= ucl:
                ax.scatter(s_mean_, n_samples - i, color='gray')
                ax.hlines(n_samples - i, lcl, ucl, color='gray')
            else:
                ax.scatter(s_mean_, n_samples - i, color='b')
                ax.hlines(n_samples - i, lcl, ucl, color='b')
            ax.set_xticks([p_mean])
            ax.set_xticklabels(['母平均'])
        plt.show()


    rv = stats.norm()
    cnt = 0
    for sample_ in samples:
        s_mean_ = np.mean(sample_)
        lcl = s_mean_ - rv.isf(0.025) * np.sqrt(p_var / n)
        ucl = s_mean_ - rv.isf(0.975) * np.sqrt(p_var / n)
        if lcl <= p_mean <= ucl:
            cnt += 1
    print(cnt / len(samples))

    sample_y = sample_u_vars * (n - 1) / p_var


    def show_chi_dist():
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        xs = np.linspace(0, 40, 100)
        rv = stats.chi2(df=n - 1)
        ax.plot(xs, rv.pdf(xs), color='gray')
        hist, _, _ = ax.hist(sample_y, bins=100, range=(0, 40), density=True)

        plt.show()


    rv = stats.chi2(df=n - 1)
    lcl = (n - 1) * u_var / rv.isf(0.025)
    hcl = (n - 1) * u_var / rv.isf(0.975)

    print(lcl, hcl)


    def show_sample_var():
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        rv = stats.chi2(df=n - 1)
        n_samples = 20
        ax.vlines(p_var, 0, 21)
        for i in range(n_samples):
            sample_ = samples[i]
            u_var_ = np.var(sample_, ddof=1)
            lcl = (n - 1) * u_var_ / rv.isf(0.025)
            ucl = (n - 1) * u_var_ / rv.isf(0.975)
            if lcl <= p_var <= ucl:
                ax.scatter(u_var_, n_samples - i, color='gray')
                ax.hlines(n_samples - i, lcl, ucl, 'gray')
            else:
                ax.scatter(u_var_, n_samples - i, color='b')
                ax.hlines(n_samples - i, lcl, ucl, 'b')
        ax.set_xticks([p_var])
        ax.set_xticklabels(['bobunnsann'])
        plt.show()

    rv = stats.chi2(df=n-1)
    cnt = 0
    for sample_ in samples:
        u_var_ = np.var(sample_, ddof=1)
        lcl = (n-1) * u_var_ / rv.isf(0.025)
        ucl = (n-1) * u_var_ / rv.isf(0.975)
        if lcl <= p_var <= ucl:
            cnt += 1
    print(cnt / len(samples))

    rv = stats.t(df=n-1)
    lcl = s_mean - rv.isf(0.025) * np.sqrt(u_var/n)
    ucl = s_mean - rv.isf(0.975) * np.sqrt(u_var/n)
    print(lcl, ucl)
    