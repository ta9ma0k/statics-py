import numpy as np
import pandas as pd
from scipy import stats

pd.options.display.precision = 3
np.random.seed(1111)

if __name__ == '__main__':
    df = pd.read_csv('../../../data/ch11_potato.csv')
    sample = np.array(df['重さ'])
    print('ポテトの重さの標本')
    print(sample)

    s_mean = np.mean(sample)
    print('標本平均')
    print(s_mean)

    rv = stats.norm(130, np.sqrt(9 / 14))
    print(rv.isf(0.95))

    z = (s_mean - 130) / np.sqrt(9/14)
    print(z)

    rv = stats.norm()
    print(rv.isf(0.95))

    print(rv.cdf(z))       
    print(rv.cdf(z) * 2)

    rv = stats.norm(130, 3)
    c = stats.norm().isf(0.95)
    n_samples = 1000
    cnt = 0
    for _ in range(n_samples):
        sample_ = np.round(rv.rvs(14), 2)
        s_mean_ = np.mean(sample_)
        z = (s_mean_ - 130) / np.sqrt(9/14)
        if z < c:
            cnt += 1
    print('第一種の過誤')
    print(cnt/n_samples)

    rv = stats.norm(128, 3)
    c = stats.norm().isf(0.95)
    n_samples = 1000
    cnt = 0
    for _ in range(n_samples):
        sample_ = np.round(rv.rvs(14), 2)
        s_mean_ = np.mean(sample_)
        z = (s_mean_ - 130) / np.sqrt(9/14)
        if z >= c:
            cnt += 1
    print('第二種の過誤')
    print(cnt/n_samples)

