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

    def pmean_test(sample, mean0, p_var, alpha=0.05):
        s_mean = np.mean(sample)
        n = len(sample)
        rv = stats.norm()
        interval = rv.interval(1 - alpha) 
        
        z = (s_mean - mean0) / np.sqrt(p_var/n)
        if interval[0] <= z <= interval[1]:
            print('帰無仮説を採択')
        else:
            print('帰無仮説を棄却')
        if z < 0:
            p = rv.cdf(z) * 2
        else:
            p = (1 - rv.cdf(z)) * 2
        print(f'p value is {p:.3f}')

    def pvar_test(sample, var0, alpha=0.05):
        u_var = np.var(sample, ddof=1)
        n = len(sample)
        rv = stats.chi2(df=n-1)
        interval = rv.interval(1-alpha)

        y = (n-1) * u_var / var0
        if interval[0] <= y <= interval[1]:
             print('帰無仮説を採択')
        else:
            print('帰無仮説を棄却')
        if y < rv.isf(0.05):
            p = rv.cdf(y) * 2
        else:
            p = (1-rv.cdf(y)) * 2
        print(f'p value is {p:.3f}')

    def pmean_test(sample, mean0, alpha=0.05):
        s_mean = np.mean(sample)
        u_var = np.var(sample, ddof=1)
        n = len(sample)
        rv = stats.t(df=n-1)
        interval = rv.interval(1-alpha)
        
        t = (s_mean - mean0) / np.sqrt(u_var/n)
        if interval[0] <= t <= interval[1]:
            print('帰無仮説を採択')
        else:
            print('帰無仮説を棄却')

        if t < 0:
            p = rv.cdf(t) * 2
        else:
            p = (1 - rv.cdf(t)) * 2
        print(f'p value is {p:.3f}')

    training_rel = pd.read_csv('../../../data/ch11_training_rel.csv')
    training_rel['差'] = training_rel['後'] - training_rel['前']

    t, p = stats.ttest_1samp(training_rel['差'], 0)
    print(f'p value = {p}')

    training_ind = pd.read_csv('../../../data/ch11_training_ind.csv')
    t, p = stats.ttest_ind(training_ind['A'], training_ind['B'], equal_var=False)
    print(f'p value = {p}')

    T, p = stats.wilcoxon(training_rel['前'], training_rel['後'])
    print(f'p value={p}')

    u, p = stats.mannwhitneyu(training_ind['A'], training_ind['B'], alternative='two-sided')
    print(f'p value={p}')
    
    ad_df = pd.read_csv('../../../data/ch11_ad.csv')
    n = len(ad_df)
    print(n)
    print(ad_df.head())

    ad_cross = pd.crosstab(ad_df['広告'], ad_df['購入'])
    print(ad_cross)

    print(ad_cross['した'] / (ad_cross['した'] + ad_cross['しなかった']))

    n_yes, n_not = ad_cross.sum()
    n_adA, n_adB = ad_cross.sum(axis=1)
    ad_ef = pd.DataFrame({'した': [n_adA * n_yes / n, n_adB * n_yes / n], 
        'しなかった': [n_adA * n_not /n, n_adB * n_not / n]}, index=['A', 'B'])
    print(ad_ef)
    print(ad_cross - ad_ef)
    y = ((ad_cross - ad_ef) ** 2 / ad_ef).sum().sum()
    print(y)

    rv = stats.chi2(1)
    print(1 - rv.cdf(y))
