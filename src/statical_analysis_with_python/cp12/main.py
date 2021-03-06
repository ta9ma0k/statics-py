import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

if __name__ == '__main__':
    df = pd.read_csv('../../../data/ch12_scores_reg.csv')
    n = len(df)
    print(f'length={n}')
    print(df.head())

    x = np.array(df['小テスト'])
    y = np.array(df['期末テスト'])
    p = 1

    poly_fit = np.polyfit(x, y, 1)
    poly_1d = np.poly1d(poly_fit)
    xs = np.linspace(x.min(), x.max())
    ys = poly_1d(xs)

   # fig = plt.figure(figsize=(10, 6))
   # ax = fig.add_subplot(111)
   # ax.set_xlabel('小テスト')
   # ax.set_ylabel('期末テスト')
   # ax.plot(xs, ys, color='gray', label=f'{poly_fit[1]:.2f}+{poly_fit[0]:.2f}x')
   # ax.scatter(x, y)
   # ax.legend()
   # plt.show()

    formula = '期末テスト ~ 小テスト'
    result = smf.ols(formula, df).fit()
    print(result.summary())

    X = np.array([np.ones_like(x), x]).T
    beta0_hat, beta1_hat = np.linalg.lstsq(X, y, rcond=1)[0]
    print(f'beta0={beta0_hat}, beta1={beta1_hat}')
    
    y_hat = beta0_hat + beta1_hat * x
    eps_hat = y - y_hat
    s_var = np.var(eps_hat, ddof=p+1)
    print(f's_var={s_var}')

    C0, C1 = np.diag(np.linalg.pinv(np.dot(X.T, X)))
    print(np.sqrt(s_var * C0), np.sqrt(s_var * C1))

    rv = stats.t(n - 2)
    lcl = beta0_hat - rv.isf(0.025) * np.sqrt(s_var * C0)
    hcl = beta0_hat - rv.isf(0.975) * np.sqrt(s_var * C0)
    print(lcl, hcl)

    rv = stats.t(n - 2)
    lcl = beta1_hat - rv.isf(0.025) * np.sqrt(s_var * C1)
    hcl = beta1_hat - rv.isf(0.975) * np.sqrt(s_var * C1)
    print(lcl, hcl)

    t = beta1_hat / np.sqrt(s_var * C1)
    print(t)
    print(1 - rv.cdf(t) * 2)

    t = beta0_hat / np.sqrt(s_var * C0)
    print(t)
    print(1 - rv.cdf(t) * 2)

    print(np.sum(eps_hat ** 2))
    total_var = np.sum((y - np.mean(y)) ** 2)
    exp_var = np.sum((y_hat - np.mean(y)) ** 2)
    unexp_var = np.sum(eps_hat ** 2)
    print(total_var, exp_var + unexp_var)
    print(exp_var / total_var)
    print(np.corrcoef(x, y)[0, 1] ** 2)

    print(1 - (unexp_var / (n - p - 1)) / ( total_var / (n - 1) ))

    f = (exp_var / p) / (unexp_var / (n - p -1))
    print(f)
    rv = stats.f(p, n-p-1)
    print(1 - rv.cdf(f))

    rv = stats.norm(y_hat, np.sqrt(unexp_var / n))
    mll = np.sum(np.log(rv.pdf(y)))
    print(mll)

    aic = -2 * mll + 2 * (p + 1)
    print(aic)

    bic = -2 * mll + np.log(n) * (p + 1)
    print(bic)

    print('#' * 40)
    formula = '期末テスト ~ 小テスト + 睡眠時間'
    result = smf.ols(formula, df).fit()
    print(result.summary())

    x1 = df['小テスト']
    x2 = df['睡眠時間']
    y = df['期末テスト']
    p = 2

    X = np.array([np.ones_like(x1), x1, x2]).T
    beta0_hat , beta1_hat, beta2_hat = np.linalg.lstsq(X, y, rcond=1)[0]
    print(beta0_hat, beta1_hat, beta2_hat)

    y_hat = beta0_hat + beta1_hat * x1 + beta2_hat * x2
    eps_hat = y - y_hat
    s_var = np.sum(eps_hat ** 2) / (n - p - 1)
    C0, C1, C2 = np.diag(np.linalg.pinv(np.dot(X.T, X)))
    rv = stats.t(n - p - 1)
    lcl = beta2_hat - rv.isf(0.025) * np.sqrt(s_var * C2)
    hcl = beta2_hat - rv.isf(0.975) * np.sqrt(s_var * C2)
    print(lcl, hcl)
    
    print('#' * 40)
    formula = '期末テスト ~ 小テスト + 睡眠時間 + 通学方法'
    result = smf.ols(formula, df).fit()
    print(result.summary())

    
