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

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel('小テスト')
    ax.set_ylabel('期末テスト')
    ax.plot(xs, ys, color='gray', label=f'{poly_fit[1]:.2f}+{poly_fit[0]:.2f}x')
    ax.scatter(x, y)
    ax.legend()
    plt.show()
