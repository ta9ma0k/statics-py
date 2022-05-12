import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.char_util import make_alphabet

LABEL_ENGLISH = '英語'
LABEL_MATH = '数学'

if __name__ == '__main__':
    df = pd.read_csv('../../data/ch2_scores_em.csv', index_col='生徒番号')

    en_scores_10 = np.array(df[LABEL_ENGLISH])[:10]
    ma_scores_10 = np.array(df[LABEL_MATH])[:10]

    scores_10_df = pd.DataFrame({LABEL_ENGLISH: en_scores_10, LABEL_MATH: ma_scores_10}, index=pd.Index(make_alphabet(10), name='生徒'))
    print('先頭１０人の英語・数学の点数一覧を表示')
    print(scores_10_df)

    summary_10_df = scores_10_df.copy()
    summary_10_df['英語の偏差'] = summary_10_df[LABEL_ENGLISH] - summary_10_df[LABEL_ENGLISH].mean()
    summary_10_df['数学の偏差'] = summary_10_df[LABEL_MATH] - summary_10_df[LABEL_MATH].mean()
    summary_10_df['偏差の積'] = summary_10_df['英語の偏差'] * summary_10_df['数学の偏差']
    print('偏差を算出')
    print(summary_10_df)
    print(f'偏差の積の平均:{summary_10_df["偏差の積"].mean()}')
    print('共分散行列')
    print(f'np.cov = {np.cov(en_scores_10, ma_scores_10, ddof=0)}')

    print(f'相関係数を算出')
    cov = np.cov(en_scores_10, ma_scores_10, ddof=0)[0, 1]
    std_product = np.std(en_scores_10) * np.std(ma_scores_10)
    corr = cov / std_product
    print(corr)

    print('相関行列を算出')
    print(np.corrcoef(en_scores_10, ma_scores_10))
    print(scores_10_df.corr())

    en_scores = np.array(df[LABEL_ENGLISH])
    ma_scores = np.array(df[LABEL_MATH])

    poly_fit = np.polyfit(en_scores, ma_scores, 1)
    poly_1d = np.poly1d(poly_fit)

    xs = np.linspace(en_scores.min(), en_scores.max())
    ys = poly_1d(xs)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    ax.set_xlabel(LABEL_ENGLISH)
    ax.set_ylabel(LABEL_MATH)
    ax.scatter(en_scores, ma_scores, label='点数')
    ax.plot(xs, ys, color='gray', label=f'{poly_fit[1]:.2f}+{poly_fit[0]:.2f}x')
    ax.legend(loc='upper left')

    plt.show()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    c = ax.hist2d(en_scores, ma_scores, bins=[9, 8], range=[(35, 80), (55, 95)])
    ax.set_xticks(c[1])
    ax.set_yticks(c[2])

    fig.colorbar(c[3], ax=ax)
    plt.show()
