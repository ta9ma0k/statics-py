import numpy as np
import pandas as pd
import matplotlib.pylab as plt

if __name__ == '__main__':
    df = pd.read_csv('../../data/ch2_scores_em.csv', index_col='生徒番号')
    print('先頭５行を表示')
    print(df.head())

    en_scores = np.array(df['英語'])[:10]
    print('先頭１０人の英語の点数リストを表示')
    print(en_scores)

    en_scores_df = pd.DataFrame({'点数': en_scores}, index=pd.Index([chr(ord("A") + i) for i in range(10)], name='生徒'))
    print('indexをつけた英語の点数リストを表示')
    print(en_scores_df)

    print('平均値')
    print(f'sum(en_scores) / len(en_scores) = {sum(en_scores) / len(en_scores)}')
    print(f'np.mean(en_scores) = {np.mean(en_scores)}')
    print(f'en_scores_df.mean() = {en_scores_df.mean()}')

    sorted_en_scores = np.sort(en_scores)
    print('sortした英語の点数リストを表示')
    print(sorted_en_scores)


    def median(sorted_array: np.ndarray) -> float:
        n = len(sorted_array)
        if n % 2 == 0:
            m0 = sorted_array[n // 2 - 1]
            m1 = sorted_array[n // 2 + 1]
            return (m0 + m1) / 2
        else:
            return sorted_array[(n + 1) / 2 - 1]


    print('中央値')
    print(f'median (udf) = {median(sorted_en_scores)}')
    print(f'np.median = {np.median(en_scores)}')
    print(f'en_scores_df.median = {en_scores_df.median()}')

    print('中央値')
    print(f'series.mode [1, 1, 1, 2, 2, 3] = {pd.Series([1, 1, 1, 2, 2, 3]).mode()}')
    print(f'series.mode [1, 1, 2, 2, 3] = {pd.Series([1, 1, 2, 2, 3]).mode()}')

    mean = np.mean(en_scores)
    deviation = en_scores - mean
    print('偏差')
    print(deviation)

    summary_df = en_scores_df.copy()
    summary_df['偏差'] = deviation
    print('偏差を付け加えた英語の点数リストを表示')
    print(summary_df)
    print(summary_df.mean())

    print('分散(numpy default)')
    print(f'np.mean(deviation ** 2) = {np.mean(deviation ** 2)}')
    print(f'np.var = {np.var(en_scores)}')
    print('不偏分散(pandas default)')
    print(f'en_scores_df.var = {en_scores_df.var()}')

    summary_df['偏差二乗'] = np.square(deviation)
    print('偏差二乗を付け加えた英語の点数リストを表示')
    print(summary_df)
    print(summary_df.mean())

    print('標準偏差')
    print(f'{np.sqrt(np.var(en_scores, ddof=0))}')
    print(f'np.std = {np.std(en_scores, ddof=0)}')

    print('範囲')
    print(f'max - min = {np.max(en_scores) - np.min(en_scores)}')
    print('四分位範囲')
    print(f'q3 - q1 = {np.percentile(en_scores, 75) - np.percentile(en_scores, 25)}')

    print('dataframe describe')
    print(pd.Series(en_scores).describe())

    z = (en_scores - np.mean(en_scores)) / np.std(en_scores)
    print('標準化（ｚスコア）')
    print(z)
    print(f'z score mean = {np.mean(z)}')
    print(f'z score std = {np.std(z, ddof=0)}')

    print('偏差値')
    en_scores_df['偏差値'] = 50 + 10 * z
    print(en_scores_df)

    all_en_scores = np.array(df['英語'])
    print(pd.Series(all_en_scores).describe())
    freq, _ = np.histogram(all_en_scores, bins=10, range=(0, 100))
    print('度数')
    print(freq)

    freq_class = [f'{i}~{i + 10}' for i in range(0, 100, 10)]
    freq_dist_df = pd.DataFrame({'度数': freq}, index=pd.Index(freq_class, name='階級'))
    print('度数分布表')
    print(freq_dist_df)

    class_value = [(i + (i + 10)) // 2 for i in range(0, 100, 10)]
    print('階級値')
    print(class_value)

    rel_freq = freq / freq.sum()
    print('相対度数')
    print(rel_freq)

    cum_rel_freq = np.cumsum(rel_freq)
    print('累積相対度数')
    print(cum_rel_freq)

    freq_dist_df['階級値'] = class_value
    freq_dist_df['相対度数'] = rel_freq
    freq_dist_df['累積相対度数'] = cum_rel_freq
    freq_dist_df = freq_dist_df[['階級値', '度数', '相対度数', '累積相対度数']]
    print(freq_dist_df)

    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)

    ax1.set_xlabel('点数')
    ax1.set_ylabel('人数')
    ax1.set_xticks(np.linspace(0, 100, 25 + 1))

    ax2 = ax1.twinx()
    weights = np.ones_like(all_en_scores) / len(all_en_scores)
    rel_freq_4_hist, _, _ = ax1.hist(all_en_scores, bins=25, range=(0, 100), weights=weights)

    cum_rel_freq_4_hist = np.cumsum(rel_freq_4_hist)
    class_value_4_hist = [(i + (i + 4))//2 for i in range(0, 100, 4)]
    ax2.plot(class_value_4_hist, cum_rel_freq_4_hist, ls='--', marker='o', color='gray')
    ax2.grid(visible=False)
    ax2.set_ylabel('累積相対度数')

    plt.show()

    fig = plt.figure(figsize=(5, 6))
    ax = fig.add_subplot(111)
    ax.boxplot(all_en_scores, labels=['英語'])
    plt.show()