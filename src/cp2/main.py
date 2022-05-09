import numpy as np
import pandas as pd

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
