import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('../../data/ch1_sport_test.csv', index_col='生徒番号')
    print('データフレームを表示')
    print(df)
    print('特定の列を表示')
    print(df['握力'])
    print('行数・列数を表示')
    print(df.shape)
