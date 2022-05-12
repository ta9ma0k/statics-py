import pandas as pd

if __name__ == '__main__':
    df1 = pd.read_excel('../../data/a001_8.xlsx')
    df2 = pd.read_excel('../../data/a001_9.xlsx')

    print(df1.head())
    print(df2.head())