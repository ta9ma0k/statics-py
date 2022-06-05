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
