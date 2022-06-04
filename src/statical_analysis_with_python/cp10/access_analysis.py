import numpy as np
import pandas as pd
from scipy import stats

pd.options.display.precision = 3


if __name__ == '__main__':
    n_access_df = pd.read_csv('../../../data/ch10_access.csv')
    n_access = np.array(n_access_df['アクセス数'])
    n = len(n_access)

    print('1時間毎のアクセス数の10件を表示')
    print(n_access[:10])

    s_mean = n_access.mean()
    print(f'標本平均={s_mean}')

    print('中心極限定理によって標本平均は近似的に正規分布に従うとするとき')
    rv = stats.norm()
    lcl = s_mean - rv.isf(0.025) * np.sqrt(s_mean/n)
    ucl = s_mean - rv.isf(0.975) * np.sqrt(s_mean/n)
    print('信頼係数95%の信頼区間は')
    print(lcl, ucl)