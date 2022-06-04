import numpy as np
import pandas as pd
from scipy import stats

pd.options.display.precision = 3


if __name__ == '__main__':
    enquete_df = pd.read_csv('../../../data/ch10_enquete.csv')
    enquete = np.array(enquete_df['知っている'])
    n = len(enquete)

    print('アンケート結果の10件を表示')
    print(enquete[:10])

    s_mean = enquete.mean()
    print(f'標本平均={s_mean}')

    print('中心極限定理によって標本平均は近似的に正規分布に従うとするとき')
    rv = stats.norm()
    lcl = s_mean - rv.isf(0.025) * np.sqrt(s_mean*(1-s_mean)/n)
    ucl = s_mean - rv.isf(0.975) * np.sqrt(s_mean*(1-s_mean)/n)
    print('信頼係数95%の信頼区間は')
    print(lcl, ucl)
