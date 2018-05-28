from Reader import read_data
from Model import Model
import numpy as np
import matplotlib.pyplot as pl
import seaborn

m = Model()
x, y, c = read_data('D:HomeCredit/application_train.csv', omit=['SK_ID_CURR', 'TARGET'])
c = np.delete(c, np.where(c == 'TARGET'), 0)

m.train(x[:-1000], y[:-1000])

print(m.model.score(x[-1000:], y[-1000:]))
'''
pl.barh(range(c.shape[0]), m.model.feature_importances_)
pl.yticks(range(c.shape[0]), c)
pl.show()
'''

with open('submission.csv', 'w') as f:
    x = read_data('D:HomeCredit/application_test.csv', istest=True)
    print(len(x))
    ans = m.model.predict_proba(x.ix[:, x.columns != 'SK_ID_CURR'].values)
    f.write('SK_ID_CURR,TARGET\n')
    [f.write('{},{}\n'.format(i, a)) for i, a in zip(x.ix[:, ['SK_ID_CURR']].values.flatten(), [b[1] for b in ans])]
