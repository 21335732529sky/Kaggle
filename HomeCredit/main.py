from Dataset import Dataset
from Model import Model
import numpy as np
import matplotlib.pyplot as pl
import seaborn
import warnings

warnings.filterwarnings('ignore')



m = Model({'n_estimators': 100,
           'max_depth': 2,
           'min_samples_split': 4,
           'learning_rate': 0.1})
d = Dataset('D:HomeCredit/application_train.csv',
            'D:HomeCredit/application_test.csv',
            omit=[['TARGET'], []], target='TARGET',
            additional=[{'path': 'D:HomeCredit/bureau.csv',
                        'useAE': True,
                        'index': 'SK_ID_CURR',
                        'omit': ['SK_ID_BUREAU']}])

x, y = d.train_data()

m.train(x.values, y.values.flatten())

x, y = d.validation_data()
print(m.model.score(x.values, y.values.flatten()))

with open('submission.csv', 'w') as f:
    x = d.test_data()
    ans = m.model.predict_proba(x.ix[:, x.columns != 'SK_ID_CURR'].values)
    f.write('SK_ID_CURR,TARGET\n')
    [f.write('{},{}\n'.format(i, a)) for i, a in zip(x.ix[:, ['SK_ID_CURR']].values.flatten(), [b[1] for b in ans])]
