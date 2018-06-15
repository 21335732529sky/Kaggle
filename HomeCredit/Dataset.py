import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from AutoEncoder import AutoEncoder
import sys

class Dataset:
    def __init__(self, path_train='', path_test='', norm_func=None,
                 omit=[], target='', additional=[], validate_size=0.2):
        self.encoders = {}
        self.train = self._read_data(path_train, target_col=target, omit=omit[0]) if path_train != '' else None
        self.test = self._read_data(path_test, istest=True, omit=omit[1]) if path_test != '' else None
        self.split = int(self.train[0].shape[0]*(1 - validate_size)) if self.train != None else None

        if self.train is not None: self.train[0] = self._normalize(self.train[0], omit=omit+['SK_ID_CURR'])
        if self.test is not None: self.test = self._normalize(self.test, omit=['SK_ID_CURR'])

        for info in additional:
            df = self._read_data(info['path'], target_col=info['index'], omit=info['omit'])[0]
            self.train[0] = self._merge(self.train[0], df, on=info['index'])
            self.test = self._merge(self.test, df, on=info['index'])

        self.train[0] = self.train.ix[:, self.train.columns != 'SK_ID_CURR']
        print(self.train[0])

    def _merge(self, base, add, on=''):
        df = pd.DataFrame(columns=add.columns)
        for i, index in enumerate(base.ix[:, [on]].values.flatten()):
            num = (add.ix[add[on] == index, :] == i).shape[0]
            tmp = sum(v for v in add.ix[add[on] == index, :].values) / num \
                  if num != 0 else np.array([0,]*add.shape[1])
            try: df[i] = np.array([index] + list(tmp))
            except KeyError: pass

        return pd.merge(base, df, on=[on])

    def _compress(self, df, index=''):
        ae = AutoEncoder(df.shape[0], int(df.shape[0]/2))
        ae.train(df.values, 2, 20, verbose=True)

        return pd.DataFrame(ae.encode(df.values.ix[:, df.columns != index]), index=index)

    def _impute(self, df):
        for key in df.ix[:, df.dtypes == 'object'].columns:
            try:
                df[key] = self.encoders[key].transform(df[key].fillna('N').values)
            except KeyError:
                self.encoders[key] = LabelEncoder().fit(df[key].fillna('N').values)
                df[key] = self.encoders[key].transform(df[key].fillna('N').values)

        return df

    def _normalize(self, df, omit=[]):
        for key in df.columns:
            if key not in omit: df[key] = MinMaxScaler().fit_transform(df[key].values)

        return df


    def _read_data(self, path, target_col='', istest=False, omit=[]):
        df = self._impute(pd.read_csv(path))
        if target_col == '': istest = True
        df = df.fillna(0)
        columns = [x not in omit for x in df.columns]
        return [df.ix[:, columns], df.ix[:, [target_col]]]\
               if not istest else df.ix[:, columns]

    def train_data(self):
        return (self.train[0][:self.split], self.train[1][:self.split])

    def validation_data(self):
        return (self.train[0][self.split:], self.train[1][self.split:])

    def test_data(self):
        return self.test
