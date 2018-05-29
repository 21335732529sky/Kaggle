import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Dataset:
    def __init__(self, path_train, path_test, omit=[], target='', validate_size=0.2):
        self.encoder = {}
        self.train = self._read_data(path_train, target_col=target, omit=omit[0])
        self.test = self._read_data(path_test, istest=True, omit=omit[1])
        self.split = int(self.train[0].shape[0]*validate_size)

    def _impute(self, df):
        if len(self.encoder.keys()) == 0:
            for key in df.ix[:, df.dtypes == 'object'].columns:
                self.encoder[key] = LabelEncoder().fit(df[key].fillna('N').values.flatten())

        for key in df.ix[:, df.dtypes == 'object'].columns:
            df[key] = self.encoder[key].transform(df[key].fillna('N').values.flatten())

        return df


    def _read_data(self, path, target_col='', istest=False, omit=[]):
        df = self._impute(pd.read_csv(path))

        df = df.fillna(0)
        columns = [x not in omit for x in df.columns]
        return (df.ix[:, columns], df.ix[:, [target_col]])\
               if not istest else df.ix[:, columns]

    def train_data(self):
        return (self.train[0][:self.split], self.train[1][:self.split])

    def validation_data(self):
        return (self.train[0][self.split:], self.train[1][self.split:])

    def test_data(self):
        return self.test
