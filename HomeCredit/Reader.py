import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_data(path, istest=False, omit=[]):
    df = pd.read_csv(path)
    le = LabelEncoder()

    for key in df.ix[:, df.dtypes == 'object'].columns:
        df[key] = le.fit_transform(df[key].fillna('N').values.flatten())

    df = df.fillna(0)

    return (df.ix[:, 2:], df.ix[:, ['TARGET']].values.flatten(), df.columns.values) if not istest else df
