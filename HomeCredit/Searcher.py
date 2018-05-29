from Model import Model
from Dataset import Dataset
from tqdm import tqdm
import itertools
from functools import reduce

class Searcher:
    def __init__(self, search_space):
        self.keys = list(search_space.keys())
        self.domains = [search_space[key] for key in self.keys]

    def run(self, dataset):
        max_ = -99999
        max_point = {}
        x_t, y_t = dataset.train_data()
        x_v, y_v = dataset.validation_data()
        for list_ in tqdm(itertools.product(*self.domains), total=reduce(lambda x, y: len(x)*len(y), self.domains)):
            params = {self.keys[i]: list_[i] for i in range(len(list_))}
            m = Model(params)
            m.train(x_t.values, y_t.values.flatten())
            score = m.model.score(x_v.values, y_v.values.flatten())
            if max_ < score:
                max_ = score
                max_point = list_

        print('Result:')
        print('Max score: {:.5f}'.format(max_))
        print('Params:')
        [print('{}: {}'.format(k, v)) for k, v in zip(self.keys, max_point)]


if __name__ == '__main__':
    s = Searcher({'n_estimators': [5,10,15,20],
                 'max_depth': [2,4,6,8]})
    d = Dataset('D:HomeCredit/application_train.csv',
                'D:HomeCredit/application_test.csv',
                omit=[['SK_ID_CURR', 'TARGET'], []], target='TARGET')

    s.run(d)
