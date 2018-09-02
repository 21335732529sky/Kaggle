from Model import Model
from Dataset import Dataset
from tqdm import tqdm
from functools import reduce
import itertools
from time import sleep
from multiprocessing import Process, cpu_count
running = 0
n_cpu = cpu_count()
filepath = "/home/u271969h/.kaggle/competitions/home-credit-default-risk/"
additional_data = [{'path': filepath + 'bureau.csv', 'useAE': True,
                    'index': 'SK_ID_CURR', 'omit': ['SK_ID_BUREAU']},
                   {'path': filepath + 'previous_application.csv', 'useAE': False,
                    'index': 'SK_ID_CURR', 'omit': ['SK_ID_PREV']},
                   {'path': filepath + 'POS_CASH_balance.csv', 'useAE': False,
                    'index': 'SK_ID_CURR', 'omit': ['SK_ID_PREV']},
                   {'path': filepath + 'installments_payments.csv', 'useAE': False,
                    'index': 'SK_ID_CURR', 'omit': ['SK_ID_PREV']}]

class Searcher:
    def __init__(self, search_space):
        self.keys = list(search_space.keys())
        self.domains = [search_space[key] for key in self.keys]

    def run(self, dataset):
        x_t, y_t = dataset.train_data()
        x_v, y_v = dataset.validation_data()
        scores = {tuple(d): 0 for d in itertools.product(*self.domains)}
        domain = list(scores.keys())
        jobs = []
        pbar = tqdm(total=reduce(lambda x,y: x*y, map(len, self.domains)))

        def calc_score(board, params):
            global running; running += 1
            params = {self.keys[i]: params[i] for i in range(len(params))}
            m = Model(params)
            m.train(x_t.values, y_t.values.flatten())
            score = m.model.score(x_v.values, y_v.values.flatten())
            board[tuple(params)] = score
            running -= 1
            pbar.update(1)

        while len(domain) > 0:
            global running, n_cpu
            if running < n_cpu:
                param = domain.pop()
                job = Process(target=calc_score, args=(scores, param))
                jobs.append(job)
                job.start()
            else:
                sleep(1)

        [job.join() for job in jobs]
        max_score = max(scores.values())

        return next((d, scores[tuple(d)]) for d in itertools.product(*self.domains)\
                    if scores[tuple(d)] == max_score)



if __name__ == '__main__':
    s = Searcher({'n_estimators': [5, 10, 20, 50, 100],
                 'max_depth': [2, 4, 6, 8, 10],
                 'min_samples_split': [2, 4, 8, 16, 32],
                 'learning_rate': [1e-4, 1e-3, 1e-2, 1e-1]})

    d = Dataset(filepath + 'application_train.csv',
                filepath + 'application_test.csv',
                omit=[['TARGET'], []], target='TARGET',
                additional=additional_data)

    print(s.run(d))
