import skimage
from skimage import io
import random
import pandas as pd
import numpy as np
import os

class DataGenerator:
    def __init__(self, train_path, test_path, label_path):
        df = pd.read_csv(label_path)
        namelist = df.loc[:, ['Id']].values.flatten()
        self.train_ids = namelist[:int(len(namelist)*0.8)]
        self.valid_ids = namelist[int(len(namelist)*0.8):]
        self.test_ids = list(set([name.rsplit('_')[0] for name in os.listdir(test_path)]))
        self.labels = {key: sum(np.eye(28)[int(s)] for s in df.ix[df.Id == key, 1].values[0].split(' '))\
                       for key in namelist}

        self.train_path = train_path
        self.test_path = test_path
        self.c = 0
        self.colors = ['_red', '_green', '_blue', '_yellow']
        random.shuffle(self.train_ids)

    def _get_image(self, name):
        return np.stack([io.imread(name + '{}.png'.format(self.colors[i])) for i in range(4)], axis=2) / 255

    def get_batch(self, size=8):
        while True:
            if self.c >= len(self.train_ids):
                self.c = 0
                random.shuffle(self.train_ids)
                raise StopIteration()
            id_ = self.train_ids[self.c:self.c + size]
            self.c += size
            images = np.stack([self._get_image('/'.join([self.train_path, name])) for name in id_])
            labels = np.array([self.labels[name] for name in id_])
            yield images, labels

    def get_validation_set(self):
        images = (self._get_image('/'.join([self.train_path, name])) for name in self.valid_ids)
        labels = np.array([self.labels[key] for key in self.valid_ids])
        return images, labels



    def get_test_set(self):
        return ((name, self._get_image('/'.join([self.test_path, name]))) for name in self.test_ids)
