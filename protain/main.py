import numpy as np
import pandas as pd
from data import DataGenerator
from model import Resnet
from tqdm import tqdm
from sklearn.metrics import f1_score

def main():
    print('loading modules ...')
    gen = DataGenerator('protain/all/train/', 'protain/all/test/', 'protain/all/train.csv')
    model = Resnet(resnet_layers=4, channels=[4, 16, 16, 32, 32])
    print('Done')
    epoch = 10

    for i in range(epoch):
        val_x, val_y = gen.get_validation_set()
        bar = tqdm(gen.get_batch(), total=len(gen.train_ids) // 8)
        for x, y in bar:
            loss = model.train(x, y)
            bar.set_description('loss = {:.5f}'.format(loss))
        preds = np.array([[int(y >= 0.5) for y in model.predict([x])[0]] for x in tqdm(val_x)])
        print('[epoch {}]: f1_macro = {}'.format(i, f1_score(val_y, preds, average='macro')))

    preds_test = [(name, [i for i, y in enumerate(model.predict([x])[0]) if y >= 0.5]) for name, x in gen.get_test_set()]
    with open('submission.csv', 'w') as f:
        f.write('Id,Predicted\n')
        for id_, preds in preds_test:
            f.write('{},{}\n'.format(id_, ' '.join(list(map(str, preds)))))

if __name__ == '__main__':
    main()
