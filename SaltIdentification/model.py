import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from skimage.filters import laplace
from keras.models import Model, load_model
from keras.layers import Input, Add, Dropout
from keras.layers.core import Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf
import itertools

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 2
CHANNEL_BASE = 16
TRAIN_PATH = './data/train_img/'
TEST_PATH = './data/test_img/'
MASK_PATH = './data/masks/'
DEPTH_PATH = './data/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[2]
test_ids = next(os.walk(TEST_PATH))[2]
depths = pd.read_csv(DEPTH_PATH + 'depths.csv', index_col='id')

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS - 1]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = np.concatenate((img, laplace(img)), axis=2)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    mask_ = imread(MASK_PATH + id_)
    mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                  preserve_range=True), axis=-1)
    mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS - 1]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = np.concatenate((img, laplace(img)), axis=2)
    X_test[n] = img

print('Done!')

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

# Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1_s = Conv2D(CHANNEL_BASE, (3, 3), activation='relu', padding='same') (s)
c1 = Conv2D(CHANNEL_BASE, (3, 3), activation='relu', padding='same') (c1_s)
c1 = Conv2D(CHANNEL_BASE, (3, 3), activation='relu', padding='same') (c1)
c1 = Add()([c1_s, c1])
p1 = MaxPooling2D((2, 2)) (c1)
#p1 = Dropout(0.3)(p1)

c2_s = Conv2D(CHANNEL_BASE*2, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(CHANNEL_BASE*2, (3, 3), activation='relu', padding='same') (c2_s)
c2 = Conv2D(CHANNEL_BASE*2, (3, 3), activation='relu', padding='same') (c2)
c2 = Add()([c2_s, c2])
p2 = MaxPooling2D((2, 2)) (c2)
#p2 = Dropout(0.3)(p2)

c3_s = Conv2D(CHANNEL_BASE*4, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(CHANNEL_BASE*4, (3, 3), activation='relu', padding='same') (c3_s)
c3 = Conv2D(CHANNEL_BASE*4, (3, 3), activation='relu', padding='same') (c3)
c3 = Add()([c3_s, c3])
p3 = MaxPooling2D((2, 2)) (c3)
#p3 = Dropout(0.3)(p3)

c4_s = Conv2D(CHANNEL_BASE*8, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(CHANNEL_BASE*8, (3, 3), activation='relu', padding='same') (c4_s)
c4 = Conv2D(CHANNEL_BASE*8, (3, 3), activation='relu', padding='same') (c4)
c4 = Add()([c4_s, c4])
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
#p4 = Dropout(0.3)(p4)

c5_s = Conv2D(CHANNEL_BASE*16, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(CHANNEL_BASE*16, (3, 3), activation='relu', padding='same') (c5_s)
c5 = Conv2D(CHANNEL_BASE*16, (3, 3), activation='relu', padding='same') (c5)
c5 = Add()([c5_s, c5])

u6 = Conv2DTranspose(CHANNEL_BASE*8, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
#u6 = Dropout(0.3)(u6)
c6_s = Conv2D(CHANNEL_BASE*8, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(CHANNEL_BASE*8, (3, 3), activation='relu', padding='same') (c6_s)
c6 = Conv2D(CHANNEL_BASE*8, (3, 3), activation='relu', padding='same') (c6)
c6 = Add()([c6_s, c6])

u7 = Conv2DTranspose(CHANNEL_BASE*4, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
#u7 = Dropout(0.3)(u7)
c7_s = Conv2D(CHANNEL_BASE*4, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(CHANNEL_BASE*4, (3, 3), activation='relu', padding='same') (c7_s)
c7 = Conv2D(CHANNEL_BASE*4, (3, 3), activation='relu', padding='same') (c7)
c7 = Add()([c7_s, c7])

u8 = Conv2DTranspose(CHANNEL_BASE*2, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
#u8 = Dropout(0.3)(u8)
c8_s = Conv2D(CHANNEL_BASE*2, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(CHANNEL_BASE*2, (3, 3), activation='relu', padding='same') (c8_s)
c8 = Conv2D(CHANNEL_BASE*2, (3, 3), activation='relu', padding='same') (c8)
c8 = Add()([c8_s, c8])

u9 = Conv2DTranspose(CHANNEL_BASE, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
#u9 = Dropout(0.3)(u9)
c9_s = Conv2D(CHANNEL_BASE, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(CHANNEL_BASE, (3, 3), activation='relu', padding='same') (c9_s)
c9 = Conv2D(CHANNEL_BASE, (3, 3), activation='relu', padding='same') (c9)
c9 = Add()([c9_s, c9])

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
model.summary()

# Fit model
earlystopper = EarlyStopping(patience=10, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=60,
                    callbacks=[earlystopper, checkpointer])

# Predict on train, val and test
model = load_model('model-dsbowl2018-1.h5', custom_objects={'mean_iou': mean_iou})
preds_train = model.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(X_test, verbose=1)

def mean_iou(y_true, y_pred):
    prec = []
    res = []
    for t in np.arange(0.1, 1, 0.05):
        tmp_p = (y_pred > t).astype(np.uint8)
        inter = np.array([sum(list((y_true[k]*tmp_p[k]).flatten())) for k in range(y_true.shape[0])])
        union = np.array([sum(list((y_true[k] + tmp_p[k] >= 1).astype(np.uint8).flatten())) for k in range(y_true.shape[0])])
        prec.append(np.mean([i / u for i, u in zip(inter, union) if u > 0]))
        res.append((prec[-1], t))
        print(res[-1])
        sys.stdout.flush()

    return max(res, key=lambda x: x[0])[1]

th = mean_iou(Y_train[int(Y_train.shape[0]*0.9):], preds_val)
print(th)
# Threshold predictions
preds_train_t = (preds_train > th).astype(np.uint8)
preds_val_t = (preds_val > th).astype(np.uint8)
preds_test_t = (preds_test > th).astype(np.uint8)

# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))

def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

pred_dict = {fn[:-4]:RLenc(np.round(preds_test_upsampled[i])) for i,fn in tqdm(enumerate(test_ids))}

sub = pd.DataFrame.from_dict(pred_dict,orient='index')
sub.index.names = ['id']
sub.columns = ['rle_mask']
sub.to_csv('submission.csv')
