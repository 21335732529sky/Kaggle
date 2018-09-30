#https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277
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
from skimage.transform import resize, rotate
from skimage.morphology import label
from skimage.filters import laplace
from keras.models import Model, load_model
from keras.layers import Input, Add, Dropout
from keras.layers.core import Lambda, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

from keras import optimizers
import tensorflow as tf
import itertools


def resnet_block(input_block, channel, shape, activation='relu', padding='same'):
    x = conv_block(activation, channel, padding, shape, input_block)
    x = conv_block(activation, channel, padding, shape, x)

    return Add()([input_block, x])

def conv_block(activation, channel, padding, shape, input_block):
    x = BatchNormalization()(input_block)
    x = Activation(activation)(x)
    x = Conv2D(channel, shape, padding=padding)(x)

    return x

# Set some parameters
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 1
CHANNEL_BASE = 8
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
    img = imread(path)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    #X_train[4*n + 1] = rotate(img, 90)
    #X_train[4*n + 2] = rotate(img, 180)
    #X_train[4*n + 3] = rotate(img, 270)
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    mask_ = imread(MASK_PATH + id_)
    mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                  preserve_range=True), axis=-1)
    mask = np.maximum(mask, mask_)
    Y_train[n] = mask
    #Y_train[4*n + 1] = rotate(mask, 90)
    #Y_train[4*n + 2] = rotate(mask, 180)
    #Y_train[4*n + 3] = rotate(mask, 270)

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path)[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
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


def get_iou_vector(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch] > 0, B[batch] > 0
        #         if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
        #             metric.append(0)
        #             continue
        #         if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
        #             metric.append(0)
        #             continue
        #         if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
        #             metric.append(1)
        #             continue

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = (np.sum(intersection > 0) + 1e-10) / (np.sum(union > 0) + 1e-10)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)


def my_iou_metric(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0.5], tf.float64)

def my_iou_metric_2(label, pred):
    return tf.py_func(get_iou_vector, [label, pred > 0], tf.float64)

def build_model():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)
    c1 = resnet_block(s, CHANNEL_BASE, (3, 3), activation='relu', padding='same')
    c1 = resnet_block(c1, CHANNEL_BASE, (3, 3), activation='relu', padding='same')
    p1 = MaxPooling2D((2, 2))(c1)
    #p1 = Dropout(0.3)(p1)

    c2 = Conv2D(CHANNEL_BASE * 2, (3, 3), activation='relu', padding='same')(p1)
    c2 = resnet_block(c2, CHANNEL_BASE * 2, (3, 3), activation='relu', padding='same')
    c2 = resnet_block(c2, CHANNEL_BASE * 2, (3, 3), activation='relu', padding='same')
    p2 = MaxPooling2D((2, 2))(c2)
    #p2 = Dropout(0.3)(p2)

    c3 = Conv2D(CHANNEL_BASE * 4, (3, 3), activation='relu', padding='same')(p2)
    c3 = resnet_block(c3, CHANNEL_BASE * 4, (3, 3), activation='relu', padding='same')
    c3 = resnet_block(c3, CHANNEL_BASE * 4, (3, 3), activation='relu', padding='same')
    p3 = MaxPooling2D((2, 2))(c3)
    #p3 = Dropout(0.3)(p3)

    c4 = Conv2D(CHANNEL_BASE * 8, (3, 3), activation='relu', padding='same')(p3)
    c4 = resnet_block(c4, CHANNEL_BASE * 8, (3, 3), activation='relu', padding='same')
    c4 = resnet_block(c4, CHANNEL_BASE * 8, (3, 3), activation='relu', padding='same')
    p4 = MaxPooling2D((2, 2))(c4)
    #p4 = Dropout(0.3)(p4)

    c5 = Conv2D(CHANNEL_BASE * 16, (3, 3), activation='relu', padding='same')(p4)
    c5 = resnet_block(c5, CHANNEL_BASE * 16, (3, 3), activation='relu', padding='same')
    c5 = resnet_block(c5, CHANNEL_BASE * 16, (3, 3), activation='relu', padding='same')

    u6 = Conv2DTranspose(CHANNEL_BASE * 8, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    #u6 = Dropout(0.3)(u6)
    c6 = Conv2D(CHANNEL_BASE * 8, (3, 3), activation='relu', padding='same')(u6)
    c6 = resnet_block(c6, CHANNEL_BASE * 8, (3, 3), activation='relu', padding='same')
    c6 = resnet_block(c6, CHANNEL_BASE * 8, (3, 3), activation='relu', padding='same')

    u7 = Conv2DTranspose(CHANNEL_BASE * 4, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    #u7 = Dropout(0.3)(u7)
    c7 = Conv2D(CHANNEL_BASE * 4, (3, 3), activation='relu', padding='same')(u7)
    c7 = resnet_block(c7, CHANNEL_BASE * 4, (3, 3), activation='relu', padding='same')
    c7 = resnet_block(c7, CHANNEL_BASE * 4, (3, 3), activation='relu', padding='same')

    u8 = Conv2DTranspose(CHANNEL_BASE * 2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    #u8 = Dropout(0.3)(u8)
    c8 = Conv2D(CHANNEL_BASE * 2, (3, 3), activation='relu', padding='same')(u8)
    c8 = resnet_block(c8, CHANNEL_BASE * 2, (3, 3), activation='relu', padding='same')
    c8 = resnet_block(c8, CHANNEL_BASE * 2, (3, 3), activation='relu', padding='same')

    u9 = Conv2DTranspose(CHANNEL_BASE, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    #u9 = Dropout(0.3)(u9)
    c9 = Conv2D(CHANNEL_BASE, (3, 3), activation='relu', padding='same')(u9)
    c9 = resnet_block(c9, CHANNEL_BASE, (3, 3), activation='relu', padding='same')
    c9 = resnet_block(c9, CHANNEL_BASE, (3, 3), activation='relu', padding='same')

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])
    model.summary()

    return model

# code download from: https://github.com/bermanmaxim/LovaszSoftmax
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss

# Build U-Net model
model = build_model()

# Fit model
earlystopper = EarlyStopping(monitor='val_my_iou_metric', mode='max', patience=10, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', monitor='val_my_iou_metric', mode='max', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size=8, epochs=60,
                    shuffle=True, callbacks=[earlystopper, checkpointer])

# Predict on train, val and test
model = load_model('model-dsbowl2018-1.h5', custom_objects={'my_iou_metric': my_iou_metric})

input_x = model.layers[0].input
output_layer = model.layers[-1].input

model2 = Model(input_x, output_layer)
c = optimizers.adam(lr=0.01)

model2.compile(loss=lovasz_loss, optimizer=c, metrics=[my_iou_metric_2])

early_stopping = EarlyStopping(monitor='val_my_iou_metric_2', mode = 'max',patience=20, verbose=1)
model_checkpoint = ModelCheckpoint('model-dsbowl2018-1.h5', monitor='val_my_iou_metric_2',
                                   mode = 'max', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_my_iou_metric_2', mode = 'max',factor=0.5, patience=5, min_lr=0.0001, verbose=1)
epochs = 50
batch_size = 32

history = model2.fit(X_train, Y_train, validation_split=0.1, epochs=epochs,
                     batch_size=batch_size, callbacks=[ model_checkpoint,reduce_lr,early_stopping], verbose=2)

model2 = load_model('model-dsbowl2018-1.h5', custom_objects={'my_iou_metric_2': my_iou_metric_2,
                                                             'lovasz_loss': lovasz_loss})

preds_train = model2.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_val = model2.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_test = model2.predict(X_test, verbose=1)

def mean_iou(y_true, y_pred):
    prec = []
    res = []
    for t in np.arange(0.1, 1, 0.05):
        tmp_p = (y_pred > t).astype(np.uint8)
        inter = np.array([sum(list((y_true[k]*tmp_p[k]).flatten())) for k in range(y_true.shape[0])])
        union = np.array([sum(list((y_true[k] + tmp_p[k] >= 1).astype(np.uint8).flatten())) for k in range(y_true.shape[0])])
        prec.append(np.mean([i / u if u > 0 else 1 for i, u in zip(inter, union)]))
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
