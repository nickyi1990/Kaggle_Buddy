from __future__ import print_function, division
import gc
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.cross_validation import KFold

import keras.backend as K
from keras.optimizers import adam, sgd
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.layers import Dense, Dropout, Input, Embedding, Flatten, Merge, Reshape, BatchNormalization


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
from IPython.display import display
from pandas_summary import DataFrameSummary


class tick_tock:
    def __init__(self, process_name):
        self.process_name = process_name
    def __enter__(self):
        print(self.process_name + " begin ......")
        self.begin_time = time.time()
    def __exit__(self, type, value, traceback):
        end_time = time.time()
        print(self.process_name + " end ......")
        print('time lapsing {0} s \n'.format(end_time - self.begin_time))


class callbacks_keras:
    def __init__(self, filepath, model,
                 base_lr=1e-3, decay_rate=1,
                 decay_after_n_epoch=10, patience=20,
                 mode='min', monitor='val_loss'):
        self.base_lr = base_lr
        self.model = model
        self.decay_rate = decay_rate
        self.decay_after_n_epoch = decay_after_n_epoch
        self.callbacks = [ModelCheckpoint(filepath = filepath,
                                          monitor = monitor,
                                          verbose = 2,
                                          save_best_only = True,
                                          save_weights_only = True,
                                          mode = mode),
                         EarlyStopping(monitor = monitor, patience = patience, verbose=2, mode = mode),
                         LearningRateScheduler(self._scheduler)]

    def _scheduler(self, epoch):
        if epoch%self.decay_after_n_epoch==0 and epoch!=0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr*self.decay_rate)
            print("lr changed to {}".format(lr**self.decay_rate))
        return K.get_value(self.model.optimizer.lr)


def pickle_dump(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def ka_xgb_r2_error(preds, dtrain):
    labels = dtrain.get_label()
    return 'error', r2_score(labels, preds)

def ka_xgb_r2_exp_error(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.clip(np.exp(preds),0, 1e10)
    return 'error', r2_score(np.exp(labels), preds)

def ka_is_numpy(df):
    '''Check if a object is numpy

       Parameters
       ----------
       df: any object
    '''
    return type(df) == np.ndarray
