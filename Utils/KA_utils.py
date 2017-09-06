from __future__ import print_function, division
import gc
import time
import pickle
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


from sklearn.metrics import r2_score, confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.model_selection import train_test_split

import keras.backend as K
from keras.optimizers import adam, sgd
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
from keras.layers import Dense, Dropout, Input, Embedding, Flatten, Merge, Reshape, BatchNormalization

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
from IPython.display import display
from pandas_summary import DataFrameSummary
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor

class WARNINGS:
    def __init__(self):
        pass
    @staticmethod
    def CLOSE_WARNING():
        import warnings
        warnings.filterwarnings('ignore')
    @staticmethod
    def OPEN_WARNING():
        import warnings
        warnings.filterwarnings('default')

class tick_tock:
    def __init__(self, process_name, verbose=1):
        self.process_name = process_name
        self.verbose = verbose
    def __enter__(self):
        if self.verbose:
            print(self.process_name + " begin ......")
            self.begin_time = time.time()
    def __exit__(self, type, value, traceback):
        if self.verbose:
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

def ka_xgb_r2_error(preds, dtrain):
    labels = dtrain.get_label()
    return 'error', r2_score(labels, preds)

def ka_xgb_r2_exp_error(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.clip(np.exp(preds),0, 1e10)
    return 'error', r2_score(np.exp(labels), preds)

def kaggle_points(n_teams, n_teammates, rank, t=1):
    return (100000 / np.sqrt(n_teammates)) * (rank ** (-0.75)) * (np.log10(1 + np.log10(n_teams))) * (np.e**(t/500))
######################################################################################################
# read and write file functions
######################################################################################################
def mkdir(path):
    '''
        If directory exsit, do not make any change
        Else make a new directory
    '''
    try:
        os.stat(path)
    except:
        os.mkdir(path)

def pickle_dump_chunks(df, path, split_size=3, inplace=False):
    """
    path = '../output/mydf'

    wirte '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'

    """
    if inplace==True:
        df.reset_index(drop=True, inplace=True)
    else:
        df = df.reset_index(drop=True)
    mkdir(path)

    for i in tqdm(range(split_size)):
        df.ix[df.index%split_size==i].to_pickle(path+'/{}.p'.format(i))

    return

def pickle_load_chunks(path, col=None):
    if col is None:
        df = pd.concat([pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*')))])
    else:
        df = pd.concat([pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*')))])
    return df

def pickle_dump(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(filename):
    with open(filename, 'rb') as f:
        # https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
        return pickle.load(f, encoding='latin1')

def ka_is_numpy(df):
    '''Check if a object is numpy

       Parameters
       ----------
       df: any object
    '''
    return type(df) == np.ndarray
