
from ..Utils.KA_utils import tick_tock, callbacks_keras

import xgboost
import lightgbm
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from sklearn.svm import SVC, SVR
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor

from lightgbm import callback

class ka_stacking_generalization(object):
    def __init__(self, X_train, X_test, y_train, kf_n, verbose=1):
        '''Stacking for models except "neuron network" and "xgboost"

           Parameters
           ----------
           X_train: numpy array
                training data
           X_test: numpy array
                testing data
           y_train: numpy array
                training target
           kf_n: KFold object from model_selection model

           Example
           -------
           kf_5 = KFold(n_splits=5, random_state=413, shuffle=True)
           stack_generater = ka_stacking_generalization(X.values, X_test.values, y.values, kf_5)
        '''
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.kf_n = kf_n
        self.verbose = verbose
        self.cv_info = {'lgbm_info':{'cv_scores':[], 'cv_rounds':[], 'cv_losses':[]}}

    def run_lgbm_stacker(self
                         , lgbm_params
                         , num_boost_round
                         , early_stopping_rounds
                         , lightgbm_verbose_eval
                         , score_metric):
        '''Stacking for xgboost

           Parameters
           ----------
           lgbm_params: python dictionary
                lightgbm parameters
           num_boost_round: int
                number of boosting rounds
                use as large as possible to overfitting, then decrese to see let CV and LB consistent.
                check difference number like 50 100 to see if they have big difference in CV, if have, one
                should notive underfitting
           early_stopping_rounds: int
                number of early stopping round, if we do not want to use earlying stopping,
                set ---> early_stopping_rounds > num_boost_round
           lightgbm_verbose_eval: 0 1 2
                verbose parameter in lightgbm
           Return
           ------
           S_train: numpy array
                            stacked training data
           S_test: numpy array
                            stacked testing data
           Example
           -------
           params = {'application': 'binary'
                     ,'learning_rate': 0.05
                     ,'metric': 'auc'
                     ,'feature_fraction': 0.8
                     ,'bagging_fraction': 0.8
                     ,'bagging_freq': 2
                     ,'num_leaves': 80
                     ,'max_depth': -1
                     ,'lambda_l2':5
                     ,'min_data_in_leaf': 10}

           S_train, S_test, cv_scores = stack_generater.run_lgbm_stacker(params, 2000, 30, 0, roc_auc_score)
        '''
        with tick_tock("stacking", self.verbose):
            n_folds = self.kf_n.n_splits
            self.cv_info['lgbm_info'] = {'cv_scores':[], 'cv_rounds':[], 'cv_losses':[]}
            S_train = np.zeros_like(self.y_train).astype(np.float32)
            S_test_i = np.zeros((self.X_test.shape[0], n_folds)).astype(np.float32)

            for i, (train_index, val_index) in enumerate(self.kf_n.split(self.X_train, self.y_train)):
                losses = dict()
                X_train_cv, X_valid_cv = self.X_train[train_index], self.X_train[val_index]
                y_train_cv, y_valid_cv = self.y_train[train_index], self.y_train[val_index]

                d_train_fold = lightgbm.Dataset(X_train_cv, y_train_cv)
                d_val_fold = lightgbm.Dataset(X_valid_cv, y_valid_cv)

                model_fold = lightgbm.train(lgbm_params
                                            , train_set=d_train_fold
                                            , valid_sets=d_val_fold
                                            , early_stopping_rounds=early_stopping_rounds
                                            , verbose_eval=lightgbm_verbose_eval
                                            , num_boost_round=num_boost_round
                                            , callbacks=[lightgbm.callback.record_evaluation(losses)])
                pred_valid = model_fold.predict(X_valid_cv)

                S_train[val_index] = pred_valid
                S_test_i[:, i] = model_fold.predict(self.X_test)

                score_tmp = score_metric(y_valid_cv, pred_valid)
                self.cv_info['lgbm_info']['cv_scores'].append(score_tmp)
                self.cv_info['lgbm_info']['cv_rounds'].append(model_fold.best_iteration)
                self.cv_info['lgbm_info']['cv_losses'].append(losses[list(losses.keys())[0]][list(losses[list(losses.keys())[0]].keys())[0]])
                print("Fold:{} --> score:{}.".format(i, score_tmp))

            S_test = S_test_i.sum(axis=1) / n_folds
            print("Mean:{}, Std:{}".format(np.mean(self.cv_info['lgbm_info']['cv_scores'])
                                           , np.std(self.cv_info['lgbm_info']['cv_scores'])))

            return S_train, S_test


    def plot_loss_curve(self, ax, info_type):
        '''
        _, ax = plt.subplots(nrows=3, ncols=2, figsize=[15,10])
        info_type: 'lgbm_info'
        '''
        pic_ith = 0
        for i in range(np.int(self.kf_n.n_splits / 2) + 1):
            for j in range(2):
                if((i+1)*(j+1) > self.kf_n.n_splits):
                    continue
                ax[i,j].set_title('round: ' + str(self.cv_info['lgbm_info']['cv_rounds'][pic_ith]) + \
                          '  --  loss: ' + str(self.cv_info['lgbm_info']['cv_losses'][pic_ith][-1]))
                ax[i,j].plot(self.cv_info[info_type]['cv_losses'][pic_ith])
                pic_ith += 1

    def run_xgboost_stacker(self
                            , xgb_params
                            , num_boost_round
                            , early_stopping_rounds
                            , xgboost_verbose_eval
                            , verbose=1):
        '''Stacking for xgboost

           Parameters
           ----------
           xgb_params: python dictionary
                xgboost parameters
           num_boost_round: int
                number of boosting rounds

           Return
           ------
           S_train: numpy array
                            stacked training data
           S_test: numpy array
                            stacked testing data

           Example
           -------
           xgb_params = {
                'eta': 0.01,
                'max_depth': 3,
                'objective': 'reg:linear',
                'eval_metric': 'rmse',
                'tree_method': 'hist',
                'colsample_bytree': 0.5,
                'base_score': data_train.y.mean(), # base prediction = mean(target)
                'silent': 1
            }

            S_train, S_test = run_xgboost_stacker(xgb_params, 741)
        '''
        with tick_tock("stacking", self.verbose):
            d_test = xgboost.DMatrix(self.X_test)
            n_folds = self.kf_n.n_folds
            S_train = np.zeros_like(self.y)
            S_test_i = np.zeros((self.X_test.shape[0], n_folds))

            for i, (train_index, val_index) in enumerate(self.kf_n):
                if verbose > 0:
                    print('xgboost is stacking fold {0} ....'.format(i+1))
                X_train_cv, X_val_cv = self.X_train[train_index], self.X_train[val_index]
                y_train_cv, y_valid_cv = self.y[train_index], self.y[val_index]
                d_train_fold = xgboost.DMatrix(X_train_cv, y_train_cv)
                d_val_fold = xgboost.DMatrix(X_val_cv, y_valid_cv)
                watchlist = [(d_train_fold,'train'),(d_val_fold,'eval')]

                model_fold = xgboost.train(xgb_params
                                           , evals=watchlist
                                           , verbose_eval=xgboost_verbose_eval
                                           , early_stopping_rounds=early_stopping_rounds
                                           , dtrain=d_train_fold
                                           , num_boost_round=num_boost_round)
                S_train[val_index] = model_fold.predict(d_val_fold)
                S_test_i[:, i] = model_fold.predict(d_test)

            S_test = S_test_i.sum(axis=1) / n_folds

            return S_train, S_test

    def run_nn_stacker(self
                       , build_model
                       , epochs
                       , batch_size
                       , saved_path
                       , saved_file_name
                       , patience
                       , verbose_nn = 2
                       , verbose=1):
        '''Stacking for neural network

           Parameters
           ----------

           Return
           ------
           S_train: numpy array
                            stacked training data
           S_test: numpy array
                            stacked testing data

           Example
           -------
            S_train, S_test = run_lgbm_stacker(xgb_params, 741)
        '''
        with tick_tock("stacking", self.verbose):
            n_folds = self.kf_n.n_folds
            S_train = np.zeros_like(self.y)
            S_test_i = np.zeros((self.X_test.shape[0], n_folds))

            for i, (train_index, val_index) in enumerate(self.kf_n):
                model = build_model()
                if verbose > 0:
                    print('neural network is stacking fold {0} ....'.format(i+1))
                X_train_cv, X_val_cv = self.X_train[train_index], self.X_train[val_index]
                y_train_cv, y_valid_cv = self.y[train_index], self.y[val_index]

                callbacks_ins = callbacks_keras(saved_path + saved_file_name + '_v'+ str(i) + ".p"
                                                , model, patience=patience)
                model.fit(X_train_cv,y_train_cv, callbacks=callbacks_ins.callbacks
                          ,validation_data=[X_val_cv, y_valid_cv]
                          ,epochs=epochs, batch_size=batch_size, verbose=verbose_nn)

                model.load_weights(saved_path + saved_file_name + '_v'+ str(i) + ".p")
                S_train[val_index] = np.squeeze(model.predict(X_val_cv))
                S_test_i[:, i] = np.squeeze(model.predict(self.X_test))

            S_test = S_test_i.sum(axis=1) / n_folds
            return S_train, S_test

    def run_other_stackers(self, base_models, score_metric):
        '''Stacking for sklearn mdoels
           Parameters
           ----------
           base_models: models in list.
               [model_1, model_2, ...], only support sklearn's models

           Return
           ------
           S_train: numpy array
                            stacked training data
           S_test: numpy array
                            stacked testing data
        '''
        with tick_tock("fitting stacking", self.verbose):
            S_train = np.zeros((self.X_train.shape[0], len(base_models))).astype(np.float32)
            S_test = np.zeros((self.X_test.shape[0], len(base_models))).astype(np.float32)
            n_folds = self.kf_n.n_splits
            cv_scores = []

            print ("Fitting base models begin")
            for i, model in enumerate(base_models):
                print ("Fitting the {0} th base models".format(i))
                S_test_i = np.zeros((self.X_test.shape[0], n_folds))
                for j, (train_index, val_index) in enumerate(self.kf_n.split(self.X_train, self.y_train)):
                    X_train_cv, X_valid_cv = self.X_train[train_index], self.X_train[val_index]
                    y_train_cv, y_valid_cv = self.y_train[train_index], self.y_train[val_index]

                    model.fit(X_train_cv, y_train_cv)
                    pred_valid = model.predict_proba(X_valid_cv)[:,1]
                    score_tmp = score_metric(y_valid_cv, pred_valid)
                    cv_scores.append(score_tmp)

                    S_train[val_index,i] = pred_valid
                    S_test_i[:,j] = model.predict_proba(self.X_test)[:,1]

                    print("Fold:{} --> score:{}.".format(i, score_tmp))
                S_test[:,i] = S_test_i.mean(1)

            print("Mean:{}, Std:{}".format(np.mean(cv_scores)
                                           , np.std(cv_scores)))
            return S_train, S_test


def ka_bagging_2class_or_reg(X_train, y_train, model, seed, bag_round
                            , X_test, update_seed=True, is_classification=True, using_notebook=True):
    '''
        Bagging for "2-class classification" model and "regression" model

        Parameters
        ----------
        X_train: numpy array 2-dimension
             training data for fitting model
        X_test: numpy array 2-dimension
             testing data for predict result
        y: numpy array 1-dimension
             training target
        model: model instance
        seed: int
             random seed
        bag_round: int
             bagging rounds
        update_seed: boolean
             update model to generate difference result
        is_classification: boolean
             classfication will predict probability by default
             regression only predict value

       Return:
       baggedpred: numpy array
             bagged prediction


       Example
       -------
       Regression:
           from sklearn.datasets import load_boston
           from sklearn.metrics import mean_squared_error

           data = load_boston()
           X = data.data
           y = data.target
           X_test = X.copy()

           model = RandomForestRegressor()
           pred_10 = bagging_2class_or_reg(X, y, model, 10, 10, X_test, is_classification=False)
           pred_1 = bagging_2class_or_reg(X, y, model, 10, 1, X_test, is_classification=False)

           print(mean_squared_error(y, pred_10)) # 1.38465739328
           print(mean_squared_error(y, pred_1)) # 2.13027490119

       Classification:
           from sklearn.datasets import load_boston
           from sklearn.metrics import roc_auc_score

           data = load_boston()
           X = data.data
           y = data.target
           X_test = X.copy()

           model = RandomForestRegressor()
           pred_10 = bagging_2class_or_reg(X, y, model, 10, 10, X_test, is_classification=False)
           pred_1 = bagging_2class_or_reg(X, y, model, 10, 1, X_test, is_classification=False)

           print(mean_squared_error(y, pred_10)) # 0.998868778281
           print(mean_squared_error(y, pred_1)) # 0.993778280543


    '''
    # create array object to hold predictions
    baggedpred=np.zeros(shape=X_test.shape[0])
    #loop for as many times as we want bags
    if using_notebook:
        for n in tqdm_notebook(range(0, bag_round)):
            #shuffle first, aids in increasing variance and forces different results
            X_train, y_train=shuffle(X_train, y_train, random_state=seed+n)

            # update seed if requested, to give a slightly different model
            # model like knn does not have random_state parameter
            if update_seed:
                model.set_params(random_state=seed + n)

            model.fit(X_train, y_train)
            if is_classification:
                pred = model.predict_proba(X_test) # predict probabilities
                if pred.ndim == 1:
                    pass
                elif pred.ndim == 2:
                    pred = pred[:,1]
                else:
                    print("this is a n>2 category problem, stacker is only suitable for 2-class and regression")
            else:
                pred = model.predict(X_test)

            baggedpred += pred/bag_round
    else:
        for n in tqdm(range(0, bag_round)):
            #shuffle first, aids in increasing variance and forces different results
            X_train, y_train=shuffle(X_train, y_train, random_state=seed+n)

            # update seed if requested, to give a slightly different model
            # model like knn does not have random_state parameter
            if update_seed:
                model.set_params(random_state=seed + n)

            model.fit(X_train, y_train)
            if is_classification:
                pred = model.predict_proba(X_test) # predict probabilities
                if pred.ndim == 1:
                    pass
                elif pred.ndim == 2:
                    pred = pred[:,1]
                else:
                    print("this is a n>2 category problem, stacker is only suitable for 2-class and regression")
            else:
                pred = model.predict(X_test)

            baggedpred += pred/bag_round
    return baggedpred

def ka_bagging_2class_or_reg_lgbm(X_train, y_train, seed, bag_round, params
                                 , X_test, using_notebook=True, num_boost_round=0):
    '''
        early version
    '''
    # create array object to hold predictions
    baggedpred=np.zeros(shape=X_test.shape[0]).astype(np.float32)
    #loop for as many times as we want bags
    if using_notebook:
        for n in tqdm_notebook(range(0, bag_round)):
            #shuffle first, aids in increasing variance and forces different results
            X_train, y_train=shuffle(X_train, y_train, random_state=seed+n)
            params['seed'] = seed + n
            model = lightgbm.train(params, lightgbm.Dataset(X_train, y_train), num_boost_round=num_boost_round)
            pred = model.predict(X_test)
            baggedpred += pred/bag_round

    return baggedpred
