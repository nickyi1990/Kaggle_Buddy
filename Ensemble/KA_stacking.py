
from ..Utils.KA_utils import tick_tock
import numpy as np
import pandas as pd
import xgboost

def ka_stacking_xgboost(train, test, y, xgb_params, num_boost_round, kf_n, verbose=1):
    '''Stacking for xgboost

       Parameters
       ----------
       train: pandas dataframe
            training data
       test: pandas dataframe
            testing data
       y: pandas series
            training target
       xgb_params: python dictionary
            xgboost parameters
       num_boost_round: int
            number of boosting rounds
       kf_n: KFold object

       Return
       ------
       pred_train_stack: numpy array
                        stacked training data
       pred_test_stack: numpy array
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

        kf_5 = KFold(data_train.shape[0], n_folds=5, shuffle=True, random_state=1024)
        pred_train_stack_xgb, pred_test_stack_xgb = ka_stacking_xgboost(data_train, data_test,
                                                                    y, xgb_params, 741, kf_5)
    '''
    with tick_tock("add stats features", verbose):
        d_test = xgboost.DMatrix(test.values)
        n_folds = kf_n.n_folds
        pred_train_stack = np.zeros_like(y)
        pred_test_stack_tmp = np.zeros((y.shape[0], n_folds))

        for i, (train_index, val_index) in enumerate(kf_n):
            X_train, X_val = train.values[train_index], train.values[val_index]
            y_train, y_val = y.values[train_index], y.values[val_index]

            d_train_fold = xgboost.DMatrix(X_train, y_train)
            d_val_fold = xgboost.DMatrix(X_val)

            model_fold = xgboost.train(xgb_params, dtrain=d_train_fold, num_boost_round=num_boost_round)
            pred_train_stack[val_index] = model_fold.predict(d_val_fold)
            pred_test_stack_tmp[:, i] = model_fold.predict(d_test)

        pred_test_stack = pred_test_stack_tmp.sum(axis=1) / n_folds

        return pred_train_stack, pred_test_stack
