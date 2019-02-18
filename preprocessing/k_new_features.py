import warnings
import numpy as np
import pandas as pd
from ..utils.k_others import tick_tock

def _deprecated(func):
    '''This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.'''
    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func

def k_create_groupby_features(df, group_columns_list, method_dict, add_to_original_data=False, verbose=1, verbose_detail="create stats features", suffix=''):
    '''Create statistical columns, group by [N columns] and compute stats on [N column]

       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       method_dict: python dictionary
          Dictionay used to create stats variables
          shoubld be {'feature': ['method_1', 'method_2']}, if method is a lambda, use function inplace.
       add_to_original_data: boolean
          only keep stats or add stats variable to raw data
       verbose: int
          1 return tick_tock info 0 do not return any info
       Return
       ------
       new pandas dataframe with original columns and new added columns

       Example
       -------
       ka_add_groupby_features(data
                               ,['class']
                               ,{'before': ['count','mean']})

       Update
       ------
       2017/09/26: pandas 0.20.3 has deprecate using just a dict to rename and create stat variables
       ,so I add another parameter method_list to fix this warning.

       2017/9/27: add verbose_detail parameter, let user specify infos they want print.
       2017/10/8: generate column names automatic
    '''
    with tick_tock(verbose_detail, verbose):
        try:
            if type(group_columns_list) == list:
                pass
            else:
                raise TypeError(group_columns_list, "should be a list")
        except TypeError as e:
            print(e)
            raise

        df_new = df.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped.agg(method_dict)
        if suffix != '':
            the_stats.columns = [''.join(group_columns_list) + '_LV_' +"_".join(x[::-1]) + '_' + str(suffix) for x in the_stats.columns.ravel()]
        else:
            the_stats.columns = [''.join(group_columns_list) + '_LV_' + "_".join(x[::-1]) for x in the_stats.columns.ravel()]
        the_stats.reset_index(inplace=True)

        if not add_to_original_data:
            df_new = the_stats
        else:
            df_new = pd.merge(left=df_new[group_columns_list], right=the_stats, on=group_columns_list, how='left').reset_index(drop=True)

    return df_new

@_deprecated
def ka_create_groupby_features_old(df, group_columns_list, agg_dict, keep_only_stats=True, verbose=1):
    '''Create statistical columns, group by [N columns] and compute stats on [N column]
       Parameters
       ----------
       df: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       agg_dict: python dictionary
          Dictionay used to create stats variables
       keep_only_stats: boolean
          only keep stats or return both raw columns and stats
       verbose: int
          1 return tick_tock info 0 do not return any info
       Return
       ------
       new pandas dataframe with original columns and new added columns
       Example
       -------
       {real_column_name: {your_specified_new_column_name : method}}
       agg_dict = {'user_id':{'prod_tot_cnts':'count'},
                   'reordered':{'reorder_tot_cnts_of_this_prod':'sum'},
                   'user_buy_product_times': {'prod_order_once':lambda x: sum(x==1),
                                              'prod_order_more_than_once':lambda x: sum(x==2)}}
       ka_add_stats_features_1_vs_n(train, ['product_id'], agg_dict)
    '''
    with tick_tock("add stats features", verbose):
        try:
            if type(group_columns_list) == list:
                pass
            else:
                raise TypeError(k + "should be a list")
        except TypeError as e:
            print(e)
            raise

        df_new = df.copy()
        grouped = df_new.groupby(group_columns_list)

        the_stats = grouped.agg(agg_dict)
        the_stats.columns = the_stats.columns.droplevel(0)
        the_stats.reset_index(inplace=True)
        if keep_only_stats:
            df_new = the_stats
        else:
            df_new = pd.merge(left=df_new, right=the_stats, on=group_columns_list, how='left')

    return df_new

def ka_replace_hash(hashes, hash_id_table):
    '''Replace "hash in hashes" to "numeric index in hash_id_table"

       Parameter
       ---------
       hashes: pandas series
       hash_id_table: pandas series

       Return
       ------
       numpy array:
           replaced numeric number


       Example
       -------
       user_ids:
       0        d9dca3cb44bab12ba313eaa681f663eb
       1        560574a339f1b25e57b0221e486907ed

       detail.USER_ID_hash:
       0         d9dca3cb44bab12ba313eaa681f663eb
       1         560574a339f1b25e57b0221e486907ed

       replace_hash(detail.USER_ID_hash, user_ids)
    '''
    replace_table = pd.Series(hash_id_table.index, index=hash_id_table.values)
    return replace_table[hashes].values

def ka_add_hash_feature(df, category_columns_list):
    '''Create hash column unique in your specified columns

       Parameters
       ----------
       df: pandas dataframe
           Features matrix

       category_columns_list: list_like
           column names in a list

       Return
       ------
       new pandas dataframe with original columns and new added columns
    '''
    with tick_tock("add hash feature"):
        df_new = df.copy()
        if(len(category_columns_list) > 8):
            df_new['hash_' + category_columns_list[0] + '_' + category_columns_list[-1]] = df_new[category_columns_list].apply(lambda x: hash(tuple(x)),
                                                                                               axis=1)
        else:
            df_new['hash_' + ''.join(category_columns_list)] = df_new[category_columns_list].apply(lambda x: hash(tuple(x)),
                                                                                               axis=1)
    return df_new
