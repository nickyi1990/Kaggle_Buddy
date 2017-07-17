from ..Utils.KA_utils import tick_tock
import numpy as np
import pandas as pd


def ka_remove_duplicate_cols(df, **kwargs):
    '''Remove duplicate columns

       Parameters
       ----------
       df: pandas dataframe
          Features matrix

       **kwargs: all parameters in drop_duplicates function
           subset : column label or sequence of labels, optional
                Only consider certain columns for identifying duplicates, by
                default use all of the columns
           keep : {'first', 'last', False}, default 'first'
               - ``first`` : Drop duplicates except for the first occurrence.
               - ``last`` : Drop duplicates except for the last occurrence.
               - False : Drop all duplicates.
           take_last : deprecated
           inplace : boolean, default False
                Whether to drop duplicates in place or to return a copy
       Return
       ------
       new pandas dataframe with "unique columns" and "removed column names"

       Example
       -------
       data_1_unique, removed_cols = ka_remove_duplicate_cols(data_1[numeric_cols])
    '''
    df_unique_columns = df.T.drop_duplicates(**kwargs).T
    return df_unique_columns, set(df.columns.tolist()) - set(df_unique_columns.columns.tolist())
