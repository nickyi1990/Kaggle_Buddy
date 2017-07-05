import numpy as np
import pandas as pd
from IPython.display import display
from pandas_summary import DataFrameSummary

def ka_display_muti_tables_summary(tables, table_names):
    '''display multi tables' summary

        Parameters
        ----------
        tables: list_like
                Pandas dataframes
        table_names: list_like
                     names of each dataframe
    '''
    for t, t_name in zip(tables, table_names):
        print(t_name + ":")
        display(DataFrameSummary(t).summary())

def ka_display_muti_tables(tables, table_names, is_head=True, n=5):
    '''Display multi tables' head

        Parameters
        ----------
        tables: list_like
                Pandas dataframes
        table_names: list_like
                     names of each dataframe
        is_head: boolean
                 If true display head else display tail
        n: int
           number of rows to display
    '''
    for t, t_name in zip(tables, table_names):
        if is_head:
            print(t_name + ":")
            display(t.shape)
            display(t.head(n=n))
        else:
            print(t_name + ":")
            display(t.shape)
            display(t.tail(n=n))

def ka_display_col_type(data):
    '''See column type distribution

       Parameters
       ----------
       data: pandas dataframe

       Return
       ------
       dataframe
    '''
    column_type = data.dtypes.reset_index()
    column_type.columns = ["count", "column type"]
    return column_type.groupby(["column type"]).agg('count').reset_index()

def ka_get_NC_col_names(data):
    '''Get column names of category and numeric

        Parameters
        ----------
        data: dataframe

        Return:
        ----------
        numerics_cols: numeric column names
        category_cols: category column names

    '''
    numerics_cols = data.select_dtypes(exclude=['O']).columns.tolist()
    category_cols = data.select_dtypes(include=['O']).columns.tolist()
    return numerics_cols, category_cols

def ka_look_missing_columns(data):
    '''show missing information

        Parameters
        ----------
        data: pandas dataframe

        Return
        ------
        df: pandas dataframe
    '''
    df_missing = data.isnull().sum().sort_values(ascending=False)
    df = pd.concat([pd.Series(df_missing.index.tolist()), pd.Series(df_missing.values),
                    pd.Series(data[df_missing.index].dtypes.apply(lambda x: str(x)).values),
                    pd.Series((df_missing / data.shape[0]).values)], axis=1, ignore_index=True)
    df.columns = ['col_name', 'missing_count', 'col_type', 'missing_rate']

    return df

def ka_look_skewnewss(data):
    '''show skewness information

        Parameters
        ----------
        data: pandas dataframe

        Return
        ------
        df: pandas dataframe
    '''
    numeric_cols = data.columns[data.dtypes != 'object'].tolist()
    skew_value = []

    for i in numeric_cols:
        skew_value += [skew(data[i])]
    df = pd.concat(
        [pd.Series(numeric_cols), pd.Series(data.dtypes[data.dtypes != 'object'].apply(lambda x: str(x)).values)
            , pd.Series(skew_value)], axis=1)
    df.columns = ['var_name', 'col_type', 'skew_value']

    return df

def ka_look_groupby_n_1_stats(X, y, iv_dv_pair, precent=25):
    '''Evaluate statistical indicators in each category

       Parameters
       ----------
       X: pandas dataframe
          Features matrix, it should not contain columns has the same value with y
       y: pandas dataframe
          Labels
       iv_dv_pair: list_like
          independent variable and dependent variable, like ['category', 'target']

       Return
       ------
       pandas dataframe

       Example
       -------
       ka_look_groupby_n_1_stats(train, y, ['x0','y'])
    '''
    _X_forplot = pd.concat([X, y], axis=1)
    _df_target = pd.DataFrame(_X_forplot.groupby(iv_dv_pair[:-1])[iv_dv_pair[-1]].\
                              agg([len, np.mean, np.median, np.min, np.max, np.std]).\
                              sort_values('mean', ascending=False)).reset_index()

    return _df_target.sort_values('mean', ascending=False)

def ka_C_Binary_ratio(y, positive=1):
    '''Find the positive ration of dependent variable

        Parameters
        ----------
        y: pandas series
           binary dependent variable
        positive: 1 or 0
                  identify which value is positive

        Return
        ------
        float value display positive rate
    '''
    return y.value_counts()[positive] / (y.value_counts().sum())

def ka_verify_primary_key(data, column_list):
    '''Verify if columns in column list can be treat as primary key

        Parameter
        ---------
        data: pandas dataframe

        column_list: list_like
                     column names in a list

        Return
        ------
        boolean: if true, these columns are unique in combination and can be used as a key
                 if false, these columns are not unique in combination and can not be used as a key
    '''

    return data.shape[0] == data.groupby(column_list).size().reset_index().shape[0]
