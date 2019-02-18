import numpy as np
import pandas as pd
from scipy.stats import skew
from IPython.display import display
from IPython.display import display_html
from pandas_summary import DataFrameSummary

####################################################################################
##                              DISPLAY BLOCK
####################################################################################

def _ka_display_col_type(data):
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

def ka_display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

def ka_display_muti_tables_summary(tables, table_names, n=5):
    '''display multi tables' summary

        Parameters
        ----------
        tables: list_like
                Pandas dataframes
        table_names: list_like
                     names of each dataframe

        Return
        ------
        1. show head of data
        2. show column types of data
        3. show summary of data
    '''
    for t, t_name in zip(tables, table_names):
        print(t_name + ":", t.shape)
        ka_display_side_by_side(t.head(n=n), _ka_display_col_type(t), DataFrameSummary(t).summary())

def ka_display_missing_columns(data):
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

def ka_display_skewnewss(data):
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

def ka_display_groupby_n_1_stats(data, group_columns_list, target_columns_list):
    '''Evaluate statistical indicators in each category

       Parameters
       ----------
       data: pandas dataframe
          Features matrix
       group_columns_list: list_like
          List of columns you want to group with, could be multiple columns
       target_columns_list: list_like
          column you want to compute stats, need to be a list with only one element

       Return
       ------
       pandas dataframe

       Example
       -------
       df = ka_display_groupby_n_1_stats(train, ['class'], ['translate_flag'])
    '''

    grouped = data.groupby(group_columns_list)
    df = grouped[target_columns_list].agg([len, np.mean, np.median, np.min, np.max, np.std]).reset_index()
    df.columns = df.columns.droplevel(0)
    df["percent"] = df.len * 100/ df.len.sum()
    df["percent"] = pd.Series(["{0:.2f}%".format(val) for val in df['percent']], index = df.index)

    return  df.sort_values('mean', ascending=False)

####################################################################################
##                              UNIVERSAL BLOCK
####################################################################################

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

####################################################################################
##                              CATEGORICAL BLOCK
####################################################################################

def k_cat_explore(x: pd.Series):
    unique_cnt = x.nunique()

    print()


def k_cat_explore(x: pd.Series):
    unique_cnt = x.nunique()
    value_cnts = x.value_counts(dropna=False)

    print("num of unique counts: {}".format(unique_cnt))
    plt_value_cnts(value_cnts.iloc[:20], x.name)
    display(value_cnts.iloc[:20])

    return unique_cnt, value_cnts

def plt_value_cnts(value_cnts, name):
    ax = value_cnts.plot(kind='barh', figsize=(10,7), color="coral", fontsize=13)
    ax.set_title(name)

    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_width())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    for i in ax.patches:
        # get_width pulls left or right; get_y pushes up or down
        ax.text(i.get_width() * 1,
                i.get_y() + 0.3,
                str(round((i.get_width()/total)*100, 2))+'%',
                fontsize=15,
                color='black')

    # invert for largest on top
    ax.invert_yaxis()
    ax.plot()

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

####################################################################################
##                              NUMERICAL BLOCK
####################################################################################
