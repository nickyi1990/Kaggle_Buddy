import pandas as pd

def ka_replace_hash(hashes, hash_id_table):
    '''
       Exchange values and index, very useful in recommendation task

       Parameters
       ----------
       hashes: Pandas Series
               Series you want to convert
       hash_id_table: Pandas Series
               index is the number u want convert hash to
               values is the has values)
       Return
       ------

       Example
       -------
    '''
    replace_table = pd.Series(hash_id_table.index, index=hash_id_table.values)
    return replace_table[hashes].values
