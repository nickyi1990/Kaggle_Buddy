import pickle

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
    """
    Parameters
    ----------
    data : any object
    filename : string

    Returns
    -------
    None
    """
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(filename):
    """
    Parameters
    ----------
    filename : string

    Returns
    -------
    None
    """
    with open(filename, 'rb') as f:
        # https://stackoverflow.com/questions/28218466/unpickling-a-python-2-object-with-python-3
        return pickle.load(f, encoding='latin1')
