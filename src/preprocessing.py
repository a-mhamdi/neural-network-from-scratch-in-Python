import numpy as np


# DATA SPLIT
def train_test_split(x: np.ndarray, y: np.ndarray, split_ratio: float = .8) -> tuple:
    """
    Split the dataset intro training and testing sets
    Parameters
    ----------
    x: array
        Features
    y: array
        Targets/Labels
    split_ratio: float
        Ratio of the split
    """
    leng = x.shape[0]
    tronc = np.floor(leng * split_ratio).astype(int) if split_ratio < 1 else x.shape[0]
    # random indices and shuffle them one more time
    indices = np.random.choice(leng, leng, replace=False)
    np.random.shuffle(indices)
    # subsets of data `train` and `test`
    train, test = indices[:tronc], indices[tronc:]
    # split data in `train` and `test` sets
    x_train, x_test = x[train], x[test]
    y_train, y_test = y[train], y[test]

    return (x_train, x_test), (y_train, y_test)


# DATA LOADER
def data_loader(x: np.ndarray, y: np.ndarray, batch_size: int = 1) -> tuple:
    """
    Parameters
    ----------
    x: array
        Features
    y: array
        Labels
    batch_size: int
        Batch size
    """
    leng = x.shape[0]
    how_many, rem = leng // batch_size, leng % batch_size

    indices = np.random.choice(leng, leng, replace=False)
    np.random.shuffle(indices)
    x = x[indices]
    y = y[indices]

    for i in range(0, how_many):
        yield x[i:i+batch_size], y[i:i+batch_size]

    if rem != 0:
        yield x[how_many*batch_size:], y[how_many*batch_size:]
