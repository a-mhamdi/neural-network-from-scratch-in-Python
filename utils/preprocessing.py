import numpy as np


### DATA SPLIT
def train_test_split(X, y, split_ratio=.8):
    """
    Split the dataset intro training and testing sets
    Parameters
    ----------
    X: array
        Features
    y: array
        Labels
    split_ratio: float
        Ratio of the split
    """

    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in range(len(X)):
        if np.random.rand() < split_ratio:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])

    return (np.array(X_train), np.array(X_test)), (np.array(y_train), np.array(y_test))


### DATA LOADER
def data_loader(X, y, batch_size=1):
    """
    Parameters
    ----------
    X: array
        Features
    y: array
        Labels
    batch_size: int
        Batch size
    """

    for i in range(0, len(X), batch_size):
        yield X[i:i + batch_size], y[i:i + batch_size]
