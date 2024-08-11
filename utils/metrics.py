import numpy as np


def loss_fct(y, yhat, hparams, is_dlt=False):
    loss, dlt = 0, 0
    if hparams['loss'].lower() == 'mae':  # mean absolute error
        loss = np.mean(np.abs(y - yhat))
        dlt = -np.sign(y - yhat)
    elif hparams['loss'].lower() == 'mse':  # mean squared error
        loss = np.mean((y - yhat) ** 2) / 2
        dlt = y - yhat
    elif hparams['loss'].lower() == 'rmse':  # root mean squared error
        loss = np.sqrt(np.mean((y - yhat) ** 2))
        dlt = np.mean((y - yhat) * yhat) / loss
    elif hparams['loss'].lower() == 'bce':  # binary cross-entropy
        loss = -np.mean(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
        dlt = (y/(yhat + 1e-10) - (1 - y)/(1 - yhat + 1e-10))

    if is_dlt:
        return loss, dlt
    else:
        return loss


def r2_score(y, yhat):
    """
    Calculate the coefficient of determination (R^2) for a given set of true values `y` and predicted values `yhat`.

    Parameters:
    y (array-like): The true values.
    yhat (array-like): The predicted values.

    Returns:
    float: The R^2 score.
    """
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return r2


def pn(y, yhat):
    yhat[yhat >= 0.5] = 1
    tp = np.sum((y == 1) & (yhat == 1))
    tn = np.sum((y == 0) & (yhat == 0))
    fp = np.sum((y == 0) & (yhat == 1))
    fn = np.sum((y == 1) & (yhat == 0))

    return tp, tn, fp, fn


def accuracy(y, yhat):
    """
    Calculate the accuracy for a given set of true values `y` and predicted values `yhat`.

    Parameters:
    y (array-like): The true values.
    yhat (array-like): The predicted values.

    Returns:
    float: The accuracy.
    """
    tp, tn, fp, fn = pn(y, yhat)
    acc = (tp+tn) / (tp+tn+fp+fn)

    return acc
