import numpy as np


def loss_fct(y: np.ndarray, yhat: np.ndarray, hparams: dict, is_dloss: bool = False) -> tuple:
    """
    This function calculates the loss `loss` and its negative gradient `dloss` with respect to the predicted values `yhat`.

    Parameters:
        y (np.ndarray): The true values.
        yhat (np.ndarray): The predicted values.
        hparams (dict): A dictionary containing the hyperparameters, including the loss function.
        is_dloss (bool): A flag indicating whether to return the `dloss`. Defaults to False.

    Returns:
        tuple: A tuple containing `loss` and `dloss` if `is_dloss` is True, otherwise just `loss`.
    """
    loss, dlt = 0, 0
    if hparams['loss'].lower() == 'mae':  # mean absolute error
        loss = np.mean(np.abs(y - yhat))
        dloss = np.sign(y - yhat)
    elif hparams['loss'].lower() == 'mse':  # mean squared error
        loss = np.mean((y - yhat) ** 2)
        dloss = (y - yhat)
    elif hparams['loss'].lower() == 'rmse':  # root mean squared error
        loss = np.sqrt(np.mean((y - yhat) ** 2))
        dloss = np.mean((y - yhat) * yhat) / loss
    elif hparams['loss'].lower() == 'bce':  # binary cross-entropy
        loss = -np.mean(y * np.log(yhat + 1e-10) + (1 - y) * np.log(1 - yhat - 1e-10))
        dloss = y/(yhat + 1e-10) - (1 - y)/(1 - yhat + 1e-10)

    if is_dloss:
        return loss, dloss
    else:
        return loss


# --- CLASSIFICATION METRICS: SINGLE TARGET ---

def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
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


def pn(y: np.ndarray, yhat: np.ndarray) -> tuple:
    """
    Calculate the number of true positives, true negatives, false positives, and false negatives.

    Parameters:
    y (array-like): The true values.
    yhat (array-like): The predicted values.

    Returns:
    tuple: A tuple containing the number of true positives, true negatives, false positives, and false negatives.
    """
    yhat = (yhat >= 0.5) * 1
    tp = np.sum((y == 1) & (yhat == 1))
    tn = np.sum((y == 0) & (yhat == 0))
    fp = np.sum((y == 0) & (yhat == 1))
    fn = np.sum((y == 1) & (yhat == 0))

    return tp, tn, fp, fn


def accuracy(y: np.ndarray, yhat: np.ndarray) -> float:
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


def precision(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Calculate the precision for a given set of true values `y` and predicted values `yhat`.

    Parameters:
    y (array-like): The true values.
    yhat (array-like): The predicted values.

    Returns:
    float: The precision.
    """
    tp, tn, fp, fn = pn(y, yhat)
    prec = tp / (tp+fp)

    return prec


def recall(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Calculate the recall for a given set of true values `y` and predicted values `yhat`.

    Parameters:
    y (array-like): The true values.
    yhat (array-like): The predicted values.

    Returns:
    float: The recall.
    """
    tp, tn, fp, fn = pn(y, yhat)
    rec = tp / (tp+fn)

    return rec


def f1_score(y: np.ndarray, yhat: np.ndarray) -> float:
    """
    Calculate the F1 score for a given set of true values `y` and predicted values `yhat`.

    Parameters:
    y (array-like): The true values.
    yhat (array-like): The predicted values.

    Returns:
    float: The F1 score.
    """
    prec = precision(y, yhat)
    rec = recall(y, yhat)
    f1 = 2 * (prec * rec) / (prec + rec)

    return f1


def cm(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """
    Calculate the confusion matrix for a given set of true values `y` and predicted values `yhat`.

    Parameters:
    y (array-like): The true values.
    yhat (array-like): The predicted values.

    Returns:
    np.ndarray: The confusion matrix.
    """
    tp, tn, fp, fn = pn(y, yhat)
    cmat = np.array([[tp, fp], [fn, tn]])

    return cmat
