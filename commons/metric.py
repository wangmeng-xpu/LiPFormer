import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray):
    assert y_true.shape == y_pred.shape
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    assert y_true.shape == y_pred.shape
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray):
    assert y_true.shape == y_pred.shape
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray):
    assert y_true.shape == y_pred.shape
    return np.abs(y_true - y_pred).sum() / y_true.sum()
