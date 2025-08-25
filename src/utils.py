import numpy as np
import pandas as pd
from typing import Tuple

def train_val_test_split_series(series: pd.Series, train_ratio=0.7, val_ratio=0.15):
    n = len(series)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = series.iloc[:n_train]
    val = series.iloc[n_train:n_train+n_val]
    test = series.iloc[n_train+n_val:]
    return train, val, test

def make_sliding_windows(arr: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(arr)):
        X.append(arr[i-seq_len:i])
        y.append(arr[i])
    X = np.array(X)
    y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y.reshape((-1, 1))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred):
    y_true = np.array(y_true).astype(float)
    y_pred = np.array(y_pred).astype(float)
    eps = 1e-8
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100.0)