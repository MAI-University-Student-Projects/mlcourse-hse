# from __future__ import annotations
#
# from typing import List
import numpy as np
from descents import LossFactory, _RidgeLoss, get_method


class LinearRegression:
    def __init__(self, loss_func: str = 'MSE', **method_config):
        self.solver_ = get_method(loss_function=LossFactory.create_loss_function(loss_func), **method_config)
        self.parameters_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, **procedure_config):
#         adding one-vector for free-weight(bias w0)
        new_shape = tuple(X.shape[i] if i < len(X.shape) - 1 else X.shape[i]+1 for i in range(len(X.shape)))
        features: np.ndarray = np.empty(new_shape)
        features[...,:-1] = X; features[..., -1] = np.ones(X.shape[:-1])
        self.parameters_ = self.solver_.descent(features, y, **procedure_config)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.parameters_[..., :-1]) + self.parameters_[..., -1]


class Ridge:
    def __init__(self, alpha: float = 1.0, loss_func: str = 'MSE', **descent_config):
        self.solver_ = get_method(
            loss_function=_RidgeLoss(parameter=alpha, loss_main=LossFactory.create_loss_function(loss_func)),
            **descent_config
        )
        self.parameters_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, **procedure_config):
#         adding one-vector for free-weight(bias w0)
        new_shape = tuple(X.shape[i] if i < len(X.shape) - 1 else X.shape[i]+1 for i in range(len(X.shape)))
        features: np.ndarray = np.empty(new_shape)
        features[...,:-1] = X; features[..., -1] = np.ones(X.shape[:-1])
        self.parameters_ = self.solver_.descent(features, y, **procedure_config)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.parameters_[..., :-1]) + self.parameters_[..., -1]
