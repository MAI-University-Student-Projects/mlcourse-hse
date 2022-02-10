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
        self.parameters_ = self.solver_.descent(X, y, **procedure_config)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.parameters_)


class Ridge:
    def __init__(self, alpha: float = 1.0, loss_func: str = 'MSE', **descent_config):
        self.solver_ = get_method(
            loss_function=_RidgeLoss(parameter=alpha, loss_main=LossFactory.create_loss_function(loss_func)),
            **descent_config
        )
        self.parameters_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, **procedure_config):
        self.parameters_ = self.solver_.descent(X, y, **procedure_config)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.parameters_)

# class LinearRegression:
#     """
#     Linear regression class
#     """
#
#     def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300):
#         """
#         :param descent_config: gradient descent config
#         :param tolerance: stopping criterion for square of euclidean norm of weight difference (float)
#         :param max_iter: stopping criterion for iterations (int)
#         """
#         self.descent: BaseDescent = get_descent(descent_config)
#
#         self.tolerance: float = tolerance
#         self.max_iter: int = max_iter
#
#         self.loss_history: List[float] = []
#
#     def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
#         """
#         Fitting descent weights for x and y dataset
#         :param x: features array
#         :param y: targets array
#         :return: self
#         """
#         # TODO: fit weights to x and y
#         raise NotImplementedError('LinearRegression fit function not implemented')
#
#     def predict(self, x: np.ndarray) -> np.ndarray:
#         """
#         Predicting targets for x dataset
#         :param x: features array
#         :return: prediction: np.ndarray
#         """
#         return self.descent.predict(x)
#
#     def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
#         """
#         Calculating loss for x and y dataset
#         :param x: features array
#         :param y: targets array
#         """
#         return self.descent.calc_loss(x, y)
