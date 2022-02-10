# from enum import auto
# from enum import Enum
# from typing import Dict
from typing import Callable, Dict, Type, Tuple, Union
import numpy as np


class _LossFunction:
    def __call__(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _MSELoss(_LossFunction):
    def __call__(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
        return np.square(y - np.dot(X, w)).mean(axis=None)

    def gradient(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
        return -2*np.dot(X.T, (y - np.dot(X, w))) / y.shape[0]


# Duck typing is a concept related to dynamic typing,
# where the type or the class of an object is less important than the methods it defines.
# When you use duck typing, you do not check types at all.
# Instead, you check for the presence of a given method or attribute

# decorator pattern
# because of duck-typing don't need any interfaces
# inheritance used in its definitive form for reuse of code
class _AbstractRegularizator(_LossFunction):
    def __init__(self, loss_main: _LossFunction):
        self.loss_main_: _LossFunction = loss_main

    def __call__(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
        return self.loss_main_(X, w, y)

    def gradient(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.loss_main_.gradient(X, w, y)


class _RidgeLoss(_AbstractRegularizator):
    def __init__(self, parameter: float,
                 loss_main: _LossFunction):
        super().__init__(loss_main)
        self.parameter_: float = parameter

    def __call__(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
        return super().__call__(X, w, y) + self.parameter_ * np.linalg.norm(w)**2

    def gradient(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
        return super().gradient(X, w, y) + 2*self.parameter_ * w


class LossFactory:
    loss_type_dict: Dict[str, Type[_LossFunction]] = {
        'MSE': _MSELoss
    }
    @staticmethod
    def create_loss_function(loss_name: str) -> _LossFunction:
        return LossFactory.loss_type_dict[loss_name]()


class LearningRate:
    def __init__(self, lambda_: float = 1e-3, s0_: float = 1.0, p_: float = 0.5):
        self.lambda_: float = lambda_
        self.s0_: float = s0_
        self.p_:float = p_
        self.iterations_ = 0

    def __call__(self) -> float:
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iterations_ += 1
        return self.lambda_ * (self.s0_ / (self.s0_ + self.iterations_)) ** self.p_


class GradientDescent:
    def __init__(self, loss_function: _LossFunction,
                 tolerance: float = 1e-3, max_iter: int = 300,
                 init_weight_generator: Callable[..., Union[float, np.ndarray]] = np.random.rand,
                 **learning_kwargs):
        self.epsilon_ = tolerance
        self.iter_limit_ = max_iter
        self.loss_func_: _LossFunction = loss_function
        self.step_info_: LearningRate = LearningRate(**learning_kwargs)
        self.w_init_method_ = init_weight_generator
        self.gradient_norm_log = [None] * self.iter_limit_
        self.weights_ = None

    def descent(self, X: np.ndarray, y: np.ndarray, **weight_init_kwargs) -> np.ndarray:
        self.weights_ = self.w_init_method_(size=X.shape[1], **weight_init_kwargs)
        g, error = self._g(X, y)
        while error > self.epsilon_:
            self._update_weights(g)
            if self.step_info_.iterations_ >= self.iter_limit_:
                break
            g, error = self._g(X, y)
        return self.weights_

    def _g(self, X, y) -> Tuple[np.ndarray, float]:
        # gradient part of step (for SGD different)
        gradient = self.loss_func_.gradient(X, self.weights_, y)
        error = np.linalg.norm(gradient)
        self.gradient_norm_log[self.step_info_.iterations_] = error
        return gradient, error

    def _update_weights(self, gradient: np.ndarray) -> None:
        # weight part of step (for Momentum different)
        self.weights_ -= self.step_info_() * gradient


class StochasticGradientDescent(GradientDescent):
    def __init__(self, batch_size: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.batch_size_ = batch_size
        self.rng_ = np.random.default_rng()

    def descent(self, X: np.ndarray, y: np.ndarray, **weight_init_kwargs) -> np.ndarray:
        self.batch_size_ = self.batch_size_ if self.batch_size_ < y.shape[0] else y.shape[0]
        self.weights_ = self.w_init_method_(size=X.shape[1], **weight_init_kwargs)
        for i in range(self.iter_limit_):
            self._update_weights(self._g(X, y)[0])
        return self.weights_

    def _g(self, X, y) -> Tuple[np.ndarray, float]:
        # replace=false: without replacement in original sample, i.e. value cannot be selected many times
        batch = self.rng_.choice(np.arange(y.shape[0]), self.batch_size_, replace=False)
        gradient = self.loss_func_.gradient(X[batch], self.weights_, y[batch])
        error = np.linalg.norm(gradient)
        self.gradient_norm_log[self.step_info_.iterations_] = error
        return gradient, error


class MomentumGradientDescent(GradientDescent):
    def __init__(self, alpha: float = 0.9, **descent_kwargs):
        super().__init__(**descent_kwargs)
        self.alpha_: float = alpha
        self.grad_momentum_ = None

    def _update_weights(self, gradient: np.ndarray) -> None:
        self.grad_momentum_ = self.alpha_*self.grad_momentum_ + self.step_info_() * gradient
        self.weights_ -= self.grad_momentum_
    
    def descent(self, X: np.ndarray, y: np.ndarray, **weight_init_kwargs) -> np.ndarray:
        self.grad_momentum_ = np.zeros(X.shape[1])
        return super().descent(X, y, **weight_init_kwargs)


class Adam(GradientDescent):
    def __init__(self, beta_1: float = 0.9, beta_2: float = 0.99, **descent_kwargs):
        super().__init__(**descent_kwargs)
        self.eps_: float = 1e-8

        # mu <-> momentum decay of gradient
        self.mu_ = None
        # v <-> elementwise normalisation of gradient for sane step value
        self.v_ = None
        self.beta1_: float = beta_1
        self.beta2_: float = beta_2

    def _update_weights(self, gradient: np.ndarray) -> None:
        self.mu_ = self.beta1_*self.mu_ + (1 - self.beta1_)*gradient
        self.v_ = self.beta2_*self.v_ + (1 - self.beta1_)*np.square(gradient)
        self.weights_ -= self.step_info_.lambda_*self.mu_ / (np.sqrt(self.v_) + self.eps_)
        self.step_info_.iterations_ += 1

    def descent(self, X: np.ndarray, y: np.ndarray, **weight_init_kwargs) -> np.ndarray:
        self.mu_ = np.zeros(X.shape[1])
        self.v_ = np.zeros(X.shape[1])
        return super().descent(X, y, **weight_init_kwargs)


def get_method(method_name: str = 'GD', **kwargs) -> GradientDescent:
    method_typedict = {
        'GD': GradientDescent,
        'SGD': StochasticGradientDescent,
        'Momentum': MomentumGradientDescent,
        'Adam': Adam
    }
    return method_typedict[method_name](**kwargs)

# @dataclass
# class LearningRate:
#     lambda_: float = 1e-3
#     s0: float = 1
#     p: float = 0.5
#
#     iteration: int = 0
#
#     def __call__(self):
#         """
#         Calculate learning rate according to lambda (s0/(s0 + t))^p formula
#         """
#         self.iteration += 1
#         return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p
#
#
# class LossFunction(Enum):
#     MSE = auto()
#     MAE = auto()
#     LogCosh = auto()
#     Huber = auto()
#
#
# class BaseDescent:
#     """
#     A base class and templates for all functions
#     """
#
#     def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
#         """
#         :param dimension: feature space dimension
#         :param lambda_: learning rate parameter
#         :param loss_function: optimized loss function
#         """
#         self.w: np.ndarray = np.random.rand(dimension)
#         self.lr: LearningRate = LearningRate(lambda_=lambda_)
#         self.loss_function: LossFunction = loss_function
#
#     def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
#         return self.update_weights(self.calc_gradient(x, y))
#
#     def update_weights(self, gradient: np.ndarray) -> np.ndarray:
#         """
#         Template for update_weights function
#         Update weights with respect to gradient
#         :param gradient: gradient
#         :return: weight difference (w_{k + 1} - w_k): np.ndarray
#         """
#         pass
#
#     def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
#         """
#         Template for calc_gradient function
#         Calculate gradient of loss function with respect to weights
#         :param x: features array
#         :param y: targets array
#         :return: gradient: np.ndarray
#         """
#         pass
#
#     def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
#         """
#         Calculate loss for x and y with our weights
#         :param x: features array
#         :param y: targets array
#         :return: loss: float
#         """
#         # TODO: implement loss calculation function
#         raise NotImplementedError('BaseDescent calc_loss function not implemented')
#
#     def predict(self, x: np.ndarray) -> np.ndarray:
#         """
#         Calculate predictions for x
#         :param x: features array
#         :return: prediction: np.ndarray
#         """
#         # TODO: implement prediction function
#         raise NotImplementedError('BaseDescent predict function not implemented')
#
#
# class VanillaGradientDescent(BaseDescent):
#     """
#     Full gradient descent class
#     """
#
#     def update_weights(self, gradient: np.ndarray) -> np.ndarray:
#         """
#         :return: weight difference (w_{k + 1} - w_k): np.ndarray
#         """
#         # TODO: implement updating weights function
#         raise NotImplementedError('VanillaGradientDescent update_weights function not implemented')
#
#     def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
#         # TODO: implement calculating gradient function
#         raise NotImplementedError('VanillaGradientDescent calc_gradient function not implemented')
#
#
# class StochasticDescent(VanillaGradientDescent):
#     """
#     Stochastic gradient descent class
#     """
#
#     def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
#                  loss_function: LossFunction = LossFunction.MSE):
#         """
#         :param batch_size: batch size (int)
#         """
#         super().__init__(dimension, lambda_, loss_function)
#         self.batch_size = batch_size
#
#     def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
#         # TODO: implement calculating gradient function
#         raise NotImplementedError('StochasticDescent calc_gradient function not implemented')
#
#
# class MomentumDescent(VanillaGradientDescent):
#     """
#     Momentum gradient descent class
#     """
#
#     def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
#         super().__init__(dimension, lambda_, loss_function)
#         self.alpha: float = 0.9
#
#         self.h: np.ndarray = np.zeros(dimension)
#
#     def update_weights(self, gradient: np.ndarray) -> np.ndarray:
#         """
#         :return: weight difference (w_{k + 1} - w_k): np.ndarray
#         """
#         # TODO: implement updating weights function
#         raise NotImplementedError('MomentumDescent update_weights function not implemented')
#
#
# class Adam(VanillaGradientDescent):
#     """
#     Adaptive Moment Estimation gradient descent class
#     """
#
#     def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
#         super().__init__(dimension, lambda_, loss_function)
#         self.eps: float = 1e-8
#
#         self.m: np.ndarray = np.zeros(dimension)
#         self.v: np.ndarray = np.zeros(dimension)
#
#         self.beta_1: float = 0.9
#         self.beta_2: float = 0.999
#
#         self.iteration: int = 0
#
#     def update_weights(self, gradient: np.ndarray) -> np.ndarray:
#         """
#         :return: weight difference (w_{k + 1} - w_k): np.ndarray
#         """
#         # TODO: implement updating weights function
#         raise NotImplementedError('Adagrad update_weights function not implemented')
#
#
# class BaseDescentReg(BaseDescent):
#     """
#     A base class with regularization
#     """
#
#     def __init__(self, *args, mu: float = 0, **kwargs):
#         """
#         :param mu: regularization coefficient (float)
#         """
#         super().__init__(*args, **kwargs)
#
#         self.mu = mu
#
#     def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
#         """
#         Calculate gradient of loss function and L2 regularization with respect to weights
#         """
#         l2_gradient: np.ndarray = np.zeros_like(x.shape[1])  # TODO: replace with L2 gradient calculation
#
#         return super().calc_gradient(x, y) + l2_gradient * self.mu
#
#
# class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
#     """
#     Full gradient descent with regularization class
#     """
#
#
# class StochasticDescentReg(BaseDescentReg, StochasticDescent):
#     """
#     Stochastic gradient descent with regularization class
#     """
#
#
# class MomentumDescentReg(BaseDescentReg, MomentumDescent):
#     """
#     Momentum gradient descent with regularization class
#     """
#
#
# class AdamReg(BaseDescentReg, Adam):
#     """
#     Adaptive gradient algorithm with regularization class
#     """
#

# def get_descent(descent_config: dict) -> BaseDescent:
#     descent_name = descent_config.get('descent_name', 'full')
#     regularized = descent_config.get('regularized', False)
#
#     descent_mapping: Dict[str, Type[BaseDescent]] = {
#         'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
#         'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
#         'momentum': MomentumDescent if not regularized else MomentumDescentReg,
#         'adam': Adam if not regularized else AdamReg
#     }
#
#     if descent_name not in descent_mapping:
#         raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')
#
#     descent_class = descent_mapping[descent_name]
#
#     return descent_class(**descent_config.get('kwargs', {}))
