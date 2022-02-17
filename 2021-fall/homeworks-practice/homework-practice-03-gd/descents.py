# from enum import auto
# from enum import Enum
# from typing import Dict
from typing import Callable, Dict, Type, Union
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

# do not regularize bias-weight(last one)
    def __call__(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> float:
        return super().__call__(X, w, y) + self.parameter_ * np.linalg.norm(w[..., :-1])**2

    def gradient(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> np.ndarray:
        grad: np.ndarray = np.empty(w.shape)
        grad[...,:-1] = super().gradient(X[...,:-1], w[...,:-1], y) + 2*self.parameter_ * w[...,:-1]
        grad[...,-1] = super().gradient(X[...,-1], w[...,-1], y)
        return grad


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
        self.p_: float = p_
        self.iterations_: int = 0

    def __call__(self) -> float:
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iterations_ += 1
        return self.lambda_ * (self.s0_ / (self.s0_ + self.iterations_)) ** self.p_


class GradientDescent:
    def __init__(self, loss_function: _LossFunction,
                 tolerance: float = 1e-3, max_iter: int = 500,
                 init_weight_generator: Callable[..., Union[float, np.ndarray]] = np.random.normal,
                 **learning_kwargs):
        self.epsilon_ = tolerance
        self.iter_limit_ = max_iter
        self.loss_func_: _LossFunction = loss_function
        self.step_info_: LearningRate = LearningRate(**learning_kwargs)
        self.w_init_method_ = init_weight_generator
        self.gradient_norm_log_ = [None] * self.iter_limit_
        self.weights_ = None

    def descent(self, X: np.ndarray, y: np.ndarray, **weight_init_kwargs) -> np.ndarray:
        self.weights_ = self.w_init_method_(size=X.shape[1:], **weight_init_kwargs)
        for i in range(self.iter_limit_):
            delta = self._update_weights(self._g(X, y))
            if delta < self.epsilon_:
                self.gradient_norm_log_ = [x for x in self.gradient_norm_log_ if x is not None]
                break
        return self.weights_

    def _g(self, X, y) -> np.ndarray:
        # gradient part of step (for SGD different)
        return self.loss_func_.gradient(X, self.weights_, y)

    def _update_weights(self, gradient: np.ndarray) -> float:
        # weight part of step (for Momentum different)
        w_prev = self.weights_.copy()
        
        self.weights_ -= self.step_info_() * gradient
        
        err = np.linalg.norm(self.weights_ - w_prev)
        self.gradient_norm_log_[self.step_info_.iterations_-1] = np.linalg.norm(gradient)
        return err


class StochasticGradientDescent(GradientDescent):
    def __init__(self, batch_size: int = 50, **kwargs):
        super().__init__(**kwargs)
        self.batch_size_ = batch_size
        self.rng_ = np.random.default_rng()

    def descent(self, X: np.ndarray, y: np.ndarray, **weight_init_kwargs) -> np.ndarray:
        self.batch_size_ = self.batch_size_ if self.batch_size_ < y.shape[0] else y.shape[0]
        return super().descent(X, y, **weight_init_kwargs)

    def _g(self, X, y) -> np.ndarray:
        # replace=false: without replacement in original sample, i.e. value cannot be selected many times
        batch = self.rng_.choice(np.arange(y.shape[0]), self.batch_size_, replace=False)
        gradient = self.loss_func_.gradient(X[batch], self.weights_, y[batch])
        return gradient


class MomentumGradientDescent(GradientDescent):
    def __init__(self, alpha: float = 0.9, **descent_kwargs):
        super().__init__(**descent_kwargs)
        self.alpha_: float = alpha
        self.grad_momentum_ = None

    def _update_weights(self, gradient: np.ndarray) -> float:
        w_prev = self.weights_.copy()
        
        self.grad_momentum_ = self.alpha_*self.grad_momentum_ + self.step_info_() * gradient
        self.weights_ -= self.grad_momentum_
        
        err = np.linalg.norm(self.weights_ - w_prev)
        self.gradient_norm_log_[self.step_info_.iterations_-1] = np.linalg.norm(gradient)
        return err
    
    def descent(self, X: np.ndarray, y: np.ndarray, **weight_init_kwargs) -> np.ndarray:
        self.grad_momentum_ = np.zeros(X.shape[1:])
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

    def _update_weights(self, gradient: np.ndarray) -> float:
        w_prev = self.weights_.copy()
        
        self.mu_ = self.beta1_*self.mu_ + (1 - self.beta1_)*gradient
        self.v_ = self.beta2_*self.v_ + (1 - self.beta1_)*np.square(gradient)
        self.weights_ -= self.step_info_.lambda_*self.mu_ / (np.sqrt(self.v_) + self.eps_)
        self.step_info_.iterations_ += 1

        err = np.linalg.norm(self.weights_ - w_prev)
        self.gradient_norm_log_[self.step_info_.iterations_-1] = np.linalg.norm(gradient)
        return err

    def descent(self, X: np.ndarray, y: np.ndarray, **weight_init_kwargs) -> np.ndarray:
        self.mu_ = np.zeros(X.shape[1:])
        self.v_ = np.zeros(X.shape[1:])
        return super().descent(X, y, **weight_init_kwargs)


def get_method(method_name: str = 'GD', **kwargs) -> GradientDescent:
    method_typedict = {
        'GD': GradientDescent,
        'SGD': StochasticGradientDescent,
        'Momentum': MomentumGradientDescent,
        'Adam': Adam
    }
    return method_typedict[method_name](**kwargs)
