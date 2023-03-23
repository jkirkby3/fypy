from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):

    @abstractmethod
    def residual_apply(self, x):
        """ Loss applied to each residual, e.g. x^2 """
        raise NotImplementedError

    @abstractmethod
    def agg_apply(self, s):
        """ Applied on final result, e.g. sqrt(...) """
        raise NotImplementedError

    def aggregate(self, s: np.ndarray):
        """ Aggregate losses, and apply final agg_apply:  sqrt(...) """
        return self.agg_apply(np.sum(self.residual_apply(s)))


class LossL2(Loss):
    """ sqrt( sum w_i * x_i^2 )"""
    def residual_apply(self, x):
        return np.power(x, 2)

    def agg_apply(self, s):
        return np.sqrt(s)


class SumLoss(Loss):
    """ Sum(x_i) """
    def residual_apply(self, x):
        return x

    def agg_apply(self, s):
        return np.sum(s)
