from sklearn.metrics import mean_squared_error
import numpy as np

from src.metrics.base_metric import BaseMetric
from src.datasets.block_corr import build_covariance as build_block_covariance


class MSE(BaseMetric):
    def __init__(self):
        super().__init__(name="MSE")

    def _evaluate(self, X_test, coef, y_test, **kwargs):
        y_pred = X_test @ coef
        return mean_squared_error(y_test, y_pred)
    
class PredictionErrorToeplitz(BaseMetric):
    def __init__(self, rho):
        super().__init__(name="PE_toeplitz")
        self.rho = rho

    def _evaluate(self, coef, theta_true, **kwargs):
        idx = np.arange(len(coef))
        Sigma = self.rho ** np.abs(idx[:, None] - idx[None, :])
        distance = coef - theta_true
        return distance.T @ Sigma @ distance
    
class PredictionErrorBlock(BaseMetric):
    def __init__(self, rho, group_size):
        super().__init__(name="PE_block")
        self.rho = rho
        self.group_size = group_size

    def _evaluate(self, coef, theta_true, **kwargs):
        Sigma = build_block_covariance(len(theta_true), self.group_size, self.rho)
        distance = coef - theta_true
        return distance.T @ Sigma @ distance
    
class EstimationError(BaseMetric):
    def __init__(self):
        super().__init__(name="EE")

    def _evaluate(self, coef, theta_true, **kwargs):
        distance = coef - theta_true
        return np.sum(distance**2)
