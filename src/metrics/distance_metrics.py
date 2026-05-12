from src.metrics.base_metric import BaseMetric

import numpy as np

EPS = 1e-12


def _get_active_set_mask(theta_true):
    return np.abs(theta_true) > EPS


class L1DistanceActiveSet(BaseMetric):
    def __init__(self):
        super().__init__(name="l1_distance_active_set")

    def _evaluate(self, coef, theta_true, **kwargs):
        active_set_mask = _get_active_set_mask(theta_true)
        diff = coef[active_set_mask] - theta_true[active_set_mask]
        return np.linalg.norm(diff)


class L1DistanceInactiveSet(BaseMetric):
    def __init__(self):
        super().__init__(name="l1_distance_inactive_set")

    def _evaluate(self, coef, theta_true, **kwargs):
        inactive_set_mask = ~_get_active_set_mask(theta_true)
        diff = coef[inactive_set_mask] - theta_true[inactive_set_mask]
        return np.linalg.norm(diff)


class L1Distance(BaseMetric):
    def __init__(self):
        super().__init__(name="l1_coefficients_distance")

    def _evaluate(self, coef, theta_true, **kwargs):
        diff = coef - theta_true
        return np.linalg.norm(diff)
