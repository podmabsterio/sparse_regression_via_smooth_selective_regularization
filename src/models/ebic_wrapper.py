import numpy as np
import math

from src.models.utils import lambda_grid

def log_choose(p: int, k_arr: np.ndarray) -> np.ndarray:
    k_arr = np.asarray(k_arr, dtype=int)
    k_arr = np.clip(k_arr, 0, p)
    lg = np.vectorize(math.lgamma)
    return lg(p + 1) - lg(k_arr + 1) - lg(p - k_arr + 1)

class EBICWrapper:
    def __init__(self, model, param_values=None, params_num=None, min_ratio=None, gamma=0.5,
                 refit_full_path=True, coef_tol=1e-10):
        self.model = model
        if param_values is None and (params_num is None or min_ratio is None):
            raise ValueError("param_values undefined")
        self.param_values = param_values
        self.params_num = params_num
        self.eps = min_ratio
        self.gamma = gamma
        self.refit_full_path = refit_full_path
        self.coef_tol = coef_tol

        self.coef_ = None
        self.best_param_ = None
        self.best_index_ = None
        self.best_score_ = None

    def fit(self, X, y, **kwargs):
        if self.param_values is None:
            self.param_values = lambda_grid(X, y, self.eps, self.params_num)

        coefs_path = np.asarray(self.model.path(X, y, self.param_values))

        preds = X @ coefs_path
        resid = preds - y[:, None]
        rss = np.sum(resid * resid, axis=0)
        rss = np.maximum(rss, np.finfo(float).tiny)

        n, p = X.shape
        k = np.sum(np.abs(coefs_path) > self.coef_tol, axis=0).astype(int)

        logC = log_choose(p, k)
        ebic = n * (np.log(rss) - np.log(n)) + k * np.log(n) + 2.0 * self.gamma * logC

        best_idx = int(np.argmin(ebic))
        self.best_index_ = best_idx
        self.best_param_ = self.param_values[best_idx]
        self.best_score_ = float(ebic[best_idx])

        self.coef_ = coefs_path[:, best_idx]
        return self
