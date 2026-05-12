import numpy as np
from sklearn.model_selection import KFold

from src.models.utils import lambda_grid

class CVWrapper:
    def __init__(self, model, param_values=None, params_num=None, min_ratio=None, cv=5, refit_full_path=True):
        self.model = model
        if param_values is None and (params_num is None or min_ratio is None):
            raise ValueError('params values undefined')
        self.param_values = param_values
        self.params_num = params_num
        self.eps = min_ratio
        self.cv = cv
        self.refit_full_path = refit_full_path
        self.coef_ = None
        self.best_param_ = None
        self.best_index_ = None
        self.best_score_ = None

    def fit(self, X, y, **kwargs):
        if self.param_values is None:
            self.param_values = lambda_grid(X, y, self.eps, self.params_num)
    
        kf = KFold(n_splits=self.cv, shuffle=False)

        scores = np.zeros(len(self.param_values), dtype=float)

        for train_idx, val_idx in kf.split(X):
            Xtr, ytr = X[train_idx], y[train_idx]
            Xva, yva = X[val_idx], y[val_idx]

            coefs_path = np.asarray(self.model.path(Xtr, ytr, self.param_values))
            preds = Xva @ coefs_path
            mse = np.mean((preds - yva[:, None]) ** 2, axis=0)
            scores += mse

        scores /= self.cv
        best_idx = int(np.argmin(scores))

        self.best_index_ = best_idx
        self.best_param_ = self.param_values[best_idx]
        self.best_score_ = float(scores[best_idx])

        if self.refit_full_path:
            full_path = np.asarray(self.model.path(X, y, self.param_values))
            self.coef_ = full_path[:, best_idx]
        else:
            one = np.asarray(self.model.path(X, y, np.asarray([self.best_param_])))
            self.coef_ = one[:, 0]

        return self
