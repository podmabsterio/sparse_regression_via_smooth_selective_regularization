import numpy as np
from sklearn.model_selection import KFold

from src.models.utils import lambda_grid

class CVWrapperOneSE:
    def __init__(self, model, param_values=None, params_num=None, min_ratio=None, cv=5, refit_full_path=True, maximize_param=False):
        self.model = model
        if param_values is None and (params_num is None or min_ratio is None):
            raise ValueError('params values undefined')
        self.param_values = param_values
        self.params_num = params_num
        self.eps = min_ratio
        self.cv = cv
        self.refit_full_path = refit_full_path
        self.maximize_param = maximize_param
        self.coef_ = None
        self.best_param_ = None
        self.best_index_ = None
        self.best_score_ = None
        self.best_score_mean_ = None
        self.best_score_se_ = None

    def fit(self, X, y, **kwargs):
        if self.param_values is None:
            self.param_values = lambda_grid(X, y, self.eps, self.params_num)
        
        kf = KFold(n_splits=self.cv, shuffle=False)

        fold_mse = []
        for train_idx, val_idx in kf.split(X):
            Xtr, ytr = X[train_idx], y[train_idx]
            Xva, yva = X[val_idx], y[val_idx]

            coefs_path = np.asarray(self.model.path(Xtr, ytr, self.param_values))
            preds = Xva @ coefs_path
            mse = np.mean((preds - yva[:, None]) ** 2, axis=0)
            fold_mse.append(mse)

        fold_mse = np.asarray(fold_mse)  # (cv, n_params)
        mean_mse = fold_mse.mean(axis=0)
        se_mse = fold_mse.std(axis=0, ddof=1) / np.sqrt(self.cv)

        best_idx = int(np.argmin(mean_mse))
        best_mean = mean_mse[best_idx]
        best_se = se_mse[best_idx]
        threshold = best_mean + best_se

        candidates = np.flatnonzero(mean_mse <= threshold)
        cand_params = self.param_values[candidates]
        chosen_param = cand_params.max() if self.maximize_param else cand_params.min()
        chosen_idx = int(np.where(self.param_values == chosen_param)[0][0])

        self.best_index_ = chosen_idx
        self.best_param_ = chosen_param
        self.best_score_ = float(mean_mse[chosen_idx])
        self.best_score_mean_ = float(best_mean)
        self.best_score_se_ = float(best_se)

        if self.refit_full_path:
            full_path = np.asarray(self.model.path(X, y, self.param_values))
            self.coef_ = full_path[:, chosen_idx]
        else:
            one = np.asarray(self.model.path(X, y, np.asarray([chosen_param])))
            self.coef_ = one[:, 0]

        return self
