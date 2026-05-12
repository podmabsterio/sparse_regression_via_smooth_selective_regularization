import numpy as np

from sklearn.linear_model import LinearRegression


class OracleEstimator:
    def __init__(self):
        self.ols = LinearRegression(fit_intercept=False)

    def fit(self, X, y, theta_true, **kwargs):
        active_mask = theta_true != 0
        X_active = X[:, active_mask]
        self.ols.fit(X_active, y)
        self.coef_ = np.zeros_like(theta_true)
        self.coef_[active_mask] = self.ols.coef_
        self.active_mask = active_mask
        return self

    def predict(self, X):
        return self.ols.predict(X[:, self.active_mask])
