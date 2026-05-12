from sklearn.linear_model import (
    Lasso,
    OrthogonalMatchingPursuit,
)

from skglm.datafits import Quadratic
from skglm.solvers import AndersonCD
from skglm.penalties import SCAD, MCPenalty as MCP

from src.models.main_model import MainModel

import numpy as np


class WrappedMainModel():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def path(self, X, y, param_values):
        path = []
        for lam in param_values:
            model = MainModel(Lambda=lam, **self.kwargs)
            model.fit(X, y)
            path.append(model.coef_)
            
        return np.column_stack(path)

class WrappedLasso():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def path(self, X, y, param_values):
        path = []
        for alpha in param_values:
            model = Lasso(alpha=alpha, fit_intercept=False, **self.kwargs)
            model.fit(X, y)
            path.append(model.coef_)
            
        return np.column_stack(path)
    
class WrappedOMP():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        
    def path(self, X, y, param_values):
        path = []
        for n_nonzero_coefs in param_values:
            model = OrthogonalMatchingPursuit(n_nonzero_coefs=int(n_nonzero_coefs), fit_intercept=False, **self.kwargs)
            model.fit(X, y)
            path.append(model.coef_)
            
        return np.column_stack(path)
    
class WrappedSCAD:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.gamma = 3.7
        
    def path(self, X, y, param_values):
        solver = AndersonCD(fit_intercept=False, **self.kwargs)
        estimator_path = solver.path(X, y, Quadratic(), SCAD(1, self.gamma), param_values)
        return estimator_path[1]
    
class WrappedMCP:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.gamma = 3.0
        
    def path(self, X, y, param_values):
        solver = AndersonCD(fit_intercept=False, **self.kwargs)
        estimator_path = solver.path(X, y, Quadratic(), MCP(1, self.gamma), param_values)
        return estimator_path[1]
    

        
        
    
        


