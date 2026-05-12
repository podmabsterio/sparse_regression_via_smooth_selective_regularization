from src.metrics.base_metric import BaseMetric

from sklearn.metrics import precision_recall_curve, auc, f1_score
import numpy as np

EPS = 1e-12

def _get_active_set_mask(theta_true) -> np.ndarray:
    return np.abs(theta_true) > EPS


class RecoveryPRAUC(BaseMetric):
    def __init__(self):
        super().__init__('recovery_pr_auc')
        
    def _evaluate(self, coef, theta_true, **kwargs):
        y_true = _get_active_set_mask(theta_true).astype(int)
        y_score = np.abs(coef)
        precision, recall, _ = precision_recall_curve(y_true, y_score)

        return auc(recall, precision)
    
class DeltaMargin(BaseMetric):
    def __init__(self):
        super().__init__('delta_margin')
        
    def _evaluate(self, coef, theta_true, **kwargs):
        active_set_mask = _get_active_set_mask(theta_true)
        min_active_set = np.min(np.abs(coef[active_set_mask]))
        max_inactive_set = np.max(np.abs(coef[~active_set_mask]))
        
        return min_active_set - max_inactive_set
    
class MAVi(BaseMetric):
    def __init__(self):
        super().__init__('MAVi')
        
    def _evaluate(self, coef, theta_true, **kwargs):
        inactive_set_mask = ~_get_active_set_mask(theta_true)
        return np.max(np.abs(coef[inactive_set_mask]))
    
class F1(BaseMetric):
    def __init__(self, threshold):
        super().__init__('f1')
        self.threshold = threshold
        
    def _evaluate(self, coef, theta_true, apply_threshold=False, **kwargs):
        if apply_threshold:
            return f1_score(np.abs(coef) > self.threshold, theta_true != 0)
        return f1_score(coef != 0, theta_true != 0)
    
class NumActive(BaseMetric):
    def __init__(self, threshold):
        super().__init__('num_active')
        self.threshold = threshold
        
    def _evaluate(self, coef, apply_threshold=False, **kwargs):
        if apply_threshold:
            return np.sum(np.abs(coef) > self.threshold)
        return np.sum(coef != 0)
    

