import numpy as np


class BaseMetric:
    def __init__(self, name):
        """
        Args:
            name (str | None): metric name.
        """
        self.name = name

    def __call__(self, **kwargs):
        return self._evaluate(**kwargs)


class BaseRecoveryMetric(BaseMetric):
    def __init__(self, name, threshold=1e-6):
        super().__init__(name)
        self.threshold = threshold

    def __call__(self, coef, **kwargs):
        coef_rounded = np.where(
            np.abs(coef) < self.threshold, np.zeros_like(coef), coef
        )
        return self._evaluate(coef=coef_rounded, **kwargs)
