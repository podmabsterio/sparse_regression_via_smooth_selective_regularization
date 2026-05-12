import numpy as np


class BaseMetricAggregator:
    def __init__(self, name, aggregation_function):
        self.name = name
        self.aggregation_function = aggregation_function

    def __call__(self, values):
        return self.aggregation_function(values)


class MeanAggregator(BaseMetricAggregator):
    def __init__(self):
        super().__init__("mean", np.mean)


class StdAggregator(BaseMetricAggregator):
    def __init__(self):
        super().__init__("std", np.std)
        
        
class MedianAggregator(BaseMetricAggregator):
    def __init__(self):
        super().__init__("median", np.median)


class IQRAggregator(BaseMetricAggregator):
    def __init__(self):
        def iqr(arr):
            q25 = np.percentile(arr, 25)
            q75 = np.percentile(arr, 75)
            return q75 - q25
        
        super().__init__("iqr", iqr)
        
        
