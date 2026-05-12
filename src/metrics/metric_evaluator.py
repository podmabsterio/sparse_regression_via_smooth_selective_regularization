from src.metrics.base_metric import BaseMetric
from src.metrics.results_array import ModelsExperimentResults
from src.metrics.metrics_aggregation import BaseMetricAggregator

from typing import List
from collections import defaultdict
import pandas as pd

from tqdm import tqdm


class PerModelMetricEvaluator:
    def __init__(
        self,
        results: ModelsExperimentResults,
        metrics: List[BaseMetric],
        aggregators: List[BaseMetricAggregator],
    ):
        self.results = results
        self.n_models = len(results.model_names)
        self.n_metrics = len(metrics)
        self.metrics = metrics
        self.aggregators = aggregators
        self.aggregated_metric_values = {
            aggregator.name: defaultdict(list)
            for aggregator in aggregators
        }

    def evaluate(self):
        pbar = tqdm(total=self.n_models * self.n_metrics)
        for model_name in self.results.model_names:
            pbar.set_description(f"Evaluating metrics for {model_name}")
            results = self.results.get_results(model_name)
            for aggregator in self.aggregators:
                self.aggregated_metric_values[aggregator.name]["model"].append(model_name)
            for metric in self.metrics:
                model_metric_values = []
                for i, result in enumerate(results):
                    if model_name == 'MainModelCV':
                        model_metric_values.append(metric(apply_threshold=True, **result))
                    else:
                        model_metric_values.append(metric(**result))

                for aggregator in self.aggregators:
                    aggregated_value = aggregator(model_metric_values)
                    self.aggregated_metric_values[aggregator.name][metric.name].append(
                        aggregated_value
                    )
                
                pbar.update(1)

        aggregated_dataframes = {}
        for name, data_dict in self.aggregated_metric_values.items():
            aggregated_dataframes[f'{name}_scalar_metrics'] = pd.DataFrame(data_dict)

        return aggregated_dataframes
