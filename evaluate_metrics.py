import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils import init_saving, save_dict_to_path, get_save_path
from src.metrics import PerModelMetricEvaluator, ModelsExperimentResults

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="src/configs", config_name="metrics")
def main(config):
    datasets = instantiate(config.datasets)
    save_path = init_saving(config, 'metrics_config')
    
    
    scalar_metrics_list = instantiate(config.scalar_metrics)
    scalar_aggregators_list = instantiate(config.scalar_aggregators)
    
    coefs_path = config.coefs_dir + '/' + config.run_name 
    
    results = ModelsExperimentResults(datasets, coefs_path)
    evaluator = PerModelMetricEvaluator(results, scalar_metrics_list, scalar_aggregators_list)
    
    scalar_metrics_values = evaluator.evaluate()
    
    save_dict_to_path(save_path, scalar_metrics_values)
    
if __name__ == "__main__":
    main()