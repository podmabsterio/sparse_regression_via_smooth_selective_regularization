import warnings

import src.hydra_utils.resolvers

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils import init_saving, save_dict_array_list

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="src/configs", config_name="run_on_dataset")
def main(config):
    datasets = instantiate(config.datasets)
    save_path = init_saving(config)
    
    trainer = instantiate(config.trainer)
    
    results = trainer.run(
        models_config=config.models,
        datasets=datasets
    )
    
    save_dict_array_list(results, save_path)
    
if __name__ == "__main__":
    main()