from hydra.utils import instantiate
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib


class ParallelTrainer:
    def __init__(self, n_jobs=-1, batch_size="auto"):
        self.n_jobs = n_jobs
        self.batch_size = batch_size

    def run(self, models_config, datasets):
        results = {}

        def run_single(dataset, model_config):
            import src.hydra_utils.resolvers
            import src.utils.silence_warnings

            model = instantiate(model_config)
            model.fit(**dataset)
            return dataset["index"], model.coef_

        n_datasets = len(datasets)
        for model_name, model_cfg in models_config.items():
            results[model_name] = [0 for _ in range(n_datasets)]
            with tqdm_joblib(tqdm(desc=f"Running {model_name}", total=n_datasets)):
                with parallel_backend("loky", inner_max_num_threads=1):
                    model_results = Parallel(
                        n_jobs=self.n_jobs, batch_size=self.batch_size
                    )(delayed(run_single)(dataset, model_cfg) for dataset in datasets)

            for index, coef in model_results:
                results[model_name][index] = coef

        return results
