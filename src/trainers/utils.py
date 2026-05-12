from src.metrics import Metrics
from hydra.utils import instantiate


def run_single(dataset, model_config: Metrics):
    model = instantiate(model_config)
    X, y, theta_true, _, _ = dataset
    run_info = {"X": X, "y": y, "theta_true": theta_true}

    model.fit(**run_info)

    return model.coef_
