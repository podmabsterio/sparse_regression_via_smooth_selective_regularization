import numpy as np
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    "linspace",
    lambda start, stop, num: np.linspace(float(start), float(stop), int(num)).tolist(),
)

OmegaConf.register_new_resolver(
    "logspace",
    lambda start, stop, num: np.logspace(float(start), float(stop), int(num)).tolist(),
)

OmegaConf.register_new_resolver(
    "range",
    lambda start, stop: np.array(list(range(start, stop)), dtype=int),
)
