from src.datasets.base_synthetic_dataset import BaseSyntheticSparseLinearDatasets

import numpy as np

class GaussianIIDatasets(BaseSyntheticSparseLinearDatasets):
    def __init__(
        self,
        n: int,
        p: int,
        s: int,
        num_datasets: int,
        test_fraction: float = 2.0,
        center_data = True,
        noise_std: float = 1.0,
        signal_amp: float = 1.0,
        seed: int = 42,
    ):
        self.noise_std = float(noise_std)
        self.signal_amp = float(signal_amp)

        super().__init__(
            n=n,
            p=p,
            s=s,
            num_datasets=num_datasets,
            test_fraction=test_fraction,
            seed=seed,
            center_data=center_data,
        )

    def _sample_theta_true(self, rng: np.random.Generator) -> np.ndarray:
        theta_true = np.zeros(self.p, dtype=float)
        if self.s > 0:
            support = rng.choice(self.p, size=self.s, replace=False)
            signs = rng.choice([-1.0, 1.0], size=self.s)
            theta_true[support] = self.signal_amp * signs
        return theta_true

    def _sample_design(self, rng: np.random.Generator, n_total: int) -> np.ndarray:
        return rng.standard_normal(size=(n_total, self.p))

    def _sample_noise(self, rng: np.random.Generator, n_total: int) -> np.ndarray:
        if self.noise_std == 0.0:
            return np.zeros(n_total, dtype=float)
        return self.noise_std * rng.standard_normal(size=n_total)
