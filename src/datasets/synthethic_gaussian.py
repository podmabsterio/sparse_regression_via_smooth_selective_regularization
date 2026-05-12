from src.datasets.base_dataset import BaseDatasets

import numpy as np


class SyntheticGaussian(BaseDatasets):
    def __init__(
        self,
        num_datasets: int,
        n: int,
        p: int,
        eff_dim: int,
        noise_std: float = 0.1,
        weight_min: float = 1,
        weight_max: float = 10,
        val_ratio: int = 10,
        rho: float = 0.0,
        block_size: int = 50,
        seed: int = 42,
    ):
        self.rng = np.random.default_rng(seed)
        self.rho = rho
        self.block_size = block_size
        val_size = val_ratio * n

        data = [
            self._generate_dataset(
                n,
                p,
                eff_dim,
                noise_std,
                weight_min,
                weight_max,
                val_size,
                rho,
                block_size,
            )
            for _ in range(num_datasets)
        ]
        super().__init__(data)

    def _generate_dataset(
        self,
        n: int,
        p: int,
        effective_dim: int,
        noise_std: float,
        weight_min: float,
        weight_max: float,
        val_size: int,
        rho: float,
        block_size: int,
    ):
        def generate_correlated_X(rows, cols, rho, block_size):
            X = np.empty((rows, cols))
            for start in range(0, cols, block_size):
                end = min(start + block_size, cols)
                b = end - start
                Z = self.rng.standard_normal((rows, 1))
                E = self.rng.standard_normal((rows, b))
                X[:, start:end] = np.sqrt(rho) * Z + np.sqrt(1 - rho) * E
            return X

        X_train = generate_correlated_X(n, p, rho, block_size)
        X_val = generate_correlated_X(val_size, p, rho, block_size)

        theta_true = np.zeros(p)
        non_zero = self.rng.choice(p, effective_dim, replace=False)
        signs = self.rng.choice([-1, 1], size=effective_dim)
        theta_true[non_zero] = signs * self.rng.uniform(
            weight_min, weight_max, size=effective_dim
        )

        y_train = X_train @ theta_true + self.rng.normal(0, noise_std, size=n)
        y_val = X_val @ theta_true + self.rng.normal(0, noise_std, size=val_size)

        return X_train, y_train, theta_true, X_val, y_val
