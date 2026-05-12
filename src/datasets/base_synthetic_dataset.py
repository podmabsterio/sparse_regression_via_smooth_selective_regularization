from src.datasets.base_dataset import BaseDatasets

import numpy as np

class BaseSyntheticSparseLinearDatasets(BaseDatasets):
    def __init__(
        self,
        n: int,
        p: int,
        s: int,
        num_datasets: int,
        test_fraction: float = 0.2,
        seed: int | None = 42,
        center_data = True,
    ):
        self.p = int(p)
        self.s = int(s)
        self.num_datasets = int(num_datasets)
        self.test_fraction = float(test_fraction)

        rng = np.random.default_rng(seed)

        n_train = int(n)
        n_test = int(np.floor(n * self.test_fraction))
        if n_train <= 0 or n_test <= 0:
            raise ValueError("test_fraction leads to empty train or test split.")

        n_total = n_train + n_test
        self.n = n_total

        data = []
        for _ in range(self.num_datasets):
            theta_true = self._sample_theta_true(rng)
            X = self._sample_design(rng, n_total)
            
            idx = rng.permutation(n_total)
            train_idx = idx[:n_train]
            test_idx = idx[n_train:]
            X_train = X[train_idx]
            X_test = X[test_idx]
            
            if center_data:
                mu = X_train.mean(axis=0, keepdims=True)
                sigma = X_train.std(axis=0, keepdims=True)
                sigma = np.maximum(sigma, 1e-12)

                X_train = (X_train - mu) / sigma
                X_test = (X_test - mu) / sigma
            
            eps = self._sample_noise(rng, n_total)
            y_train = X_train @ theta_true + eps[train_idx]
            y_test = X_test @ theta_true + eps[test_idx]

            data.append((X_train, y_train, theta_true.copy(), X_test, y_test))

        super().__init__(data)

    def _sample_theta_true(self, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def _sample_design(self, rng: np.random.Generator, n_total: int) -> np.ndarray:
        raise NotImplementedError

    def _sample_noise(self, rng: np.random.Generator, n_total: int) -> np.ndarray:
        raise NotImplementedError
