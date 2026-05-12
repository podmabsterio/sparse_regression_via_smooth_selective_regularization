from src.datasets.gaussianiid import GaussianIIDatasets

import numpy as np

class ToeplitzCorrelatedDatasets(GaussianIIDatasets):
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
        rho=0.5,
        seed: int = 42,
    ):
        self.rho = float(rho)

        super().__init__(
            n=n,
            p=p,
            s=s,
            noise_std=noise_std,
            signal_amp=signal_amp,
            num_datasets=num_datasets,
            test_fraction=test_fraction,
            seed=seed,
            center_data=center_data,
        )


    def _sample_design(self, rng: np.random.Generator, n_total: int) -> np.ndarray:
        idx = np.arange(self.p)
        Sigma = self.rho ** np.abs(idx[:, None] - idx[None, :])
        L = np.linalg.cholesky(Sigma)
        Z = rng.standard_normal(size=(n_total, self.p))
        X = Z @ L.T
        return X