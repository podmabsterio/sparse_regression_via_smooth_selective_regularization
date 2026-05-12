import numpy as np

from src.datasets.gaussianiid import GaussianIIDatasets

def build_covariance(p, g, rho):
    n_groups = p // g
    Sigma = np.zeros((p, p), dtype=float)

    for k in range(n_groups):
        start = k * g
        end = start + g
        block = np.full((g, g), rho, dtype=float)
        np.fill_diagonal(block, 1.0)
        Sigma[start:end, start:end] = block

    return Sigma
    

class BlockCorrDatasets(GaussianIIDatasets):
    def __init__(
        self,
        n: int,
        p: int,
        s: int,
        group_size,
        active_groups,
        rho,
        num_datasets: int,
        test_fraction: float = 2.0,
        center_data = True,
        noise_std: float = 1.0,
        signal_amp: float = 1.0,
        seed: int = 42,
    ):
        
        self.group_size = group_size
        self.rho = rho
        self.active_groups = active_groups
        
        self.n_groups = p // group_size

        self.Sigma = build_covariance(p, group_size, rho)
        self.L = np.linalg.cholesky(self.Sigma)
        
        super().__init__(
            n=n,
            p=p,
            s=s,
            num_datasets=num_datasets,
            test_fraction=test_fraction,
            noise_std = noise_std,
            signal_amp=signal_amp,
            seed=seed,
            center_data=center_data,
        )

    def _sample_design(self, rng: np.random.Generator, n_total: int) -> np.ndarray:
        Z = rng.standard_normal(size=(n_total, self.p))
        return Z @ self.L.T

    def _sample_theta_true(self, rng: np.random.Generator) -> np.ndarray:
        theta_true = np.zeros(self.p, dtype=float)
        if self.s <= 0:
            return theta_true

        per_group = self.s // self.active_groups

        chosen_groups = rng.choice(self.n_groups, size=self.active_groups, replace=False)

        support = []
        for g_idx in chosen_groups:
            start = g_idx * self.group_size
            end = start + self.group_size

            local = rng.choice(np.arange(start, end), size=per_group, replace=False)
            support.append(local)

        support = np.concatenate(support)

        signs = rng.choice([-1.0, 1.0], size=self.s)
        theta_true[support] = self.signal_amp * signs
        return theta_true
