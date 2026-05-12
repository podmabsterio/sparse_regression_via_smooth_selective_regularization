import numpy as np

def lambda_grid(X, y, eps=1e-3, num=100):
    lambda_max = np.linalg.norm(X.T @ y, ord=np.inf) / len(y)
    lambda_min = eps * lambda_max
    lambdas = np.logspace(
        np.log10(lambda_max),
        np.log10(lambda_min),
        num=num
    )

    return lambdas
