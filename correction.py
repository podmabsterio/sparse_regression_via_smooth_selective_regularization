import warnings

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.metrics.results_array import ModelsExperimentResults

import src.hydra_utils.resolvers

from pathlib import Path
import pickle

from tqdm import tqdm

import matplotlib.pyplot as plt

from src.utils import init_saving, save_dict_array_list
from src.models import CVWrapper, MainModel

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

def pen_H_der2(x, lambda_, M):
    return M * (2 - 5 * lambda_ * x**2 + lambda_**2 * x**4) * np.exp(-lambda_ * x**2 * 0.5)

def get_fisher_matrix(X, sigma, theta_lam, Lambda, M):
    return (X.T @ X / sigma**2) + np.diag(pen_H_der2(theta_lam, Lambda, M * 2 * sigma**2))

def get_score_vector(X, sigma, theta_true, y):
    eps = y - X @ theta_true
    return X.T @ eps / sigma**2


def stable_pinv_via_eig(fisher, score_vector, eig_min=1e-3, eig_max=1e3):
    eigvals, eigvecs = np.linalg.eigh(fisher)
    
    np.clip(eigvals, eig_min, eig_max, out=eigvals)
    inv_eigvals = 1.0 / eigvals

    return eigvecs @ (inv_eigvals * (eigvecs.T @ score_vector))

def get_linear_approximation(X, y, sigma, theta_lam, theta_true, Lambda, M):
    fisher = get_fisher_matrix(X, sigma, theta_lam, Lambda, M)
    score_vector = get_score_vector(X, sigma, theta_true, y)
    return stable_pinv_via_eig(fisher, score_vector)


@hydra.main(version_base=None, config_path="src/configs", config_name="correction")
def main(config):
    datasets = instantiate(config.datasets)
    
    results = ModelsExperimentResults(datasets, path_to_coefs=Path(config.save_dir) / config.run_name)
    model_results = results.get_results('MainModel')
    model_noiseless_results = results.get_results('MainModelNoiseless')
    
    M, Lambda = config.M, config.Lambda
    sigma = datasets.noise_std
    
    corrected_arr = []
    not_corrected_arr = []
    for model, model_n in tqdm(zip(model_results, model_noiseless_results), total=len(model_results)):
        tilde_theta = model['coef']
        theta_lam = model_n['coef']
        X, y, theta_true = model['X'], model['y'], model['theta_true']
        linear_approx = get_linear_approximation(X, y, sigma, theta_lam, theta_true, Lambda, M)
        
        corrected = np.linalg.norm(tilde_theta - theta_true)
        not_corrected = np.linalg.norm(tilde_theta - theta_true + linear_approx)
        
        corrected_arr.append(corrected)
        not_corrected_arr.append(not_corrected)
        
    corrected_path = f"correction/corrected_lam{int(Lambda)}"
    not_corrected_path = f"correction/not_corrected_lam{int(Lambda)}"
    
    with open(corrected_path, "wb") as f:
        pickle.dump(corrected_arr, f)
        
    with open(not_corrected_path, "wb") as f:
        pickle.dump(not_corrected_arr, f)
    
if __name__ == "__main__":
    main()