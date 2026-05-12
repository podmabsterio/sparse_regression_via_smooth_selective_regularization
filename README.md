# High-Dimensional Sparse Regression via Smooth Selective Regularization: Sharp Non-Asymptotic Guarantees

This repository contains the official implementation accompanying the paper:

> **High-Dimensional Sparse Regression via Smooth Selective Regularization: Sharp Non-Asymptotic Guarantees**

The repository provides:

* An implementation of the proposed smooth selective regularization penalty
* The optimization procedure introduced in the paper
* Code for all numerical experiments reported in the manuscript
* Scripts for evaluating empirical performance and validating theoretical corrections

All datasets used in the experiments are synthetic and are generated on the fly by the provided scripts. No external datasets are required.

---

## Installing dependencies

This project requires **Python 3.11**.

Install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The main baseline dependencies are `scikit-learn` for Lasso/OMP and `skglm` for SCAD/MCP.
Both are distributed under the BSD 3-Clause License.

---

## Repository Structure

The core implementation of the proposed method is located in:

```
src/models/main_model.py
```

This file contains the implementation of the regularization penalty and the corresponding optimization algorithm described in the paper.

---

## Training Models

To train the proposed model and all baseline methods used for comparison, run:

```
python run.py -cn=<CONFIG_NAME>
```

The training script:

* Instantiates the dataset
* Trains the proposed model and all competing methods
* Saves trained model weights

All trained model weights are stored in the directory:

```
saved/
```

---

## Experimental Configuration

We use **Hydra** for experiment configuration management.
Each dataset used in the paper has a dedicated configuration file.

### Available Configurations

#### 1. `gaus_iid_sigma05.yaml`

Uncorrelated Gaussian design with:

* $n = 125$
* $p = 1000$
* $|\alpha| = 10$
* $\sigma = 0.5$

#### 2. `gaus_iid_sigma1.yaml`

Uncorrelated Gaussian design with:

* $n = 125$
* $p = 1000$
* $|\alpha| = 10$
* $\sigma = 1$

#### 3. `toeplitz_corr.yaml`

Toeplitz-correlated Gaussian design with:

* $n = 125$
* $p = 1000$
* $|\alpha| = 10$
* $\sigma = 1$
* $\rho = 1$

#### 4. `block_corr.yaml`

Block-correlated Gaussian design with:

* $n = 250$
* $p = 1000$
* $|\alpha| = 10$
* $\sigma = 1$
* $\rho = 1$

---

## Evaluating Metrics

To compute evaluation metrics for a specific dataset, run:

```
python evaluate_metrics.py -cn=metrics_<CONFIG_NAME>
```

where `<CONFIG_NAME>` corresponds to one of the configuration names listed above.

All computed metrics are saved to:

```
saved_metrics/
```

---

## Theoretical Correction Validation

To verify the theoretical results presented in the paper, including the Fisher correction procedure, use:

```
python correction.py -cn=correction.yaml
```

This script:

* Applies the Fisher correction
* Saves model weights before and after correction

Results are stored in:

```
correction/
```

---

## Reproducibility

All experiments reported in the paper can be reproduced using the provided configuration files and scripts.
Hydra ensures consistent experiment management and reproducible configurations.

