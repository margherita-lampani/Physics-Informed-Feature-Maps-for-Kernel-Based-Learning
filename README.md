# Physics-Informed-Feature-Maps-for-Kernel-Based-Learning
Physics‑informed approach to feature‑based ML that builds non‑linear feature maps from physical laws and dimensional analysis. These maps define physics‑aware kernels that embed domain knowledge in the RKHS geometry, improving interpretability and enabling mechanism discovery via feature ranking.

This repository contains the Python notebooks used to reproduce the synthetic and real-data experiments presented in the manuscript by Lampani et al. (2025):  
https://arxiv.org/abs/2504.17112

All experiments implement the physics-informed feature map framework described in the paper, covering both regression and classification tasks.

---------------------------------------------------------------------

## Repository Structure

| Notebook | Description | 
|---|---|
| `regression_bernoulli.ipynb` | Fluid dynamics regression (Section 3.1) | 
| `regression_pulsar.ipynb` | Pulsar magnetic dissipation regression (Section 3.2) | 
| `classification_binary.ipynb` | Gravitational binary system classification (Section 3.3) | 
| `classification_flares.ipynb` | Solar flare forecasting — requires data in `Data/` (Section 4) | 

---------------------------------------------------------------------

## Notebook Overview

### `regression_bernoulli.ipynb`
Reproduces the fluid dynamics regression experiment (Section 3.1). Starting from seven physical features — static pressure $p$, fluid density $\rho$, speed $v$, volumetric flow rate $Q$, cross-sectional area $A$, dynamic viscosity $\mu$, and height $h$ — the notebook constructs seven physics-informed features (PIFs) with the dimension of pressure, the first three of which correspond exactly to the terms of Bernoulli's equation. Ridge Regression and SVR with a linear kernel are trained on both standardized features (SFs) and standardized PIFs (SPIFs) under three noise levels (10%, 30%, 50%). The notebook then applies the sequential feature ranking algorithm to identify which PIFs drive prediction saturation and recovers the physical coefficients of Bernoulli's equation by de-standardizing the regression weights.

### `regression_pulsar.ipynb`
Reproduces the pulsar magnetic dissipation regression experiment (Section 3.2). The notebook generates synthetic data from the magnetic energy dissipation law, builds seven PIFs with the dimension of power, and runs the experiment twice: once with all SPIFs and once excluding PIF₁ — the feature corresponding to the correct physical equation — to test how performance degrades when the governing law is absent from the feature set. Results from both runs are compared against the standard SF baseline.

### `classification_binary.ipynb`
Reproduces the gravitational binary system classification experiment (Section 3.3). The task is to predict whether two celestial bodies are gravitationally bound, using the sign of the total mechanical energy. Two PIFs with the dimension of energy are constructed from the four input features. SVC with both linear and Gaussian kernels is applied to SFs and SPIFs across three class-imbalance configurations (50/50, 70/30, 90/10), and the results are evaluated through confusion matrices and skill scores (TSS, HSS, Sensitivity, Specificity, Accuracy).

### `classification_flares.ipynb` *(coming soon)*
Will reproduce the solar flare forecasting experiment (Section 4), the only experiment based on real data. Nine magnetogram-derived features are used to construct six PIFs with the dimension of magnetic energy, and SVC with linear and Gaussian kernels is trained to predict whether an active region will produce an M-class flare within the next 24 hours. A sequential PIF ranking procedure identifies PIF₂ = ΦI as the most predictive feature. The required data must be placed in the `Data/` folder before running the notebook.

---------------------------------------------------------------------

## Citation

If you use this code, please cite:

Lampani, M., Guastavino, S., Piana, M., Benvenuto, F. (2025).  
Physics-Informed Feature Maps for Kernel-Based Learning: A Reinterpretation of Symbolic Regression.

Preprint available at:  
https://arxiv.org/abs/2504.17112
