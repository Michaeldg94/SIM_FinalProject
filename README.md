# Statistical Modelling of Galaxy Morphology

**Albert Lamb - Michael Duarte Gonçalves - Tina Sikharulidze**

Statistical Modelling & Inference — Final Project, December 2025

---

## Welcome

This repository contains our exploration of galaxy morphology classification using the [Galaxy Zoo](https://data.galaxyzoo.org/) dataset. We approach the problem from four complementary angles: regularized regression, generalized additive models, deep representation learning, and Bayesian clustering.

Each method brings something different to the table. The baseline models identify which astronomical features matter most. GAMs capture non-linear relationships. The VAE learns directly from images. And the Bayesian GMM discovers natural groupings without any labels. Together, they paint a fuller picture of what makes galaxies look the way they do.

---

## Data

Due to capacity constraints, we can not put the data in the `data/` folder. Download from [Galaxy Zoo](https://data.galaxyzoo.org/) and place in `data/`:

- `gz2sample.csv.gz`
- `zoo2MainSpecz.csv.gz`
- `galaxy_metadata.csv` (created by `03_VAE/` folder)
- `latent_representations.npy` (created by `03_VAE/` folder)

For the VAE pipeline, also grab images from [Zenodo](https://zenodo.org/records/3565489).

---

## Project Structure

```
SMI_FinalProject/
├── data/                        # Datasets (not tracked)
├── preprocess.py                # Shared preprocessing utilities
│
├── 01_Baseline/                 # Regularized Linear Models
├── 02_GAM/                      # Generalized Additive Models
├── 03_VAE/                      # Variational Autoencoder
└── 04_BMoG/                     # Bayesian Mixture of Gaussians
```

---

## Methods

### 01_Baseline — Regularized Regression

Predicts P(smooth) using tabular astronomical features. Compares OLS, Ridge, Lasso, and Adaptive Lasso with logit-transformed targets. Identifies the most important features through coefficient analysis across regularization strengths.

**Key results:** $R^2\approx 0.33$ are similar between different regularization methods, suggesting the existence of non-linear relationships.

---

### 02_GAM — Generalized Additive Models

Extends the baseline with spline-based transformations to capture non-linear feature effects. Uses `SplineTransformer` + `LinearGAM` from `pygam` after feature selection via Lasso/Ridge/Adaptive Lasso.

**Key results:** Best $R^2$ = 0.69 with Ridge (43 features, 5-knot splines). Provides smooth partial dependence plots showing how each feature influences morphology.

---

### 03_VAE — Variational Autoencoder

Learns 16-dimensional latent representations directly from 128×128 galaxy images. A convolutional encoder-decoder architecture trained with reconstruction + KL loss. No hand-crafted features required.

**Key results**: Clean separation between smooth and disk galaxies in latent space (visualized via t-SNE). Outputs feed directly into Stage 2 clustering (aka `04_BMoG/` folder).

---

### 04_BMoG — Bayesian Mixture of Gaussians

Clusters galaxies in the VAE latent space using Gibbs sampling with conjugate priors (Normal-Inverse-Wishart). Discovers natural groupings without using Galaxy Zoo labels during training. Labels are only used afterward to validate cluster meaning.

**Key results**: BIC selects $K=4$ clusters. Validation reveals that these clusters do not correspond to human morphological classifications (ARI = 0.003, NMI = 0.001). The VAE latent space appears to encode visual features distinct from the smooth vs. disk distinction used by Galaxy Zoo volunteers.
