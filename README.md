# Federated Medical Image Classification with Physics-Informed Deep Learning

A modular research framework for federated learning on medical image datasets, combining **grid-wise Perona-Malik PIDL regularization**, **Flower SecAgg+ secure aggregation**, and **robustness experiments** (data-poisoning attacks + update-clipping defense).

---

## Overview

| Component | Choice |
|---|---|
| Feature extractor | ResNet18 (ImageNet pretrained) |
| Classification loss | Cross-entropy |
| Physics regularizer | Grid-wise Perona-Malik anisotropic diffusion (PIDL) |
| Federated framework | Flower 1.29 (`flwr[simulation]`) |
| Secure aggregation | SecAgg+ (simulated overhead; disabled in `run_simulation`) |
| Differential privacy | None (by design) |
| Datasets | Brain Tumor MRI · Colon Cancer · COVID-19 Radiography |
| Client counts | 3, 4, 5 |
| Data partitioning | Stratified IID (80 % train / 20 % test global split) |

---

## Project Structure

```
medical_fl_pidl/
├── configs/
│   ├── dataset_configs.py        Dataset registry + DatasetConfig dataclass
│   └── experiment_config.py      Hyperparameter config (ModelConfig, etc.)
├── data/
│   ├── kaggle_loader.py          KaggleHub download + find_image_root()
│   ├── dataset_utils.py          build_federated_dataloaders() — main entry point
│   └── partitioning.py           Stratified IID partitioning
├── models/
│   └── resnet_pidl.py            ResNet18 with intermediate feature-map exposure
├── losses/
│   └── pidl_loss.py              Perona-Malik grid-wise PIDL loss
├── federated/
│   ├── task.py                   Stateless train() / evaluate() / evaluate_full()
│   ├── client_app.py             Flower ClientApp — attack injection + update clipping
│   ├── server_app.py             Flower ServerApp — FedAvg + comprehensive logging
│   └── strategy_logging.py       LoggingFedAvg — per-round JSONL + SecAgg timing sim
├── metrics/
│   ├── classification_metrics.py Accuracy, F1, AUC, sensitivity, specificity, per-class
│   └── calibration_metrics.py    ECE, mean confidence, mean entropy
├── robustness/
│   ├── robustness_config.py      RobustnessConfig dataclass
│   ├── attacks.py                GaussianNoiseDataset, LabelFlipDataset
│   ├── defenses.py               clip_model_update(), compute_update_norm()
│   └── __init__.py
├── utils/
│   ├── logging_utils.py          ExperimentLogger — CSV/JSON structured output
│   ├── path_utils.py             Cross-platform path helpers
│   └── seed_utils.py             Reproducibility seeding
├── notebooks/
│   ├── 01_clean_multidataset_experiments.ipynb   Main FL runner (GitHub + Colab)
│   ├── 02_result_analysis_and_plots.ipynb        Analysis, plots, summary tables
│   ├── 03_robustness_experiments_optional.ipynb  Attack/defense experiments
│   └── 04_ablation_study.ipynb                  PIDL type / grid size / lambda ablation
├── results/                      Per-run CSV/JSON output (main experiments)
├── results_ablation/             Per-run output (ablation study)
├── results_robustness/           Per-run output (robustness experiments)
├── requirements.txt
├── pyproject.toml
├── context.md
└── README.md
```

---

## Workflow

### Step 1 — Run main experiments (Colab)

1. Push the repo to GitHub.
2. Open `notebooks/01_clean_multidataset_experiments.ipynb` in Colab.
3. Set `GITHUB_REPO` to your repo URL, then run all cells.
4. Results are saved to `results/{dataset}/{n}_clients/`.
5. Download and commit the `results/` folder.

### Step 2 — Analyse results (Colab or local)

Open `notebooks/02_result_analysis_and_plots.ipynb`.  
The notebook auto-generates `fl_summary.json` from `fl_rounds.csv` if missing,
then produces all plots and tables.

### Step 3 — Robustness experiments (optional)

Open `notebooks/03_robustness_experiments_optional.ipynb`.  
Results are saved separately to `results_robustness/`.

### Step 4 — Ablation study (optional)

Open `notebooks/04_ablation_study.ipynb`.  
Runs 18 new experiments (6 variants × 3 datasets, 3 clients) and loads the existing
3-client baseline from `results/` without re-running it.  
Results are saved to `results_ablation/`.

---

## Data Pipeline

```
Full dataset (100 %)
│
├── Global test set  (20 %)  ──► server evaluation every round
│
└── Training pool   (80 %)
    ├── Client 0  (~26–27 %)  ──► local train  fit()
    ├── Client 1  (~26–27 %)  ──► local train  fit()
    └── Client 2  (~26–27 %)  ──► local train  fit()
         (clients use global test set for their evaluate() call too)
```

Partitioning is **stratified IID** — every client receives roughly equal samples
from every class, controlled by `random_seed=42`.

---

## Perona-Malik PIDL Regularization

The physics-informed regularizer enforces the **Perona-Malik anisotropic diffusion PDE** as a soft constraint on intermediate ResNet feature maps.

**PDE (steady-state):**
```
div( c(|∇F|) · ∇F ) = 0
```

**Lorentzian diffusivity:**
```
c(s) = 1 / (1 + (s/K)²)
```

**Grid-wise PIDL loss:**  
The feature map `(B, C, H, W)` is divided into `grid_size × grid_size` non-overlapping local patches. The PM residual is computed independently per patch and averaged. This is intentional — pathology regions are *local*, not global.

**Total loss:**
```
L_total = L_CE + λ · L_PIDL
```

Default: `feature_layer = layer2`, `grid_size = 4`, `λ = 0.1`, `K = 1.0`.

---

## Datasets

| Key | Dataset | Classes | Kaggle Slug |
|---|---|---|---|
| `brain_tumor_mri` | Brain Tumor MRI | 4 (glioma, meningioma, notumor, pituitary) | masoudnickparvar/brain-tumor-mri-dataset |
| `colon_cancer_or_pathology` | Lung & Colon Cancer Histopathology | 5 | andrewmvd/lung-and-colon-cancer-histopathological-images |
| `covid` | COVID-19 Radiography | 4 (COVID, Lung Opacity, Normal, Viral Pneumonia) | tawsifurrahman/covid19-radiography-database |

---

## Experimental Results

All results: 5 FL rounds · 2 local epochs · ResNet18 · layer2 PIDL · SecAgg+ overhead simulated (~0.65–0.68 s/round).  
`*` = best in group per dataset.

---

### Main experiments — final accuracy (grid-wise PIDL 4×4, λ=0.10)

| Dataset | 3 clients | 4 clients | 5 clients |
|---|---|---|---|
| Brain Tumor MRI | 95.27 % | 94.37 % | 95.00 % |
| Colon Cancer | 99.95 % | 99.75 % | 99.90 % |
| COVID-19 | 91.52 % | 90.29 % | 90.20 % |

---

### Ablation Group 1 — PIDL Regulariser Type (3 clients, λ=0.10, grid 4×4)

| Dataset | Method | Acc % | Bal.Acc % | F1-Mac % | Prec % | Recall % | ROC-AUC % | ECE % | Train (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Brain Tumor MRI | No PIDL | **95.54** \* | **95.54** | **95.56** | **95.64** | **95.54** | 99.59 | **1.49** | 361 |
| Brain Tumor MRI | Global PIDL | 94.46 | 94.46 | 94.50 | 94.78 | 94.46 | 99.54 | 1.19 | 349 |
| Brain Tumor MRI | Grid-wise 4×4 | 95.27 | 95.27 | 95.24 | 95.30 | 95.27 | **99.64** | 2.69 | 462 |
| Colon Cancer | No PIDL | 99.25 | 99.25 | 99.25 | 99.26 | 99.25 | 100.00 | 0.61 | 1132 |
| Colon Cancer | Global PIDL | 99.80 | 99.80 | 99.80 | 99.80 | 99.80 | 100.00 | 0.32 | 1098 |
| Colon Cancer | Grid-wise 4×4 | **99.95** \* | **99.95** | **99.95** | **99.95** | **99.95** | 100.00 | **0.23** | 1347 |
| COVID-19 | No PIDL | 88.59 | 91.81 | 88.26 | 85.90 | 91.81 | 98.39 | 3.47 | 1252 |
| COVID-19 | Global PIDL | 90.88 | 92.51 | 91.27 | 90.46 | 92.51 | 98.80 | 1.80 | 1247 |
| COVID-19 | Grid-wise 4×4 | **91.52** \* | **92.42** | **91.84** | **91.43** | **92.42** | 98.75 | 2.95 | 1690 |

---

### Ablation Group 2 — Grid Size (3 clients, λ=0.10, Perona-Malik)

| Dataset | Grid | Acc % | Bal.Acc % | F1-Mac % | Prec % | Recall % | ROC-AUC % | ECE % | Train (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Brain Tumor MRI | 2×2 | 94.46 | 94.46 | 94.43 | 94.74 | 94.46 | 99.60 | 2.30 | 363 |
| Brain Tumor MRI | 4×4 (baseline) | **95.27** \* | **95.27** | 95.24 | 95.30 | **95.27** | **99.64** | 2.69 | 462 |
| Brain Tumor MRI | 8×8 | **95.27** \* | **95.27** | **95.29** | **95.39** | **95.27** | 99.62 | **1.19** | 352 |
| Colon Cancer | 2×2 | **100.00** \* | **100.00** | **100.00** | **100.00** | **100.00** | 100.00 | 0.18 | 1106 |
| Colon Cancer | 4×4 (baseline) | 99.95 | 99.95 | 99.95 | 99.95 | 99.95 | 100.00 | 0.23 | 1347 |
| Colon Cancer | 8×8 | 99.95 | 99.95 | 99.95 | 99.95 | 99.95 | 100.00 | **0.09** | 1087 |
| COVID-19 | 2×2 | **92.13** \* | **93.31** | **92.08** | 91.02 | **93.31** | 98.71 | 1.89 | 1252 |
| COVID-19 | 4×4 (baseline) | 91.52 | 92.42 | 91.84 | 91.43 | 92.42 | 98.75 | 2.95 | 1690 |
| COVID-19 | 8×8 | 90.53 | 89.99 | 91.18 | **93.05** | 89.99 | 98.73 | **1.36** | 1246 |

---

### Ablation Group 3 — Lambda Weight λ (3 clients, grid 4×4, Perona-Malik)

| Dataset | λ | Acc % | Bal.Acc % | F1-Mac % | Prec % | Recall % | ROC-AUC % | ECE % | Train (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Brain Tumor MRI | 0.01 | 95.98 | 95.98 | 95.99 | 96.00 | 95.98 | 99.67 | 3.03 | 363 |
| Brain Tumor MRI | 0.10 (baseline) | 95.27 | 95.27 | 95.24 | 95.30 | 95.27 | 99.64 | 2.69 | 462 |
| Brain Tumor MRI | 0.50 | **96.52** \* | **96.52** | **96.51** | **96.51** | **96.52** | **99.70** | 2.55 | 365 |
| Colon Cancer | 0.01 | 99.85 | 99.85 | 99.85 | 99.85 | 99.85 | 100.00 | 0.22 | 1116 |
| Colon Cancer | 0.10 (baseline) | 99.95 | 99.95 | 99.95 | 99.95 | 99.95 | 100.00 | 0.23 | 1347 |
| Colon Cancer | 0.50 | **100.00** \* | **100.00** | **100.00** | **100.00** | **100.00** | 100.00 | **0.08** | 1115 |
| COVID-19 | 0.01 | **92.42** \* | **93.08** | **92.58** | 92.17 | **93.08** | **98.77** | 1.62 | 1237 |
| COVID-19 | 0.10 (baseline) | 91.52 | 92.42 | 91.84 | 91.43 | 92.42 | 98.75 | 2.95 | 1690 |
| COVID-19 | 0.50 | 92.27 | 92.67 | **92.76** | **92.94** | 92.67 | 98.67 | **1.57** | 1230 |

---

### Robustness experiments — Brain Tumor MRI, 3 clients, 1 malicious client

| Experiment | Final Acc | Macro F1 | ECE |
|---|---|---|---|
| Clean baseline | 95.45 % | 95.43 % | 1.03 % |
| Gaussian noise attack | 95.18 % | 95.18 % | 1.58 % |
| Label-flip attack | 91.96 % | 91.93 % | 10.98 % |
| Noise + update clipping | 34.02 % | 25.67 % | 13.08 % |
| Label-flip + update clipping | 37.77 % | 34.59 % | 5.82 % |

> Update clipping (clip_norm=3.0) dramatically reduces accuracy when combined with attacks,  
> indicating the defense is overly aggressive at this norm threshold for this model size.

---

## Secure Aggregation

`SecAggPlusWorkflow` is **conditionally enabled** via `use_secagg=true` in config.  
In `run_simulation` (Flower 1.29) SecAgg+ is incompatible with the simulation
backend, so it is disabled for local runs.  A **timing simulation** in
`strategy_logging.py` measures the wall-clock cost of SecAgg+ mask generation
and summation on actual model parameters, providing an honest overhead estimate.

---

## Configuration

All hyperparameters are set via `FL_RUN_OVERRIDE` (JSON env var) in notebook 01,
which overrides defaults in `pyproject.toml`.  Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `num_server_rounds` | 5 | FL communication rounds |
| `local_epochs` | 2 | Local training epochs per round |
| `feature_layer` | `layer2` | ResNet layer for PIDL regularization |
| `lambda_pm` | 0.1 | PIDL regularization weight |
| `grid_size` | 4 | Patch grid for grid-wise PIDL |
| `use_secagg` | false | Enable SecAgg+ (CLI deployment only) |
| `enable_attack` | false | Enable data-poisoning attack |
| `attack_type` | `gaussian_noise` | `gaussian_noise` or `label_flip` |
| `enable_update_clipping` | false | Enable update-norm clipping defense |
| `clip_norm` | 3.0 | Max L2 norm of client update |

---

## Output Files

Each experiment folder (`results/{dataset}/{n}_clients/`) contains:

| File | Contents |
|---|---|
| `config.json` | Full hyperparameter snapshot |
| `dataset_summary.json` | Class counts, client partition sizes |
| `fl_rounds.csv` | Per-round global metrics |
| `fl_clients.csv` | Per-round per-client metrics |
| `round_metrics.jsonl` | Raw per-round strategy history (incl. secagg timing) |
| `per_class_metrics.csv` | Per-class precision, recall, F1 |
| `fl_summary.json` | Best/final metrics summary |

---

## Reproducibility

All runs use `random_seed=42` for dataset splits, client partitioning, model
initialisation, and data augmentation. Re-running with the same seed produces
identical results.
