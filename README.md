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
│   └── 03_robustness_experiments_optional.ipynb  Attack/defense experiments
├── results/                      Per-run CSV/JSON output (main experiments)
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

### Main experiments — final accuracy (5 FL rounds, 2 local epochs)

| Dataset | 3 clients | 4 clients | 5 clients |
|---|---|---|---|
| Brain Tumor MRI | 0.9527 | 0.9437 | 0.9500 |
| Colon Cancer | 0.9995 | 0.9975 | 0.9990 |
| COVID-19 | 0.9152 | 0.9029 | 0.9020 |

### Robustness experiments — Brain Tumor MRI, 3 clients, 1 malicious client

| Experiment | Final Acc | Macro F1 | ECE |
|---|---|---|---|
| Clean baseline | 0.9545 | 0.9543 | 0.0103 |
| Gaussian noise attack | 0.9518 | 0.9518 | 0.0158 |
| Label-flip attack | 0.9196 | 0.9193 | 0.1098 |
| Noise + update clipping | 0.3402 | 0.2567 | 0.1308 |
| Label-flip + update clipping | 0.3777 | 0.3459 | 0.0582 |

SecAgg+ simulated aggregation overhead: **~0.65–0.83 s per round**.

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
