# Federated Medical Image Classification with Physics-Informed Deep Learning

A modular research framework for federated learning on medical image datasets, featuring **grid-wise Perona-Malik PIDL regularization** and **Flower SecAgg+** secure aggregation.

---

## Overview

| Component | Choice |
|-----------|--------|
| Feature extractor | ResNet18 (pretrained on ImageNet) |
| Classification loss | Cross-entropy |
| Physics regularizer | Grid-wise Perona-Malik anisotropic diffusion (PIDL) |
| Federated framework | Flower (flwr ≥ 1.9) |
| Secure aggregation | Flower SecAgg+ |
| Differential privacy | None (by design) |
| Datasets | Chest X-Ray, HAM10000, Brain Tumor MRI, Retinal OCT |

---

## Project Structure

```
medical_fl_pidl/
├── configs/
│   ├── dataset_configs.py       # Per-dataset KaggleHub handles, class info
│   └── experiment_config.py     # Federated, model, training, PIDL hyperparams
├── data/
│   ├── kaggle_loader.py         # KaggleHub download + PyTorch Dataset wrappers
│   ├── dataset_utils.py         # Transforms, augmentation, DataLoader builders
│   └── partitioning.py          # IID and Dirichlet non-IID FL partitioning
├── models/
│   └── resnet_pidl.py           # ResNet18 with feature-map capture for PIDL
├── losses/
│   └── pidl_loss.py             # Perona-Malik grid-wise PIDL regularization loss
├── federated/
│   ├── task.py                  # Core train / evaluate functions
│   ├── client_app.py            # Flower ClientApp (NumPyClient)
│   ├── server_app.py            # Flower ServerApp with SecAgg+ workflow
│   └── strategy_logging.py      # FedAvg subclass with per-round metric logging
├── metrics/
│   ├── classification_metrics.py  # Accuracy, F1, AUC, per-class breakdown
│   └── calibration_metrics.py     # ECE, MCE, reliability diagrams
├── utils/
│   ├── logging_utils.py         # JSON / CSV metric logging helpers
│   ├── path_utils.py            # Cross-platform path resolution (Colab aware)
│   └── seed_utils.py            # Reproducibility seeding
├── notebooks/
│   ├── 01_clean_multidataset_experiments.ipynb   # Main FL experiment runner
│   ├── 02_result_analysis_and_plots.ipynb        # Metric plots & analysis
│   └── 03_robustness_experiments_optional.ipynb  # Optional robustness tests
├── results/                     # Auto-created, stores per-run JSON/CSV logs
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Quick Start

### Local (Python ≥ 3.9)

```bash
pip install -r requirements.txt

# Run a 3-client experiment on Chest X-Ray
python -m medical_fl_pidl.federated.server_app \
    --dataset chest_xray \
    --num_clients 3 \
    --num_rounds 20 \
    --lambda_pidl 0.01
```

### Google Colab

Open `notebooks/01_clean_multidataset_experiments.ipynb` in Colab.  
All install, auth, and run steps are contained in the first few cells.

---

## Perona-Malik PIDL Regularization

The physics-informed loss enforces the **Perona-Malik anisotropic diffusion** PDE as a soft constraint on intermediate ResNet feature maps.

**PDE (steady-state):**
```
div( c(|∇F|) · ∇F ) = 0
```

**Diffusivity (Lorentzian):**
```
c(s) = 1 / (1 + (s / K)²)
```

**Grid-wise PIDL loss:**
```
L_PIDL = (1 / B·C·H·W) · ‖ div( c(|∇F|) · ∇F ) ‖²_F
```

**Total loss:**
```
L_total = L_CE + λ · L_PIDL
```

Gradients and divergence are computed via **finite differences on the (H × W) spatial grid** of the selected ResNet layer (default: `layer3`, spatial size 14 × 14 for 224 × 224 input).

---

## Supported Datasets

| Key | Dataset | Classes | Source |
|-----|---------|---------|--------|
| `chest_xray` | Chest X-Ray Pneumonia | 2 | paultimothymooney/chest-xray-pneumonia |
| `ham10000` | HAM10000 Skin Lesion | 7 | kmader/skin-cancer-mnist-ham10000 |
| `brain_tumor` | Brain Tumor MRI | 4 | masoudnickparvar/brain-tumor-mri-dataset |
| `retinal_oct` | Retinal OCT | 4 | paultimothymooney/kermany2018 |

---

## Client Counts

The framework is designed to run with **3, 4, or 5 clients** with one config change:

```python
federated_config.num_clients = 4   # or 3, 5
```

SecAgg+ parameters auto-adjust to maintain 1-client dropout tolerance.

---

## Secure Aggregation

SecAgg+ is enabled by default in `server_app.py` via Flower's `SecAggPlusWorkflow`.  
No differential privacy noise is added.

---

## Reproducibility

Every run logs:
- Per-round train loss, CE loss, PIDL loss, accuracy
- Per-round server-side validation metrics
- Final test accuracy, F1 (macro), AUC, ECE
- Experiment config snapshot

Results are saved to `results/<experiment_name>_<timestamp>/`.
