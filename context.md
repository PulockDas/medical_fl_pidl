# Project Context — `medical_fl_pidl`

> **Purpose of this file:** Full session context for AI assistants in future conversations.
> Read this before touching any file in the project.

---

## 1. Project Goal

Federated learning framework for medical image classification with:

| Component | Choice |
|-----------|--------|
| Feature extractor | ResNet18 (ImageNet pretrained) |
| Classification loss | Cross-entropy |
| Physics regularizer | **Grid-wise Perona-Malik PIDL** on intermediate feature maps |
| FL framework | **Flower ≥ 1.9** (`flwr[simulation]`) |
| Secure aggregation | **Flower SecAgg+** (`SecAggPlusWorkflow`) |
| Differential privacy | **None** (by design) |
| Datasets | brain_tumor_mri, colon_cancer_pathology, covid (KaggleHub) |
| Client counts | 3, 4, or 5 |
| Partitioning | Stratified IID (default) or Dirichlet non-IID |

---

## 2. Complete File Tree

```
medical_fl_pidl/                            ← project root
├── configs/
│   ├── dataset_configs.py      532 lines   ← Dataset registry + DatasetConfig class
│   └── experiment_config.py    221 lines   ← Full hyperparameter config hierarchy
├── data/
│   ├── kaggle_loader.py        696 lines   ← Download + find_image_root + Dataset classes
│   ├── dataset_utils.py        586 lines   ← build_federated_dataloaders() (PRIMARY ENTRY)
│   └── partitioning.py         291 lines   ← Stratified IID + Dirichlet partitioning
├── models/
│   └── resnet_pidl.py          260 lines   ← ResNet18 with feature-map capture
├── losses/
│   └── pidl_loss.py            309 lines   ← Perona-Malik PIDL loss (core physics)
├── federated/
│   ├── task.py                 262 lines   ← Stateless train() / evaluate() functions
│   ├── client_app.py           236 lines   ← Flower ClientApp (NumPyClient)
│   ├── server_app.py           200 lines   ← Flower ServerApp + SecAgg+ workflow
│   └── strategy_logging.py     189 lines   ← FedAvg subclass with per-round JSONL logging
├── metrics/
│   ├── classification_metrics.py  162 lines  ← Accuracy, F1, AUC, per-class breakdown
│   └── calibration_metrics.py     240 lines  ← ECE, MCE, Brier, reliability diagram
├── utils/
│   ├── logging_utils.py        229 lines   ← JSONL/CSV writers, config snapshot, summary print
│   ├── path_utils.py            88 lines   ← Colab-aware project root resolution
│   └── seed_utils.py            37 lines   ← set_all_seeds() for full reproducibility
├── notebooks/
│   ├── 01_clean_multidataset_experiments.ipynb   ← Main FL experiment runner (Colab-ready)
│   ├── 02_result_analysis_and_plots.ipynb        ← Curves, calibration, multi-dataset compare
│   └── 03_robustness_experiments_optional.ipynb  ← Dirichlet sweep, layer ablation, n-client
├── results/                    ← Auto-created; one subdir per run
├── requirements.txt
├── pyproject.toml
├── README.md
└── context.md                  ← THIS FILE
```

---

## 3. The Physics — Perona-Malik PIDL

### PDE (Perona & Malik, 1990)
```
∂u/∂t = div( c(|∇u|) · ∇u )
```

**Diffusivity functions** (edge-stopping, K = threshold):
```
Lorentzian:  c(s) = 1 / (1 + (s/K)²)    [default]
Gaussian:    c(s) = exp( -(s/K)² )
```

### PIDL application (steady-state residual)
At steady state `∂u/∂t = 0`, so the residual is:
```
R(F) = div( c(|∇F|) · ∇F )
```

Applied to intermediate ResNet feature maps `F ∈ ℝ^(B×C×H×W)`:
```
L_PIDL = (1 / B·C·H·W) · ‖R(F)‖²_F
```

**Total training loss:**
```
L = L_CE  +  λ · L_PIDL
```

### Numerical scheme (grid-wise finite differences)
- **Gradients:** forward differences on the (H×W) spatial grid
- **Divergence:** backward differences on the flux field (adjoint pair → operator is negative semi-definite)

### Implementation: `losses/pidl_loss.py`

Key functions:
```python
perona_malik_residual(feature_map, K, diffusivity_type, normalize) -> Tensor  # core physics
PIDLLoss(K, diffusivity_type, normalize, reduction)                            # nn.Module
CEWithPIDLLoss(lambda_pidl, K, diffusivity_type, normalize)                    # composite
```

`CEWithPIDLLoss.forward(logits, labels, feature_map)` returns `(total, ce_loss, pidl_loss)` — all three scalars for logging.

---

## 4. Dataset Configuration System

### `configs/dataset_configs.py`

**The only place to change Kaggle slugs:**
```python
# ── EDIT THESE ─────────────────────────────────────────────────
SLUG_BRAIN_TUMOR  = "masoudnickparvar/brain-tumor-mri-dataset"
SLUG_COLON_CANCER = "andrewmvd/lung-and-colon-cancer-histopathological-images"
SLUG_COVID        = "tawsifurrahman/covid19-radiography-database"
# ───────────────────────────────────────────────────────────────
```

**Three registered datasets:**

| Key | Object | `num_classes` | `class_names` | `has_presplit_dirs` |
|-----|--------|---------------|---------------|---------------------|
| `"brain_tumor_mri"` | `BRAIN_TUMOR` | `4` (known) | `[glioma, meningioma, notumor, pituitary]` | `True` |
| `"colon_cancer_pathology"` | `COLON_CANCER` | `None` (auto) | `None` (auto) | `False` |
| `"covid"` | `COVID` | `None` (auto) | `None` (auto) | `False` |

`None` means auto-detected from folder structure after download.

**`DatasetConfig` key methods:**
```python
cfg = get_dataset_config("covid")

cfg.set_data_root(path, auto_detect=True)  # fills class_names + num_classes
cfg.auto_detect_classes()                  # explicit scan
n, names = cfg.resolve_classes()           # safe getter; auto-detects if needed
cfg.is_ready()                             # True when root + classes are known

list_available_datasets(verbose=False)     # print formatted table
```

**Backward-compatible properties** (used silently by kaggle_loader and experiment_config):
```python
cfg.name           # → cfg.dataset_name
cfg.kaggle_handle  # → cfg.kaggle_slug
cfg.is_folder_based  # → expected_structure == "imagefolder"
cfg.image_subdir   # → None (always)
cfg.csv_filename   # → None (always)
```

---

## 5. Data Pipeline

### Step-by-step flow

```
1. download_kaggle_dataset(slug, name)  →  raw_path (str)
        ↓
2. find_image_root(raw_path)            →  img_root (str)
   [4 detection strategies: train splits → root → depth-1 → depth-2]
        ↓
3. build_federated_dataloaders(
       image_root = img_root,
       num_clients = 3,
       test_split  = 0.20,
       batch_size  = 32,
       image_size  = (224, 224),
       augment     = True,
       random_seed = 42,
       save_summary_to = "results/run_01",
   )
   → {
       "client_train_loaders": list[DataLoader],   # one per client
       "global_test_loader":   DataLoader,
       "num_classes":          int,
       "class_names":          list[str],
       "dataset_summary":      dict,               # also → dataset_summary.json
     }
```

### `dataset_utils.py` — internal steps
1. `ImageFolder(root, transform=None)` — auto-detects classes from folder names
2. `_is_grayscale_dataset()` — samples 5 images; inserts `Grayscale(3)` if needed
3. `sklearn.train_test_split(stratify=labels)` — stratified global train/test split
4. `partition_indices(strategy="iid")` — stratified IID distribution across clients
5. `Subset → _TransformDataset(train_tf)` per client → `DataLoader(shuffle=True)`
6. `Subset → _TransformDataset(test_tf)` → `DataLoader(shuffle=False)`
7. `_build_summary()` → optionally `_save_summary()` as `dataset_summary.json`

### `partitioning.py` — strategies
```python
# Default — each client mirrors global class distribution
stratified_iid_partition(train_indices, all_targets, num_clients, seed)

# Optional — non-IID heterogeneity (lower alpha = more skewed)
dirichlet_partition(train_indices, all_targets, num_clients, alpha, seed)

# Dispatcher
partition_indices(train_indices, all_targets, num_clients,
                  strategy="iid"|"dirichlet", dirichlet_alpha=0.5, seed)

# Diagnostics
partition_stats(client_index_lists, all_targets, class_names)
```

All functions operate on NumPy index arrays — no PyTorch dependency.

### `kaggle_loader.py` — additional functions
```python
download_kaggle_dataset(slug, name, force_download=False) -> str
find_image_root(downloaded_path: str) -> str
preview_dataset_structure(root_path: str, max_depth: int = 3) -> None

# Internal builders (used by build_full_dataset_splits, kept for old pipeline)
FolderMedicalDataset    # samples: list[(Path, int)], scan_directory() classmethod
CSVMedicalDataset       # for CSV-based datasets (HAM10000 pattern)
build_full_dataset_splits(config, root, train_tf, val_tf, seed) -> dict[str, Dataset]
download_dataset(config, cache_dir)  # backward-compat alias
```

---

## 6. Model — `models/resnet_pidl.py`

```python
class ResNetPIDL(nn.Module):
    # Backbone decomposed into named blocks (no hooks)
    conv1, bn1, relu, maxpool
    layer1  # (B,  64, 56, 56)
    layer2  # (B, 128, 28, 28)
    layer3  # (B, 256, 14, 14)  ← default PIDL layer
    layer4  # (B, 512,  7,  7)
    avgpool
    classifier  # Dropout → Linear(512, num_classes)

    def forward(x) -> (logits, feature_map)
    # feature_map is from config.pidl_feature_layer ("layer2"|"layer3"|"layer4")
```

**Factory functions:**
```python
model = build_model(num_classes, config)        # ModelConfig → ResNetPIDL
params = get_model_parameters(model)            # → list[np.ndarray] for Flower
set_model_parameters(model, parameters)         # load Flower params in-place
```

**`ModelConfig` fields:** `backbone`, `pretrained`, `freeze_backbone`, `dropout_rate`, `pidl_feature_layer`

---

## 7. Federated Learning Layer

### `federated/task.py` — stateless functions (shared by client + notebooks)
```python
build_optimizer(model, config: TrainingConfig)  # → Adam / AdamW / SGD

train(model, dataloader, criterion, optimizer, device, num_epochs, scheduler)
# returns {"train_loss", "train_ce_loss", "train_pidl_loss", "train_accuracy"}

evaluate(model, dataloader, criterion, device)
# returns {"val_loss", "val_accuracy", "num_samples"}

evaluate_full(model, dataloader, device, num_classes)
# returns {"loss", "accuracy", "all_logits", "all_labels", "all_probs", "num_samples"}

resolve_device("auto"|"cpu"|"cuda"|"mps") -> torch.device
```

### `federated/client_app.py`
```python
class MedicalFLClient(NumPyClient):
    def fit(parameters, config)      # load global → train → return updated params + metrics
    def evaluate(parameters, config) # load global → eval on local val set

def make_client_fn(config, all_partitions, val_dataset, device)
# Returns the client_fn closure required by Flower simulation
# Uses _TransformedSubset to apply transforms without modifying shared datasets
```

### `federated/server_app.py`
```python
server_app, strategy = build_server_app(config, log_dir)
# Builds LoggingFedAvg strategy + SecAgg+ workflow (or DefaultWorkflow fallback)
# SecAgg+ requires flwr >= 1.9 with SecAggPlusWorkflow in flwr.server.workflow

run_simulation(config, client_fn, server_app, backend_config=None)
# Launches flwr.simulation.run_simulation()
```

**SecAgg+ parameters auto-set by `ExperimentConfig.finalize()`:**
- `secagg_num_shares` = `num_clients`
- `secagg_reconstruction_threshold` = `num_clients - 1` (1-client dropout tolerance)

### `federated/strategy_logging.py`
```python
class LoggingFedAvg(FedAvg):
    # Overrides aggregate_fit + aggregate_evaluate
    # Writes one JSONL record per round to round_metrics.jsonl
    # Prints: "Round  3 | Train Acc: 72.14%  Loss: 0.8123  PIDL: 0.000421 | ..."

    get_history() -> list[dict]   # full per-round record list
    close()                       # flush + close log file
```

---

## 8. Configuration System

### Hierarchy
```
ExperimentConfig
├── dataset:      DatasetConfig      (from configs/dataset_configs.py)
├── federated:    FederatedConfig    (num_clients, num_rounds, SecAgg+ params)
├── model:        ModelConfig        (backbone, dropout, pidl_feature_layer)
├── training:     TrainingConfig     (lr, batch_size, local_epochs, optimizer)
├── pidl:         PIDLConfig         (lambda_pidl, diffusivity_k, diffusivity_type)
├── partitioning: PartitioningConfig (strategy, dirichlet_alpha)
└── logging:      LoggingConfig      (log_format, run_name)
```

### Quickest way to create a config
```python
from configs.experiment_config import make_config

cfg = make_config(
    dataset_key   = "brain_tumor_mri",  # or "colon_cancer_pathology" / "covid"
    num_clients   = 3,                  # 3, 4, or 5
    num_rounds    = 20,
    lambda_pidl   = 0.01,               # 0.0 to disable PIDL
    use_secagg_plus = True,
    local_epochs  = 3,
    partitioning  = "iid",              # or "dirichlet"
    seed          = 42,
)
# cfg.finalize() is called automatically by make_config()
```

### Important: `finalize()` must be called after changing `num_clients`
```python
cfg.federated.num_clients = 5
cfg.finalize()  # updates SecAgg+ shares, min_fit_clients, etc.
```

---

## 9. Metrics

### `metrics/classification_metrics.py`
```python
compute_classification_metrics(probs, labels, class_names, prefix="")
# Returns: accuracy, f1_macro, f1_weighted, precision_macro, recall_macro,
#          auc_macro (OvR), per-class accuracy (acc_class_<name>)

compute_confusion_matrix(probs, labels, num_classes) -> np.ndarray

aggregate_round_metrics(history, keys=None) -> dict[str, list[float]]
# Converts list of per-round dicts → dict of metric curves for plotting
```

### `metrics/calibration_metrics.py`
```python
compute_ece(probs, labels, n_bins=15) -> float     # Expected Calibration Error
compute_mce(probs, labels, n_bins=15) -> float     # Maximum Calibration Error
compute_brier_score(probs, labels) -> float

compute_calibration_metrics(probs, labels, n_bins=15, prefix="")
# Returns: {prefix+"ece", prefix+"mce", prefix+"brier_score"}

reliability_diagram_data(probs, labels, n_bins=15)
# Returns: {"bin_acc", "bin_conf", "bin_frac", "bin_edges"} for plotting
```

---

## 10. Utilities

### `utils/path_utils.py`
```python
get_project_root() -> Path   # checks MEDICAL_FL_ROOT env → Colab Drive → local
get_results_dir(run_name=None) -> Path   # creates if missing
get_kaggle_cache_dir() -> Path | None    # /content/kaggle_cache in Colab, else None
is_colab() -> bool
```

### `utils/logging_utils.py`
```python
make_run_dir(results_dir, experiment_name, run_name=None) -> Path
save_config_snapshot(config_dict, run_dir) -> Path    # writes config.json
save_metrics(metrics, run_dir, format="both")         # "json" | "csv" | "both"
load_metrics_jsonl(path) -> list[dict]
load_metrics_csv(path) -> list[dict]
print_experiment_summary(final_metrics, config_dict)  # formatted table
```

### `utils/seed_utils.py`
```python
set_all_seeds(seed: int)   # Python, NumPy, torch.manual_seed, CUDA, cudnn deterministic
# Each client should use seed + client_id to differ while remaining reproducible
```

---

## 11. Key Import Graph (no circular dependencies)

```
configs/dataset_configs.py   ← (no local imports)
configs/experiment_config.py ← configs/dataset_configs
utils/*                      ← (no local imports)
losses/pidl_loss.py          ← (no local imports)
models/resnet_pidl.py        ← configs/experiment_config
data/partitioning.py         ← (no local imports)
data/kaggle_loader.py        ← configs/dataset_configs
data/dataset_utils.py        ← data/partitioning
metrics/*                    ← (no local imports)
federated/task.py            ← configs/experiment_config, losses, models
federated/strategy_logging.py← (flwr only)
federated/client_app.py      ← configs, data, federated/task, losses, models, utils
federated/server_app.py      ← configs, federated/strategy_logging, models
```

**Critical:** `configs/dataset_configs.py` must never import from `data/` (would create a circular dependency with `kaggle_loader.py` which imports `DatasetConfig`). The `_scan_for_classes()` function in `dataset_configs.py` implements its own minimal folder scanner for this reason.

---

## 12. Notebook Responsibilities

| Notebook | Purpose |
|----------|---------|
| `01_clean_multidataset_experiments.ipynb` | Main runner: install, auth, config, download, partition, simulate, evaluate, save |
| `02_result_analysis_and_plots.ipynb` | Load saved JSONL/CSV → training curves, calibration diagram, multi-dataset bar chart, PIDL λ ablation |
| `03_robustness_experiments_optional.ipynb` | Dirichlet α sweep, diffusivity type compare, layer ablation, 3/4/5-client comparison |

All notebooks are **Colab-compatible**: Drive mount, `pip install`, Kaggle credential setup, `sys.path.insert` to project root.

---

## 13. `dataset_summary.json` Schema

Written by `build_federated_dataloaders(save_summary_to=...)` and `_save_summary()`.

```json
{
  "image_root": "/path/to/Training",
  "num_classes": 4,
  "class_names": ["glioma", "meningioma", "notumor", "pituitary"],
  "total_images": 5712,
  "class_counts": { "glioma": 1321, "meningioma": 1339, "notumor": 1595, "pituitary": 1457 },
  "is_grayscale_source": false,
  "splits": {
    "train": { "total": 4569, "class_counts": { ... } },
    "test":  { "total": 1143, "class_counts": { ... } }
  },
  "partitioning": {
    "strategy": "stratified_iid",
    "num_clients": 3,
    "client_sizes": [1523, 1523, 1523],
    "client_class_distributions": [ { "glioma": 440, ... }, { ... }, { ... } ]
  },
  "settings": {
    "image_size": [224, 224],
    "batch_size": 32,
    "augment": true,
    "random_seed": 42,
    "test_split": 0.2,
    "num_workers": 2
  }
}
```

---

## 14. Results Directory Layout

Each run creates:
```
results/
└── federated_pidl_20260504_172300/
    ├── config.json            ← ExperimentConfig snapshot
    ├── dataset_summary.json   ← from build_federated_dataloaders
    ├── round_metrics.jsonl    ← one JSON record per FL round (from LoggingFedAvg)
    ├── round_metrics.csv      ← same data as CSV
    ├── final_metrics.jsonl    ← test-set classification + calibration metrics
    ├── final_metrics.csv
    ├── training_curves.png    ← written by notebook 02
    ├── reliability_diagram.png
    └── ...
```

---

## 15. Known Conventions and Gotchas

### SecAgg+ availability
`SecAggPlusWorkflow` is in `flwr.server.workflow` (requires `flwr >= 1.9`).
`server_app.py` has a `try/except ImportError` that falls back to `DefaultWorkflow` gracefully.

### `class_names=None` in DatasetConfig
`colon_cancer_pathology` and `covid` configs ship with `class_names=None`.
`kaggle_loader._build_folder_splits()` calls `config.auto_detect_classes()` automatically before building `class_to_idx`. Always safe to call `build_full_dataset_splits()` — it handles None.

### `build_federated_dataloaders` vs `build_full_dataset_splits`
These are **two separate entry points** for two usage patterns:

| Function | Where used | Returns |
|----------|-----------|---------|
| `build_federated_dataloaders()` | Notebooks, direct scripts | Per-client `DataLoader`s + `dataset_summary` |
| `build_full_dataset_splits()` | `federated/client_app.py` pipeline | `{"train", "val", "test"}` `Dataset` dict |

The `client_app.py` still uses the older `build_full_dataset_splits` flow. Both work; the newer `build_federated_dataloaders` is the preferred entry point for notebooks.

### Grayscale handling
`build_train_transform(is_grayscale=True)` inserts `transforms.Grayscale(num_output_channels=3)` after `Resize` and before `ColorJitter`. This replicates the single channel across 3 identical channels, making the image tensor shape `(3, H, W)` as ResNet18 expects.

### `num_workers` in Colab
Colab's multiprocessing can cause deadlocks with `num_workers > 0`. If DataLoaders hang, pass `num_workers=0` to `build_federated_dataloaders()`.

### `finalize()` is idempotent
Safe to call multiple times. `make_config()` always calls it. Call it again manually after changing `num_clients` mid-experiment.

### Flower simulation vs production
`run_simulation()` uses `flwr.simulation.run_simulation()` (new API). The `SecAggPlusWorkflow` is only effective in the new workflow API, not in `start_simulation()` (legacy).

---

## 16. Dependencies (requirements.txt summary)

```
torch >= 2.1.0
torchvision >= 0.16.0
flwr[simulation] >= 1.9.0      # SecAgg+ requires >= 1.9
kagglehub >= 0.2.0
scikit-learn >= 1.3.0          # stratified train_test_split
numpy >= 1.24.0
pandas >= 2.0.0
matplotlib >= 3.7.0
seaborn >= 0.12.0
scipy >= 1.11.0
Pillow >= 10.0.0
tqdm >= 4.66.0
pyyaml >= 6.0.0
```

---

## 17. Quick-start Minimal Example

```python
import sys
sys.path.insert(0, "/path/to/medical_fl_pidl")

from configs.dataset_configs import get_dataset_config
from data.kaggle_loader import download_kaggle_dataset, find_image_root
from data.dataset_utils import build_federated_dataloaders, print_dataset_summary
from federated.client_app import make_client_fn
from federated.server_app import build_server_app, run_simulation
from federated.task import resolve_device
from utils.logging_utils import make_run_dir, save_config_snapshot
from utils.path_utils import get_results_dir
from utils.seed_utils import set_all_seeds
from configs.experiment_config import make_config

# 1. Configure
cfg = make_config(dataset_key="brain_tumor_mri", num_clients=3, num_rounds=10)
set_all_seeds(cfg.seed)
device = resolve_device(cfg.device)

# 2. Download + detect structure
raw_path = download_kaggle_dataset(cfg.dataset.kaggle_slug, cfg.dataset.dataset_name)
img_root = find_image_root(raw_path)
cfg.dataset.set_data_root(raw_path)  # auto-detects classes if needed

# 3. Build federated DataLoaders
result = build_federated_dataloaders(
    image_root=img_root,
    num_clients=cfg.federated.num_clients,
    test_split=0.20,
    batch_size=cfg.training.batch_size,
    random_seed=cfg.seed,
    save_summary_to=get_results_dir("run_01"),
)
print_dataset_summary(result["dataset_summary"])

# 4. Run simulation
run_dir = make_run_dir(get_results_dir(), cfg.experiment_name)
save_config_snapshot(cfg.to_dict(), run_dir)
server_app, strategy = build_server_app(cfg, log_dir=run_dir)
# client_fn requires partitions from partitioning.py — see notebook 01 for full flow
```
