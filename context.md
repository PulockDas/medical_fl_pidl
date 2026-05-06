# Project Context — `medical_fl_pidl`

> **Purpose:** Full session context for AI assistants. Read this before touching any file.

---

## 1. Project Goal

Federated learning research framework for medical image classification combining:
- ResNet18 feature extraction
- Grid-wise Perona-Malik PIDL spatial regularization
- Flower 1.29 federated learning (`run_simulation` Python API)
- SecAgg+ secure aggregation (simulated overhead; incompatible with `run_simulation` in Flower 1.29)
- Data-poisoning attack experiments + update-clipping defense
- Three datasets via KaggleHub, 3/4/5 clients, comprehensive metric logging

---

## 2. Complete File Tree

```
medical_fl_pidl/
├── configs/
│   ├── dataset_configs.py        DatasetConfig dataclass, SLUG_* constants
│   └── experiment_config.py      ModelConfig dataclass
├── data/
│   ├── kaggle_loader.py          download_kaggle_dataset(), find_image_root(),
│   │                             _try_flatten_class_images() (Strategy 5 for COVID)
│   ├── dataset_utils.py          build_federated_dataloaders() — primary entry point
│   └── partitioning.py           stratified_iid_partition()
├── models/
│   └── resnet_pidl.py            ResNetPIDL, build_model(), get/set_model_parameters()
├── losses/
│   └── pidl_loss.py              PIDLLoss, gridwise_perona_malik_loss()
├── federated/
│   ├── task.py                   train(), evaluate(), evaluate_full(), get_federated_data()
│   │                             _DATA_CACHE (thread-safe, key by data_root/clients/batch/seed)
│   ├── client_app.py             MedicalFLClient, client_fn, _make_client_app()
│   │                             attack injection in __init__, update clipping in fit()
│   ├── server_app.py             server_fn, LoggingFedAvg strategy, finalize_experiment()
│   │                             _active_exp_logger / _active_final_params_ref globals
│   └── strategy_logging.py       LoggingFedAvg, _simulate_secagg_time()
├── metrics/
│   ├── classification_metrics.py compute_classification_metrics()
│   └── calibration_metrics.py    compute_calibration_metrics()
├── robustness/
│   ├── robustness_config.py      RobustnessConfig dataclass, from_dict(), is_client_malicious()
│   ├── attacks.py                GaussianNoiseDataset, LabelFlipDataset, wrap_dataset_with_attack()
│   ├── defenses.py               clip_model_update(), compute_update_norm()
│   └── __init__.py
├── utils/
│   ├── logging_utils.py          ExperimentLogger — fl_rounds.csv, fl_summary.json, etc.
│   │                             _cfg stored on save_config() for finalize_experiment()
│   ├── path_utils.py
│   └── seed_utils.py
├── notebooks/
│   ├── 01_clean_multidataset_experiments.ipynb
│   ├── 02_result_analysis_and_plots.ipynb
│   └── 03_robustness_experiments_optional.ipynb
├── results/                      Main experiment outputs
├── results_robustness/           Robustness experiment outputs
├── requirements.txt
├── pyproject.toml
├── context.md
└── README.md
```

---

## 3. Datasets

| Key | Kaggle Slug | Classes | Notes |
|---|---|---|---|
| `brain_tumor_mri` | masoudnickparvar/brain-tumor-mri-dataset | 4 | find_image_root works directly |
| `colon_cancer_or_pathology` | andrewmvd/lung-and-colon-cancer-histopathological-images | 5 | Choose `colon_image_sets` or `lung_image_sets` |
| `covid` | tawsifurrahman/covid19-radiography-database | 4 | Strategy 5 (_try_flatten_class_images) handles class/images/ nesting |

---

## 4. Data Pipeline

```
Full dataset (100 %)
├── Global test set (20 %)   ──► server evaluate_full() each round
└── Training pool   (80 %)
    ├── Client 0 (~26-27 %)  ──► local train + fit()
    ├── Client 1 (~26-27 %)  ──► local train + fit()
    └── Client 2 (~26-27 %)  ──► local train + fit()
```

- Stratified IID partitioning: every client gets equal class distribution
- No separate validation split — clients use global test set for evaluate()
- `random_seed=42` everywhere for reproducibility
- `_DATA_CACHE` in `task.py` ensures data is built once per simulation run

---

## 5. FL Pipeline (run_simulation)

### Configuration flow
1. Notebook sets `os.environ['FL_RUN_OVERRIDE'] = json.dumps(run_cfg)` before `run_simulation()`
2. `_parse_run_config()` in both `server_app.py` and `client_app.py` reads this env var and merges with pyproject.toml defaults
3. After `run_simulation()` returns, notebook calls `from federated.server_app import finalize_experiment; finalize_experiment()`

### Server side (`server_app.py`)
- `server_fn(context)` → parses config → loads data → creates `ExperimentLogger` → creates `LoggingFedAvg` strategy → sets module globals (`_active_exp_logger`, `_active_final_params_ref`, `_active_strategy_ref`) → returns `ServerAppComponents`
- `evaluate_fn` runs `evaluate_full()` + classification/calibration metrics every round, logs to ExperimentLogger, captures last parameters into `_final_params_ref`
- `finalize_experiment()` (public function): reads module globals, calls `logger.finalize()`, saves `final_model.pth` — called explicitly by notebook after `run_simulation()`
- **atexit is NOT reliable** inside `run_simulation` — must call `finalize_experiment()` explicitly

### Client side (`client_app.py`)
- `client_fn(context)` → parses config → loads data from cache → builds model and PIDLLoss → creates MedicalFLClient
- `MedicalFLClient.__init__`: if client is malicious (`enable_attack=True` and `client_id in malicious_client_ids`), wraps train DataLoader with `wrap_dataset_with_attack()`
- `MedicalFLClient.fit()`: trains locally → if `enable_update_clipping`, calls `clip_model_update()` → logs `update_norm_before_clip`, `update_norm_after_clip`, `is_malicious`

### SecAgg+
- `use_secagg=False` by default — incompatible with `run_simulation` in Flower 1.29
- `_simulate_secagg_time()` in `strategy_logging.py` measures wall-clock time of SecAgg math ops on actual parameters → recorded as `secagg_overhead_sec` in `round_metrics.jsonl`
- Real SecAgg+ only works with `flwr run` CLI (not simulation)

---

## 6. Robustness Module

### Attacks (`robustness/attacks.py`)
- **Gaussian noise**: wraps Dataset, adds `N(0, noise_std)` to normalised image tensors
- **Label flip**: wraps Dataset, replaces label with wrong class with probability `flip_probability`
- Both are DataLoader-level wrappers — server is unaware

### Defense (`robustness/defenses.py`)
- **Update clipping**: clips L2 norm of `(new_params - old_params)` to `clip_norm`
- Applied after training, before sending to server
- Applied to ALL clients (honest + malicious) — this is the expected behavior

### Config keys (injected via FL_RUN_OVERRIDE)
```json
{
  "enable_attack":            false,
  "attack_type":              "gaussian_noise",
  "malicious_client_ids":     "0",
  "noise_std":                0.8,
  "label_flip_probability":   0.30,
  "enable_update_clipping":   false,
  "clip_norm":                3.0
}
```

---

## 7. Logging Output Files

Per experiment folder (`results/{dataset}/{n}_clients/`):

| File | Written by | Contents |
|---|---|---|
| `config.json` | `exp_logger.save_config()` | Full hyperparameter snapshot; also stores `self._cfg` for finalize |
| `dataset_summary.json` | `exp_logger.save_dataset_summary()` | Class counts, `client_class_distributions` (list of dicts) |
| `fl_rounds.csv` | `exp_logger.log_round()` | Per-round global metrics — all metric columns |
| `fl_clients.csv` | `exp_logger.log_client_rounds_from_history()` | Per-round per-client metrics |
| `round_metrics.jsonl` | `LoggingFedAvg.aggregate_fit()` | Raw strategy history incl. `secagg_overhead_sec` — **appended each run** |
| `per_class_metrics.csv` | `exp_logger.log_round()` | Per-class precision/recall/F1 per round |
| `fl_summary.json` | `exp_logger.finalize()` via `finalize_experiment()` | Best/final metrics summary |
| `fl_eval.json` | `exp_logger.finalize()` via `finalize_experiment()` | Full per-round eval array |
| `final_model.pth` | `finalize_experiment()` | Final global model state dict |

**Important**: `fl_summary.json` and `final_model.pth` require `finalize_experiment()` to be called explicitly after `run_simulation()`. They are NOT written by atexit.

**Notebook 2 fallback**: If `fl_summary.json` is missing, notebook 2's `_build_summary_from_csv()` auto-generates it from `fl_rounds.csv` (without confusion matrix).

---

## 8. Robustness Experiment Results (Brain Tumor MRI, 3 clients)

| Experiment | Final Acc | Macro F1 | ECE |
|---|---|---|---|
| Clean | 0.9545 | 0.9543 | 0.0103 |
| Gaussian noise (1/3 clients) | 0.9518 | 0.9518 | 0.0158 |
| Label-flip (1/3 clients) | 0.9196 | 0.9193 | 0.1098 |
| Noise + update clipping | 0.3402 | 0.2567 | 0.1308 |
| Label-flip + update clipping | 0.3777 | 0.3459 | 0.0582 |

Note: Aggressive clipping (clip_norm=3.0) also limits honest clients, severely hurting learning in only 5 rounds.

---

## 9. Main Experiment Results

| Dataset | 3c Acc | 4c Acc | 5c Acc |
|---|---|---|---|
| Brain Tumor MRI | 0.9527 | 0.9437 | 0.9500 |
| Colon Cancer | 0.9995 | 0.9975 | 0.9990 |
| COVID-19 | 0.9152 | 0.9029 | 0.9020 |

SecAgg simulated overhead: ~0.65–0.83 s/round.

---

## 10. Known Issues and Decisions

| Issue | Resolution |
|---|---|
| `run_simulation()` doesn't accept `run_config` param (Flower 1.29) | Use `FL_RUN_OVERRIDE` env var |
| SecAgg+ incompatible with `run_simulation` | `use_secagg=False` default; timing simulated instead |
| atexit doesn't fire reliably in `run_simulation` | Call `finalize_experiment()` explicitly in notebook |
| `round_metrics.jsonl` appends across runs | Takes last N lines (most recent run) in notebook 2 |
| COVID-19 dataset has `class/images/` nesting | Strategy 5 in `find_image_root()` creates symlink-flattened temp dir |
| Colon cancer dataset has `colon_image_sets/` and `lung_image_sets/` | User must set `COLON_OR_LUNG` variable in notebook 1 |
| `fl_rounds.csv` has `ce_loss=0, reg_loss=0` if finalize didn't run | Notebook 2 enriches from `round_metrics.jsonl` in § 2b |

---

## 11. Notebook Workflow

### Notebook 01 (main experiments)
1. Clone/pull from GitHub
2. Install from `requirements.txt` (not editable install — avoids setuptools issue)
3. Download datasets via KaggleHub
4. For each (dataset, num_clients): set `FL_RUN_OVERRIDE`, call `run_simulation()`, call `finalize_experiment()`
5. Aggregate master summary

### Notebook 02 (analysis)
1. Clone/pull from GitHub (results already committed)
2. Auto-generate `fl_summary.json` if missing (from `fl_rounds.csv`)
3. Enrich `rounds_dfs` with ce_loss/reg_loss/training_time/secagg from JSONL (§ 2b)
4. Generate all tables and plots
5. Download `figures.zip`

### Notebook 03 (robustness)
1. Clone/pull from GitHub
2. Download dataset via KaggleHub, call `find_image_root()` for `DATA_ROOT`
3. Run 5 experiments: clean / noisy / label_flip / noisy+clip / flip+clip
4. Results saved to `results_robustness/`

---

## 12. Pyproject.toml Default Config

```toml
[tool.flwr.app.config]
num-server-rounds = 5
num-clients = 3
local-epochs = 2
batch-size = 32
learning-rate = 0.001
image-size = 224
feature-layer = "layer2"
regularizer-type = "perona_malik"
lambda-pm = 0.1
use-grid-loss = true
grid-size = 4
k = 1.0
use-secagg = false
secagg-num-shares = 3
secagg-reconstruction-threshold = 2
secagg-max-weight = 1048575
```
