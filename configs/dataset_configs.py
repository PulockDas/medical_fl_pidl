"""
Dataset configurations for three medical image datasets.

CHANGING A DATASET SLUG
------------------------
Edit only the SLUG_* constants in the "EDIT HERE" block below.
Nothing else in this file or the pipeline needs to change.

AUTO-DETECTION
--------------
num_classes and class_names are set to None for datasets whose folder
structure varies by Kaggle source. They are filled automatically when
you call config.set_data_root(path) after downloading, or explicitly
via config.auto_detect_classes().

QUICK USAGE
-----------
::

    from configs.dataset_configs import get_dataset_config

    cfg = get_dataset_config("brain_tumor_mri")
    print(cfg)                          # see current state

    # After downloading:
    cfg.set_data_root("/path/to/data")  # auto-detects classes
    n_classes, names = cfg.resolve_classes()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
#  EDIT HERE – Kaggle dataset slugs
#  Format: "owner/dataset-name"  (copy from the Kaggle dataset URL)
# ══════════════════════════════════════════════════════════════════════════════

SLUG_BRAIN_TUMOR = "masoudnickparvar/brain-tumor-mri-dataset"
# Structure: root/Training/{glioma, meningioma, notumor, pituitary}  4 classes
#            root/Testing/{...}  (merged into train split, re-split at runtime)
# find_image_root strategy: 1 — named training split

SLUG_COLON_CANCER = "andrewmvd/lung-and-colon-cancer-histopathological-images"
# Structure: root/lung_colon_image_set/colon_image_sets/{colon_aca, colon_n}
#                                     /lung_image_sets/{lung_aca, lung_n, lung_scc}
# NOTE: data_root must be set explicitly to one of the two sub-sets.
#   colon_image_sets  → 2 classes (adenocarcinoma vs normal)  — default for this project
#   lung_image_sets   → 3 classes (adenocarcinoma, normal, squamous cell carcinoma)
# find_image_root strategy: 4 — depth-2 (returns lung_image_sets by count)
#   Override manually in the notebook to point at colon_image_sets.

SLUG_COVID = "tawsifurrahman/covid19-radiography-database"
# Structure: root/COVID-19_Radiography_Dataset/{COVID,Lung_Opacity,Normal,Viral Pneumonia}/images/*.jpg
#            Images are nested one level deeper inside each class dir (images/ subdir).
#            Masks are also present (masks/ subdir) — ignored.
# find_image_root strategy: 5 — auto-flattened via /tmp/ symlinks  (4 classes)

# ══════════════════════════════════════════════════════════════════════════════


# ---------------------------------------------------------------------------
# Internal: folder scanner (no imports from data/ to avoid circular deps)
# ---------------------------------------------------------------------------

_IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)

_TRAIN_DIR_NAMES: list[str] = [
    "Training", "training", "train", "Train", "TRAIN",
    "train_set", "train_images", "imgs_train",
]


def _has_image_files(directory: Path) -> bool:
    """Return True if `directory` directly contains at least one image file."""
    try:
        for f in directory.iterdir():
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS:
                return True
    except PermissionError:
        pass
    return False


def _visible_subdirs(directory: Path) -> list[Path]:
    """Return sorted non-hidden subdirectories of `directory`."""
    try:
        return sorted(
            [d for d in directory.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=lambda d: d.name,
        )
    except PermissionError:
        return []


def _classes_from_dir(directory: Path) -> list[str] | None:
    """If `directory`'s subdirs directly contain images, return their names.

    Returns None if the directory is not an ImageFolder-compatible root
    (fewer than 2 class subdirs or none contain images).
    """
    subdirs = _visible_subdirs(directory)
    if len(subdirs) < 2:
        return None
    class_dirs = [d for d in subdirs if _has_image_files(d)]
    if not class_dirs:
        return None
    return sorted(d.name for d in class_dirs)


def _scan_for_classes(root: Path) -> list[str]:
    """Auto-detect class names from a downloaded Kaggle dataset directory.

    Handles all common Kaggle ImageFolder layouts:

    ① ``root/Training/class_A/``  ``root/Training/class_B/``
    ② ``root/class_A/``  ``root/class_B/``
    ③ ``root/some_subdir/class_A/``  (depth-1 nesting)
    ④ ``root/outer/inner/class_A/``  (depth-2 nesting)

    Args:
        root: Downloaded dataset root directory.

    Returns:
        Sorted list of class-name strings, or ``[]`` if not detected.
    """
    # ① Preferred training split directories
    for train_name in _TRAIN_DIR_NAMES:
        train_dir = root / train_name
        if train_dir.is_dir():
            classes = _classes_from_dir(train_dir)
            if classes:
                return classes

    # ② Root itself is an ImageFolder root
    classes = _classes_from_dir(root)
    if classes:
        return classes

    # ③ Depth-1 subdirectory
    for d1 in _visible_subdirs(root):
        classes = _classes_from_dir(d1)
        if classes:
            return classes

    # ④ Depth-2 subdirectory (e.g. root/dataset_name/dataset_name/class_A)
    for d1 in _visible_subdirs(root):
        for d2 in _visible_subdirs(d1):
            classes = _classes_from_dir(d2)
            if classes:
                return classes

    return []


# ---------------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Configuration for one medical image dataset.

    Fields you set once
    -------------------
    dataset_name      Short identifier used in filenames and logs.
    kaggle_slug       Kaggle "owner/dataset-name" handle (edit SLUG_* above).
    image_size        (H, W) resize target; (224, 224) for ResNet18.
    task_type         "multi_class" or "binary".
    description       One-line human description.
    expected_structure  Always "imagefolder" for these datasets.

    Fields filled after download
    ----------------------------
    data_root         Local path set by set_data_root().
    num_classes       Auto-detected or pre-specified integer.
    class_names       Auto-detected or pre-specified list.

    Pipeline settings (rarely changed)
    -----------------------------------
    train_split / val_split / test_split
    has_presplit_dirs   True if the dataset ships with train/ and test/ dirs.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    dataset_name: str
    kaggle_slug: str

    # ── Vision ────────────────────────────────────────────────────────────
    image_size: Tuple[int, int]
    task_type: Literal["multi_class", "binary"]
    description: str
    expected_structure: str = "imagefolder"

    # ── Filled after download (None = auto-detect) ────────────────────────
    data_root: Optional[str] = None
    num_classes: Optional[int] = None
    class_names: Optional[List[str]] = None

    # ── Pipeline splits ───────────────────────────────────────────────────
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    has_presplit_dirs: bool = False

    # ── Backward-compatibility aliases ────────────────────────────────────
    @property
    def name(self) -> str:
        """Alias for dataset_name (backward-compatible)."""
        return self.dataset_name

    @property
    def kaggle_handle(self) -> str:
        """Alias for kaggle_slug (backward-compatible)."""
        return self.kaggle_slug

    @property
    def is_folder_based(self) -> bool:
        """True when expected_structure == 'imagefolder'."""
        return self.expected_structure == "imagefolder"

    @property
    def image_subdir(self) -> Optional[str]:
        """Always None; structure is resolved via find_image_root()."""
        return None

    @property
    def csv_filename(self) -> Optional[str]:
        """Always None; these datasets use ImageFolder layout."""
        return None

    # ── Post-download helpers ──────────────────────────────────────────────

    def set_data_root(
        self,
        path: str,
        auto_detect: bool = True,
    ) -> "DatasetConfig":
        """Set the local dataset root and optionally auto-detect classes.

        Call this immediately after :func:`~data.kaggle_loader.download_kaggle_dataset`
        returns a path.

        Args:
            path:        Local directory path returned by KaggleHub.
            auto_detect: If True and class_names is still None, call
                         :meth:`auto_detect_classes` automatically.

        Returns:
            ``self`` for chaining.

        Example::

            cfg = get_dataset_config("covid")
            local_path = download_kaggle_dataset(cfg.kaggle_slug, cfg.dataset_name)
            cfg.set_data_root(local_path)
            print(cfg.class_names)   # ['COVID', 'Normal', 'Viral Pneumonia']
        """
        self.data_root = str(path)
        if auto_detect and self.class_names is None:
            self.auto_detect_classes()
        return self

    def auto_detect_classes(self) -> "DatasetConfig":
        """Scan data_root and fill in class_names and num_classes.

        Uses a four-strategy waterfall that handles all common Kaggle
        ImageFolder structures (see :func:`_scan_for_classes`).

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If data_root has not been set yet.
        """
        if not self.data_root:
            raise ValueError(
                "data_root is not set. Call set_data_root(path) first, "
                "or pass the downloaded path directly: "
                "config.auto_detect_classes() after config.data_root = path"
            )

        root = Path(self.data_root)
        detected = _scan_for_classes(root)

        if detected:
            self.class_names = detected
            self.num_classes = len(detected)

            if self.task_type == "binary" and self.num_classes != 2:
                print(
                    f"[DatasetConfig] Warning: task_type='binary' but detected "
                    f"{self.num_classes} classes: {detected}.\n"
                    f"  Update task_type or change the Kaggle slug."
                )

            print(
                f"[DatasetConfig] Auto-detected {self.num_classes} classes "
                f"for '{self.dataset_name}': {self.class_names}"
            )
        else:
            print(
                f"[DatasetConfig] Warning: could not detect classes in '{root}'.\n"
                f"  Call preview_dataset_structure('{root}') to inspect the layout,\n"
                f"  then set class_names manually: config.class_names = [...]"
            )

        return self

    def resolve_classes(self) -> Tuple[int, List[str]]:
        """Return (num_classes, class_names), auto-detecting if needed.

        Safe to call at any point in the pipeline. If class_names is already
        set, returns immediately. Otherwise calls auto_detect_classes().

        Returns:
            ``(num_classes, class_names)`` tuple.

        Raises:
            RuntimeError: If auto-detection also fails (no data_root set or
                          folder structure is unrecognised).
        """
        if self.class_names is not None and self.num_classes is not None:
            return self.num_classes, self.class_names

        if self.data_root:
            self.auto_detect_classes()

        if self.class_names is None:
            raise RuntimeError(
                f"Cannot resolve classes for '{self.dataset_name}'.\n"
                f"  Either set data_root and call auto_detect_classes(), or\n"
                f"  set class_names manually: config.class_names = ['A', 'B', ...]"
            )

        return self.num_classes, self.class_names

    def is_ready(self) -> bool:
        """Return True if data_root is set and classes are known."""
        return (
            self.data_root is not None
            and self.class_names is not None
            and self.num_classes is not None
        )

    def summary(self) -> str:
        """Return a single-line human-readable status string."""
        cls_str = (
            f"{self.num_classes} classes" if self.num_classes else "classes=auto"
        )
        root_str = (
            f"root={os.path.basename(self.data_root)}" if self.data_root else "root=not set"
        )
        return (
            f"{self.dataset_name} | {self.task_type} | {cls_str} | "
            f"{self.image_size[0]}×{self.image_size[1]} | {root_str}"
        )

    def __str__(self) -> str:
        cls_info = (
            f"{self.num_classes} classes: {self.class_names}"
            if self.class_names
            else "classes: auto-detect after download"
        )
        root_info = self.data_root or "(not yet downloaded)"
        return (
            f"DatasetConfig(\n"
            f"  dataset_name : {self.dataset_name}\n"
            f"  kaggle_slug  : {self.kaggle_slug}\n"
            f"  task_type    : {self.task_type}\n"
            f"  image_size   : {self.image_size}\n"
            f"  {cls_info}\n"
            f"  data_root    : {root_info}\n"
            f"  description  : {self.description}\n"
            f")"
        )

    def __post_init__(self) -> None:
        total = self.train_split + self.val_split + self.test_split
        assert abs(total - 1.0) < 1e-6, (
            f"train/val/test splits must sum to 1.0 for '{self.dataset_name}' "
            f"(got {total:.4f})."
        )
        if self.class_names is not None and self.num_classes is None:
            self.num_classes = len(self.class_names)
        if self.num_classes is not None and self.class_names is not None:
            assert len(self.class_names) == self.num_classes, (
                f"len(class_names)={len(self.class_names)} != "
                f"num_classes={self.num_classes} for '{self.dataset_name}'."
            )


# ---------------------------------------------------------------------------
# Dataset instances
# ---------------------------------------------------------------------------

BRAIN_TUMOR = DatasetConfig(
    dataset_name="brain_tumor_mri",
    kaggle_slug=SLUG_BRAIN_TUMOR,
    image_size=(224, 224),
    task_type="multi_class",
    description="Brain MRI scans: glioma, meningioma, no tumor, pituitary (4-class)",
    # Classes are known for the default slug; update if you change SLUG_BRAIN_TUMOR
    num_classes=4,
    class_names=["glioma", "meningioma", "notumor", "pituitary"],
    has_presplit_dirs=True,   # ships with Training/ and Testing/ directories
)

COLON_CANCER = DatasetConfig(
    dataset_name="colon_cancer_pathology",
    kaggle_slug=SLUG_COLON_CANCER,
    image_size=(224, 224),
    task_type="multi_class",
    description=(
        "Histopathological images for colon / lung cancer classification. "
        "Classes are auto-detected because they vary by Kaggle source."
    ),
    # num_classes and class_names are intentionally None:
    # they are auto-detected after download via set_data_root()
    num_classes=None,
    class_names=None,
    has_presplit_dirs=False,
)

COVID = DatasetConfig(
    dataset_name="covid",
    kaggle_slug=SLUG_COVID,
    image_size=(224, 224),
    task_type="multi_class",
    description=(
        "Chest X-ray / CT images for COVID-19 detection. "
        "Classes are auto-detected because they vary by Kaggle source."
    ),
    # num_classes and class_names are intentionally None:
    # they are auto-detected after download via set_data_root()
    num_classes=None,
    class_names=None,
    has_presplit_dirs=False,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

AVAILABLE_DATASETS: Dict[str, DatasetConfig] = {
    "brain_tumor_mri":       BRAIN_TUMOR,
    "colon_cancer_pathology": COLON_CANCER,
    "covid":                  COVID,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Retrieve a DatasetConfig by its dataset_name key.

    Args:
        dataset_name: One of the keys in AVAILABLE_DATASETS.

    Returns:
        The corresponding DatasetConfig instance.

    Raises:
        KeyError: With a helpful message listing available names.

    Example::

        cfg = get_dataset_config("covid")
        cfg.set_data_root(download_kaggle_dataset(cfg.kaggle_slug, cfg.dataset_name))
    """
    if dataset_name not in AVAILABLE_DATASETS:
        available = "\n  ".join(AVAILABLE_DATASETS.keys())
        raise KeyError(
            f"Unknown dataset name '{dataset_name}'.\n"
            f"Available datasets:\n  {available}\n"
            f"Add new datasets to AVAILABLE_DATASETS in configs/dataset_configs.py."
        )
    return AVAILABLE_DATASETS[dataset_name]


def list_available_datasets(verbose: bool = False) -> None:
    """Print a formatted table of all registered datasets.

    Args:
        verbose: If True, also print class names and data_root status.

    Example::

        from configs.dataset_configs import list_available_datasets
        list_available_datasets()
        list_available_datasets(verbose=True)
    """
    SEP = "═" * 72
    print(f"\n{SEP}")
    print("  Available Medical Image Datasets")
    print(SEP)

    header = f"  {'Key':<26} {'Task':<14} {'Classes':<10} {'Image Size':<12} {'Ready'}"
    print(header)
    print("  " + "─" * 68)

    for key, cfg in AVAILABLE_DATASETS.items():
        cls_str = str(cfg.num_classes) if cfg.num_classes else "auto"
        size_str = f"{cfg.image_size[0]}×{cfg.image_size[1]}"
        ready_str = "✓" if cfg.is_ready() else "—"
        print(
            f"  {key:<26} {cfg.task_type:<14} {cls_str:<10} {size_str:<12} {ready_str}"
        )

        if verbose:
            print(f"    slug  : {cfg.kaggle_slug}")
            print(f"    desc  : {cfg.description}")
            if cfg.class_names:
                print(f"    classes: {cfg.class_names}")
            if cfg.data_root:
                print(f"    root  : {cfg.data_root}")

    print(SEP)
    print(
        "  Tip: change any slug by editing the SLUG_* constants at the top of\n"
        "       configs/dataset_configs.py, then re-download.\n"
        "  Tip: for datasets with classes=auto, call\n"
        "       cfg.set_data_root(downloaded_path) to auto-detect classes.\n"
    )
