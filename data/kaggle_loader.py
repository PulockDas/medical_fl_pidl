"""
KaggleHub dataset downloader with automatic ImageFolder root detection.

Public API
----------
    download_kaggle_dataset(slug, name, force_download) -> str
        Thin wrapper around ``kagglehub.dataset_download()``.
        Public Kaggle datasets work without any API key or credentials.

    find_image_root(downloaded_path) -> str
        Auto-detect the ImageFolder-compatible subdirectory within a
        downloaded dataset, handling all common Kaggle directory layouts.

    preview_dataset_structure(root_path, max_depth)
        Print a tree view of the dataset directory with image counts.

Example
-------
::

    import kagglehub
    from data.kaggle_loader import find_image_root, preview_dataset_structure

    # Public dataset — no credentials required
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    root = find_image_root(path)
    preview_dataset_structure(root)
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)

_TRAIN_SPLIT_NAMES: list[str] = [
    "Training", "training", "train", "Train", "TRAIN",
    "train_set", "train_images", "imgs_train",
]

_ALL_SPLIT_NAMES: list[str] = _TRAIN_SPLIT_NAMES + [
    "Testing", "testing", "test", "Test", "TEST",
    "test_set", "val", "validation", "Validation",
]

# Subdirectory names that are NOT class labels (used to reject false ImageFolder matches)
_NON_CLASS_NAMES: frozenset[str] = frozenset(
    {"images", "imgs", "image", "masks", "mask", "annotations",
     "labels", "metadata", "segmentation", "thumbnails"}
)


# ---------------------------------------------------------------------------
# 1. download_kaggle_dataset
# ---------------------------------------------------------------------------


def download_kaggle_dataset(
    dataset_slug: str,
    dataset_name: str = "",
    force_download: bool = False,
) -> str:
    """Download a Kaggle dataset and return its local path.

    This is a thin wrapper around ``kagglehub.dataset_download()``.
    Public Kaggle datasets require **no API key or credentials**.
    KaggleHub caches downloads locally; re-running this is instant.

    Args:
        dataset_slug:   Kaggle dataset identifier ``"owner/dataset-name"``.
        dataset_name:   Optional human-readable label for display.
        force_download: Re-download even if a local cache exists.

    Returns:
        Absolute path string to the downloaded dataset directory.

    Example::

        path = download_kaggle_dataset(
            "masoudnickparvar/brain-tumor-mri-dataset",
            "Brain Tumor MRI",
        )
    """
    import kagglehub

    label = dataset_name or dataset_slug
    print(f"  Downloading: {label}")
    path = kagglehub.dataset_download(dataset_slug, force_download=force_download)
    print(f"  Cached at  : {path}")
    return str(path)


# ---------------------------------------------------------------------------
# 2. find_image_root
# ---------------------------------------------------------------------------


def find_image_root(downloaded_path: str) -> str:
    """Automatically find the ImageFolder-compatible root inside a dataset.

    An ImageFolder-compatible directory has at least two subdirectories where
    each subdirectory represents a class and directly contains image files.

    Detection order (first match wins):

    1. ``root/Training/`` (or ``train/``, ``Train/``, …)
    2. ``root/`` itself
    3. Any depth-1 subdirectory that is ImageFolder-compatible
       (prefers non-test dirs; breaks ties by image count)
    4. Depth-2 subdirs — handles ``root/X/X/class_folders`` nesting
    5. Fallback: returns ``root`` with a warning

    Args:
        downloaded_path: Path returned by ``kagglehub.dataset_download()``
                         or :func:`download_kaggle_dataset`.

    Returns:
        Absolute path string of the best ImageFolder-compatible directory.
    """
    root = Path(downloaded_path)

    if not root.is_dir():
        raise NotADirectoryError(f"Downloaded path is not a directory: {root}")

    # Strategy 1: named training-split directory
    for name in _TRAIN_SPLIT_NAMES:
        candidate = root / name
        if candidate.is_dir() and _is_imagefolder_compatible(candidate):
            _report(candidate, "named training split")
            return str(candidate)

    # Strategy 2: root itself
    if _is_imagefolder_compatible(root):
        _report(root, "root directory")
        return str(root)

    # Strategy 3: depth-1 subdirectory
    depth1 = [d for d in _subdirs(root) if _is_imagefolder_compatible(d)]
    if depth1:
        non_test = [d for d in depth1 if "test" not in d.name.lower()
                    and "val" not in d.name.lower()]
        pool = non_test if non_test else depth1
        pool.sort(key=lambda d: _count_images(d), reverse=True)
        _report(pool[0], "depth-1 subdirectory")
        return str(pool[0])

    # Strategy 4: depth-2 subdirectory (e.g. root/dataset/dataset/classes)
    # Guard: skip candidates whose own subdirectories are utility names like
    # "images" / "masks" — those are NOT class labels (COVID-19 dataset layout).
    for d1 in _subdirs(root):
        depth2 = [
            d for d in _subdirs(d1)
            if _is_imagefolder_compatible(d) and not _is_utility_layout(d)
        ]
        if depth2:
            depth2.sort(key=lambda d: _count_images(d), reverse=True)
            _report(depth2[0], "depth-2 nested subdirectory")
            return str(depth2[0])

    # Strategy 5: class/images/ nesting (e.g. COVID-19 Radiography Database)
    # Pattern:  root/ClassA/images/*.jpg  +  root/ClassA/masks/*.jpg (ignored)
    #           root/ClassB/images/*.jpg  +  root/ClassB/masks/*.jpg (ignored)
    # Fix:  create a flat /tmp/ directory where class symlinks point to images/
    for try_root in [root] + list(_subdirs(root)):
        flat = _try_flatten_class_images(try_root)
        if flat is not None:
            _report(flat, "class/images/ nesting — auto-flattened via symlinks")
            return str(flat)

    print(
        f"[find_image_root] Warning: could not detect an ImageFolder-compatible "
        f"structure under '{root}'. Returning root. "
        f"Call preview_dataset_structure() to inspect the layout."
    )
    return str(root)


# ---------------------------------------------------------------------------
# 3. preview_dataset_structure
# ---------------------------------------------------------------------------


def preview_dataset_structure(root_path: str, max_depth: int = 3) -> None:
    """Print a tree view of the dataset directory with image counts.

    Example output::

        brain_mri/                     [2 subdirs]
        ├── yes/                       [155 images]
        └── no/                        [98 images]

    Args:
        root_path: Path to inspect.
        max_depth: Maximum tree depth (default 3).
    """
    root = Path(root_path)
    if not root.is_dir():
        print(f"[preview_dataset_structure] '{root_path}' is not a directory.")
        return
    print(f"\n{root.name}/")
    _print_tree(root, prefix="", depth=0, max_depth=max_depth)
    print()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_utility_layout(path: Path) -> bool:
    """Return True if all direct subdirs of `path` are utility names (not classes).

    Used to guard Strategy 4: prevents ``COVID/`` (whose subdirs are ``images/``
    and ``masks/``) from being treated as an ImageFolder-compatible class dir.
    """
    subdirs = _subdirs(path)
    if not subdirs:
        return False
    return all(d.name.lower() in _NON_CLASS_NAMES for d in subdirs)


def _try_flatten_class_images(path: Path) -> "Path | None":
    """Create a flat ImageFolder directory if `path` matches the class/images/ pattern.

    Detects layouts where images live inside a named sub-subdir::

        path/
        ├── ClassA/
        │   ├── images/   ← actual images here
        │   └── masks/    ← ignored
        └── ClassB/
            ├── images/
            └── masks/

    For each matching class, creates a symlink in ``/tmp/fl_flat_<hash>/``
    pointing to ``ClassA/images/``, making it an ImageFolder-compatible root::

        /tmp/fl_flat_xxxx/
        ├── ClassA  -> /path/ClassA/images/
        └── ClassB  -> /path/ClassB/images/

    Returns the flat directory path, or ``None`` if the pattern is not matched.
    """
    import hashlib

    candidates = _subdirs(path)
    if len(candidates) < 2:
        return None

    # Only trigger when every class dir has an images/ subdir with actual images.
    # This is strict on purpose — avoids false positives on other datasets.
    valid = [
        (d.name, d / "images")
        for d in candidates
        if (d / "images").is_dir() and _has_images(d / "images")
    ]
    if len(valid) < 2:
        return None

    flat = Path(f"/tmp/fl_flat_{hashlib.md5(str(path).encode()).hexdigest()[:8]}")
    flat.mkdir(exist_ok=True)

    for class_name, img_dir in valid:
        link = flat / class_name
        if not link.exists():
            try:
                link.symlink_to(img_dir.resolve())
            except OSError:
                # Symlinks unsupported on this OS/filesystem — cannot flatten
                return None

    return flat if sum(1 for _ in flat.iterdir()) >= 2 else None


def _subdirs(path: Path) -> list[Path]:
    """Return sorted non-hidden subdirectories of `path`."""
    try:
        return sorted(
            [d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=lambda d: d.name,
        )
    except PermissionError:
        return []


def _has_images(directory: Path) -> bool:
    """Return True if `directory` directly contains any image files."""
    try:
        for entry in directory.iterdir():
            if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS:
                return True
    except PermissionError:
        pass
    return False


def _count_images(directory: Path, cap: int = 50_000) -> int:
    """Count image files recursively under `directory` (capped at `cap`)."""
    count = 0
    try:
        for p in directory.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                count += 1
                if count >= cap:
                    break
    except PermissionError:
        pass
    return count


def _is_imagefolder_compatible(path: Path, min_classes: int = 2) -> bool:
    """Return True if `path` has ≥ `min_classes` subdirs that contain images."""
    subdirs = _subdirs(path)
    if len(subdirs) < min_classes:
        return False
    return any(_has_images(d) for d in subdirs[:8])


def _report(path: Path, strategy: str) -> None:
    classes = [d.name for d in _subdirs(path)]
    print(
        f"[find_image_root] Found ({strategy}): '{path.name}'\n"
        f"  Classes: {classes}"
    )


def _count_direct_images(directory: Path) -> int:
    """Count images directly inside `directory` (non-recursive)."""
    try:
        return sum(
            1 for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
    except PermissionError:
        return 0


def _print_tree(path: Path, prefix: str, depth: int, max_depth: int) -> None:
    if depth >= max_depth:
        return
    subdirs = _subdirs(path)
    for i, subdir in enumerate(subdirs):
        is_last = i == len(subdirs) - 1
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "
        n_img = _count_direct_images(subdir)
        child_subdirs = _subdirs(subdir)
        annotation = (
            f"[{n_img:,} images]" if n_img > 0
            else f"[{len(child_subdirs)} subdirs]" if child_subdirs
            else "[empty]"
        )
        print(f"{prefix}{connector}{subdir.name + '/':<30} {annotation}")
        if child_subdirs and depth + 1 < max_depth:
            _print_tree(subdir, prefix + extension, depth + 1, max_depth)
