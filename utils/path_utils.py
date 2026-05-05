"""
Cross-platform path resolution with Google Colab awareness.

Provides a single function, get_project_root(), that works correctly whether
the code is run locally (Windows/Linux/Mac) or inside a Colab notebook with
optional Google Drive mounting.
"""

from __future__ import annotations

import os
from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute path to the medical_fl_pidl project root.

    Resolution order:
      1. MEDICAL_FL_ROOT environment variable (explicit override).
      2. /content/drive/MyDrive/medical_fl_pidl (Google Colab + Drive).
      3. /content/medical_fl_pidl (Google Colab, no Drive).
      4. Parent of this file's directory (local development).

    Returns:
        Path object pointing to the project root.
    """
    # Explicit override
    if env_root := os.environ.get("MEDICAL_FL_ROOT"):
        return Path(env_root).resolve()

    # Google Colab with Drive mounted
    colab_drive = Path("/content/drive/MyDrive/medical_fl_pidl")
    if colab_drive.exists():
        return colab_drive

    # Google Colab without Drive
    colab_local = Path("/content/medical_fl_pidl")
    if colab_local.exists():
        return colab_local

    # Local development: two levels up from utils/path_utils.py
    return Path(__file__).resolve().parent.parent


def get_results_dir(run_name: str | None = None) -> Path:
    """Return the path to the results directory, creating it if needed.

    Args:
        run_name: Optional subdirectory name for this experiment run.
                  If None, returns the top-level results/ directory.

    Returns:
        Path to the results directory (created if it does not exist).
    """
    root = get_project_root()
    results = root / "results"
    if run_name:
        results = results / run_name
    results.mkdir(parents=True, exist_ok=True)
    return results


def get_kaggle_cache_dir() -> Path | None:
    """Return a suitable KaggleHub cache directory (Colab-friendly).

    In Colab, use /content/kaggle_cache to avoid filling the small root FS.
    Locally, return None to use KaggleHub's default (~/.cache/kagglehub).

    Returns:
        Path or None.
    """
    colab_cache = Path("/content/kaggle_cache")
    # Check if we are inside Google Colab
    try:
        import google.colab  # noqa: F401  (import just to detect Colab)
        colab_cache.mkdir(parents=True, exist_ok=True)
        return colab_cache
    except ImportError:
        return None


def is_colab() -> bool:
    """Return True if the code is running inside Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False
