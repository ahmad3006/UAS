"""Shared paths and artifact resolution for the XPQRS project."""

from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = ROOT / "archive" / "XPQRS"
MAT_PATH = ARCHIVE_DIR / "5Kfs_1Cycle_50f_1000Sam_1A.mat"

MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
VISUALIZATIONS_DIR = ROOT / "visualizations"

LOADED_DATA_PATH = ROOT / "loaded_dataset.pkl"
PROCESSED_FEATURES_PKL = ROOT / "processed_features.pkl"
PROCESSED_FEATURES_CSV = ROOT / "processed_features.csv"


def first_existing(*paths: Path) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def ensure_dirs() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)


def rf_model_path() -> Path:
    return first_existing(MODELS_DIR / "trained_model.pkl", ROOT / "trained_model.pkl") or MODELS_DIR / "trained_model.pkl"


def dnn_model_path() -> Path:
    return first_existing(MODELS_DIR / "trained_dnn.pkl", ROOT / "trained_dnn.pkl") or MODELS_DIR / "trained_dnn.pkl"


def cnn_model_path() -> Path:
    return first_existing(MODELS_DIR / "trained_cnn.pkl", ROOT / "trained_cnn.pkl") or MODELS_DIR / "trained_cnn.pkl"


def dnn_test_split_path() -> Path:
    return first_existing(RESULTS_DIR / "test_split.pkl", ROOT / "test_split.pkl") or RESULTS_DIR / "test_split.pkl"


def cnn_test_split_path() -> Path:
    return first_existing(RESULTS_DIR / "test_split_cnn.pkl", ROOT / "test_split_cnn.pkl") or RESULTS_DIR / "test_split_cnn.pkl"


def rf_test_split_path() -> Path:
    return first_existing(RESULTS_DIR / "test_split_rf.pkl", ROOT / "test_split.pkl") or RESULTS_DIR / "test_split_rf.pkl"


def dnn_history_path() -> Path:
    return first_existing(RESULTS_DIR / "training_history_dnn.pkl", ROOT / "training_history_dnn.pkl") or RESULTS_DIR / "training_history_dnn.pkl"


def cnn_history_path() -> Path:
    return first_existing(RESULTS_DIR / "training_history_cnn.pkl", ROOT / "training_history_cnn.pkl") or RESULTS_DIR / "training_history_cnn.pkl"
