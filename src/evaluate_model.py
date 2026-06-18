"""Evaluate a saved model using processed features."""

import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.paths import (
    PROCESSED_FEATURES_CSV,
    PROCESSED_FEATURES_PKL,
    rf_model_path,
    rf_test_split_path,
)

DATA_DIR = Path(__file__).resolve().parent.parent
FEATURES_PKL = PROCESSED_FEATURES_PKL
FEATURES_CSV = PROCESSED_FEATURES_CSV
MODEL_PATH = rf_model_path()
TEST_SPLIT_PATH = rf_test_split_path()

try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
except ImportError:
    accuracy_score = None  # type: ignore
    classification_report = None  # type: ignore
    confusion_matrix = None  # type: ignore


def load_features_from_pickle(path: Path) -> List[Dict[str, Any]]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_features_from_csv(path: Path) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: Dict[str, Any] = {}
            for key, value in row.items():
                if key in {"file_name", "label"}:
                    parsed[key] = value
                else:
                    parsed[key] = float(value) if value is not None and value != "" else 0.0
            features.append(parsed)
    return features


def load_feature_dataset() -> List[Dict[str, Any]]:
    if FEATURES_PKL.exists():
        return load_features_from_pickle(FEATURES_PKL)
    if FEATURES_CSV.exists():
        return load_features_from_csv(FEATURES_CSV)
    raise FileNotFoundError(
        f"No processed features found. Run src/preprocess.py first.\nExpected {FEATURES_PKL} or {FEATURES_CSV}"
    )


def load_trained_model(path: Path) -> Tuple[Any, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found. Run src/train_model.py first.\nExpected {path}"
        )
    with path.open("rb") as handle:
        data = pickle.load(handle)
    return data["model"], data["label_encoder"]


def load_test_split(path: Path) -> Tuple[List[List[float]], List[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Test split not found. Run src/train_model.py first to save the test split.\nExpected {path}"
        )
    with path.open("rb") as handle:
        data = pickle.load(handle)
    return data["X_test"], data["y_test"]


def prepare_dataset(features: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[str]]:
    if not features:
        raise ValueError("Feature dataset is empty.")
    feature_names = [name for name in features[0].keys() if name not in {"file_name", "label"}]
    X = [[float(row[name]) for name in feature_names] for row in features]
    y = [row["label"] for row in features]
    return X, y


def summarize_misclassifications(y_true: List[str], y_pred: List[str], top_n: int = 5) -> None:
    errors = Counter((true, pred) for true, pred in zip(y_true, y_pred) if true != pred)
    if not errors:
        print("No misclassifications detected.")
        return

    print("\nTop misclassified true->predicted pairs:")
    for (true, pred), count in errors.most_common(top_n):
        print(f"  {count}x: {true} -> {pred}")


def evaluate_model(model: Any, encoder: Any, X: List[List[float]], y: List[str]) -> None:
    if accuracy_score is None:
        raise ImportError("scikit-learn is required for evaluation. Install it with pip install scikit-learn.")
    y_pred = model.predict(X)
    if hasattr(encoder, "inverse_transform"):
        try:
            y_pred = encoder.inverse_transform(y_pred)
        except Exception:
            pass

    print(f"Evaluating on {len(y)} samples using the saved model.")
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    summarize_misclassifications(y, y_pred)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained classification model with processed features.")
    parser.add_argument("--features", type=Path, help="Optional custom processed features file to load.")
    parser.add_argument("--model", type=Path, help="Optional trained model file to load.")
    parser.add_argument("--use-full", action="store_true", help="Evaluate on the full feature dataset instead of the saved test split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_path = args.features if args.features is not None else (FEATURES_PKL if FEATURES_PKL.exists() else FEATURES_CSV)
    model_path = args.model if args.model is not None else MODEL_PATH

    print(f"Loading model from: {model_path}")
    model, encoder = load_trained_model(model_path)

    if args.use_full or args.features is not None:
        print(f"Loading features from: {feature_path}")
        features = load_feature_dataset() if args.features is None else (
            load_features_from_pickle(feature_path) if feature_path.suffix == ".pkl" else load_features_from_csv(feature_path)
        )
        X, y = prepare_dataset(features)
    else:
        print(f"Loading test split from: {TEST_SPLIT_PATH}")
        X, y = load_test_split(TEST_SPLIT_PATH)

    evaluate_model(model, encoder, X, y)


if __name__ == "__main__":
    main()
