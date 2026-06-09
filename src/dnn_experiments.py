"""Compare DNN training on raw signal versus engineered features for the XPQRS dataset."""

import pickle
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.train_dnn import build_model, extract_features, load_dataset, split_dataset

DATA_DIR = Path(__file__).resolve().parent.parent
REPORT_PATH = DATA_DIR / "dnn_experiments_report.txt"


def train_and_evaluate(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = build_model()
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    return {
        "validation_accuracy": accuracy_score(y_val, val_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "val_report": classification_report(y_val, val_pred, zero_division=0),
        "test_report": classification_report(y_test, test_pred, zero_division=0),
    }


def run_experiment(use_feature_set: bool) -> Tuple[str, Dict[str, float]]:
    X, y = load_dataset()
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    if use_feature_set:
        X = extract_features(X)
        mode = "engineered features"
    else:
        mode = "raw signal"

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y_encoded)
    results = train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test)
    return mode, results


def format_report(experiment_name: str, results: Dict[str, float]) -> str:
    return (
        f"Experiment: {experiment_name}\n"
        f"Validation accuracy: {results['validation_accuracy']:.4f}\n"
        f"Test accuracy: {results['test_accuracy']:.4f}\n"
        f"\nValidation report:\n{results['val_report']}\n"
        f"Test report:\n{results['test_report']}\n"
        f"{'-' * 80}\n"
    )


def main() -> None:
    reports = []
    for use_features in (False, True):
        mode, result = run_experiment(use_features)
        report = format_report(mode, result)
        reports.append(report)
        print(report)

    REPORT_PATH.write_text("\n".join(reports), encoding="utf-8")
    print(f"Saved experiment report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
