"""Train a classification model from processed feature vectors."""

import csv
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.paths import MODELS_DIR, RESULTS_DIR, ensure_dirs

DATA_DIR = Path(__file__).resolve().parent.parent
FEATURES_PKL = DATA_DIR / "processed_features.pkl"
FEATURES_CSV = DATA_DIR / "processed_features.csv"
MODEL_PATH = MODELS_DIR / "trained_model.pkl"
REPORT_PATH = RESULTS_DIR / "training_rf_report.txt"
TEST_SPLIT_PATH = RESULTS_DIR / "test_split_rf.pkl"

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
except ImportError:
    RandomForestClassifier = None  # type: ignore
    train_test_split = None  # type: ignore
    LabelEncoder = None  # type: ignore
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
        print(f"Loading features from pickle: {FEATURES_PKL}")
        return load_features_from_pickle(FEATURES_PKL)
    if FEATURES_CSV.exists():
        print(f"Loading features from CSV: {FEATURES_CSV}")
        return load_features_from_csv(FEATURES_CSV)
    raise FileNotFoundError(
        f"No processed features found. Run src/preprocess.py first.\nExpected {FEATURES_PKL} or {FEATURES_CSV}"
    )


def prepare_dataset(features: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[str]]:
    if not features:
        raise ValueError("Feature dataset is empty.")

    labels = []
    X: List[List[float]] = []
    feature_names = [name for name in features[0].keys() if name not in {"file_name", "label"}]

    for row in features:
        labels.append(row["label"])
        X.append([float(row[name]) for name in feature_names])

    print(f"Using feature names: {feature_names}")
    return X, labels


def encode_labels(labels: List[str]) -> Tuple[List[int], Any]:
    if LabelEncoder is None:
        raise ImportError("scikit-learn is required to encode labels. Install it with pip install scikit-learn.")
    encoder = LabelEncoder()
    return encoder.fit_transform(labels).tolist(), encoder


def train_model(X: List[List[float]], y: List[int]) -> Any:
    if RandomForestClassifier is None:
        raise ImportError("scikit-learn is required to train the model. Install it with pip install scikit-learn.")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf


def save_model(model: Any, label_encoder: Any, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump({"model": model, "label_encoder": label_encoder}, handle)


def save_report(report: str, path: Path) -> None:
    path.write_text(report, encoding="utf-8")


def main() -> None:
    if RandomForestClassifier is None:
        print("scikit-learn not installed. Please install it with:\n  pip install scikit-learn")
        return

    features = load_feature_dataset()
    print(f"Loaded {len(features)} feature rows.")

    X, labels = prepare_dataset(features)
    y, encoder = encode_labels(labels)

    if len(X) < 6:
        print("Warning: dataset is very small for training. The model may overfit.")

    if train_test_split is None:
        raise ImportError("scikit-learn is required for train/test split.")

    label_counts = Counter(y)
    stratify = None
    if len(label_counts) > 1 and all(count >= 2 for count in label_counts.values()):
        stratify = y
    else:
        print("Warning: not enough examples per class for stratified split. Using random split only.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=42,
        stratify=stratify,
    )

    ensure_dirs()
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)

    y_test_names = encoder.inverse_transform(y_test)
    y_pred_names = encoder.inverse_transform(y_pred)

    acc = accuracy_score(y_test_names, y_pred_names)
    report = classification_report(y_test_names, y_pred_names, zero_division=0)
    cm = confusion_matrix(y_test_names, y_pred_names)

    with TEST_SPLIT_PATH.open("wb") as handle:
        pickle.dump({"X_test": X_test, "y_test": y_test_names}, handle)
    print(f"Saved test split to: {TEST_SPLIT_PATH}")

    output_report = [
        f"Accuracy: {acc:.4f}",
        "\nClassification Report:",
        report,
        "\nConfusion Matrix:",
        str(cm),
    ]
    save_report("\n".join(output_report), REPORT_PATH)
    save_model(model, encoder, MODEL_PATH)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training report saved to: {REPORT_PATH}")
    print(f"Accuracy on test set: {acc:.4f}")


if __name__ == "__main__":
    main()
