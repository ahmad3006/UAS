"""Train a deep learning model on the XPQRS dataset stored in .mat format."""

import argparse
import pickle
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss

DATA_DIR = Path(__file__).resolve().parent.parent
MAT_PATH = DATA_DIR / "archive" / "XPQRS" / "5Kfs_1Cycle_50f_1000Sam_1A.mat"
MODEL_PATH = DATA_DIR / "trained_dnn.pkl"
REPORT_PATH = DATA_DIR / "training_dnn_report.txt"
TEST_SPLIT_PATH = DATA_DIR / "test_split.pkl"
SCALER_PATH = DATA_DIR / "scaler_dnn.pkl"
HISTORY_PATH = DATA_DIR / "training_history_dnn.pkl"

CLASS_NAMES = [
    "Pure Sinusoidal",
    "Sag",
    "Swell",
    "Interruption",
    "Transient",
    "Oscillatory Transient",
    "Harmonics",
    "Harmonics with Sag",
    "Harmonics with Swell",
    "Flicker",
    "Flicker with Sag",
    "Flicker with Swell",
    "Sag with Oscillatory Transient",
    "Swell with Oscillatory Transient",
    "Sag with Harmonics",
    "Swell with Harmonics",
    "Notch",
]


def load_dataset(mat_path: Path = MAT_PATH) -> Tuple[np.ndarray, List[str]]:
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT dataset not found: {mat_path}")

    data = loadmat(mat_path)
    if "Out" not in data:
        raise KeyError("Expected variable 'Out' in MAT file.")

    arr = data["Out"]
    signals_per_class, timesteps, num_classes = arr.shape
    if num_classes != len(CLASS_NAMES):
        raise ValueError(f"Expected {len(CLASS_NAMES)} classes, found {num_classes}")

    X = arr.reshape((-1, timesteps))
    y = [CLASS_NAMES[class_idx] for class_idx in range(num_classes) for _ in range(signals_per_class)]
    return X.astype(np.float32), y


def extract_features(X: np.ndarray, fft_bins: int = 20) -> np.ndarray:
    if X.ndim != 2:
        raise ValueError("Expected X to be 2D array of shape (n_samples, n_timesteps)")

    time_features = np.stack(
        [
            X.mean(axis=1),
            X.std(axis=1),
            X.min(axis=1),
            X.max(axis=1),
            np.median(X, axis=1),
            X.max(axis=1) - X.min(axis=1),
            skew(X, axis=1, bias=False),
            kurtosis(X, axis=1, fisher=True, bias=False),
        ],
        axis=1,
    )
    freq = np.abs(np.fft.rfft(X, axis=1))[:, :fft_bins]
    freq = np.log1p(freq)
    return np.concatenate([time_features, freq], axis=1)


def split_dataset(
    X: np.ndarray,
    y: List[str],
    test_size: float = 0.10,
    val_size: float = 0.111111,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, np.array(y_train), np.array(y_val), np.array(y_test)


def build_model() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        max_iter=1,
        warm_start=True,
        early_stopping=False,
        tol=1e-4,
        random_state=42,
        verbose=False,
    )


def save_pickle(obj: Any, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def train_with_history(
    model: MLPClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 200,
) -> dict:
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    class_labels = np.unique(y_train)

    for epoch in range(1, max_epochs + 1):
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        train_loss = log_loss(y_train, model.predict_proba(X_train), labels=class_labels)
        val_loss = log_loss(y_val, model.predict_proba(X_val), labels=class_labels)
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )

    return history


def evaluate_report(
    model: MLPClassifier,
    encoder: LabelEncoder,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[float, float, List[str]]:
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

    y_val_names = encoder.inverse_transform(y_val)
    y_val_pred_names = encoder.inverse_transform(y_val_pred)
    y_test_names = encoder.inverse_transform(y_test)
    y_test_pred_names = encoder.inverse_transform(y_test_pred)

    val_acc = accuracy_score(y_val_names, y_val_pred_names)
    test_acc = accuracy_score(y_test_names, y_test_pred_names)

    report_lines = [
        f"Validation accuracy: {val_acc:.4f}",
        f"Test accuracy: {test_acc:.4f}",
        "\nValidation classification report:",
        classification_report(y_val_names, y_val_pred_names, zero_division=0),
        "\nTest classification report:",
        classification_report(y_test_names, y_test_pred_names, zero_division=0),
        "\nTest confusion matrix:",
        str(confusion_matrix(y_test_names, y_test_pred_names)),
    ]
    return val_acc, test_acc, report_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and save a DNN model for the XPQRS dataset.")
    parser.add_argument(
        "--feature-set",
        action="store_true",
        help="Use engineered time-frequency features instead of raw signal samples.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=MODEL_PATH,
        help="Output path for the trained model pickle file.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=REPORT_PATH,
        help="Output path for the training report text file.",
    )
    parser.add_argument(
        "--test-split-output",
        type=Path,
        default=TEST_SPLIT_PATH,
        help="Output path for the saved test split.",
    )
    parser.add_argument(
        "--scaler-output",
        type=Path,
        default=SCALER_PATH,
        help="Output path for the saved scaler.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading dataset from: {MAT_PATH}")
    X, y = load_dataset(MAT_PATH)
    print(f"Loaded {X.shape[0]} examples with {X.shape[1]} timesteps each.")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    if args.feature_set:
        print("Extracting engineered time-frequency features...")
        X = extract_features(X)
        print(f"Feature matrix shape: {X.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y_encoded)
    print(f"Train/val/test shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    save_pickle(scaler, args.scaler_output)
    print(f"Saved scaler to: {args.scaler_output}")

    model = build_model()
    print("Training MLPClassifier...")
    history = train_with_history(model, X_train, y_train, X_val, y_val, max_epochs=200)
    save_pickle(history, HISTORY_PATH)
    print(f"Saved training history to: {HISTORY_PATH}")

    val_acc, test_acc, report = evaluate_report(model, encoder, X_val, y_val, X_test, y_test)

    save_pickle(
        {"model": model, "scaler": scaler, "label_encoder": encoder},
        args.model_output,
    )
    save_pickle({"X_test": X_test, "y_test": encoder.inverse_transform(y_test)}, args.test_split_output)
    save_pickle({"X_val": X_val, "y_val": encoder.inverse_transform(y_val)}, DATA_DIR / "val_split.pkl")
    save_pickle({"X_train": X_train, "y_train": encoder.inverse_transform(y_train)}, DATA_DIR / "train_split.pkl")

    args.report_output.write_text("\n".join(report), encoding="utf-8")
    print(f"Saved model to: {args.model_output}")
    print(f"Saved test split to: {args.test_split_output}")
    print(f"Saved training report to: {args.report_output}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
