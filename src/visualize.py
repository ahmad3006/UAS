"""Visualize XPQRS signal samples and model evaluation results."""

import argparse
from pathlib import Path
import pickle
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix

from src.paths import (
    MAT_PATH,
    VISUALIZATIONS_DIR,
    cnn_history_path,
    cnn_model_path,
    cnn_test_split_path,
    dnn_history_path,
    dnn_model_path,
    dnn_test_split_path,
    ensure_dirs,
)
from src.train_cnn import CNN1D

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

SHOW_PLOTS = False


def configure_display(show: bool) -> None:
    global SHOW_PLOTS
    SHOW_PLOTS = show
    if show:
        try:
            matplotlib.use("TkAgg")
            plt.switch_backend("TkAgg")
        except Exception:
            pass


def maybe_show() -> None:
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def save_current_figure(filename: str) -> Path:
    ensure_dirs()
    output_path = VISUALIZATIONS_DIR / filename
    plt.gcf().savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    return output_path


def load_signal_data(mat_path: Path = MAT_PATH) -> Tuple[np.ndarray, List[str]]:
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT dataset not found: {mat_path}")

    data = loadmat(mat_path)
    if "Out" not in data:
        raise KeyError("Expected variable 'Out' in MAT file.")

    arr = data["Out"]
    signals_per_class, timesteps, num_classes = arr.shape
    X = arr.reshape((-1, timesteps))
    y = [CLASS_NAMES[class_idx] for class_idx in range(num_classes) for _ in range(signals_per_class)]
    return X, y


def sample_indices_by_class(class_index: int) -> List[int]:
    signals_per_class = 1000
    start = class_index * signals_per_class
    return list(range(start, start + min(5, signals_per_class)))


def plot_waveforms(X: np.ndarray, y: List[str], class_indexes: List[int]) -> None:
    plt.figure(figsize=(12, 8))
    plot_count = 0
    for class_index in class_indexes:
        indices = sample_indices_by_class(class_index)
        for i, idx in enumerate(indices[:3]):
            plot_count += 1
            plt.subplot(len(class_indexes), 3, plot_count)
            plt.plot(X[idx], linewidth=1)
            plt.title(f"{CLASS_NAMES[class_index]} (sample {i + 1})")
            plt.xlabel("Timestep")
            plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.suptitle("Raw XPQRS Signal Waveforms", y=1.02)
    save_current_figure("waveforms.png")
    maybe_show()


def plot_fft(X: np.ndarray, y: List[str], class_index: int) -> None:
    sample_idx = class_index * 1000
    signal = X[sample_idx]
    fft_mag = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(signal.size, d=1.0)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, fft_mag)
    plt.title(f"FFT Magnitude for {CLASS_NAMES[class_index]} (sample 1)")
    plt.xlabel("Frequency bin")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.tight_layout()
    save_current_figure("fft_sample.png")
    maybe_show()


def plot_dnn_confusion_matrix(model_path: Optional[Path] = None, test_split_path: Optional[Path] = None) -> None:
    model_path = model_path or dnn_model_path()
    test_split_path = test_split_path or dnn_test_split_path()

    if not model_path.exists() or not test_split_path.exists():
        raise FileNotFoundError(
            f"DNN model or test split not found.\n"
            f"  model: {model_path}\n"
            f"  test split: {test_split_path}\n"
            f"Run: python -m src.train_dnn"
        )

    with model_path.open("rb") as handle:
        model_data = pickle.load(handle)
    model = model_data["model"]
    label_encoder = model_data["label_encoder"]

    with test_split_path.open("rb") as handle:
        split_data = pickle.load(handle)
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    y_pred = model.predict(X_test)
    if hasattr(label_encoder, "inverse_transform"):
        try:
            y_pred = label_encoder.inverse_transform(y_pred)
        except Exception:
            pass

    cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for DNN Model")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=90, fontsize=8)
    plt.yticks(tick_marks, CLASS_NAMES, fontsize=8)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    save_current_figure("dnn_confusion_matrix.png")
    maybe_show()


def plot_cnn_confusion_matrix(model_path: Optional[Path] = None, test_split_path: Optional[Path] = None) -> None:
    model_path = model_path or cnn_model_path()
    test_split_path = test_split_path or cnn_test_split_path()

    if not model_path.exists() or not test_split_path.exists():
        raise FileNotFoundError(
            f"CNN model or test split not found.\n"
            f"  model: {model_path}\n"
            f"  test split: {test_split_path}\n"
            f"Run: python -m src.train_cnn"
        )

    with model_path.open("rb") as handle:
        model_data = pickle.load(handle)

    with test_split_path.open("rb") as handle:
        split_data = pickle.load(handle)
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    scaler = model_data["scaler"]
    label_encoder = model_data["label_encoder"]

    X_test_scaled = scaler.transform(X_test)
    model = CNN1D(num_classes=len(CLASS_NAMES), input_length=X_test_scaled.shape[1])
    model.load_state_dict(model_data["model_state"])
    model.eval()

    with torch.no_grad():
        X_tensor = torch.from_numpy(X_test_scaled).unsqueeze(1).float()
        output = model(X_tensor)
        _, predicted = torch.max(output, 1)

    y_pred = label_encoder.inverse_transform(predicted.cpu().numpy())
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_NAMES)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for CNN Model")
    plt.colorbar()
    tick_marks = np.arange(len(CLASS_NAMES))
    plt.xticks(tick_marks, CLASS_NAMES, rotation=90, fontsize=8)
    plt.yticks(tick_marks, CLASS_NAMES, fontsize=8)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    save_current_figure("cnn_confusion_matrix.png")
    maybe_show()


def load_training_history(history_path: Path) -> dict:
    if not history_path.exists():
        raise FileNotFoundError(f"Training history not found: {history_path}")
    with history_path.open("rb") as handle:
        return pickle.load(handle)


def plot_dnn_training_history(history_path: Optional[Path] = None) -> None:
    history_path = history_path or dnn_history_path()
    try:
        history = load_training_history(history_path)
    except FileNotFoundError:
        print("DNN training history not found. Skipping DNN training plot.")
        return

    train_losses = history.get("train_loss", [])
    val_losses = history.get("val_loss", [])
    train_accs = history.get("train_acc", [])
    val_accs = history.get("val_acc", [])

    if not train_losses or not train_accs:
        print("DNN training history is incomplete.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label="Train Loss", color="tab:blue", linewidth=2)
    axes[0].plot(val_losses, label="Val Loss", color="tab:orange", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss (DNN)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(train_accs, label="Train Accuracy", color="tab:green", linewidth=2)
    axes[1].plot(val_accs, label="Val Accuracy", color="tab:red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy (DNN)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    ensure_dirs()
    plot_path = VISUALIZATIONS_DIR / "training_history_dnn.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved DNN training history plot to: {plot_path}")
    maybe_show()


def plot_cnn_training_history(history_path: Optional[Path] = None) -> None:
    history_path = history_path or cnn_history_path()
    if not history_path.exists():
        print("CNN training history not found. Skipping training plot.")
        return

    history = load_training_history(history_path)
    train_losses = history.get("train_losses", [])
    val_losses = history.get("val_losses", [])
    train_accs = history.get("train_accs", [])
    val_accs = history.get("val_accs", [])

    if not train_losses or not train_accs:
        print("CNN training history is incomplete.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="Train Loss", linewidth=2)
    axes[0].plot(val_losses, label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss (CNN)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(train_accs, label="Train Accuracy", linewidth=2, color="tab:green")
    axes[1].plot(val_accs, label="Val Accuracy", linewidth=2, color="tab:red", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy (CNN)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    ensure_dirs()
    plot_path = VISUALIZATIONS_DIR / "training_history_cnn.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved CNN training history plot to: {plot_path}")
    maybe_show()


def plot_signal_comparison() -> None:
    X, _ = load_signal_data()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    classes_to_plot = [0, 3]

    for idx, class_idx in enumerate(classes_to_plot):
        sample_idx = class_idx * 1000
        signal = X[sample_idx]
        fft_mag = np.abs(np.fft.rfft(signal))

        axes[idx, 0].plot(signal, linewidth=1.5)
        axes[idx, 0].set_title(f"{CLASS_NAMES[class_idx]} - Time Domain")
        axes[idx, 0].set_xlabel("Timestep")
        axes[idx, 0].set_ylabel("Amplitude")
        axes[idx, 0].grid(True)

        axes[idx, 1].plot(fft_mag, linewidth=1.5, color="orange")
        axes[idx, 1].set_title(f"{CLASS_NAMES[class_idx]} - Frequency Domain (FFT)")
        axes[idx, 1].set_xlabel("Frequency Bin")
        axes[idx, 1].set_ylabel("Magnitude")
        axes[idx, 1].grid(True)

    plt.tight_layout()
    ensure_dirs()
    plot_path = VISUALIZATIONS_DIR / "signal_comparison.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved signal comparison plot to: {plot_path}")
    maybe_show()


def plot_class_statistics() -> None:
    X, _ = load_signal_data()

    class_means = []
    class_stds = []

    for class_idx in range(len(CLASS_NAMES)):
        start = class_idx * 1000
        end = start + 1000
        signals = X[start:end]
        class_means.append(signals.mean())
        class_stds.append(signals.std())

    x_positions = np.arange(len(CLASS_NAMES))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x_positions, class_means)
    axes[0].set_xlabel("Class Index")
    axes[0].set_ylabel("Mean Amplitude")
    axes[0].set_title("Mean Amplitude by Class")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels([str(i) for i in x_positions], rotation=45)

    axes[1].bar(x_positions, class_stds, color="orange")
    axes[1].set_xlabel("Class Index")
    axes[1].set_ylabel("Std Amplitude")
    axes[1].set_title("Std Deviation by Class")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels([str(i) for i in x_positions], rotation=45)

    plt.tight_layout()
    ensure_dirs()
    plot_path = VISUALIZATIONS_DIR / "class_statistics.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved class statistics plot to: {plot_path}")
    maybe_show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate XPQRS visualizations.")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after saving them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_display(args.show)
    ensure_dirs()

    X, y = load_signal_data()
    print("Loaded raw XPQRS signal data.")

    class_indexes = [0, 3, 6, 9, 12]
    print("Plotting raw waveforms for several classes...")
    plot_waveforms(X, y, class_indexes)

    print(f"Plotting FFT for class {class_indexes[0]} ({CLASS_NAMES[class_indexes[0]]})...")
    plot_fft(X, y, class_indexes[0])

    print("Plotting signal comparison (time-domain vs frequency-domain)...")
    plot_signal_comparison()

    print("Plotting class statistics (mean and std)...")
    plot_class_statistics()

    print("Plotting CNN training history...")
    plot_cnn_training_history()

    print("Plotting DNN training history...")
    plot_dnn_training_history()

    cnn_model = cnn_model_path()
    cnn_split = cnn_test_split_path()
    if cnn_model.exists() and cnn_split.exists():
        print("Plotting confusion matrix for saved CNN model...")
        plot_cnn_confusion_matrix(cnn_model, cnn_split)
    else:
        print("CNN model/test split tidak ditemukan; lewati plot CNN confusion matrix.")

    dnn_model = dnn_model_path()
    dnn_split = dnn_test_split_path()
    if dnn_model.exists() and dnn_split.exists():
        print("Plotting confusion matrix for saved DNN model...")
        plot_dnn_confusion_matrix(dnn_model, dnn_split)
    else:
        print("DNN model/test split tidak ditemukan; lewati plot DNN confusion matrix.")

    print(f"\nSemua plot disimpan di: {VISUALIZATIONS_DIR}")


if __name__ == "__main__":
    main()
