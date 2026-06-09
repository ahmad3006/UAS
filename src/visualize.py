"""Visualize XPQRS signal samples and DNN evaluation results."""

from pathlib import Path
import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix

DATA_DIR = Path(__file__).resolve().parent.parent
MAT_PATH = DATA_DIR / "archive" / "XPQRS" / "5Kfs_1Cycle_50f_1000Sam_1A.mat"
MODEL_PATH = DATA_DIR / "trained_dnn.pkl"
TEST_SPLIT_PATH = DATA_DIR / "test_split.pkl"
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
    plt.show()


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
    plt.show()


def plot_confusion_matrix() -> None:
    if not MODEL_PATH.exists() or not TEST_SPLIT_PATH.exists():
        raise FileNotFoundError("Trained model or test split not found. Run src/train_dnn.py first.")

    with MODEL_PATH.open("rb") as handle:
        model_data = pickle.load(handle)
    model = model_data["model"]
    label_encoder = model_data["label_encoder"]

    with TEST_SPLIT_PATH.open("rb") as handle:
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
    plt.show()


def plot_training_history() -> None:
    history_path = DATA_DIR / "training_history_cnn.pkl"
    if not history_path.exists():
        print("Training history not found. Skipping training plot.")
        return

    with history_path.open("rb") as handle:
        history = pickle.load(handle)

    train_losses = history.get("train_losses", [])
    val_losses = history.get("val_losses", [])
    val_accs = history.get("val_accs", [])

    if not train_losses:
        print("Training history is empty.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses, label="Train Loss", linewidth=2)
    axes[0].plot(val_losses, label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss (CNN)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(val_accs, label="Val Accuracy", linewidth=2, color="green")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy (CNN)")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_signal_comparison() -> None:
    X, y = load_signal_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    classes_to_plot = [0, 3]  # Pure Sinusoidal dan Sag
    
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
    plt.show()


def plot_class_statistics() -> None:
    X, y = load_signal_data()
    
    class_means = []
    class_stds = []
    
    for class_idx in range(len(CLASS_NAMES)):
        start = class_idx * 1000
        end = start + 1000
        signals = X[start:end]
        class_means.append(signals.mean())
        class_stds.append(signals.std())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(range(len(CLASS_NAMES)), class_means)
    axes[0].set_xlabel("Class Index")
    axes[0].set_ylabel("Mean Amplitude")
    axes[0].set_title("Mean Amplitude by Class")
    axes[0].set_xticklabels([str(i) for i in range(len(CLASS_NAMES))], rotation=45)
    
    axes[1].bar(range(len(CLASS_NAMES)), class_stds, color="orange")
    axes[1].set_xlabel("Class Index")
    axes[1].set_ylabel("Std Amplitude")
    axes[1].set_title("Std Deviation by Class")
    axes[1].set_xticklabels([str(i) for i in range(len(CLASS_NAMES))], rotation=45)
    
    plt.tight_layout()
    plt.show()


def main() -> None:
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

    if (DATA_DIR / "training_history_cnn.pkl").exists():
        print("Plotting CNN training history...")
        plot_training_history()

    if MODEL_PATH.exists() and TEST_SPLIT_PATH.exists():
        print("Plotting confusion matrix for saved DNN model...")
        plot_confusion_matrix()
    else:
        print("Model/test split tidak ditemukan; lewati plot confusion matrix.")


if __name__ == "__main__":
    main()
