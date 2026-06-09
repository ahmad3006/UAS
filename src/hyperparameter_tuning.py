"""Hyperparameter tuning for MLP and CNN models on XPQRS dataset."""

import pickle
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

DATA_DIR = Path(__file__).resolve().parent.parent
MAT_PATH = DATA_DIR / "archive" / "XPQRS" / "5Kfs_1Cycle_50f_1000Sam_1A.mat"
RESULTS_PATH = DATA_DIR / "hyperparameter_tuning_results.txt"

CLASS_NAMES = [
    "Pure Sinusoidal", "Sag", "Swell", "Interruption", "Transient",
    "Oscillatory Transient", "Harmonics", "Harmonics with Sag",
    "Harmonics with Swell", "Flicker", "Flicker with Sag",
    "Flicker with Swell", "Sag with Oscillatory Transient",
    "Swell with Oscillatory Transient", "Sag with Harmonics",
    "Swell with Harmonics", "Notch",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(mat_path: Path = MAT_PATH) -> Tuple[np.ndarray, np.ndarray]:
    data = loadmat(mat_path)
    arr = data["Out"]
    signals_per_class, timesteps, num_classes = arr.shape
    X = arr.reshape((-1, timesteps)).astype(np.float32)
    y_labels = [CLASS_NAMES[class_idx] for class_idx in range(num_classes) for _ in range(signals_per_class)]
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)
    return X, y


def prepare_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111111, stratify=y_temp, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def tune_mlp(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> List[Dict[str, Any]]:
    """Grid search for MLP hyperparameters."""
    hidden_layers_options = [
        (128, 64),
        (256, 128, 64),
        (256, 128, 64, 32),
        (512, 256, 128),
    ]
    learning_rates = [1e-4, 1e-3, 1e-2]
    batch_sizes = [64, 128, 256]

    results = []
    total_combinations = len(hidden_layers_options) * len(learning_rates) * len(batch_sizes)
    current = 0

    for hidden_layers, lr, batch_size in product(hidden_layers_options, learning_rates, batch_sizes):
        current += 1
        print(f"MLP Tuning {current}/{total_combinations}: hidden={hidden_layers}, lr={lr}, batch={batch_size}")

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation="relu",
            solver="adam",
            alpha=lr,
            batch_size=batch_size,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
            verbose=False,
        )

        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)

        results.append({
            "model": "MLP",
            "hidden_layers": hidden_layers,
            "learning_rate": lr,
            "batch_size": batch_size,
            "validation_accuracy": val_acc,
        })

    return results


def tune_cnn(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
) -> List[Dict[str, Any]]:
    """Grid search for CNN hyperparameters using PyTorch."""
    kernel_sizes = [3, 5]
    dropouts = [0.3, 0.5]
    learning_rates = [1e-4, 1e-3]

    results = []
    total_combinations = len(kernel_sizes) * len(dropouts) * len(learning_rates)
    current = 0

    X_train_t = torch.from_numpy(X_train).unsqueeze(1).float()
    X_val_t = torch.from_numpy(X_val).unsqueeze(1).float()
    y_train_t = torch.from_numpy(y_train).long()
    y_val_t = torch.from_numpy(y_val).long()

    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)

    for kernel_size, dropout, lr in product(kernel_sizes, dropouts, learning_rates):
        current += 1
        print(f"CNN Tuning {current}/{total_combinations}: kernel={kernel_size}, dropout={dropout}, lr={lr}")

        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv1d(1, 32, kernel_size=kernel_size, padding=1)
                self.relu1 = nn.ReLU()
                self.pool1 = nn.MaxPool1d(2)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=kernel_size, padding=1)
                self.relu2 = nn.ReLU()
                self.pool2 = nn.MaxPool1d(2)
                self.fc1 = nn.Linear(64 * 25, 128)
                self.dropout = nn.Dropout(dropout)
                self.fc2 = nn.Linear(128, len(CLASS_NAMES))

            def forward(self, x):
                x = self.pool1(self.relu1(self.conv1(x)))
                x = self.pool2(self.relu2(self.conv2(x)))
                x = x.view(x.size(0), -1)
                x = self.dropout(torch.relu(self.fc1(x)))
                x = self.fc2(x)
                return x

        model = SimpleCNN().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)

        best_val_acc = 0.0
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    output = model(X_batch)
                    _, predicted = torch.max(output, 1)
                    total += y_batch.size(0)
                    correct += (predicted == y_batch).sum().item()
            val_acc = correct / total
            best_val_acc = max(best_val_acc, val_acc)

        results.append({
            "model": "CNN",
            "kernel_size": kernel_size,
            "dropout": dropout,
            "learning_rate": lr,
            "validation_accuracy": best_val_acc,
        })

    return results


def main() -> None:
    print("Loading XPQRS dataset...")
    X, y = load_dataset()
    print(f"Dataset shape: {X.shape}")

    print("Preparing train/val/test splits...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(X, y)

    print("\n" + "=" * 60)
    print("STARTING MLP HYPERPARAMETER TUNING")
    print("=" * 60)
    mlp_results = tune_mlp(X_train, X_val, y_train, y_val)

    print("\n" + "=" * 60)
    print("STARTING CNN HYPERPARAMETER TUNING")
    print("=" * 60)
    cnn_results = tune_cnn(X_train, X_val, y_train, y_val, epochs=20)

    print("\n" + "=" * 60)
    print("TUNING RESULTS")
    print("=" * 60)

    all_results = mlp_results + cnn_results
    all_results_sorted = sorted(all_results, key=lambda x: x["validation_accuracy"], reverse=True)

    report_lines = [
        "HYPERPARAMETER TUNING RESULTS",
        "=" * 60,
        "\nTop 10 Configurations:\n",
    ]

    for i, result in enumerate(all_results_sorted[:10], 1):
        report_lines.append(f"{i}. {result['model']} - Val Acc: {result['validation_accuracy']:.4f}")
        for key, value in result.items():
            if key not in ["model", "validation_accuracy"]:
                report_lines.append(f"   {key}: {value}")
        report_lines.append("")

    report_lines.append("\nBest MLP Configuration:")
    best_mlp = max(mlp_results, key=lambda x: x["validation_accuracy"])
    for key, value in best_mlp.items():
        report_lines.append(f"  {key}: {value}")

    report_lines.append("\nBest CNN Configuration:")
    best_cnn = max(cnn_results, key=lambda x: x["validation_accuracy"])
    for key, value in best_cnn.items():
        report_lines.append(f"  {key}: {value}")

    report_text = "\n".join(report_lines)
    print(report_text)

    RESULTS_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
