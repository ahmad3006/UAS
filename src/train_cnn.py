"""Train a 1D CNN model on XPQRS dataset using PyTorch."""

import argparse
import pickle
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = Path(__file__).resolve().parent.parent
MAT_PATH = DATA_DIR / "archive" / "XPQRS" / "5Kfs_1Cycle_50f_1000Sam_1A.mat"
MODEL_PATH = DATA_DIR / "trained_cnn.pkl"
REPORT_PATH = DATA_DIR / "training_cnn_report.txt"
TEST_SPLIT_PATH = DATA_DIR / "test_split_cnn.pkl"

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN1D(nn.Module):
    """1D Convolutional Neural Network for signal classification."""

    def __init__(self, num_classes: int = 17, input_length: int = 100) -> None:
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2)

        # Calculate flattened size after convolutions and pooling
        flattened_size = 128 * (input_length // 8)

        self.fc1 = nn.Linear(flattened_size, 256)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128)
        self.relu_fc2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = self.dropout1(self.relu_fc1(self.fc1(x)))
        x = self.dropout2(self.relu_fc2(self.fc2(x)))
        x = self.fc3(x)

        return x


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


def train_epoch(
    model: CNN1D,
    loader: torch.utils.data.DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimizer: optim.Optimizer,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(output.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def validate_epoch(
    model: CNN1D,
    loader: torch.utils.data.DataLoader,
    criterion: nn.CrossEntropyLoss,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            output = model(X_batch)
            loss = criterion(output, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(output.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / total
    return avg_loss, accuracy


def predict(model: CNN1D, X: np.ndarray) -> np.ndarray:
    model.eval()
    X_tensor = torch.from_numpy(X).unsqueeze(1).float().to(DEVICE)

    with torch.no_grad():
        output = model(X_tensor)
        _, predicted = torch.max(output, 1)

    return predicted.cpu().numpy()


def save_pickle(obj: Any, path: Path) -> None:
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 1D CNN model for XPQRS signal classification.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading dataset from: {MAT_PATH}")
    X, y = load_dataset(MAT_PATH)
    print(f"Loaded {X.shape[0]} examples with {X.shape[1]} timesteps each.")

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y_encoded)
    print(f"Train/val/test shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train).unsqueeze(1).float(),
        torch.from_numpy(y_train).long(),
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val).unsqueeze(1).float(),
        torch.from_numpy(y_val).long(),
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = CNN1D(num_classes=len(CLASS_NAMES), input_length=X.shape[1]).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Training CNN1D on {DEVICE.type.upper()} for {args.epochs} epochs...")
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch + 1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), DATA_DIR / "best_cnn_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    model.load_state_dict(torch.load(DATA_DIR / "best_cnn_model.pth"))

    y_val_pred = predict(model, X_val)
    y_test_pred = predict(model, X_test)

    y_val_names = encoder.inverse_transform(y_val)
    y_val_pred_names = encoder.inverse_transform(y_val_pred)
    y_test_names = encoder.inverse_transform(y_test)
    y_test_pred_names = encoder.inverse_transform(y_test_pred)

    val_acc = accuracy_score(y_val_names, y_val_pred_names)
    test_acc = accuracy_score(y_test_names, y_test_pred_names)

    report = [
        f"CNN1D Training Report",
        f"=" * 50,
        f"Validation accuracy: {val_acc:.4f}",
        f"Test accuracy: {test_acc:.4f}",
        f"\nBest epoch: {len(train_losses) - patience_counter}",
        f"Final train loss: {train_losses[-1]:.4f}",
        f"Final val loss: {val_losses[-1]:.4f}",
        f"\nValidation classification report:",
        classification_report(y_val_names, y_val_pred_names, zero_division=0),
        f"\nTest classification report:",
        classification_report(y_test_names, y_test_pred_names, zero_division=0),
        f"\nTest confusion matrix:",
        str(confusion_matrix(y_test_names, y_test_pred_names)),
    ]

    save_pickle({"model_state": model.state_dict(), "scaler": scaler, "label_encoder": encoder}, MODEL_PATH)
    save_pickle({"X_test": X_test, "y_test": y_test_names}, TEST_SPLIT_PATH)
    save_pickle(
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_accs": train_accs,
            "val_accs": val_accs,
        },
        DATA_DIR / "training_history_cnn.pkl",
    )

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")
    print(f"\nSaved CNN model to: {MODEL_PATH}")
    print(f"Saved test split to: {TEST_SPLIT_PATH}")
    print(f"Saved training report to: {REPORT_PATH}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
