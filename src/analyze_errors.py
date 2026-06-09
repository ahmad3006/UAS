"""Analyze model errors and generate detailed confusion matrix analysis."""

import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = DATA_DIR / "trained_dnn.pkl"
TEST_SPLIT_PATH = DATA_DIR / "test_split.pkl"
ANALYSIS_PATH = DATA_DIR / "error_analysis_report.txt"

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


def load_model_and_data(
    model_path: Path = MODEL_PATH,
    test_split_path: Path = TEST_SPLIT_PATH,
) -> Tuple[object, object, np.ndarray, List[str]]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not test_split_path.exists():
        raise FileNotFoundError(f"Test split not found: {test_split_path}")

    with model_path.open("rb") as handle:
        model_data = pickle.load(handle)
    model = model_data["model"]
    encoder = model_data["label_encoder"]

    with test_split_path.open("rb") as handle:
        test_data = pickle.load(handle)
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    return model, encoder, X_test, y_test


def analyze_errors(y_true: List[str], y_pred: List[str]) -> Dict[str, List[Tuple[str, int]]]:
    """Analyze which classes are confused with which."""
    error_counts = {}
    for true_class in CLASS_NAMES:
        error_counts[true_class] = []

    for true, pred in zip(y_true, y_pred):
        if true != pred:
            error_counts[true].append(pred)

    result = {}
    for true_class in CLASS_NAMES:
        counter = Counter(error_counts[true_class])
        result[true_class] = counter.most_common(5)

    return result


def class_level_accuracy(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Calculate per-class accuracy."""
    accuracy_by_class = {}
    for class_name in CLASS_NAMES:
        mask = np.array(y_true) == class_name
        if mask.sum() == 0:
            accuracy_by_class[class_name] = 0.0
        else:
            correct = np.sum((np.array(y_true)[mask] == np.array(y_pred)[mask]))
            accuracy_by_class[class_name] = correct / mask.sum()
    return accuracy_by_class


def confusion_analysis(y_true: List[str], y_pred: List[str]) -> Tuple[np.ndarray, Dict[str, List[str]]]:
    """Generate confusion matrix and analyze hardest-to-distinguish classes."""
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES)

    # Find off-diagonal elements with highest counts (most confused pairs)
    hardest_pairs = []
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            if i != j:
                hardest_pairs.append((CLASS_NAMES[i], CLASS_NAMES[j], cm[i, j]))

    hardest_pairs = sorted(hardest_pairs, key=lambda x: x[2], reverse=True)[:10]

    return cm, hardest_pairs


def main() -> None:
    print("Loading model and test data...")
    model, encoder, X_test, y_test = load_model_and_data()

    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_names = encoder.inverse_transform(y_pred)

    print("Analyzing errors...")
    errors_by_class = analyze_errors(y_test, y_pred_names)
    class_accuracies = class_level_accuracy(y_test, y_pred_names)
    cm, hardest_pairs = confusion_analysis(y_test, y_pred_names)

    report_lines = [
        "=" * 80,
        "ERROR ANALYSIS REPORT - XPQRS SIGNAL CLASSIFICATION",
        "=" * 80,
        "",
        "OVERALL METRICS:",
        f"Total samples: {len(y_test)}",
        f"Correct predictions: {sum(1 for t, p in zip(y_test, y_pred_names) if t == p)}",
        f"Incorrect predictions: {sum(1 for t, p in zip(y_test, y_pred_names) if t != p)}",
        "",
        "PER-CLASS ACCURACY:",
        "-" * 50,
    ]

    # Sort by accuracy
    sorted_accuracies = sorted(class_accuracies.items(), key=lambda x: x[1], reverse=True)
    for class_name, acc in sorted_accuracies:
        status = "✓ GOOD" if acc > 0.8 else "⚠ MEDIUM" if acc > 0.5 else "✗ BAD"
        report_lines.append(f"{class_name:35} {acc:6.2%} {status}")

    report_lines.extend([
        "",
        "TOP 10 HARDEST-TO-DISTINGUISH CLASS PAIRS:",
        "-" * 50,
    ])

    for true_class, pred_class, count in hardest_pairs:
        pct = (count / sum(1 for t in y_test if t == true_class)) * 100 if sum(1 for t in y_test if t == true_class) > 0 else 0
        report_lines.append(f"{true_class:30} → {pred_class:30} ({count} times, {pct:.1f}%)")

    report_lines.extend([
        "",
        "CLASSES WITH MOST ERRORS:",
        "-" * 50,
    ])

    errors_by_count = sorted(
        [(cls, sum(count for _, count in errors)) for cls, errors in errors_by_class.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    for class_name, error_count in errors_by_count[:10]:
        total_samples = sum(1 for t in y_test if t == class_name)
        error_rate = (error_count / total_samples) * 100 if total_samples > 0 else 0
        report_lines.append(f"{class_name:35} {error_count:3} errors ({error_rate:5.1f}%)")

    report_lines.extend([
        "",
        "CONFUSION MATRIX SUMMARY:",
        "-" * 50,
        "Shape: (17, 17) - rows are true labels, columns are predictions",
        f"Trace (correct predictions): {np.trace(cm)}",
        f"Off-diagonal (incorrect predictions): {cm.sum() - np.trace(cm)}",
        "",
        "RECOMMENDATIONS:",
        "-" * 50,
    ])

    # Find classes that need improvement
    worst_classes = sorted_accuracies[-5:]
    report_lines.append("Classes with lowest accuracy (suggestions):")
    for class_name, acc in worst_classes:
        hardest_confused = [pred for _, pred, _ in hardest_pairs if any(pair[0] == class_name for pair in [(true_class, CLASS_NAMES[j]) for true_class, pairs in zip(CLASS_NAMES, cm) for j, _ in enumerate(pairs)])]
        report_lines.append(f"  - {class_name}: Try collecting more samples or use data augmentation")
        if errors_by_class[class_name]:
            top_confusion = errors_by_class[class_name][0]
            report_lines.append(f"    (Most often confused with: {top_confusion[0]})")

    report_lines.extend([
        "",
        "DETAILED MISCLASSIFICATION BY CLASS:",
        "-" * 50,
    ])

    for class_name in CLASS_NAMES:
        if errors_by_class[class_name]:
            report_lines.append(f"\n{class_name}:")
            for pred_class, count in errors_by_class[class_name][:3]:
                report_lines.append(f"  → {pred_class}: {count} times")
        else:
            report_lines.append(f"\n{class_name}: (No misclassifications)")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    ANALYSIS_PATH.write_text(report_text, encoding="utf-8")
    print(f"\nDetailed report saved to: {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
