"""Preprocess loaded time-series data into feature vectors."""

from pathlib import Path
import math
import pickle
import csv
from statistics import mean, stdev
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent.parent
LOADED_DATA_PATH = DATA_DIR / "loaded_dataset.pkl"
OUTPUT_CSV = DATA_DIR / "processed_features.csv"
OUTPUT_PKL = DATA_DIR / "processed_features.pkl"


def ensure_float_sequence(raw_rows: List[List[Any]]) -> List[float]:
    """Convert raw row data into a flat 1D float signal."""
    if not raw_rows:
        return []

    parsed = []
    if len(raw_rows) == 1:
        parsed = raw_rows[0]
    else:
        if len(raw_rows[0]) == 2 and all(len(row) == 2 for row in raw_rows):
            parsed = [float(row[1]) for row in raw_rows if _is_number(row[1])]
        else:
            parsed = [value for row in raw_rows for value in row]

    signal = []
    for value in parsed:
        if _is_number(value):
            signal.append(float(value))
    return signal


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def rms(signal: List[float]) -> float:
    return math.sqrt(sum(x * x for x in signal) / len(signal)) if signal else 0.0


def peak_to_peak(signal: List[float]) -> float:
    return max(signal) - min(signal) if signal else 0.0


def zero_crossings(signal: List[float]) -> int:
    count = 0
    for a, b in zip(signal, signal[1:]):
        if a == 0.0 or b == 0.0:
            continue
        if a * b < 0:
            count += 1
    return count


def mean_abs_deviation(signal: List[float]) -> float:
    if not signal:
        return 0.0
    m = mean(signal)
    return sum(abs(x - m) for x in signal) / len(signal)


def describe_signal(signal: List[float]) -> Dict[str, float]:
    if not signal:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "ptp": 0.0,
            "rms": 0.0,
            "zero_crossings": 0,
            "mean_abs_dev": 0.0,
        }

    return {
        "mean": mean(signal),
        "std": stdev(signal) if len(signal) > 1 else 0.0,
        "min": min(signal),
        "max": max(signal),
        "ptp": peak_to_peak(signal),
        "rms": rms(signal),
        "zero_crossings": zero_crossings(signal),
        "mean_abs_dev": mean_abs_deviation(signal),
    }


def extract_features(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []
    for record in records:
        signal = ensure_float_sequence(record.get("raw_rows", []))
        if not signal:
            continue

        stats = describe_signal(signal)
        signal_index = record.get("signal_index", 0)
        features.append(
            {
                "file_name": f"{record['file_path'].stem}_{signal_index}",
                "label": record["label"],
                "n_values": len(signal),
                **stats,
            }
        )
    return features


def save_features_to_csv(features: List[Dict[str, Any]], output_path: Path) -> None:
    if not features:
        raise ValueError("No features to save.")

    fieldnames = list(features[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(features)


def save_features_to_pickle(features: List[Dict[str, Any]], output_path: Path) -> None:
    with output_path.open("wb") as handle:
        pickle.dump(features, handle)


def load_loaded_dataset(path: Path = LOADED_DATA_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Loaded dataset not found: {path}. Run src/load_data.py first."
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


def main() -> None:
    print(f"Loading loaded dataset from: {LOADED_DATA_PATH}")
    records = load_loaded_dataset(LOADED_DATA_PATH)
    print(f"Records loaded: {len(records)}")

    features = extract_features(records)
    print(f"Extracted feature vectors: {len(features)}")

    save_features_to_csv(features, OUTPUT_CSV)
    save_features_to_pickle(features, OUTPUT_PKL)

    print(f"Saved features to CSV: {OUTPUT_CSV}")
    print(f"Saved features to pickle: {OUTPUT_PKL}")


if __name__ == "__main__":
    main()
