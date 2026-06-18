"""Load time-series data from archive/XPQRS and infer labels from file names."""

from pathlib import Path
import csv
import pickle
from typing import Any, Dict, List

DATA_DIR = Path(__file__).resolve().parent.parent / "archive" / "XPQRS"


def list_csv_files(data_dir: Path = DATA_DIR) -> List[Path]:
    """Return all CSV files in the dataset folder."""
    return sorted(data_dir.glob("*.csv"))


def infer_label_from_filename(file_path: Path) -> str:
    """Infer a human-readable class label from the file name."""
    return file_path.stem.replace("_", " ")


def parse_value(value: str) -> Any:
    """Convert a CSV value to float when possible, otherwise keep as string."""
    try:
        return float(value)
    except ValueError:
        return value


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def load_csv_file(file_path: Path) -> List[Dict[str, Any]]:
    """Load a single CSV file; each row is one signal sample."""
    label = infer_label_from_filename(file_path)
    with file_path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader if row]

    if not rows:
        raise ValueError(f"CSV file is empty: {file_path}")

    records: List[Dict[str, Any]] = []
    for signal_index, row in enumerate(rows):
        parsed_row = [parse_value(cell) for cell in row]
        signal = [float(value) for value in parsed_row if _is_number(value)]
        if not signal:
            continue

        records.append(
            {
                "file_path": file_path,
                "label": label,
                "signal_index": signal_index,
                "columns": len(parsed_row),
                "n_rows": 1,
                "signal": signal,
                "raw_rows": [parsed_row],
            }
        )
    return records


def load_all_csv_data(data_dir: Path = DATA_DIR) -> List[Dict[str, Any]]:
    """Load every signal from all CSV files in the dataset folder."""
    csv_files = list_csv_files(data_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    records: List[Dict[str, Any]] = []
    for file_path in csv_files:
        records.extend(load_csv_file(file_path))
    return records


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Summarize the number of files per inferred label."""
    counts: Dict[str, int] = {}
    for record in records:
        counts[record["label"]] = counts.get(record["label"], 0) + 1
    return counts


def save_records(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Save loaded records to a pickle file for faster reuse."""
    with output_path.open("wb") as handle:
        pickle.dump(records, handle)


def main() -> None:
    print(f"Loading CSV dataset from: {DATA_DIR}")
    records = load_all_csv_data(DATA_DIR)
    summary = summarize_records(records)

    print(f"Loaded {len(records)} signal samples from {len(summary)} class files.")
    print("Label counts:")
    for label, count in summary.items():
        print(f"  - {label}: {count}")

    first = records[0]
    print("\nSample record:")
    print(f"  file_path: {first['file_path'].name}")
    print(f"  label: {first['label']}")
    print(f"  signal_index: {first['signal_index']}")
    print(f"  timesteps: {first['columns']}")
    print(f"  first 10 values: {first['signal'][:10]}")

    output_path = Path(__file__).resolve().parent.parent / "loaded_dataset.pkl"
    save_records(records, output_path)
    print(f"Saved loaded dataset to: {output_path}")


if __name__ == "__main__":
    main()
