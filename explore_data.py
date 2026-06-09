"""Explore data in archive/XPQRS.

This script performs step-by-step exploration for the dataset:
1. Scan the dataset folder
2. List CSV/.mat files and their classes
3. Load example files
4. Print metadata and basic statistics
5. Optionally plot one sample signal
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None


DATA_DIR = Path(__file__).resolve().parent / "archive" / "XPQRS"


def list_data_files(data_dir: Path):
    csv_files = sorted(data_dir.glob("*.csv"))
    mat_files = sorted(data_dir.glob("*.mat"))
    return csv_files, mat_files


def infer_label_from_filename(file_path: Path):
    return file_path.stem.replace("_", " ")


def inspect_csv_file(file_path: Path):
    print(f"\n--- Inspecting CSV: {file_path.name}")
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        print(f"Failed to read CSV: {exc}")
        return None

    print(f"Shape: {df.shape}")
    print("Columns:", list(df.columns))
    print("First 5 rows:")
    print(df.head(5).to_string(index=False))

    print("\nBasic stats:")
    print(df.describe(include="all"))

    missing = df.isna().sum()
    if missing.any():
        print("\nMissing values by column:")
        print(missing[missing > 0])
    else:
        print("\nNo missing values found.")

    return df


def inspect_mat_file(file_path: Path):
    print(f"\n--- Inspecting MAT: {file_path.name}")
    if loadmat is None:
        print("scipy is not installed, cannot read .mat files.")
        return None

    try:
        data = loadmat(file_path)
    except Exception as exc:
        print(f"Failed to read MAT file: {exc}")
        return None

    keys = [k for k in data.keys() if not k.startswith("__")]
    print("Keys in MAT file:", keys)
    for key in keys:
        value = data[key]
        print(f" - {key}: shape={np.shape(value)}, dtype={value.dtype}")
    return data


def print_summary(csv_files, mat_files):
    print(f"Dataset folder: {DATA_DIR}")
    print(f"Found CSV files: {len(csv_files)}")
    for file_path in csv_files:
        print(f"  - {file_path.name}")

    print(f"Found MAT files: {len(mat_files)}")
    for file_path in mat_files:
        print(f"  - {file_path.name}")


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data folder not found: {DATA_DIR}")

    csv_files, mat_files = list_data_files(DATA_DIR)
    print_summary(csv_files, mat_files)

    if not csv_files and not mat_files:
        print("No data files found in the dataset folder.")
        return

    # Inspect the first CSV file as a representative example.
    if csv_files:
        first_csv = csv_files[0]
        df = inspect_csv_file(first_csv)

        if df is not None:
            if df.shape[1] == 1:
                print("\nThis file looks like a single-column time series.")
            elif df.shape[1] == 2:
                print("\nThis file looks like a two-column time series, possibly time + value.")
            else:
                print("\nThis file has multiple columns. Investigate which columns represent the signal, label, or metadata.")

    if mat_files:
        inspect_mat_file(mat_files[0])

    print("\nStep-by-step next actions:")
    print("1. Periksa file CSV satu per satu untuk memastikan format konsisten.")
    print("2. Tentukan label kelas dari nama file atau kolom metadata.")
    print("3. Buat loader data yang membaca semua file dan menyimpan label.")
    print("4. Lakukan ekstraksi fitur atau preprocessing sebelum melatih model.")


if __name__ == "__main__":
    main()
