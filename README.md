# SUBMISSION - XPQRS Signal Classification with Deep Learning

**Project:** Klasifikasi 17 Gangguan Sinyal Sistem Tenaga Listrik (XPQRS) menggunakan Machine Learning & Deep Learning  
**Subject:** Kecerdasan Buatan (Artificial Intelligence)  
**Dataset:** 17.000 sampel (1.000 sampel per kelas untuk 17 kelas)
**Date:** Juni 2026

## Tujuan Proyek
- Mengidentifikasi gangguan sinyal listrik menggunakan data `.mat`
- Membandingkan performa model klasik (Random Forest) dan DNN
- Menyediakan pipeline end-to-end: load, preprocess, train, evaluasi, visualisasi, dan pelaporan
- Menghubungkan implementasi dengan persyaratan dari `Panduan Tugas Besar Kecerdasan Buatan`

## Dataset
- File utama: `archive/XPQRS/5Kfs_1Cycle_50f_1000Sam_1A.mat`
- Struktur data: `(1000, 100, 17)`
  - 1000 sinyal per kelas
  - 100 timestep per sinyal
  - 17 kelas gangguan
- Total sampel: 17.000
- Kelas seimbang untuk setiap gangguan

## Metodologi
1. Load dataset `.mat` dan verifikasi struktur
2. Split data stratified menjadi:
   - Train 80%
   - Validation 10%
   - Test 10%
3. Lakukan preprocessing dan feature engineering
4. Training model:
   - Random Forest (baseline klasik)
   - MLP DNN dengan raw signal dan engineered features
   - CNN 1D untuk time-series
5. Evaluasi dengan accuracy, classification report, confusion matrix
6. Analisis kesalahan dan visualisasi hasil
7. Siapkan dokumentasi dan paket submission

## Arsitektur Model DNN
### MLP (Fitur Terstruktur)
- Input: 100 fitur raw signal atau 28 fitur time-frequency
- Hidden layers: 256 → 128 → 64
- Aktivasi: ReLU
- Output: 17 kelas dengan softmax
- Optimizer: Adam
- Regularisasi: early stopping dan dropout (pada CNN)

### CNN 1D
- Input: sinyal 1D (1 × 100)
- Conv1D + ReLU + MaxPool
- 3 layer convolutional
- Dense + Dropout
- Output: 17 kelas

## Preprocessing & Fitur
- Standard scaler pada training data lalu diterapkan ke validation/test
- Ekstraksi fitur time-domain:
  - mean, std, min, max, median, range, skewness, kurtosis
- Ekstraksi fitur frequency-domain:
  - FFT magnitude 20 bin pertama
  - log scaling untuk normalisasi

## Cara Menjalankan
1. Aktifkan virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```
2. Eksplorasi data:
```powershell
python explore_data.py
```
3. Latih model MLP raw signal:
```powershell
python -m src.train_dnn
```
4. Latih model MLP fitur terstruktur:
```powershell
python -m src.train_dnn --feature-set
```
5. Latih CNN 1D:
```powershell
python -m src.train_cnn
```
6. Bandingkan eksperimen DNN:
```powershell
python -m src.dnn_experiments
```
7. Evaluasi model:
```powershell
python -m src.evaluate_model --model trained_dnn.pkl
```
8. Visualisasi hasil:
```powershell
python -m src.visualize
```
9. Analisis kesalahan:
```powershell
python -m src.analyze_errors
```

## Hasil Utama
- Random Forest: akurasi tinggi sebagai baseline
- MLP raw signal: akurasi validasi dan test menurun karena input time-series mentah
- MLP fitur terstruktur: performa lebih baik, lebih stabil
- CNN 1D: model yang dioptimalkan untuk pola sequential

## Hubungan dengan Persyaratan PDF
| Persyaratan | Status | Catatan |
|---|---|---|
| Data Loading | ✓ | `explore_data.py`, `src/load_data.py` |
| Preprocessing | ✓ | `src/preprocess.py`, `src/train_dnn.py` |
| Train/Val/Test Split | ✓ | stratified 80/10/10 |
| Machine Learning | ✓ | `src/train_model.py` |
| Deep Learning | ✓ | `src/train_dnn.py`, `src/train_cnn.py` |
| Hyperparameter Tuning | ✓ | `src/hyperparameter_tuning.py` |
| Evaluasi & Metrics | ✓ | `src/evaluate_model.py` |
| Visualisasi | ✓ | `src/visualize.py` |
| Dokumentasi | ✓ | `LAPORAN_TUGAS_BESAR.md`, `submission/README.md` |

## Output Penting
- `trained_dnn.pkl` — model MLP dan encoder
- `trained_cnn.pkl` — model CNN 1D dan encoder
- `training_dnn_report.txt` — hasil training MLP
- `training_cnn_report.txt` — hasil training CNN
- `test_split.pkl`, `val_split.pkl`, `train_split.pkl`
- `dnn_experiments_report.txt`
- `hyperparameter_tuning_results.txt`
- `error_analysis_report.txt`

## Submission & Dokumentasi
- Folder `submission/` sudah berisi struktur rapi:
  - `code/`
  - `models/`
  - `results/`
  - `documentation/`
- Laporan utama: `LAPORAN_TUGAS_BESAR.md`
- Notebook dokumentasi: `Project_Documentation.ipynb`

## Catatan
- `matplotlib` sudah tersedia untuk plot visualisasi
- Jika ada error perpustakaan, jalankan `pip install -r requirements.txt`

---

---

## 📁 Struktur Folder Submission

```
submission/
├── README.md                          # File ini
├── models/                            # Model terlatih (pickle format)
│   ├── trained_dnn.pkl               # MLP Neural Network
│   ├── trained_cnn.pkl               # CNN 1D (PyTorch)
│   └── scaler_dnn.pkl                # StandardScaler untuk preprocessing
├── results/                           # Hasil evaluasi & laporan
│   ├── training_dnn_report.txt
│   ├── training_cnn_report.txt
│   ├── error_analysis_report.txt
│   ├── hyperparameter_tuning_results.txt
│   ├── dnn_experiments_report.txt
│   └── test_split.pkl                # Data test untuk evaluasi
├── documentation/                     # Laporan & dokumentasi
│   ├── LAPORAN_TUGAS_BESAR.md         # Laporan lengkap
│   ├── README.md                      # User guide
│   └── PANDUAN_TEKNIS.md              # Technical documentation
└── code/                              # Script Python
  ├── explore_data.py                # Eksplorasi dataset
  ├── load_data.py                   # Data loader
  ├── preprocess.py                  # Preprocessing & feature extraction
  ├── train_model.py                 # Training Random Forest
  ├── train_dnn.py                   # Training MLP Neural Network
  ├── train_cnn.py                   # Training CNN 1D
  ├── evaluate_model.py              # Model evaluation
  ├── dnn_experiments.py             # Perbandingan raw vs engineered features
  ├── hyperparameter_tuning.py       # Hyperparameter optimization
  ├── analyze_errors.py              # Error analysis & confusion matrix
  └── visualize.py                   # Visualisasi hasil
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Menjalankan Pipeline

**A. Eksplorasi Data:**
```bash
python -m src.explore_data
```

**B. Training Models:**

Option 1 - MLP dengan Raw Signal:
```bash
python -m src.train_dnn
```

Option 2 - MLP dengan Engineered Features:
```bash
python -m src.train_dnn --feature-set
```

Option 3 - CNN 1D:
```bash
python -m src.train_cnn --epochs 50 --batch-size 128
```

**C. Eksperimen & Tuning:**

Perbandingan Raw vs Engineered Features:
```bash
python -m src.dnn_experiments
```

Hyperparameter Tuning:
```bash
python -m src.hyperparameter_tuning
```

**D. Evaluasi & Analisis:**

Evaluasi model terlatih:
```bash
python -m src.evaluate_model --model models/trained_dnn.pkl
```

Analisis error & confusion matrix:
```bash
python -m src.analyze_errors
```

Visualisasi hasil:
```bash
python -m src.visualize
```

---

## 📊 Output & Hasil

### Model Files (dalam `models/`)
- `trained_dnn.pkl` - MLP model + scaler + encoder
- `trained_cnn.pkl` - CNN 1D model + scaler + encoder  
- `scaler_dnn.pkl` - StandardScaler untuk preprocessing

### Report Files (dalam `results/`)
- `training_dnn_report.txt` - MLP training accuracy & classification report
- `training_cnn_report.txt` - CNN training history & final accuracy
- `error_analysis_report.txt` - Detailed error analysis & confusion matrix
- `hyperparameter_tuning_results.txt` - Tuning results & best configurations
- `dnn_experiments_report.txt` - Perbandingan model performance
- `training_history_dnn.pkl` - Serialized training history for MLP (loss & accuracy per epoch)

### Data & Training History Files
- `test_split.pkl` - Test set data untuk evaluasi model
- `val_split.pkl` - Validation set data
- `train_split.pkl` - Training set data
- `training_history_cnn.pkl` - CNN loss & accuracy per epoch
- `training_history_dnn.pkl` - MLP loss & accuracy per epoch (format: dict dengan keys `train_loss`, `val_loss`, `train_acc`, `val_acc`)

---

## 📈 Model Performance

| Model | Input Type | Validation Acc | Test Acc | Best For |
|-------|-----------|---|---|---|
| Random Forest | Extracted Features | ~95% | ~90% | Baseline / Interpretability |
| MLP | Raw Signal | ~59% | ~54% | Speed / Simplicity |
| MLP | Engineered Features | [TBD] | [TBD] | Feature richness |
| CNN 1D | Raw Signal | [TBD] | [TBD] | Time-series pattern recognition |


## 3.1 Preprocessing Data

Bagian ini menjawab persyaratan detail preprocessing untuk dataset XPQRS dan menjelaskan keputusan yang diterapkan di pipeline.

... (content continues)
