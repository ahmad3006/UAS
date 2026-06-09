# XPQRS Deep Learning Project

Proyek ini membangun pipeline lengkap untuk klasifikasi 17 jenis gangguan sinyal listrik pada dataset XPQRS. Tujuan utama adalah menguji model machine learning klasik dan deep learning, lalu menghasilkan dokumentasi serta hasil yang siap dipresentasikan.

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

README ini sekarang lebih lengkap untuk presentasi tugas besar, karena mencakup tujuan, dataset, metodologi, arsitektur, hasil, dan kaitan dengan persyaratan PDF.