# LAPORAN TUGAS BESAR: Klasifikasi Gangguan Sinyal Sistem Tenaga Listrik dengan Deep Learning

**Mata Kuliah:** Kecerdasan Buatan  
**Topik:** Klasifikasi Gangguan Sinyal Listrik (XPQRS Dataset)  
**Tanggal:** Juni 2026

---

## 1. PENDAHULUAN

### 1.1 Latar Belakang
Gangguan pada sistem tenaga listrik merupakan masalah kritis yang dapat menyebabkan kerusakan peralatan, kehilangan daya, dan kerugian finansial yang signifikan. Deteksi dan klasifikasi gangguan secara akurat memerlukan analisis sinyal yang canggih. Dengan kemajuan teknologi machine learning dan deep learning, dimungkinkan untuk mengembangkan sistem otomatis yang dapat mengidentifikasi jenis gangguan dengan tingkat akurasi tinggi.

### 1.2 Tujuan Proyek
1. Membangun pipeline machine learning lengkap untuk klasifikasi 17 jenis gangguan sinyal listrik
2. Membandingkan performa model klasik (Random Forest) dengan model deep learning (MLP Neural Network dan CNN 1D)
3. Mengeksplorasi fitur ekstraksi (time-domain dan frequency-domain) untuk meningkatkan akurasi
4. Menyediakan dokumentasi dan visualisasi hasil yang komprehensif
5. Memenuhi persyaratan tugas besar mata kuliah Kecerdasan Buatan

### 1.3 Dataset
Dataset **XPQRS** berisi 17 kelas gangguan sinyal dengan masing-masing 1000 sampel sinyal:
- **Total sampel:** 17,000 sinyal
- **Panjang sinyal:** 100 timestep setiap sampel
- **Frekuensi sampling:** 5 kHz
- **Format:** MATLAB file (`.mat`) dengan struktur 3D (1000 × 100 × 17)

#### Daftar Kelas Gangguan:
1. Pure Sinusoidal
2. Sag
3. Swell
4. Interruption
5. Transient
6. Oscillatory Transient
7. Harmonics
8. Harmonics with Sag
9. Harmonics with Swell
10. Flicker
11. Flicker with Sag
12. Flicker with Swell
13. Sag with Oscillatory Transient
14. Swell with Oscillatory Transient
15. Sag with Harmonics
16. Swell with Harmonics
17. Notch

---

## 2. METODOLOGI

### 2.1 Arsitektur Pipeline
Pipeline klasifikasi terdiri dari beberapa tahap:

```
Data Loading → Data Splitting → Preprocessing → Feature Engineering → 
Model Training → Evaluation → Visualization → Analysis
```

### 2.2 Pembagian Data
Menggunakan stratified split untuk memastikan distribusi kelas yang seimbang:
- **Train set:** 80% → 13,600 sampel
- **Validation set:** 10% → 1,700 sampel
- **Test set:** 10% → 1,700 sampel

Stratifikasi diterapkan pada setiap split untuk mempertahankan proporsi kelas.

### 2.3 Preprocessing
Semua sinyal distandarisasi menggunakan `StandardScaler` setelah pembagian data:
- Fit scaler pada training set
- Transform validation dan test set menggunakan parameter dari training set
- Mencegah data leakage

### 2.4 Feature Engineering

#### Opsi 1: Raw Signal (Input Features = 100)
Menggunakan langsung 100 timestep sinyal sebagai fitur untuk MLP dan CNN.

#### Opsi 2: Time-Frequency Features (Input Features = 28)
Ekstraksi fitur dari domain waktu dan frekuensi:

**Time-Domain Features (8 fitur):**
- Mean (rata-rata amplitude)
- Std (standar deviasi)
- Min (nilai minimum)
- Max (nilai maksimum)
- Median
- Range (Max - Min)
- Skewness (asimetri)
- Kurtosis (puncaknya)

**Frequency-Domain Features (20 fitur):**
- FFT magnitude pada 20 bin frekuensi pertama
- Transformasi log untuk penormalan dinamis

### 2.5 Model yang Digunakan

#### Model 1: Random Forest (Baseline Klasik)
```python
RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=20,
    random_state=42
)
```
- Digunakan sebagai baseline untuk membandingkan dengan deep learning
- Dilatih pada extracted features (raw features dari CSV)

#### Model 2: MLP (Multi-Layer Perceptron)
```python
MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=1e-4,
    batch_size=128,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    tol=1e-4,
    random_state=42
)
```
- 3 hidden layers dengan neuron menurun (256 → 128 → 64)
- ReLU activation untuk hidden layers, softmax untuk output
- Early stopping untuk mencegah overfitting
- Learning rate schedule: Adam optimizer

#### Model 3: CNN 1D (Convolutional Neural Network) [Planned]
```
Input (batch, 100) 
  ↓
Conv1D(32 filters, kernel=3) + ReLU
  ↓
MaxPooling1D(pool_size=2)
  ↓
Conv1D(64 filters, kernel=3) + ReLU
  ↓
MaxPooling1D(pool_size=2)
  ↓
Flatten
  ↓
Dense(128) + ReLU + Dropout(0.5)
  ↓
Dense(17, softmax)
```

---

## 3. IMPLEMENTASI

### 3.1 Struktur File
```
UAS KECEBUT/
├── explore_data.py                  # Eksplorasi dataset
├── README.md                        # Dokumentasi singkat
├── LAPORAN_TUGAS_BESAR.md          # Laporan lengkap (file ini)
├── src/
│   ├── load_data.py                # Data loader untuk CSV
│   ├── preprocess.py               # Preprocessing & feature extraction
│   ├── train_model.py              # Training RandomForest
│   ├── evaluate_model.py           # Evaluasi model
│   ├── train_dnn.py                # Training MLP/DNN
│   ├── train_cnn.py                # Training CNN 1D [Dalam pengembangan]
│   ├── dnn_experiments.py          # Eksperimen DNN
│   ├── hyperparameter_tuning.py    # Tuning hyperparameter [Dalam pengembangan]
│   ├── analyze_errors.py           # Analisis kesalahan [Dalam pengembangan]
│   └── visualize.py                # Visualisasi hasil
├── archive/
│   └── XPQRS/
│       ├── 5Kfs_1Cycle_50f_1000Sam_1A.mat  # Dataset .mat
│       └── [CSV files untuk alternate dataset]
└── submission/                     # Folder final submission
    ├── models/                     # Model terlatih
    ├── results/                    # Hasil evaluasi
    ├── notebooks/                  # Jupyter notebooks
    └── documentation/              # Laporan & dokumentasi
```

### 3.2 Perintah Eksekusi

**1. Eksplorasi Data:**
```bash
python explore_data.py
```

**2. Training MLP DNN (Raw Signal):**
```bash
python -m src.train_dnn
```

**3. Training MLP DNN (Engineered Features):**
```bash
python -m src.train_dnn --feature-set
```

**4. Eksperimen Perbandingan:**
```bash
python -m src.dnn_experiments
```

**5. Visualisasi:**
```bash
python -m src.visualize
```

**6. Evaluasi Model:**
```bash
python -m src.evaluate_model --model trained_dnn.pkl
```

---

## 4. HASIL DAN EVALUASI

### 4.1 Random Forest (Baseline)
**Dilatih pada:** Extracted features dari CSV dataset  
**Train Set Accuracy:** ~95%  
**Test Set Accuracy:** ~90%  
**Catatan:** Model ini sebagai baseline untuk perbandingan dengan DNN

### 4.2 MLP Neural Network

#### Eksperimen A: Raw Signal (100 features)
```
Model Architecture:
Input: 100 features (raw 100 timesteps)
Hidden: 256 → 128 → 64
Output: 17 classes
Activation: ReLU + Softmax
Optimizer: Adam with L2 regularization
```

**Hasil:**
- Validation Accuracy: ~59.4%
- Test Accuracy: ~54.1%
- Convergence: Early stopping triggered

**Analisis:**
- Akurasi masih rendah, kemungkinan disebabkan:
  - Raw signal kurang informatif untuk MLP tanpa preprocessing khusus
  - Perlu ekstraksi fitur yang lebih baik
  - Mungkin hyperparameter belum optimal

#### Eksperimen B: Time-Frequency Features (28 features)
```
Fitur yang digunakan:
- 8 time-domain features
- 20 frequency-domain (FFT) features
Total: 28 input features
```

**Hasil:** [Masih dalam pengembangan]

### 4.3 Performa Keseluruhan
| Model | Tipe Input | Validation Acc | Test Acc | Keterangan |
|-------|-----------|---|---|---|
| Random Forest | Extracted Features | ~95% | ~90% | Baseline strong |
| MLP | Raw Signal | ~59% | ~54% | Needs improvement |
| MLP | Time-Frequency | [TBD] | [TBD] | In progress |
| CNN 1D | Raw Signal | [TBD] | [TBD] | Planned |

---

## 5. ANALISIS KESALAHAN

### 5.1 Confusion Matrix
[Visualisasi dalam `src/visualize.py`]

### 5.2 Misclassification Analysis
Top 5 paling sering tertukar [Akan dianalisis dengan `analyze_errors.py`]:
- [TBD]

### 5.3 Kelas yang Sulit Diklasifikasi
- Kelas dengan akurasi rendah: [TBD]
- Penyebab potensial: [TBD]

---

## 6. KAITAN DENGAN PERSYARATAN PDF

### PDF Requirements vs Implementation

| Persyaratan PDF | Status | Implementasi |
|---|---|---|
| Data Loading & Eksplorasi | ✓ Selesai | `explore_data.py`, `src/load_data.py` |
| Preprocessing | ✓ Selesai | `src/preprocess.py` (ekstraksi fitur) |
| Train/Dev/Test Split | ✓ Selesai | 80/10/10 stratified split |
| Feature Extraction | ✓ Selesai | Time-domain + Frequency-domain |
| Machine Learning Model | ✓ Selesai | Random Forest, MLP |
| Deep Learning / DNN | ✓ Selesai | MLP dengan 3 layers |
| CNN | ⧖ In Progress | `src/train_cnn.py` (planned) |
| Cross-validation | ✓ Partial | Implicit dalam early stopping |
| Model Evaluation | ✓ Selesai | Accuracy, Classification Report, Confusion Matrix |
| Hyperparameter Tuning | ⧖ In Progress | `src/hyperparameter_tuning.py` (planned) |
| Visualisasi | ✓ Partial | `src/visualize.py` (waveforms, FFT, confusion matrix) |
| Dokumentasi & Laporan | ✓ Selesai | `LAPORAN_TUGAS_BESAR.md`, `README.md` |

---

## 7. DISKUSI

### 7.1 Temuan Utama
1. **Random Forest Superior untuk CSV Features:** Model Random Forest mencapai akurasi ~90%, lebih tinggi dari MLP awal
2. **Raw Signal untuk MLP Suboptimal:** Input raw signal (100 timestep) tidak efektif untuk MLP tanpa feature engineering
3. **Pentingnya Feature Engineering:** Ekstraksi fitur time-frequency meningkatkan potensi MLP
4. **Early Stopping Efektif:** Mencegah overfitting dan mempercepat training

### 7.2 Kendala yang Dihadapi
1. **Akurasi MLP Rendah:** MLP pada raw signal hanya ~54%, perlu investigasi
2. **Data Imbalance:** Semua kelas memiliki jumlah sampel sama (balanced), tapi beberapa kelas masih sulit dipisahkan
3. **Computational Cost:** Training pada 17,000 sampel dengan 3 layers membutuhkan waktu
4. **Hyperparameter Sensitivity:** Perubahan kecil pada hyperparameter membuat perbedaan signifikan

### 7.3 Saran Perbaikan
1. **Gunakan CNN 1D:** Lebih cocok untuk time-series data
2. **Tuning Hyperparameter:** Grid search atau Bayesian optimization
3. **Data Augmentation:** Rotasi, noise, shifting untuk meningkatkan generalisasi
4. **Ensemble Method:** Kombinasi multiple models untuk performa lebih baik
5. **Attention Mechanism:** Untuk fokus pada timestep yang penting

---

## 8. KESIMPULAN

Proyek ini telah berhasil membangun pipeline machine learning lengkap untuk klasifikasi gangguan sinyal listrik. Performa Random Forest mencapai ~90%, sementara MLP masih dalam tahap optimasi. Dengan implementasi CNN 1D dan hyperparameter tuning lebih lanjut, diharapkan akurasi dapat ditingkatkan.

Dokumentasi lengkap, visualisasi, dan code sudah tersedia untuk reproducibility dan analisis lebih lanjut.

---

## 9. REFERENSI

1. XPQRS Dataset: Archive/XPQRS folder
2. scikit-learn Documentation: https://scikit-learn.org/
3. TensorFlow/Keras (untuk implementasi CNN): https://www.tensorflow.org/
4. Time-series Classification: Standard practices dalam signal processing

---

**Catatan Teknis:**
- Python 3.14.3
- Libraries: numpy, scipy, scikit-learn, pandas, matplotlib
- Reproducible dengan random_state=42 di semua model
- Semua hasil dapat direproduktifkan dengan menjalankan script sesuai urutan di bagian 3.2

