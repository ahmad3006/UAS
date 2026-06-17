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
## 3.1 Preprocessing Data — Jawaban Lengkap

Berikut penjelasan rinci setiap langkah preprocessing yang dilakukan pada pipeline XPQRS, beserta alasan dan potongan kode yang relevan.

1) Apakah ada missing values? Bagaimana menanganinya (drop/imputation) dan mengapa?

  - Temuan: file MAT `archive/XPQRS/...` berisi sinyal lengkap per kelas; tidak ditemukan missing values numerik eksplisit.
  - Implementasi defensif: `src/preprocess.py` mengabaikan (drop) record yang tidak menghasilkan urutan angka valid saat parsing (`ensure_float_sequence`). Contoh logika:

    ```python
    signal = ensure_float_sequence(record.get("raw_rows", []))
    if not signal:
      continue  # drop malformed / empty records
    ```

  - Alasan memilih drop vs impute: untuk data time-series listrik, imputasi (mengisi nilai yang hilang) dapat merusak pola temporal (transien, frekuensi). Karena jumlah observasi yang valid besar (17.000 sampel), strategi drop lebih aman dan tidak mengurangi representativitas kelas.

2) Apakah perlu normalisasi atau standardisasi fitur? Metode yang dipilih dan risikonya jika tidak dilakukan.

  - Pipeline menggunakan `StandardScaler` (mean=0, std=1) pada fitur numerik sebelum pelatihan (baik untuk MLP maupun untuk fitur engineered):

    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    ```

  - Alasan: model berbasis gradien (MLP dan optimizers pada CNN) sensitif terhadap skala fitur; standardisasi mempercepat konvergensi, mencegah fitur dengan skala besar mendominasi gradien, dan membuat training lebih stabil.
  - Risiko jika tidak dilakukan: pelatihan bisa menjadi lambat, tidak stabil, atau menghasilkan model suboptimal; pada CNN1D, variasi skala juga dapat mempersulit pembelajaran filter konvolusi.

  - Catatan: untuk sinyal mentah, selain `StandardScaler` ada alternatif `MinMaxScaler` (mengubah ke [0,1]) — dipilih `StandardScaler` karena distribusi fitur time-frequency menangani outlier lebih baik saat menggunakan mean/std.

**Catatan kritis — Data Leakage**

Normalisasi/standardisasi harus dilakukan setelah pembagian data (split). Scaler hanya boleh di-'fit' pada training set, lalu digunakan untuk mentransformasikan validation/test set. Jika scaler di-fit pada seluruh dataset sebelum split, ini menyebabkan data leakage dan estimasi performa yang terlalu optimistis.

Contoh yang benar:

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit hanya pada training set
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
```

Hindari melakukan `scaler.fit_transform` pada `X` penuh sebelum melakukan `train_test_split`.

3) Apakah ada fitur kategorikal? Bagaimana encoding-nya (one-hot, label encoding)? Perbedaannya.

  - Target/label: kategorikal (17 kelas). Encoding yang digunakan: `LabelEncoder` (mengubah label ke integer 0..16).

    ```python
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    ```

  - Perbedaan: `LabelEncoder` memberi indeks integer per kelas (ringkas, cocok untuk loss fungsi cross-entropy). `One-hot` menghasilkan vektor biner panjang K; berguna saat model memerlukan reprezentasi eksplisit (mis. training manual dengan multi-label vector). Karena scikit-learn `MLPClassifier` dan PyTorch `CrossEntropyLoss` menerima integer class labels, `LabelEncoder` dipilih untuk target.

  - Untuk fitur kategorikal non-target (tidak ada pada dataset sinyal mentah), strategi umum adalah one-hot atau embedding jika jumlah kategori besar.

4) Untuk CNN 1D: bagaimana merestrukturisasi data tabular menjadi bentuk `(n_samples, timesteps, channels)`?

  - Implementasi PyTorch pada `src/train_cnn.py`:

    ```python
    # X_train shape awal -> (n_samples, timesteps)
    X_tensor = torch.from_numpy(X_train).unsqueeze(1).float()
    # Tensor shape -> (n_samples, 1, timesteps)
    ```

  - Penjelasan: PyTorch `Conv1d` mengharapkan input `(batch, channels, length)`. `unsqueeze(1)` menambahkan dimensi channel tunggal. Saat membuat DataLoader, `TensorDataset` menerima `(X_tensor, y_tensor)` sehingga batch bertipe `(batch_size, 1, timesteps)`.

5) Untuk CNN 2D: apakah gambar perlu diubah ukurannya, dinormalisasi ke [0,1], atau dikonversi grayscale?

  - Catatan penting: repo ini tidak menggunakan CNN 2D (tidak ada pipeline citra). Untuk panduan umum jika mengadaptasi ke CNN2D:

    - Resize semua citra ke ukuran konsisten (mis. 224×224) untuk batch training yang seragam.
    - Normalisasi pixel ke [0,1] atau standarize menggunakan mean/std (mis. ImageNet mean/std) untuk transfer learning.
    - Konversi ke grayscale hanya jika warna tidak membawa informasi; mengurangi channel dari 3 → 1 dapat mengurangi beban komputasi.

6) Apakah kelas target seimbang (balanced)? Jika tidak, langkah mitigasi.

  - Dataset XPQRS: seimbang oleh desain — 1.000 sampel per kelas (17 kelas), sehingga tidak perlu oversampling.
  - Split stratified digunakan untuk menjaga proporsi kelas:

    ```python
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)
    ```

  - Jika dataset imbalanced, opsi mitigasi yang tersedia di repo/implementasi:
    - Oversampling (SMOTE) pada fitur tabular
    - Menambahkan `class_weight` di loss (PyTorch `CrossEntropyLoss(weight=...)` atau scikit-learn `class_weight='balanced'`)
    - Undersampling kelas mayor


Ringkasan: implementasi saat ini sudah menjawab kebutuhan preprocessing untuk XPQRS — missing records di-drop, `StandardScaler` diterapkan, labels di-encode dengan `LabelEncoder`, dan data untuk CNN1D diubah ke bentuk `(n_samples, 1, timesteps)`. Dataset berimbang sehingga mitigasi imbalance tidak diperlukan.

## 3.2 Penjelasan Setiap Layer dalam Arsitektur

Berikut penjelasan matematis dan konseptual setiap layer utama yang digunakan dalam proyek.

- Dense (Fully-Connected) Layer

  - Operasi matematis: $$y=\phi(W\cdot x + b)$$ di mana $x\in\mathbb{R}^n$ (input), $W\in\mathbb{R}^{m\times n}$ (bobot), $b\in\mathbb{R}^m$ (bias), dan $\phi$ fungsi aktivasi.
  - Pemilihan jumlah neuron: pertimbangkan trade-off bias-variance. Terlalu sedikit neuron → underfitting; terlalu banyak → overfitting dan biaya komputasi tinggi. Pilihan arsitektur (mis. 256→128→64) didasarkan pada eksperimen awal dan jumlah fitur (100 timesteps atau fitur terstruktur ~28).
  - Kedalaman layer: lebih banyak layer (kedalaman) memungkinkan komposisi fungsi kompleks tetapi membutuhkan lebih banyak data dan regularisasi (dropout, weight decay). Gunakan validasi silang untuk menentukan kedalaman optimal.

- Conv1D / Conv2D Layer

  - Kernel/filter: vektor (1D) atau matriks (2D) yang melakukan konvolusi (sliding window). Untuk Conv1D di posisi $t$:

    $$y[t]=\sum_{i=0}^{k-1}w[i]\,x[t+i]+b$$

  - Parameter penting:
    - `filters`: jumlah kernel yang dipelajari (output channels).
    - `kernel_size` ($k$): ukuran jendela lokal.
    - `strides`: langkah pergeseran jendela; pengaruh pada output length.
    - `padding`: mengontrol apakah tepi dipertahankan (`same`) atau dipotong (`valid`).
  - Mengapa efektif: CNN memanfaatkan lokalitas dan translational invariance—satu filter mendeteksi pola serupa di posisi berbeda, sehingga parameter jauh lebih hemat dibanding MLP untuk data berstruktur lokal.

- Pooling (Max/Average/GlobalAveragePooling)

  - Fungsi: mereduksi resolusi spasial/temporal sambil mempertahankan fitur penting (mis. nilai maksimum pada jendela). Pooling mengurangi dimensi, menekan noise, dan menurunkan parameter di layer berikutnya.
  - Hilang vs dipertahankan: kehilangan presisi posisi dan detail kecil, namun mempertahankan keberadaan pola yang relevan.
  - GlobalAveragePooling (GAP) vs Flatten: GAP mereduksi tiap feature map menjadi satu nilai, mengurangi parameter signifikan sebelum Dense; GAP sering lebih aman terhadap overfitting dibanding Flatten yang mempertahankan semua aktivasi.

- Flatten

  - Saat digunakan: ketika ingin menyambungkan seluruh aktivasi spasial ke layer Dense. Output Flatten berukuran `filters * spatial_dim` vs GAP yang menghasilkan `filters`.

## 3.3 Fungsi Aktivasi Output Layer

Pemilihan aktivasi output sesuai jenis tugas:

- Biner: `sigmoid` → probabilitas dalam (0,1).
- Multi-kelas (mutually exclusive): `softmax` → vektor probabilitas yang jumlahnya 1.
- Multi-label: `sigmoid` per neuron → setiap kelas bernilai probabilitas independen.
- Regresi kontinu: `linear` (tanpa aktivasi).

Konsekuensi aktivasi salah: contoh, `sigmoid` untuk regresi membatasi output ke (0,1) sehingga tak mampu memprediksi nilai di luar rentang. Memakai `linear` untuk klasifikasi memerlukan penggunaan loss yang menerima logits (mis. `BCEWithLogitsLoss`); jika tidak, kombinasi activasi/loss yang salah menghasilkan gradien yang tidak sesuai dan optimasi gagal.

Contoh matematis: BCE untuk satu sampel

$$L=-y\log p-(1-y)\log(1-p),\quad p=\sigma(z)$$

Gradien terhadap logit $z$ adalah $(p-y)$; bila aktivasi/loss tidak sesuai, gradien bisa menjadi sangat kecil atau tidak menggambarkan kesalahan sebenarnya.

## 3.4 Fungsi Loss

- Binary Cross-Entropy (BCE): tepat untuk klasifikasi biner karena berasal dari likelihood Bernoulli; MSE tidak cocok karena menyebabkan isu saturasi gradien pada sigmoid.
- Categorical Cross-Entropy: gunakan `categorical_crossentropy` untuk one-hot label; gunakan `sparse_categorical_crossentropy` untuk label integer (lebih efisien).
- MSE: untuk regresi; sensitif terhadap outlier — pertimbangkan Huber atau MAE bila outlier mengganggu.

Diskusi: menggunakan loss yang salah (mis. MSE untuk klasifikasi) mengubah landscape optimasi dan biasanya memperlambat atau mencegah konvergensi yang baik.

## 3.5 Pemisahan Data: Train / Dev / Test

- Rasio yang digunakan: **80/10/10** (cukup untuk 17.000 sampel: banyak data train, dev/test memadai untuk validasi dan evaluasi final).
- Gunakan `stratify=y` untuk menjaga proporsi kelas pada setiap split.
- Untuk dataset sangat kecil (<500), pertimbangkan stratified K-Fold daripada single split; jika hanya train-val dipakai, test independen tetap direkomendasikan untuk laporan akhir.
- Hindari data leakage dengan memastikan preprocessing (scaler, feature selection) hanya dipelajari dari training set.

## 3.6 Cross-Validation

- K-Fold: robust untuk dataset kecil; model dilatih K kali menggunakan fold berbeda sebagai validation.
- Stratified K-Fold: wajib untuk klasifikasi tak seimbang.
- Penggabungan hasil: rata-rata metrik ± std; atau gunakan model ensemble (averaging/voting) bila ingin meningkatkan performa prediksi.

## 3.7 Batch Size dalam Training

- Full-batch: stabil tapi mahal memori.
- Mini-batch (umum): compromise antara stabilitas & efisiensi (32–256).
- Stochastic (1): sangat noisy.

Dalam proyek ini default `batch_size=128` di `train_cnn.py`. Pengaruh batch size terhadap learning rate: Linear Scaling Rule menyarankan menaikkan LR sebanding dengan kenaikan batch size, tetapi ini bukan aturan pasti — perlu validasi.

## 3.8 Visualisasi Fitur dan Aktivasi Model

Minimal yang harus ada:
- Kurva training vs validation loss per epoch.
- Confusion matrix untuk 17 kelas.

Tambahan berguna:
- MLP: histogram bobot per layer, distribusi aktivasi, feature importance (permutation/SHAP).
- CNN1D: visualisasi feature maps, plot kernel filter, Grad-CAM / saliency untuk interpretabilitas.

Catatan implementasi: `src/visualize.py` sudah menghasilkan training history dan confusion matrix; tambahkan fungsi untuk menyimpan histogram bobot dan feature map bila dibutuhkan.


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

## Artifacts Status (Sinkronisasi)

Ringkasan lokasi artefak yang dirujuk di README:

- Saat ini file model dan laporan utama tersedia di root repository:
  - `trained_dnn.pkl` — ditemukan di repository root
  - `trained_cnn.pkl` — ditemukan di repository root
  - `scaler_dnn.pkl` — ditemukan di repository root
  - `training_dnn_report.txt`, `training_cnn_report.txt` — ditemukan di repository root
  - `training_history_dnn.pkl`, `training_history_cnn.pkl` — ditemukan di repository root
  - `test_split.pkl`, `val_split.pkl`, `train_split.pkl` — ditemukan di repository root

Jika Anda ingin struktur sesuai README (mis. `models/` dan `results/`), jalankan perintah berikut di PowerShell dari root repository untuk memindahkan file:

```powershell
mkdir models
mkdir results
move trained_dnn.pkl models\trained_dnn.pkl
move trained_cnn.pkl models\trained_cnn.pkl
move scaler_dnn.pkl models\scaler_dnn.pkl
move training_dnn_report.txt results\training_dnn_report.txt
move training_cnn_report.txt results\training_cnn_report.txt
move training_history_dnn.pkl results\training_history_dnn.pkl
move training_history_cnn.pkl results\training_history_cnn.pkl
move test_split.pkl results\test_split.pkl
move val_split.pkl results\val_split.pkl
move train_split.pkl results\train_split.pkl
```

Catatan: `Panduan Tugas Besar Kecerdasan Buatan.pdf` menekankan pemahaman konsep, justification, dan visualisasi. Struktur folder `models/` dan `results/` di README ini adalah packaging praktis untuk submission dan bukan instruksi eksplisit dari dokumen PDF.

Atau biarkan file berada di root; pada perintah evaluasi gunakan path relatif yang sesuai, mis:

```powershell
python -m src.evaluate_model --model trained_dnn.pkl
```


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
