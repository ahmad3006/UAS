# SUBMISSION - XPQRS Signal Classification with Deep Learning

**Project:** Klasifikasi 17 Gangguan Sinyal Sistem Tenaga Listrik (XPQRS) menggunakan Machine Learning & Deep Learning  
**Subject:** Kecerdasan Buatan (Artificial Intelligence)  
**Dataset:** 17.000 sampel (1.000 sampel per kelas untuk 17 kelas)
**Date:** Juni 2026

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
cd submission
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Menjalankan Pipeline

**A. Eksplorasi Data:**
```bash
python code/explore_data.py
```

**B. Training Models:**

Option 1 - MLP dengan Raw Signal:
```bash
python code/train_dnn.py
```

Option 2 - MLP dengan Engineered Features:
```bash
python code/train_dnn.py --feature-set
```

Option 3 - CNN 1D:
```bash
python code/train_cnn.py --epochs 50 --batch-size 128
```

**C. Eksperimen & Tuning:**

Perbandingan Raw vs Engineered Features:
```bash
python code/dnn_experiments.py
```

Hyperparameter Tuning:
```bash
python code/hyperparameter_tuning.py
```

**D. Evaluasi & Analisis:**

Evaluasi model terlatih:
```bash
python code/evaluate_model.py --model models/trained_dnn.pkl
```

Analisis error & confusion matrix:
```bash
python code/analyze_errors.py
```

Visualisasi hasil:
```bash
python code/visualize.py
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

- **Apakah ada missing values?**

   Dataset XPQRS yang digunakan (file MAT pada `archive/XPQRS`) tidak mengandung missing values eksplisit pada level sampel sinyal; namun modul `src/preprocess.py` mengabaikan (skip) record yang tidak menghasilkan sekumpulan nilai numerik valid saat parsing. Implementasi saat ini:

   - Jika sinyal tidak berisi angka yang dapat diparsing, record tersebut dilewati (dropped) pada tahap ekstraksi fitur (`ensure_float_sequence` + `extract_features`).
   - Alasan: sinyal kosong biasanya menandakan korupsi atau format yang tidak sesuai; meng-drop lebih aman dibanding meng-impute, karena imputasi pada urutan waktu dapat merusak pola temporal.

- **Normalisasi / Standardisasi fitur**

   Pipeline training menggunakan `StandardScaler` untuk mentransformasikan fitur numerik sebelum pelatihan:

   - Contoh (dari `src/train_dnn.py` / `src/train_cnn.py`):

      ```python
      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_train)
      X_val = scaler.transform(X_val)
      X_test = scaler.transform(X_test)
      ```

   - Metode dipilih: `StandardScaler` (mean=0, std=1). Alasan: banyak model (MLP, optimizers, dan gradient-based training) sensitif terhadap skala fitur sehingga standardisasi biasanya mempercepat konvergensi dan menstabilkan training.
   - Risiko jika tidak dilakukan: fitur dengan skala besar dapat mendominasi gradien, menyebabkan training lambat atau tidak stabil; untuk CNN 1D pada raw signal, scaling juga membantu agar bobot konvolusi terkontrol.

- **Fitur kategorikal dan encoding**

   - Target label adalah kategorikal (17 kelas). Di seluruh pipeline label dikonversi menggunakan `LabelEncoder`:

      ```python
      encoder = LabelEncoder()
      y_encoded = encoder.fit_transform(y)
      ```

   - Perbedaan singkat: `LabelEncoder` memberi integer per kelas (0..K-1) — cocok untuk API scikit-learn (`MLPClassifier`) dan untuk penggunaan loss categorical (softmax + CrossEntropy). `One-hot encoding` membuat representasi vektor biner panjang K — berguna jika model eksplisit memerlukan vektor target (mis. Keras dengan to_categorical) atau untuk feature kategorikal input.

   - Pada pipeline ini, `LabelEncoder` digunakan untuk target karena training API (sklearn/PyTorch) menerima label berbentuk integer dan internal loss function mengharapkan integer class index.

- **Reshape untuk CNN 1D**

   - Untuk `CNN 1D`, data time-series direpresentasikan sebagai array 2D `(n_samples, timesteps)` lalu diubah menjadi tensor PyTorch 3D dengan channel tunggal:

      ```python
      # di train_cnn.py
      torch.from_numpy(X_train).unsqueeze(1).float()
      # hasil shape -> (n_samples, 1, timesteps)
      ```

   - Penjelasan: `unsqueeze(1)` menambahkan dimensi channel sehingga input memenuhi format `(batch, channels, sequence_length)` yang diharapkan oleh `nn.Conv1d`.

- **Catatan untuk CNN 2D (tidak digunakan di repo ini)**

   - Repo saat ini tidak mengimplementasikan CNN 2D. Jika menggunakan CNN 2D untuk citra, praktik umum yang direkomendasikan:
      - Resize citra ke ukuran input konsisten (mis. 224x224)
      - Normalisasi nilai pixel ke rentang [0, 1] atau dengan mean/std dataset
      - Konversi ke grayscale jika warna tidak diperlukan (mengurangi channel dari 3 ke 1)

- **Keseimbangan kelas (class balance)**

   - Dataset XPQRS di repo ini berimbang per desain: 1.000 sampel per kelas (17 kelas → 17.000 sampel). Oleh karena itu langkah oversampling tidak diperlukan.
   - Split train/val/test dilakukan dengan `stratify=y` untuk mempertahankan proporsi kelas pada setiap split (lihat `src/train_cnn.py` dan `src/train_dnn.py` yang menggunakan `train_test_split(..., stratify=y, ...)`).
   - Jika dataset tidak seimbang, opsi mitigasi yang tersedia: oversampling (SMOTE), undersampling, atau menggunakan `class_weight` di loss function (PyTorch `CrossEntropyLoss(weight=...)` atau scikit-learn `class_weight='balanced'`).

---

Ringkasan: implementasi kode saat ini sudah menerapkan standardisasi (`StandardScaler`), encoding label (`LabelEncoder`), dan reshape untuk `CNN 1D`. Preprocessing mem-drop sample yang tidak mengandung data numerik dan split menggunakan `stratify` sehingga keseimbangan kelas tetap terjaga.

## 📝 Dokumentasi Lengkap

1. **LAPORAN_TUGAS_BESAR.md** - Laporan akademik lengkap
   - Latar belakang & metodologi
   - Deskripsi dataset & preprocessing
   - Arsitektur model & implementasi
   - Hasil & evaluasi
   - Analisis & diskusi

2. **README.md** - User guide singkat
   - Isi proyek & struktur file
   - Cara menjalankan scripts
   - Output penting & catatan

3. **PANDUAN_TEKNIS.md** - Technical documentation
   - Penjelasan setiap module
   - Input/output format
   - API documentation

---

## 📐 Detail Arsitektur & Matematika

Bagian ini menyajikan ringkasan matematis untuk layer utama yang digunakan dalam implementasi MLP dan CNN 1D sesuai Panduan Tugas Besar.

- **Dense / Fully-Connected Layer (MLP)**

   Setiap neuron pada layer dense menghitung output sebagai fungsi afine diikuti aktivasi:

   $$z = W \cdot x + b$$
   $$y = \phi(z)$$

   di mana $x$ adalah vektor input, $W$ matriks bobot, $b$ bias, dan $\phi$ adalah fungsi aktivasi (contoh: ReLU atau softmax untuk layer akhir).

- **1D Convolution (Conv1D)**

   Conv1D menerapkan kernel (filter) pada sinyal menggunakan konsep *sliding window* di sepanjang dimensi waktu. Untuk filter $k$ berukuran $m$, output pada posisi $t$ dinyatakan sebagai:

   $$y[t] = \sum_{i=0}^{m-1} w[i] \cdot x[t + i] + b$$

   Diimplementasikan berulang dengan pergeseran stride (biasanya stride=1) sehingga pola lokal temporal dapat dideteksi.

- **Pooling (dimensionality reduction)**

   Pooling seperti max-pooling merangkum nilai dalam jendela lokal untuk mengurangi resolusi temporal dan menekan variansi:

   $$y_{pool}[j] = \max_{i\in W_j} x[i]$$

   di mana $W_j$ adalah indeks elemen di jendela pooling ke-$j$. Pooling mengurangi panjang urutan (sequence length) dan membantu mendapatkan representasi yang lebih ringkas.

Referensi implementasi matematika juga tercermin langsung pada `src/train_dnn.py` (MLP) dan `src/train_cnn.py` (Conv1D + pooling).

---

## 🔍 Metodologi Visualisasi (Update)

`src/visualize.py` sekarang mendukung studi komparatif antara `CNN 1D` dan `MLP` (MLP juga disebut DNN dalam codebase). Fitur utama:

- Plot training history untuk kedua model (loss & accuracy per epoch):
   - MLP: membaca `training_history_dnn.pkl` (keys: `train_loss`, `val_loss`, `train_acc`, `val_acc`).
   - CNN: membaca `training_history_cnn.pkl` (keys: `train_losses`, `val_losses`, `train_accs`, `val_accs`).
- Confusion matrix untuk 17 kelas gangguan dari model yang tersimpan (`trained_dnn.pkl` dan `trained_cnn.pkl`).
- Visualisasi perbandingan waktu-domain vs frekuensi-domain, statistik kelas, dan waveform sampling.

Untuk menjalankan visualisasi:

```bash
python code/visualize.py
```

Hasil plot disimpan di folder `visualizations/` dan juga ditampilkan interaktif.

## 🛠️ Teknologi & Library

- **Python 3.14.3**
- **Machine Learning:** scikit-learn, numpy, scipy
- **Deep Learning:** PyTorch (CNN)
- **Data Processing:** pandas, scikit-learn
- **Visualization:** matplotlib
- **PDF Parsing:** PyPDF2, pdfplumber, pymupdf

---

## 📋 Persyaratan PDF

Semua persyaratan dari "Panduan Tugas Besar Kecerdasan Buatan.pdf" telah diimplementasikan:

✓ Data Loading & Eksplorasi  
✓ Preprocessing & Feature Extraction  
✓ Train/Dev/Test Split (80/10/10)  
✓ Machine Learning Models (Random Forest)  
✓ Deep Learning Models (MLP, CNN 1D)  
✓ Model Evaluation & Metrics  
✓ Hyperparameter Tuning  
✓ Error Analysis  
✓ Visualisasi & Reporting  
✓ Dokumentasi Lengkap  

---

## 🔧 Customization

### Mengubah Hyperparameter

**MLP Neural Network** (`code/train_dnn.py`):
```python
# Edit build_model() function
hidden_layer_sizes=(256, 128, 64),  # Ubah jumlah & ukuran layer
learning_rate = 1e-3,               # Ubah learning rate
batch_size = 128,                   # Ubah batch size
```

**CNN 1D** (`code/train_cnn.py`):
```bash
python code/train_cnn.py --epochs 100 --batch-size 256 --lr 5e-4
```

### Data Augmentation

Edit `code/train_dnn.py` atau `code/train_cnn.py` untuk menambahkan:
- Signal shifting
- Noise injection
- Amplitude scaling
- Time warping

---

## 📞 Support

Untuk pertanyaan atau issue:
1. Lihat error message & check `code/` untuk debug
2. Lihat documentation di `documentation/`
3. Jalankan `explore_data.py` untuk dataset info

---

## 📜 License & Citation

Proyek ini untuk keperluan akademik Mata Kuliah Kecerdasan Buatan.  
Dataset XPQRS: Archive/XPQRS folder

---

**Created:** Juni 2026  
**Last Updated:** 2026-06-18
