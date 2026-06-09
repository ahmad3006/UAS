# PANDUAN TEKNIS - XPQRS Signal Classification Project

---

## 1. MODULE DESCRIPTIONS

### 1.1 explore_data.py
**Tujuan:** Eksplorasi initial dataset dan validasi struktur data

**Input:**
- Archive folder dengan file `.mat` dan CSV

**Output:**
- Console output dengan informasi dataset
- Dataset statistics (shape, classes, samples per class)

**Cara Menggunakan:**
```bash
python explore_data.py
```

**Output Contoh:**
```
MAT file: (1000, 100, 17)
  - 1000 signals per class
  - 100 timesteps each
  - 17 classes

CSV files found: 17
  - Each class has its own CSV file
```

---

### 1.2 code/load_data.py
**Tujuan:** Load dataset CSV dan inferensi label dari nama file

**Input:**
- CSV files dalam archive/XPQRS/

**Output:**
- `loaded_dataset.pkl` (dictionary dengan X dan y)

**Usage di Script Lain:**
```python
from code.load_data import load_dataset
X, y, encoder = load_dataset()
```

---

### 1.3 code/preprocess.py
**Tujuan:** Preprocessing dan feature extraction

**Input:**
- `loaded_dataset.pkl`

**Output:**
- `processed_features.pkl` (list of dictionaries)
- `processed_features.csv` (CSV format)

**Features yang Diekstrak:**
- Mean, Std, Min, Max, Median, Range
- Skewness, Kurtosis (statistical features)

**Cara Menggunakan:**
```bash
python code/preprocess.py
```

---

### 1.4 code/train_model.py
**Tujuan:** Training baseline Random Forest model

**Input:**
- `processed_features.pkl`

**Output:**
- `trained_model.pkl` (Random Forest model)
- `test_split.pkl` (test data)
- `training_report.txt` (accuracy report)

**Hyperparameter:**
```python
RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=20,
    random_state=42
)
```

---

### 1.5 code/train_dnn.py
**Tujuan:** Training MLP Neural Network pada raw signal atau engineered features

**Input:**
- `archive/XPQRS/5Kfs_1Cycle_50f_1000Sam_1A.mat`

**Output:**
- `trained_dnn.pkl` (MLP model + scaler + encoder)
- `training_dnn_report.txt` (accuracy & classification report)
- `test_split.pkl`, `val_split.pkl`, `train_split.pkl`
- `scaler_dnn.pkl` (StandardScaler)

**Options:**
```bash
# Raw signal input (100 features)
python code/train_dnn.py

# Engineered features input (28 features)
python code/train_dnn.py --feature-set
```

**Model Architecture:**
```
Input Layer: 100 atau 28 features
↓
Dense(256) + ReLU
↓
Dense(128) + ReLU
↓
Dense(64) + ReLU
↓
Output(17) + Softmax

Optimizer: Adam (lr default=0.001)
Loss: Cross-entropy
Early Stopping: Yes (patience=15)
```

---

### 1.6 code/train_cnn.py
**Tujuan:** Training 1D CNN model untuk signal classification

**Input:**
- `archive/XPQRS/5Kfs_1Cycle_50f_1000Sam_1A.mat`

**Output:**
- `trained_cnn.pkl` (CNN model state + scaler + encoder)
- `training_cnn_report.txt` (report & confusion matrix)
- `best_cnn_model.pth` (best model checkpoint)
- `training_history_cnn.pkl` (loss & accuracy per epoch)

**Options:**
```bash
python code/train_cnn.py --epochs 50 --batch-size 128 --lr 1e-3 --patience 10
```

**Model Architecture:**
```
Input: (batch, 1, 100) - 100 timesteps
↓
Conv1D(32, kernel=3) + ReLU
↓
MaxPool1D(2)
↓
Conv1D(64, kernel=3) + ReLU
↓
MaxPool1D(2)
↓
Conv1D(128, kernel=3) + ReLU
↓
MaxPool1D(2)
↓
Flatten
↓
Dense(256) + ReLU + Dropout(0.5)
↓
Dense(128) + ReLU + Dropout(0.5)
↓
Output(17) + Softmax
```

**Device:** Automatic (GPU jika available, CPU fallback)

---

### 1.7 code/evaluate_model.py
**Tujuan:** Evaluasi model terlatih pada test set

**Input:**
- Model pickle file (--model)
- Test split data

**Output:**
- Console output: Accuracy, Classification Report, Confusion Matrix

**Usage:**
```bash
# Evaluate DNN model
python code/evaluate_model.py --model trained_dnn.pkl

# Evaluate custom model
python code/evaluate_model.py --model models/trained_dnn.pkl

# Evaluate on full dataset
python code/evaluate_model.py --use-full
```

---

### 1.8 code/dnn_experiments.py
**Tujuan:** Eksperimen perbandingan MLP dengan raw signal vs engineered features

**Input:**
- `archive/XPQRS/5Kfs_1Cycle_50f_1000Sam_1A.mat`

**Output:**
- Console output dengan comparison results
- `dnn_experiments_report.txt`

**Cara Menggunakan:**
```bash
python code/dnn_experiments.py
```

**Output Format:**
```
Experiment: raw signal
Validation accuracy: 0.5941
Test accuracy: 0.5412
...

Experiment: engineered features
Validation accuracy: [TBD]
Test accuracy: [TBD]
...
```

---

### 1.9 code/hyperparameter_tuning.py
**Tujuan:** Grid search untuk mencari hyperparameter optimal

**Input:**
- `archive/XPQRS/5Kfs_1Cycle_50f_1000Sam_1A.mat`

**Output:**
- Console output dengan tuning results
- `hyperparameter_tuning_results.txt` (top 10 configurations)

**Parameter yang di-tune:**

MLP:
- hidden_layer_sizes: [(128,64), (256,128,64), (256,128,64,32), (512,256,128)]
- learning_rate (alpha): [1e-4, 1e-3, 1e-2]
- batch_size: [64, 128, 256]

CNN:
- kernel_size: [3, 5]
- dropout: [0.3, 0.5]
- learning_rate: [1e-4, 1e-3]

**Cara Menggunakan:**
```bash
python code/hyperparameter_tuning.py
```

**Warning:** Proses ini memakan waktu lama (~1-2 jam)

---

### 1.10 code/analyze_errors.py
**Tujuan:** Analisis kesalahan model secara detail

**Input:**
- `trained_dnn.pkl` (model)
- `test_split.pkl` (test data)

**Output:**
- Console output dengan detailed error analysis
- `error_analysis_report.txt`

**Analysis Details:**
- Per-class accuracy
- Top 10 hardest-to-distinguish pairs
- Classes dengan paling banyak error
- Detailed confusion matrix
- Rekomendasi improvement

**Cara Menggunakan:**
```bash
python code/analyze_errors.py
```

**Output Contoh:**
```
PER-CLASS ACCURACY:
Pure Sinusoidal:     95.23% ✓ GOOD
Sag:                 87.65% ✓ GOOD
Swell:               82.11% ⚠ MEDIUM
Harmonics:           45.32% ✗ BAD
...

TOP 10 HARDEST-TO-DISTINGUISH CLASS PAIRS:
Pure Sinusoidal   → Harmonics   (234 times, 45.2%)
Sag               → Sag+Harm    (178 times, 34.6%)
...
```

---

### 1.11 code/visualize.py
**Tujuan:** Visualisasi sinyal, FFT, training history, confusion matrix

**Input:**
- `archive/XPQRS/5Kfs_1Cycle_50f_1000Sam_1A.mat` (untuk signal plotting)
- `trained_dnn.pkl` (untuk confusion matrix)
- `training_history_cnn.pkl` (untuk training history plot)

**Output:**
- Multiple matplotlib figures (interactive display)

**Plot Types:**
1. Raw waveforms untuk beberapa kelas
2. FFT magnitude spectrum
3. Time-domain vs Frequency-domain comparison
4. Class statistics (mean & std)
5. Training loss & validation accuracy (jika CNN model ada)
6. Confusion matrix heatmap

**Cara Menggunakan:**
```bash
python code/visualize.py
```

---

## 2. DATA FORMAT SPECIFICATIONS

### 2.1 MAT File Structure
```
5Kfs_1Cycle_50f_1000Sam_1A.mat
  ├── Out: (1000, 100, 17) ndarray
  │   ├── Dimension 1: 1000 signals per class
  │   ├── Dimension 2: 100 timesteps per signal
  │   └── Dimension 3: 17 signal classes
  
Data type: float64
Value range: typically [-1.0, 1.0]
Sampling rate: 5 kHz
Duration: 100 samples / 5000 Hz = 0.02 seconds (20 ms)
```

### 2.2 CSV Format
```
filename: Pure_Sinusoidal.csv, Sag.csv, etc.
columns: [mean, std, min, max, median, range, skewness, kurtosis] + feature_1...feature_N
```

### 2.3 Model Pickle Format
```python
# trained_dnn.pkl structure
{
    "model": MLPClassifier object,
    "scaler": StandardScaler object,
    "label_encoder": LabelEncoder object
}

# trained_cnn.pkl structure
{
    "model_state": torch.state_dict,
    "scaler": StandardScaler object,
    "label_encoder": LabelEncoder object
}

# test_split.pkl structure
{
    "X_test": ndarray (1700, 100),
    "y_test": list of strings (class names)
}
```

---

## 3. PREPROCESSING PIPELINE

### 3.1 Data Split Strategy
```
Total: 17,000 samples (1000 per class)
  ├── Train: 13,600 (80%)
  ├── Val: 1,700 (10%)
  └── Test: 1,700 (10%)

Split method: Stratified train_test_split
Random state: 42 (for reproducibility)
```

### 3.2 Scaling Strategy
```python
# StandardScaler fitted on training set only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Prevents data leakage
```

### 3.3 Feature Engineering
```python
# Time-domain features
mean = X.mean(axis=1)
std = X.std(axis=1)
min_ = X.min(axis=1)
max_ = X.max(axis=1)
median = np.median(X, axis=1)
range_ = max_ - min_
skewness = scipy.stats.skew(X, axis=1)
kurtosis = scipy.stats.kurtosis(X, axis=1)

# Frequency-domain features
fft = np.abs(np.fft.rfft(X, axis=1))[:, :20]  # First 20 bins
fft_log = np.log1p(fft)  # Log scaling

# Total: 8 + 20 = 28 features
```

---

## 4. MODEL COMPARISON

| Aspect | Random Forest | MLP Raw | MLP Features | CNN 1D |
|--------|---|---|---|---|
| Input Shape | (n, 8-20) | (n, 100) | (n, 28) | (n, 1, 100) |
| Interpretability | High | Low | Medium | Medium |
| Training Time | Fast | Medium | Medium | Slow |
| GPU Required | No | No | No | Yes (optional) |
| Memory Usage | Low | Low | Low | High |
| Best For | Quick baseline | Speed | Feature richness | Time-series patterns |

---

## 5. TROUBLESHOOTING

### Issue: Memory Error saat Training
**Solution:**
- Reduce batch_size: `--batch-size 64`
- Reduce hidden layers: Edit model architecture
- Use validation_fraction lebih kecil

### Issue: Poor Model Performance (Low Accuracy)
**Solutions:**
- Run hyperparameter_tuning.py untuk find optimal params
- Gunakan engineered features: `--feature-set`
- Try CNN 1D: `train_cnn.py`
- Increase training epochs

### Issue: GPU Not Detected (CNN)
**Solution:**
- Otomatis fallback ke CPU
- Check: `python -c "import torch; print(torch.cuda.is_available())"`

### Issue: Model Not Found Error
**Solution:**
- Ensure model training selesai terlebih dahulu
- Check file path: `trained_dnn.pkl` harus di root directory

---

## 6. API REFERENCE

### Loading Trained Model
```python
import pickle
from pathlib import Path

model_path = Path("trained_dnn.pkl")
with model_path.open("rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
scaler = model_data["scaler"]
encoder = model_data["label_encoder"]

# Predict on new data
X_new_scaled = scaler.transform(X_new)
y_pred_encoded = model.predict(X_new_scaled)
y_pred_names = encoder.inverse_transform(y_pred_encoded)
```

### Loading CNN Model (PyTorch)
```python
import torch
import pickle

model_path = Path("trained_cnn.pkl")
with model_path.open("rb") as f:
    model_data = pickle.load(f)

model_state = model_data["model_state"]
scaler = model_data["scaler"]
encoder = model_data["label_encoder"]

# Load model (require defining model architecture first)
# ... [define CNN1D class] ...
model = CNN1D()
model.load_state_dict(model_state)
model.eval()
```

---

## 7. PERFORMANCE METRICS

### Classification Metrics
```
Accuracy = TP + TN / Total
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
```

### Confusion Matrix
- Rows: True labels
- Columns: Predicted labels
- Diagonal: Correct predictions
- Off-diagonal: Misclassifications

---

**Last Updated:** Juni 2026
