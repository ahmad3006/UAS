# Project Status and PDF Requirement Mapping

## Summary
Proyek telah menyelesaikan pipeline dasar untuk dataset pada folder `archive/XPQRS`:
- `explore_data.py`: eksplorasi data dan inspeksi file CSV/.mat
- `src/load_data.py`: pemuatan semua file CSV dan inferensi label dari nama file
- `src/preprocess.py`: ekstraksi fitur numerik dasar dan penyimpanan hasil
- `src/train_model.py`: pelatihan model RandomForest, split train/test, simpan model
- `src/evaluate_model.py`: evaluasi model menggunakan test split dan ringkasan misclassification
- `src/train_dnn.py`: pelatihan MLP DNN pada dataset .mat yang lebih lengkap, simpan model dan split
- `src/dnn_experiments.py`: eksperimen perbandingan raw signal vs engineered time-frequency features

## PDF Requirements Mapping
### Sudah Ada
- Data loading dan eksplorasi awal (`explore_data.py`, `src/load_data.py`)
- Preprocessing dasar dan feature extraction (`src/preprocess.py`)
- Pembagian data train/test dan evaluasi model (`src/train_model.py`, `src/evaluate_model.py`)
- Pencatatan hasil model tersimpan, test split, dan laporan pelatihan sederhana

### Belum atau Perlu Ditingkatkan
- **Deep learning / DNN**: panduan meminta DNN (MLP/CNN 1D/CNN 2D atau varian). Skrip saat ini menggunakan `RandomForest`, bukan jaringan saraf.
- **Model architecture explanation**: belum ada arsitektur neural network, `model.summary()`, atau penjelasan setiap layer.
- **Output activation dan loss**: belum ada penyesuaian aktivasi/output layer khusus (sigmoid/softmax/linear) karena belum menggunakan DNN.
- **Preprocessing yang benar untuk DNN**: belum ada normalisasi/standardisasi sekuensial yang jelas setelah split untuk pelatihan DNN.
- **Train/Dev/Test split**: saat ini hanya `train/test`; belum ada `dev/validation` yang eksplisit.
- **Cross-validation**: belum diterapkan, meskipun PDF menganjurkan terutama untuk dataset kecil.
- **Batch size dan optimizer**: belum ada konfigurasi batch size, optimizer (Adam/SGD), atau early stopping.
- **Visualisasi model internal**: belum ada visualisasi filter/feature map, weight distribution, activation distribution, Grad-CAM, atau saliency.
- **Dokumentasi dan laporan**: belum ada laporan final, README, atau ringkasan eksperimen.
- **Presentasi/YouTube**: belum ada konten presentasi, video, atau deskripsi.

## Next Actions
1. Tambahkan solusi DNN yang sesuai dengan jenis data (misalnya CNN 1D untuk sinyal atau MLP jika fitur tabular).
2. Tambahkan `train/dev/test` split eksplisit dan/atau cross-validation.
3. Implementasikan preprocessing yang sesuai untuk DNN (normalisasi/reshape/time-series handling).
4. Tambahkan visualisasi hasil: kurva loss, confusion matrix, feature importance/aktivasi, dan interpretasi.
5. Buat `README.md` atau laporan terpisah yang menjelaskan:
   - Tujuan proyek
   - Pipeline yang dibangun
   - Model yang dipilih dan alasan
   - Hasil evaluasi
   - Saran perbaikan

## Notes
- Penginstalan PDF parsing berhasil (`PyPDF2`, `pdfplumber`, `pymupdf`).
- Ekstraksi teks PDF berhasil dan disimpan ke `panduan_text.txt`.
- Dokumen ini membantu menunjukkan perbedaan antara implementasi saat ini dan kebutuhan tugas besar.
