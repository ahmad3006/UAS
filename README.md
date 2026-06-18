# UAS KECEBUT — XPQRS Signal Classification

Proyek tugas besar **Kecerdasan Buatan**: klasifikasi 17 gangguan sinyal listrik (dataset XPQRS).

## Laporan

Laporan lengkap BAB 1–6: **[LAPORAN_TUGAS_BESAR.md](LAPORAN_TUGAS_BESAR.md)**  
Notebook interaktif: **[Project_Documentation.ipynb](Project_Documentation.ipynb)**

## Quick Start

```powershell
pip install -r requirements.txt
python -m src.load_data
python -m src.preprocess
python -m src.train_model
python -m src.train_dnn
python -m src.train_cnn --epochs 50 --batch-size 128
python -m src.visualize
```

## Output

| Folder | Isi |
|--------|-----|
| `models/` | Model terlatih (.pkl) |
| `results/` | Laporan training & data split |
| `visualizations/` | Gambar PNG hasil visualisasi |
