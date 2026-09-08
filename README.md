# Prediksi Penyakit Jantung (Heart Disease Prediction)

Proyek **Machine Learning end-to-end** untuk memprediksi risiko seseorang terkena
penyakit jantung (klasifikasi biner: *berisiko* / *tidak berisiko*) berdasarkan
fitur medis dan gaya hidup. Dikembangkan dengan **Python + Scikit-Learn**.

---

## 1. Ringkasan Masalah

- **Tipe masalah**: Klasifikasi biner (_supervised learning_).
- **Dataset**: UCI Heart Disease (Cleveland) — 303 pasien, 13 fitur, 1 target.
- **Target**: `0` = sehat, `1` = berpenyakit jantung (nilai 1–4 pada data mentah
  dibinerkan menjadi 1).
- **Konteks medis**: metrik paling penting adalah **Recall (sensitivitas)** —
  *false negative* (pasien jantung dianggap sehat) jauh lebih berbahaya daripada
  *false positive*.

## 2. Hasil Utama

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (baseline) | 0.839 | 0.877 | 0.766 | 0.814 | **0.907** |
| Decision Tree | 0.735 | 0.750 | 0.639 | 0.687 | 0.738 |
| Random Forest (tuned) | 0.810 | 0.812 | 0.774 | 0.787 | 0.894 |
| KNN | 0.822 | 0.844 | 0.756 | 0.795 | 0.881 |
| SVM (RBF) | 0.835 | 0.858 | 0.783 | 0.813 | 0.886 |
| XGBoost | 0.781 | 0.766 | 0.757 | 0.760 | 0.856 |

*Nilai pada tabel = rata-rata 5-fold cross-validation.*

**Model final: Random Forest hasil GridSearchCV**
> pada test set (20% data): **Accuracy 0.885**, **Precision 0.839**,
> **Recall 0.929**, **F1 0.881**, **ROC-AUC 0.952**.

Fitur paling berpengaruh: **oldpeak** (depresi segmen ST), **cp** (tipe nyeri
dada), **thal** (hasil tes thalassemia), **thalach** (denyut max) dan **slope**.

## 3. Fitur Dataset

| Fitur | Keterangan | Tipe |
|---|---|---|
| age | Usia (tahun) | numerik |
| sex | 1 = pria, 0 = wanita | biner |
| cp | Tipe nyeri dada (1–4) | kategorikal |
| trestbps | Tekanan darah istirahat (mm Hg) | numerik |
| chol | Kolesterol serum (mg/dl) | numerik |
| fbs | Gula darah puasa > 120 mg/dl (1/0) | biner |
| restecg | Hasil EKG istirahat (0–2) | kategorikal |
| thalach | Denyut jantung maksimum tercapai | numerik |
| exang | Angina akibat olahraga (1/0) | biner |
| oldpeak | Depresi segmen ST saat olahraga | numerik |
| slope | Kemiringan puncak ST (1–3) | kategorikal |
| ca | Jumlah pembuluh darah utama (0–3) | diskret |
| thal | Hasil tes thalassemia (3/6/7) | kategorikal |

## 4. Struktur Proyek

```
prediksi-jantung/
├── README.md                  ← Dokumen ini
├── requirements.txt           ← Dependensi
├── data/
│   ├── heart.csv              ← Dataset mentah (UCI Cleveland)
│   └── sample_patients.csv    ← Contoh data pasien untuk prediksi batch
├── app.py                     ← Web app Streamlit (lihat bagian 5.5)
├── notebooks/
│   ├── 01_EDA.ipynb           ← Eksplorasi data (sudah tereksekusi)
│   └── 02_Modeling.ipynb      ← Pemodelan & evaluasi (sudah tereksekusi)
├── src/
│   ├── utils.py               ← Path, gaya visual, penyimpanan figure
│   ├── preprocessing.py       ← Cleaning, encoding, scaling, pipeline
│   ├── train.py               ← Training end-to-end (CV, grid search, eval)
│   └── predict.py             ← Inferensi (single/batch)
├── models/                    ← Artefak model tersimpan (joblib)
│   ├── heart_disease_model.pkl
│   ├── preprocessor.pkl
│   ├── model_comparison.csv
│   └── model_info.json
└── reports/figures/           ← Visualisasi (PNG)
```

## 5. Cara Menjalankan

### 5.1 Instalasi dependensi

```bash
pip install -r requirements.txt
```

> Catatan: versi minimum Python 3.10 (direkomendasikan 3.11+).

### 5.2 Menjalankan pipeline training (hasil tersimpan otomatis)

```bash
python -m src.train
```

Menghasilkan:
- `models/heart_disease_model.pkl` — model final terbaik
- `models/preprocessor.pkl` — pipeline preprocessing yang sudah fit
- `models/model_comparison.csv` — tabel metrik 6 model
- `models/model_info.json` — metadata model
- `reports/figures/*.png` — confusion matrix, ROC, feature importance, dll.

### 5.3 Prediksi baru

**Mode tunggal (pasien contoh):**
```bash
python -m src.predict
```

**Mode batch dari file CSV:**
```bash
python -m src.predict --csv data/sample_patients.csv
```

Format CSV batch: 13 kolom tanpa target
(`age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal`).

### 5.4 Menjalankan notebook

```bash
jupyter notebook notebooks/01_EDA.ipynb
jupyter notebook notebooks/02_Modeling.ipynb
```

Notebook sudah tereksekusi jadi bisa langsung dibaca tanpa menjalankan ulang.

### 5.5 Menjalankan Web App (Streamlit)

Model juga bisa diakses lewat **website** berbasis **Streamlit**:

```bash
python -m streamlit run app.py
```

> **Catatan Windows:** jika `streamlit` tidak dikenali sebagai perintah, jalankan dengan
> prefix `python -m` seperti di atas. (Ini terjadi karena folder `Scripts` Python belum masuk PATH.)

- Buka `http://localhost:8501` di browser.
- **Tab "Prediksi Perorangan"**: isi 13 fitur pasien → klik "Prediksi Risiko" →
  muncul label, probabilitas, serta **threshold** yang bisa diatur (makin rendah
  threshold, makin tinggi recall — sesuai konteks medis).
- **Tab "Prediksi Batch (CSV)"**: unggah CSV banyak pasien sekaligus.
- **Tab "Evaluasi Model"**: ringkasan model final + perbandingan 6 model + kurva ROC
  dan feature importance.

> **Hosting gratis di cloud:** Streamlit Community Cloud maupun Hugging Face Spaces
> menerima repo ini langsung (`streamlit run app.py`). Untuk data pasien nyata,
> pastikan memakai server sendiri karena data medis bersifat sensitif.

## 6. Alur Kerja (Pipeline)

1. **EDA** → struktur data, missing values (hanya `ca` & `thal` = 6 nilai),
   distribusi & korelasi, pengecekan imbalance (kelas hampir seimbang 54:46).
2. **Preprocessing** → `?` → NaN → imputasi (median/modus) → One-Hot Encoding
   untuk fitur kategorikal → StandardScaler untuk fitur numerik
   (dibungkus `ColumnTransformer` agar **tidak ada data leakage**).
3. **Modeling** → 6 model dibandingkan dengan *stratified 5-fold CV*:
   Logistic Regression (baseline), Decision Tree, Random Forest, KNN, SVM, XGBoost.
4. **Tuning** → GridSearchCV (scoring `roc_auc`) pada Random Forest & XGBoost.
5. **Evaluasi** → test set: accuracy, precision, recall, F1, ROC-AUC +
   confusion matrix + kurva ROC semua model.
6. **Interpretasi** → feature importance model pohon.

## 7. Insight Utama & Rekomendasi

1. **Model terbaik: Random Forest** (AUC test 0.952). Logistic Regression sanggup
   bersaing (AUC 0.907) → berguna sebagai model sederhana yang mudah dijelaskan.
2. **Recall > accuracy** di konteks medis: lebih baik salah alarm daripada
   melewatkan pasien jantung. Recall model final 0.929.
3. **Fitur penentu**: oldpeak, cp, thal, thalach, slope. Sebaliknya `fbs` dan
   `chol` berpengaruh kecil di dataset ini.
4. **Keterbatasan**: dataset kecil (303 baris, 61 di test). Untuk penerapan nyata,
   gunakan dataset lebih besar dan validasi eksternal. Ini adalah *screening tool*,
   **bukan pengganti diagnosis dokter**.

## 8. Tech Stack

- Python 3.13, pandas, numpy
- Scikit-Learn (utamakan): pipeline, CV, GridSearchCV
- XGBoost
- Matplotlib & Seaborn (visualisasi)
- Joblib (persistensi model)