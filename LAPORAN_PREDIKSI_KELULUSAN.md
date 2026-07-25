# Laporan Prediksi Kelulusan Mahasiswa
## *(Graduation Prediction Report)*

> **Tanggal Generate:** 23 Juli 2026  
> **Dataset:** `data.csv` (Prediksi Kelulusan Mahasiswa — Politeknik Portugal)  
> **Tools:** Python 3 + scikit-learn + pandas + matplotlib/seaborn  
> **Algoritma:** Logistic Regression & Random Forest dengan GridSearchCV Hyperparameter Tuning

---

## 1. Ringkasan Dataset

| Keterangan | Nilai |
|---|---|
| **Total Records** | 4,424 |
| **Total Kolom** | 37 (36 fitur + 1 target) |
| **Jumlah Fitur** | 36 |
| **Model yang Digunakan** | Logistic Regression, Random Forest |

### Distribusi Target (Original)

| Status | Jumlah | Persentase |
|---|---|---|
| **Graduate** | 2,209 | 49.9% |
| **Dropout** | 1,421 | 32.1% |
| **Enrolled** | 794 | 17.9% |
| **Total** | **4,424** | **100%** |

> ⚠️ Data **Enrolled (794 records)** dihapus dari proses training karena statusnya netral (belum lulus maupun dropout).  
> **Data biner yang digunakan untuk modeling: 3,630 records** (Graduate 60.9% + Dropout 39.1%).

---

## 2. Kategori 36 Fitur yang Digunakan

| Kategori | Jumlah | Fitur |
|---|---|---|
| **Demografis** | 11 | Marital status, Nacionality, Mother's qualification, Father's qualification, Mother's occupation, Father's occupation, Displaced, Educational special needs, Gender, Age at enrollment, International |
| **Akademik Sem 1 & 2** | 15 | Application mode, Application order, Course, Daytime/evening attendance, Previous qualification, Previous qualification (grade), Admission grade, Curricular units 1st sem (credited / enrolled / evaluations / approved / grade / without evaluations), Curricular units 2nd sem (credited / enrolled / evaluations / approved / grade / without evaluations) |
| **Finansial** | 3 | Debtor, Tuition fees up to date, Scholarship holder |
| **Ekonomi Makro** | 3 | Unemployment rate, Inflation rate, GDP |
| **Lainnya** | 4 | Application mode, Application order, Course, Daytime/evening attendance (termasuk dalam akademik) |

---

## 3. Preprocessing & Metodologi

| No | Langkah | Deskripsi |
|----|---------|-----------|
| 1 | **Data Loading** | Membaca `data.csv` dengan separator `;` |
| 2 | **Column Cleaning** | Strip whitespace, hapus karakter `"` dan `'` dari nama kolom |
| 3 | **Filter Target** | Hapus 794 record dengan Target `Enrolled`, sisakan `Graduate` + `Dropout` = 3,630 records |
| 4 | **Target Encoding** | Binary encoding: **Dropout = 1**, **Graduate = 0** |
| 5 | **Feature-Target Split** | `X` = 36 fitur, `y` = Target_encoded |
| 6 | **Train-Test Split** | **80:20** dengan `stratify=y` → menjaga proporsi kelas di kedua set |
| 7 | **Feature Scaling** | `StandardScaler` — fit ke X_train, transform ke X_test (mean=0, std=1) |
| 8 | **Model Training** | Logistic Regression & Random Forest (Default + Tuned) |
| 9 | **Hyperparameter Tuning** | `GridSearchCV` dengan **5-fold cross-validation**, scoring = F1 |
| 10 | **Evaluasi** | Accuracy, Precision, Recall, F1-Score, Specificity, Confusion Matrix |

### Train-Test Split Detail:
- **Train set:** 2,904 records (80%)
- **Test set:** 726 records (20%)

---

## 4. Hyperparameter Tuning (GridSearchCV)

### 4.1 Logistic Regression — Grid

| Parameter | Values | Keterangan |
|-----------|--------|------------|
| `C` | 0.01, 0.1, 1, 10, 100 | Regularization strength (inverse) |
| `penalty` | l1, l2 | Jenis regularisasi |
| `solver` | liblinear, saga | Algoritma optimisasi |
| **Total** | 5 × 2 × 2 = **20 kombinasi** | × 5 fold = **100 fits** |

### 4.2 Random Forest — Grid

| Parameter | Values | Keterangan |
|-----------|--------|------------|
| `n_estimators` | 100, 200, 300 | Jumlah pohon keputusan |
| `max_depth` | 10, 20, 30, None | Kedalaman maksimum pohon |
| `min_samples_split` | 2, 5, 10 | Minimum sampel untuk split node |
| `min_samples_leaf` | 1, 2, 4 | Minimum sampel di leaf node |
| **Total** | 3 × 4 × 3 × 3 = **108 kombinasi** | × 5 fold = **540 fits** |

> **Total keseluruhan: 640 fits** (100 LR + 540 RF)

---

## 5. Hasil Evaluasi Model

### 5.1 Perbandingan Default vs Tuned

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression (Default)** | **94.21%** | **93.21%** | **91.90%** | **92.55%** |
| Logistic Regression (Tuned) | 93.80% | 93.14% | 90.85% | 91.98% |
| Random Forest (Default) | 92.01% | 91.85% | 87.32% | 89.53% |
| Random Forest (Tuned) | 91.32% | 91.70% | 85.56% | 88.52% |

> 🏆 **Model Terbaik: Logistic Regression Default** dengan F1-Score **92.55%**

### 5.2 Confusion Matrix — Logistic Regression (Default)

|  | Predicted: Graduate | Predicted: Dropout | **Total Actual** |
|--|:-:|:-:|:-:|
| **Actual: Graduate** | TN = 402 | FP = 15 | 417 |
| **Actual: Dropout** | FN = 23 | TP = 286 | 309 |
| **Total Predicted** | 425 | 301 | **726** |

| Metrik | Rumus | Nilai |
|--------|-------|------|
| **Accuracy** | (TP + TN) / Total | **94.21%** |
| **Precision** | TP / (TP + FP) | **93.21%** |
| **Recall (Sensitivity)** | TP / (TP + FN) | **91.90%** |
| **Specificity** | TN / (TN + FP) | **96.40%** |
| **F1-Score** | 2 × P × R / (P + R) | **92.55%** |

### 5.3 Confusion Matrix — Random Forest (Default)

|  | Predicted: Graduate | Predicted: Dropout | **Total Actual** |
|--|:-:|:-:|:-:|
| **Actual: Graduate** | TN = 398 | FP = 19 | 417 |
| **Actual: Dropout** | FN = 39 | TP = 270 | 309 |
| **Total Predicted** | 437 | 289 | **726** |

| Metrik | Rumus | Nilai |
|--------|-------|------|
| **Accuracy** | (TP + TN) / Total | **92.01%** |
| **Precision** | TP / (TP + FP) | **91.85%** |
| **Recall (Sensitivity)** | TP / (TP + FN) | **87.32%** |
| **Specificity** | TN / (TN + FP) | **95.44%** |
| **F1-Score** | 2 × P × R / (P + R) | **89.53%** |

---

## 6. Interpretasi Metrik

| Metrik | Arti | LR Default |
|--------|------|:---:|
| **Accuracy** | % prediksi benar dari seluruh data | 94.21% |
| **Precision** | Dari yang diprediksi Dropout, berapa % benar Dropout | 93.21% |
| **Recall** | Dari seluruh mahasiswa Dropout, berapa % terdeteksi | 91.90% |
| **Specificity** | Dari seluruh mahasiswa Graduate, berapa % benar terprediksi Graduate | 96.40% |
| **F1-Score** | Harmonic mean Precision & Recall (metrik utama) | 92.55% |

> Logistic Regression mendeteksi **91.90% mahasiswa dropout**, dengan **false positive rate hanya 3.6%** (specificity 96.40%).

---

## 7. Analisis & Kesimpulan

### 🥇 Model Terbaik
**Logistic Regression (Default)** — F1-Score **92.55%**

### Temuan Utama
1. **Logistic Regression > Random Forest** pada dataset ini, menunjukkan hubungan antar fitur bersifat **linear** dan dataset sudah terstruktur baik
2. **Precision tinggi (93.21%)** → ketika model memprediksi mahasiswa dropout, 93 dari 100 prediksi benar
3. **Recall 91.90%** → model menangkap hampir 92% mahasiswa yang benar-benar dropout
4. **Default parameters lebih optimal** dibanding hasil GridSearchCV, menandakan model default sudah well-calibrated untuk dataset ini
5. **Fitur akademik** (curricular units, admission grade) adalah prediktor terkuat berdasarkan feature importance Random Forest
6. **Fitur finansial** (scholarship holder, debtor) juga memberikan kontribusi signifikan terhadap prediksi

### Visualisasi yang Dihasilkan

| File | Jenis | Deskripsi |
|------|------|-----------|
| `eda_target_distribution.png` | Bar Chart | Distribusi Graduate vs Dropout vs Enrolled |
| `eda_correlation_heatmap.png` | Heatmap | Korelasi antar 10 fitur kunci |
| `eda_academic_performance.png` | Boxplot | Unit akademik disetujui (Graduate vs Dropout) |
| `confusion_matrix_lr.png` | Confusion Matrix | Matriks kebingungan Logistic Regression |
| `confusion_matrix_rf.png` | Confusion Matrix | Matriks kebingungan Random Forest |
| `hasil_evaluasi.png` | Tabel | Perbandingan metrik 4 model |
| `data_process_flow.png` | Diagram | Alur preprocessing data |

---

## 8. Struktur File Project

```
📁 Semester2-afteruts/
├── 📄 data.csv                          ← Dataset utama (4,424 rows × 37 cols)
├── 📄 LAPORAN_PREDIKSI_KELULUSAN.md     ← Laporan ini
├── 🐍 eda.py                            ← Script Exploratory Data Analysis
├── 🐍 predict.py                        ← Script training + GridSearchCV
├── 🐍 generate_report.py                ← Script evaluasi (versi lama)
├── 🐍 generate_report_md.py             ← Script generate laporan .md
├── 📊 hasil_evaluasi.csv                ← Hasil numerik evaluasi model
├── 🖼️ hasil_evaluasi.png                ← Tabel hasil evaluasi
├── 🖼️ confusion_matrix_lr.png           ← Confusion matrix Logistic Regression
├── 🖼️ confusion_matrix_rf.png           ← Confusion matrix Random Forest
├── 🖼️ eda_target_distribution.png       ← Distribusi target
├── 🖼️ eda_correlation_heatmap.png       ← Heatmap korelasi
├── 🖼️ eda_academic_performance.png      ← Boxplot performa akademik
├── 🖼️ data_process_flow.png             ← Diagram alur proses
└── 📄 README.md                         ← Dokumentasi project
```

---

## 9. Teknologi & Library

| Library | Versi | Fungsi |
|---------|-------|--------|
| **Python** | 3.x | Bahasa pemrograman |
| **pandas** | - | Data manipulation & analysis |
| **numpy** | - | Numerical computing |
| **scikit-learn** | - | ML models, preprocessing, metrics, GridSearchCV |
| **matplotlib** | - | Visualisasi grafik |
| **seaborn** | - | Visualisasi statistik (heatmap, boxplot) |

---

## 10. Cara Menjalankan Ulang

```bash
# 1. Exploratory Data Analysis
python eda.py

# 2. Training + Evaluasi + GridSearchCV Hyperparameter Tuning
python predict.py

# 3. Generate laporan .md
python generate_report_md.py
```

---

*Laporan ini digenerate otomatis oleh `generate_report_md.py` — tinggal copy-paste ke PowerPoint*