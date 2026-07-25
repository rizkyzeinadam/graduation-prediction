# Prediksi Kelulusan Mahasiswa — Graduation Prediction

> **Dataset:** Prediksi Kelulusan Mahasiswa Politeknik Portugal  
> **Algoritma:** Logistic Regression & Random Forest dengan GridSearchCV Hyperparameter Tuning  
> **Tools:** Python 3 + scikit-learn + pandas + matplotlib + seaborn

---

## Deskripsi Project

Project ini membangun model machine learning untuk memprediksi status kelulusan mahasiswa (**Graduate** vs **Dropout**) menggunakan 36 fitur dari data akademik, demografis, finansial, dan ekonomi makro. Dua algoritma klasifikasi digunakan: **Logistic Regression** sebagai model baseline dan **Random Forest** sebagai model ensemble, kemudian dioptimasi dengan **GridSearchCV 5-fold cross-validation**.

---

## Dataset

| Keterangan | Nilai |
|---|---|
| Total Records | 4,424 |
| Total Fitur | 36 |
| Kolom Target | Target (Graduate, Dropout, Enrolled) |
| Data Training | 3,630 (Enrolled dihapus) |
| Graduate (lulus) | 2,209 (60.9% dari data biner) |
| Dropout (tidak lulus) | 1,421 (39.1% dari data biner) |

---

## Algoritma yang Digunakan

### Logistic Regression (Baseline)
Algoritma klasifikasi linear yang sederhana, cepat dilatih, dan mudah diinterpretasikan. Digunakan sebagai acuan awal performa minimum dataset.

### Random Forest (Pembanding)
Algoritma ensemble learning berbasis pohon keputusan yang mampu menangkap hubungan non-linear antar fitur dan tahan terhadap overfitting.

### GridSearchCV Hyperparameter Tuning
Kedua model di-tuning menggunakan **GridSearchCV 5-fold cross-validation** untuk mencari kombinasi hyperparameter optimal.

| Model | Parameter Grid | Total Fits |
|-------|---------------|:---:|
| **Logistic Regression** | C (5) × penalty (2) × solver (2) | **100** |
| **Random Forest** | n_estimators (3) × max_depth (4) × min_samples_split (3) × min_samples_leaf (3) | **540** |
| **Total** | | **640 fits** |

---

## Preprocessing Pipeline

| No | Langkah | Deskripsi |
|:--:|---------|-----------|
| 1 | Data Loading | Membaca `data.csv` (separator `;`) |
| 2 | Column Cleaning | Strip whitespace + hapus `"` dan `'` dari nama kolom |
| 3 | Filter Target | Hapus 794 record `Enrolled`, sisakan Graduate + Dropout |
| 4 | Target Encoding | Dropout = 1, Graduate = 0 |
| 5 | Train-Test Split | 80:20 stratify=y (2,904 train / 726 test) |
| 6 | StandardScaler | Fit ke X_train, transform ke X_test |
| 7 | Training | LR & RF (Default + Tuned via GridSearchCV) |
| 8 | Evaluasi | Accuracy, Precision, Recall, F1-Score, Specificity, Confusion Matrix |

---

## Hasil Evaluasi Model

### Perbandingan Default vs Tuned

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression (Default)** | **94.21%** | **93.21%** | **91.90%** | **92.55%** |
| Logistic Regression (Tuned) | 93.80% | 93.14% | 90.85% | 91.98% |
| Random Forest (Default) | 92.01% | 91.85% | 87.32% | 89.53% |
| Random Forest (Tuned) | 91.32% | 91.70% | 85.56% | 88.52% |

> 🏆 **Model Terbaik: Logistic Regression Default** — F1-Score **92.55%**

### Confusion Matrix — Logistic Regression (Default)

|  | Predicted Graduate | Predicted Dropout |
|--|:-:|:-:|
| **Actual Graduate** | TN = 402 | FP = 15 |
| **Actual Dropout** | FN = 23 | TP = 286 |

### Confusion Matrix — Random Forest (Default)

|  | Predicted Graduate | Predicted Dropout |
|--|:-:|:-:|
| **Actual Graduate** | TN = 398 | FP = 19 |
| **Actual Dropout** | FN = 39 | TP = 270 |

---

## Analisis Hasil

Berdasarkan hasil pengujian, **Logistic Regression memberikan performa terbaik** secara konsisten di seluruh metrik evaluasi:

- **Recall 91.90%** → mendeteksi hampir 92% mahasiswa dropout (False Negative Rate hanya 8.1%)
- **Precision 93.21%** → dari 100 prediksi dropout, hanya ~7 yang false alarm
- **Specificity 96.40%** → mahasiswa graduate hampir tidak pernah salah diprediksi sebagai dropout

**Mengapa Logistic Regression > Random Forest?** Variabel penentu utama kelulusan (curricular units approved, admission grade) memiliki korelasi linear yang kuat dengan status kelulusan. Logistic Regression mampu memetakan batas keputusan linear tanpa overfitting. Sebaliknya, Random Forest default cenderung membuat pohon terlalu kompleks sehingga menurunkan generalisasi — recall turun drastis ke 87.32%.

Hyperparameter tuning melalui GridSearchCV tidak meningkatkan performa secara signifikan karena default parameters sudah well-calibrated untuk dataset ini.

---

## Kesimpulan

Model **Logistic Regression** memberikan performa terbaik dengan **F1-Score 92.55%** dan **Recall 91.90%**, efektif mendeteksi mahasiswa berisiko dropout. Model ini dapat digunakan sebagai dasar pengembangan **sistem early warning** pada institusi pendidikan untuk intervensi dini mahasiswa bermasalah.

---

## Struktur Project

```
📁 Semester2-afteruts/
├── 📄 data.csv                          ← Dataset utama (4,424 × 37)
├── 📄 README.md                         ← File ini
├── 📄 LAPORAN_PREDIKSI_KELULUSAN.md     ← Laporan lengkap siap PPT
├── 🐍 eda.py                            ← Exploratory Data Analysis
├── 🐍 predict.py                        ← Training + GridSearchCV
├── 🐍 generate_report.py                ← Script evaluasi (lama)
├── 🐍 generate_report_md.py             ← Generate laporan .md
├── 📊 hasil_evaluasi.csv                ← Hasil numerik (4 model)
├── 🖼️ hasil_evaluasi.png                ← Tabel hasil evaluasi
├── 🖼️ confusion_matrix_lr.png           ← Confusion matrix LR
├── 🖼️ confusion_matrix_rf.png           ← Confusion matrix RF
├── 🖼️ eda_target_distribution.png       ← Distribusi target
├── 🖼️ eda_correlation_heatmap.png       ← Heatmap korelasi
├── 🖼️ eda_academic_performance.png      ← Boxplot akademik
└── 🖼️ data_process_flow.png             ← Diagram alur proses
```

---

## Cara Menjalankan

```bash
# 1. Exploratory Data Analysis
python eda.py

# 2. Training + GridSearchCV + Evaluasi  
python predict.py

# 3. Generate laporan lengkap (.md)
python generate_report_md.py
```

---

## Teknologi

| Library | Fungsi |
|---------|--------|
| **pandas** | Data manipulation & analysis |
| **numpy** | Numerical computing |
| **scikit-learn** | LogisticRegression, RandomForest, GridSearchCV, StandardScaler, metrics |
| **matplotlib** | Visualisasi grafik |
| **seaborn** | Heatmap, boxplot, countplot |