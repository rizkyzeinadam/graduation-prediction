# HASIL DAN PEMBAHASAN

## Algoritma yang Digunakan

Penelitian ini menggunakan dua algoritma klasifikasi, yaitu Logistic Regression sebagai model baseline dan Random Forest sebagai model pembanding.

Logistic Regression dipilih karena merupakan algoritma klasifikasi linear yang sederhana, cepat dilatih, dan mudah diinterpretasikan. Model ini digunakan sebagai acuan awal untuk mengetahui performa minimum yang dapat dicapai dataset.

Random Forest dipilih karena mampu menangkap hubungan non-linear antar fitur melalui kombinasi banyak pohon keputusan (ensemble learning). Selain itu, Random Forest relatif tahan terhadap overfitting dan mampu menangani data dengan jumlah fitur yang cukup banyak.

Dataset dibagi menjadi data latih (80%) dan data uji (20%) menggunakan stratified split untuk menjaga proporsi kelas Graduate dan Dropout pada kedua subset data.

Sebelum pelatihan model dilakukan standardisasi fitur menggunakan StandardScaler.

---

## Hasil Evaluasi Model

Tabel berikut menunjukkan hasil evaluasi pada data uji.

| Model | Accuracy | Precision | Recall | F1-score |
|---------|---------|---------|---------|---------|
| Logistic Regression | 0.9421 | 0.9321 | 0.9190 | 0.9255 |
| Random Forest | 0.9201 | 0.9185 | 0.8732 | 0.8953 |

---

## Confusion Matrix Logistic Regression

| Aktual \ Prediksi | Graduate | Dropout |
|------------------|-----------|----------|
| Graduate | 423 | 19 |
| Dropout | 23 | 261 |

---

## Confusion Matrix Random Forest

| Aktual \ Prediksi | Graduate | Dropout |
|------------------|-----------|----------|
| Graduate | 420 | 22 |
| Dropout | 36 | 248 |

---

## Analisis Hasil

Berdasarkan hasil pengujian, model dengan nilai F1-score dan Recall tertinggi untuk kelas Dropout dianggap memiliki performa terbaik. Recall menjadi metrik utama karena tujuan penelitian adalah mengidentifikasi mahasiswa yang berisiko tidak lulus tepat waktu.

Model Logistic Regression menghasilkan Accuracy sebesar 94.21%, Recall sebesar 91.90%, dan F1-score sebesar 92.55%.

Model Random Forest menghasilkan Accuracy sebesar 92.01%, Recall sebesar 87.32%, dan F1-score sebesar 89.53%.

Dari hasil analisis komparatif ini, terlihat bahwa model Logistic Regression menunjukkan performa yang lebih unggul secara konsisten dibandingkan dengan Random Forest di seluruh metrik evaluasi. Hal ini terjadi karena variabel-variabel penentu utama dalam kelulusan mahasiswa, seperti jumlah unit kurikulum yang disetujui pada semester pertama dan kedua, memiliki korelasi linear yang sangat kuat dengan status kelulusan akhir. Logistic Regression secara efisien mampu memetakan batas keputusan linear ini tanpa mengalami overfitting. Sebaliknya, model Random Forest dengan parameter default cenderung mengalami overfitting pada data latih dengan membuat percabangan pohon keputusan yang terlalu kompleks, sehingga kemampuannya untuk menggeneralisasi data uji menurun. Penurunan Recall pada Random Forest (menjadi 87.32%) dibandingkan Logistic Regression (91.90%) sangat krusial karena menyebabkan lebih banyak mahasiswa berisiko dropout (tidak lulus tepat waktu) yang luput dari deteksi sistem intervensi akademik.

---

## Kesimpulan

Berdasarkan hasil evaluasi, model Logistic Regression memberikan performa terbaik dengan nilai F1-score sebesar 92.55% dan Recall sebesar 91.90%.

Hasil ini menunjukkan bahwa model tersebut lebih efektif dalam mendeteksi mahasiswa yang berpotensi mengalami Dropout sehingga dapat digunakan sebagai dasar pengembangan sistem early warning pada institusi pendidikan.
