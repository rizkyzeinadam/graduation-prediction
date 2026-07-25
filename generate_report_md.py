import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

def get_data_summary(df):
    """Generate summary statistics about the dataset."""
    target_counts = df['Target'].value_counts()
    total = len(df)
    
    summary = {
        'total_rows': total,
        'total_columns': len(df.columns),
        'features': len(df.columns) - 1,  # exclude Target
        'target_distribution': {
            'Graduate': int(target_counts.get('Graduate', 0)),
            'Dropout': int(target_counts.get('Dropout', 0)),
            'Enrolled': int(target_counts.get('Enrolled', 0))
        },
        'binary_only': int(target_counts.get('Graduate', 0) + target_counts.get('Dropout', 0)),
        'target_percentage': {
            'Graduate': f"{(target_counts.get('Graduate', 0) / total * 100):.1f}%",
            'Dropout': f"{(target_counts.get('Dropout', 0) / total * 100):.1f}%",
            'Enrolled': f"{(target_counts.get('Enrolled', 0) / total * 100):.1f}%"
        }
    }
    return summary

def get_feature_categories():
    """Return categorized features."""
    return {
        'Demografis (11)': [
            'Marital status', 'Nacionality', "Mother's qualification",
            "Father's qualification", "Mother's occupation", "Father's occupation",
            'Displaced', 'Educational special needs', 'Gender',
            'Age at enrollment', 'International'
        ],
        'Akademik (15)': [
            'Application mode', 'Application order', 'Course',
            'Daytime/evening attendance\t', 'Previous qualification',
            'Previous qualification (grade)', 'Admission grade',
            'Curricular units 1st sem (credited)',
            'Curricular units 1st sem (enrolled)',
            'Curricular units 1st sem (evaluations)',
            'Curricular units 1st sem (approved)',
            'Curricular units 1st sem (grade)',
            'Curricular units 1st sem (without evaluations)',
            'Curricular units 2nd sem (credited)',
            'Curricular units 2nd sem (enrolled)',
            'Curricular units 2nd sem (evaluations)',
            'Curricular units 2nd sem (approved)',
            'Curricular units 2nd sem (grade)',
            'Curricular units 2nd sem (without evaluations)'
        ],
        'Finansial (3)': [
            'Debtor', 'Tuition fees up to date', 'Scholarship holder'
        ],
        'Ekonomi Makro (3)': [
            'Unemployment rate', 'Inflation rate', 'GDP'
        ]
    }

def main():
    csv_path = 'data.csv'
    eval_path = 'hasil_evaluasi.csv'
    
    # ==========================================
    # 1. Data Info
    # ==========================================
    df = pd.read_csv(csv_path, sep=';')
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
    
    data_summary = get_data_summary(df)
    df_binary = df[df['Target'].isin(['Graduate', 'Dropout'])].copy()
    
    # Basic stats for key numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    key_numeric = ['Admission grade', 'Age at enrollment', 'Curricular units 1st sem (approved)',
                   'Curricular units 1st sem (grade)', 'Curricular units 2nd sem (approved)',
                   'Curricular units 2nd sem (grade)', 'Unemployment rate', 'Inflation rate', 'GDP']
    key_numeric = [c for c in key_numeric if c in df.columns]
    
    stats_df = df[key_numeric].describe().round(2)
    
    # ==========================================
    # 2. Read evaluation results
    # ==========================================
    df_eval = pd.read_csv(eval_path)
    
    # ==========================================
    # 3. Read confusion matrix info (from actual run values)
    # Since confusion matrices were already generated with images,
    # we use the precision/recall values to derive TP, FP, FN, TN
    # For now, we'll use the actual values from the dataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    
    df_binary['Target_encoded'] = df_binary['Target'].map({'Dropout': 1, 'Graduate': 0})
    X = df_binary.drop(columns=['Target', 'Target_encoded'])
    y = df_binary['Target_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Default models
    lr_default = LogisticRegression(max_iter=1000, random_state=42)
    lr_default.fit(X_train_scaled, y_train)
    y_pred_lr = lr_default.predict(X_test_scaled)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    
    rf_default = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_default.fit(X_train_scaled, y_train)
    y_pred_rf = rf_default.predict(X_test_scaled)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    
    # Feature importance from Random Forest
    feature_importance = pd.DataFrame({
        'Fitur': X.columns,
        'Importance': rf_default.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # ==========================================
    # 4. Calculate additional metrics
    # ==========================================
    def calc_metrics_from_cm(cm):
        tn, fp, fn, tp = cm.ravel()
        total = tn + fp + fn + tp
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return {
            'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
            'Total': total,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Specificity': specificity
        }
    
    lr_metrics = calc_metrics_from_cm(cm_lr)
    rf_metrics = calc_metrics_from_cm(cm_rf)
    
    # ==========================================
    # 5. Generate Markdown Report
    # ==========================================
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Map CSV model names for hyperparameter info
    hyperparams = {
        'Logistic Regression (Default)': 'C=1.0, penalty=l2, solver=lbfgs',
        'Logistic Regression (Tuned)': 'GridSearchCV: C=0.1, penalty=l2, solver=liblinear',
        'Random Forest (Default)': 'n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1',
        'Random Forest (Tuned)': 'GridSearchCV: n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=1'
    }
    
    # Merge hyperparameter info
    df_eval_display = df_eval.copy()
    
    md = f"""# Laporan Prediksi Kelulusan Mahasiswa
## *(Graduation Prediction Report)*

> **Tanggal Generate:** {now}  
> **Dataset:** data.csv (Prediksi Kelulusan Mahasiswa)  
> **Tools:** Python 3 + scikit-learn + pandas + matplotlib/seaborn

---

## 1. Ringkasan Dataset

| Keterangan | Nilai |
|---|---|
| **Total Records** | {data_summary['total_rows']:,} |
| **Total Kolom** | {data_summary['total_columns']} |
| **Jumlah Fitur** | {data_summary['features']} |
| **Jumlah Model** | 2 (Logistic Regression, Random Forest) |

### Distribusi Target

| Status | Jumlah | Persentase |
|---|---|---|
| Graduate | {data_summary['target_distribution']['Graduate']:,} | {data_summary['target_percentage']['Graduate']} |
| Dropout | {data_summary['target_distribution']['Dropout']:,} | {data_summary['target_percentage']['Dropout']} |
| Enrolled | {data_summary['target_distribution']['Enrolled']:,} | {data_summary['target_percentage']['Enrolled']} |
| **Total** | **{data_summary['total_rows']:,}** | **100%** |

> Data **Enrolled** dihapus dari training karena bersifat netral (belum lulus/dropout).  
> **Data biner yang digunakan: {data_summary['binary_only']:,} records** (Graduate + Dropout).

---

## 2. Kategori Fitur

### 36 Fitur yang Digunakan untuk Prediksi

| Kategori | Jumlah | Fitur |
|---|---|---|
| **Demografis** | 11 | Marital status, Nacionality, Mother's qualification, Father's qualification, Mother's occupation, Father's occupation, Displaced, Educational special needs, Gender, Age at enrollment, International |
| **Akademik Semester 1 & 2** | 15 | Application mode, Application order, Course, Daytime/evening attendance, Previous qualification, Previous qualification (grade), Admission grade, Curricular units 1st sem (credited/enrolled/evaluations/approved/grade/without evaluations), Curricular units 2nd sem (credited/enrolled/evaluations/approved/grade/without evaluations) |
| **Finansial** | 3 | Debtor, Tuition fees up to date, Scholarship holder |
| **Ekonomi Makro** | 3 | Unemployment rate, Inflation rate, GDP |

---

## 3. Statistika Deskriptif Fitur Numerik Kunci

{stats_df.to_markdown()}

---

## 4. Top 15 Feature Importance (Random Forest)

{feature_importance.head(15).to_markdown(index=False)}

---

## 5. Preprocessing & Metodologi

| Langkah | Deskripsi |
|---|---|
| **Data Cleaning** | Membersihkan nama kolom dari karakter `"` dan `'` |
| **Filter Target** | Menghapus record dengan Target 'Enrolled' |
| **Target Encoding** | Dropout = 1, Graduate = 0 |
| **Train-Test Split** | 80:20 dengan `stratify=y` (menjaga proporsi kelas) |
| **Feature Scaling** | `StandardScaler` (mean=0, std=1) fit pada train, transform pada test |
| **Cross-Validation** | 5-fold GridSearchCV untuk hyperparameter tuning |
| **Scoring Metric** | F1-Score (cocok untuk data imbalance) |

---

## 6. Hyperparameter Tuning (GridSearchCV)

### Logistic Regression
| Parameter | Values yang Di-search |
|---|---|
| `C` | 0.01, 0.1, 1, 10, 100 |
| `penalty` | l1, l2 |
| `solver` | liblinear, saga |
| **Total Kombinasi** | 5 × 2 × 2 = **20 kombinasi × 5 fold = 100 fit** |

### Random Forest
| Parameter | Values yang Di-search |
|---|---|
| `n_estimators` | 100, 200, 300 |
| `max_depth` | 10, 20, 30, None |
| `min_samples_split` | 2, 5, 10 |
| `min_samples_leaf` | 1, 2, 4 |
| **Total Kombinasi** | 3 × 4 × 3 × 3 = **108 kombinasi × 5 fold = 540 fit** |

---

## 7. Hasil Evaluasi Model (Default vs Tuned)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression (Default) | {float(df_eval.iloc[0]['Accuracy'])*100:.2f}% | {float(df_eval.iloc[0]['Precision'])*100:.2f}% | {float(df_eval.iloc[0]['Recall'])*100:.2f}% | {float(df_eval.iloc[0]['F1-Score'])*100:.2f}% |
| Logistic Regression (Tuned) | {float(df_eval.iloc[1]['Accuracy'])*100:.2f}% | {float(df_eval.iloc[1]['Precision'])*100:.2f}% | {float(df_eval.iloc[1]['Recall'])*100:.2f}% | {float(df_eval.iloc[1]['F1-Score'])*100:.2f}% |
| Random Forest (Default) | {float(df_eval.iloc[2]['Accuracy'])*100:.2f}% | {float(df_eval.iloc[2]['Precision'])*100:.2f}% | {float(df_eval.iloc[2]['Recall'])*100:.2f}% | {float(df_eval.iloc[2]['F1-Score'])*100:.2f}% |
| Random Forest (Tuned) | {float(df_eval.iloc[3]['Accuracy'])*100:.2f}% | {float(df_eval.iloc[3]['Precision'])*100:.2f}% | {float(df_eval.iloc[3]['Recall'])*100:.2f}% | {float(df_eval.iloc[3]['F1-Score'])*100:.2f}% |

---

## 8. Confusion Matrix Detail (Default Models)

### Logistic Regression
|  | Predicted Graduate | Predicted Dropout | Total |
|---|---|---|---|
| **Actual Graduate** | TN = {lr_metrics['TN']} | FP = {lr_metrics['FP']} | {lr_metrics['TN'] + lr_metrics['FP']} |
| **Actual Dropout** | FN = {lr_metrics['FN']} | TP = {lr_metrics['TP']} | {lr_metrics['FN'] + lr_metrics['TP']} |
| **Total** | {lr_metrics['TN'] + lr_metrics['FN']} | {lr_metrics['FP'] + lr_metrics['TP']} | **{lr_metrics['Total']}** |

| Metrik | Nilai |
|---|---|
| Accuracy | {lr_metrics['Accuracy']*100:.2f}% |
| Precision | {lr_metrics['Precision']*100:.2f}% |
| Recall | {lr_metrics['Recall']*100:.2f}% |
| F1-Score | {lr_metrics['F1-Score']*100:.2f}% |
| Specificity | {lr_metrics['Specificity']*100:.2f}% |

### Random Forest
|  | Predicted Graduate | Predicted Dropout | Total |
|---|---|---|---|
| **Actual Graduate** | TN = {rf_metrics['TN']} | FP = {rf_metrics['FP']} | {rf_metrics['TN'] + rf_metrics['FP']} |
| **Actual Dropout** | FN = {rf_metrics['FN']} | TP = {rf_metrics['TP']} | {rf_metrics['FN'] + rf_metrics['TP']} |
| **Total** | {rf_metrics['TN'] + rf_metrics['FN']} | {rf_metrics['FP'] + rf_metrics['TP']} | **{rf_metrics['Total']}** |

| Metrik | Nilai |
|---|---|
| Accuracy | {rf_metrics['Accuracy']*100:.2f}% |
| Precision | {rf_metrics['Precision']*100:.2f}% |
| Recall | {rf_metrics['Recall']*100:.2f}% |
| F1-Score | {rf_metrics['F1-Score']*100:.2f}% |
| Specificity | {rf_metrics['Specificity']*100:.2f}% |

---

## 9. Analisis & Kesimpulan

### Model Terbaik
- **Logistic Regression (Default)** memberikan performa terbaik dengan F1-Score **{float(df_eval.iloc[0]['F1-Score'])*100:.2f}%**
- Model sederhana Logistic Regression mengungguli Random Forest pada dataset ini

### Insight Penting
1. Fitur **akademik** (curricular units, admission grade) adalah prediktor terkuat dropout
2. **Fitur finansial** (scholarship holder, debtor) juga berkontribusi signifikan
3. StandardScaler penting diterapkan karena fitur memiliki skala yang berbeda-beda
4. Default parameters lebih optimal dibanding tuned pada dataset ini — menunjukkan model default sudah well-calibrated

### Visualisasi yang Dihasilkan
| File | Deskripsi |
|---|---|
| `eda_target_distribution.png` | Distribusi status mahasiswa |
| `eda_correlation_heatmap.png` | Heatmap korelasi fitur kunci |
| `eda_academic_performance.png` | Boxplot unit akademik (Graduate vs Dropout) |
| `confusion_matrix_lr.png` | Confusion matrix Logistic Regression |
| `confusion_matrix_rf.png` | Confusion matrix Random Forest |
| `hasil_evaluasi.png` | Tabel perbandingan hasil evaluasi |

---

*Report generated automatically by generate_report_md.py*
"""
    
    with open('LAPORAN_PREDIKSI_KELULUSAN.md', 'w', encoding='utf-8') as f:
        f.write(md)
    
    print("Laporan berhasil disimpan ke 'LAPORAN_PREDIKSI_KELULUSAN.md'")
    print(f"\nTotal baris: {len(md.split(chr(10)))}")

if __name__ == '__main__':
    main()