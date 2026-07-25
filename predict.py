import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

def main():
    csv_path = 'data.csv'
    
    # 1. Membaca dataset
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' tidak ditemukan.")
        return
        
    print("Membaca dataset...")
    df = pd.read_csv(csv_path, sep=';')
    
    # 2. Membersihkan nama kolom
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
    
    # 3. Hapus data dengan target 'Enrolled'
    if 'Target' not in df.columns:
        print("Error: Kolom 'Target' tidak ditemukan.")
        return
    df_binary = df[df['Target'].isin(['Graduate', 'Dropout'])].copy()
    
    # 4. Lakukan encoding target: Graduate = 0, Dropout = 1
    df_binary['Target_encoded'] = df_binary['Target'].map({'Dropout': 1, 'Graduate': 0})
    
    # Memisahkan fitur dan target
    X = df_binary.drop(columns=['Target', 'Target_encoded'])
    y = df_binary['Target_encoded']
    
    # 5. Train-test split 80:20 menggunakan stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 6. StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ================================================================
    # 7a. Melatih Logistic Regression (DEFAULT)
    # ================================================================
    print("\n" + "="*50)
    print("LOGISTIC REGRESSION - Default")
    print("="*50)
    lr_default = LogisticRegression(max_iter=1000, random_state=42)
    lr_default.fit(X_train_scaled, y_train)
    y_pred_lr_default = lr_default.predict(X_test_scaled)
    
    acc_lr_default = accuracy_score(y_test, y_pred_lr_default)
    prec_lr_default = precision_score(y_test, y_pred_lr_default)
    rec_lr_default = recall_score(y_test, y_pred_lr_default)
    f1_lr_default = f1_score(y_test, y_pred_lr_default)
    
    print(f"Accuracy:  {acc_lr_default:.4f}")
    print(f"Precision: {prec_lr_default:.4f}")
    print(f"Recall:    {rec_lr_default:.4f}")
    print(f"F1-Score:  {f1_lr_default:.4f}")
    
    # 7b. GridSearchCV — Logistic Regression
    print("\nMelakukan GridSearchCV untuk Logistic Regression...")
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=2000, random_state=42),
        param_grid=lr_param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    lr_grid.fit(X_train_scaled, y_train)
    
    print(f"Best Parameters: {lr_grid.best_params_}")
    print(f"Best CV F1-Score: {lr_grid.best_score_:.4f}")
    
    lr_best = lr_grid.best_estimator_
    y_pred_lr_tuned = lr_best.predict(X_test_scaled)
    
    acc_lr_tuned = accuracy_score(y_test, y_pred_lr_tuned)
    prec_lr_tuned = precision_score(y_test, y_pred_lr_tuned)
    rec_lr_tuned = recall_score(y_test, y_pred_lr_tuned)
    f1_lr_tuned = f1_score(y_test, y_pred_lr_tuned)
    cm_lr = confusion_matrix(y_test, y_pred_lr_tuned)
    
    print(f"Accuracy (Tuned):  {acc_lr_tuned:.4f}")
    print(f"Precision (Tuned): {prec_lr_tuned:.4f}")
    print(f"Recall (Tuned):    {rec_lr_tuned:.4f}")
    print(f"F1-Score (Tuned):  {f1_lr_tuned:.4f}")
    
    # ================================================================
    # 8a. Melatih Random Forest (DEFAULT)
    # ================================================================
    print("\n" + "="*50)
    print("RANDOM FOREST - Default")
    print("="*50)
    rf_default = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_default.fit(X_train_scaled, y_train)
    y_pred_rf_default = rf_default.predict(X_test_scaled)
    
    acc_rf_default = accuracy_score(y_test, y_pred_rf_default)
    prec_rf_default = precision_score(y_test, y_pred_rf_default)
    rec_rf_default = recall_score(y_test, y_pred_rf_default)
    f1_rf_default = f1_score(y_test, y_pred_rf_default)
    
    print(f"Accuracy:  {acc_rf_default:.4f}")
    print(f"Precision: {prec_rf_default:.4f}")
    print(f"Recall:    {rec_rf_default:.4f}")
    print(f"F1-Score:  {f1_rf_default:.4f}")
    
    # 8b. GridSearchCV — Random Forest
    print("\nMelakukan GridSearchCV untuk Random Forest...")
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=rf_param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    rf_grid.fit(X_train_scaled, y_train)
    
    print(f"Best Parameters: {rf_grid.best_params_}")
    print(f"Best CV F1-Score: {rf_grid.best_score_:.4f}")
    
    rf_best = rf_grid.best_estimator_
    y_pred_rf_tuned = rf_best.predict(X_test_scaled)
    
    acc_rf_tuned = accuracy_score(y_test, y_pred_rf_tuned)
    prec_rf_tuned = precision_score(y_test, y_pred_rf_tuned)
    rec_rf_tuned = recall_score(y_test, y_pred_rf_tuned)
    f1_rf_tuned = f1_score(y_test, y_pred_rf_tuned)
    cm_rf = confusion_matrix(y_test, y_pred_rf_tuned)
    
    print(f"Accuracy (Tuned):  {acc_rf_tuned:.4f}")
    print(f"Precision (Tuned): {prec_rf_tuned:.4f}")
    print(f"Recall (Tuned):    {rec_rf_tuned:.4f}")
    print(f"F1-Score (Tuned):  {f1_rf_tuned:.4f}")
    
    # ================================================================
    # 9. Membuat DataFrame hasil evaluasi (perbandingan default vs tuned)
    # ================================================================
    data_eval = {
        'Model': [
            'Logistic Regression (Default)', 
            'Logistic Regression (Tuned)',
            'Random Forest (Default)', 
            'Random Forest (Tuned)'
        ],
        'Accuracy': [acc_lr_default, acc_lr_tuned, acc_rf_default, acc_rf_tuned],
        'Precision': [prec_lr_default, prec_lr_tuned, prec_rf_default, prec_rf_tuned],
        'Recall': [rec_lr_default, rec_lr_tuned, rec_rf_default, rec_rf_tuned],
        'F1-Score': [f1_lr_default, f1_lr_tuned, f1_rf_default, f1_rf_tuned]
    }
    df_eval = pd.DataFrame(data_eval)
    
    # 10. Simpan hasil evaluasi ke CSV
    df_eval.to_csv('hasil_evaluasi.csv', index=False)
    print("\nHasil evaluasi berhasil disimpan ke 'hasil_evaluasi.csv'")
    
    # 11. Simpan hasil evaluasi menjadi file PNG hasil_evaluasi.png
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')
    
    df_eval_formatted = df_eval.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        df_eval_formatted[col] = df_eval_formatted[col].map(lambda x: f"{x:.4f}")
        
    table = ax.table(
        cellText=df_eval_formatted.values, 
        colLabels=df_eval_formatted.columns, 
        loc='center', 
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    # Style header baris tabel
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
        # Highlight baris tuned
        if row == 2 or row == 4:
            cell.set_facecolor('#e8f4e8')
            
    plt.title("Tabel Perbandingan Hasil Evaluasi Model (Default vs Tuned)", fontsize=10, pad=15, weight='bold')
    plt.savefig('hasil_evaluasi.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Tabel hasil evaluasi berhasil disimpan ke 'hasil_evaluasi.png'")
    
    # 12. Simpan Confusion Matrix Logistic Regression (Tuned)
    disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['Graduate', 'Dropout'])
    disp_lr.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix - Logistic Regression (Tuned)", pad=15, weight='bold')
    plt.savefig('confusion_matrix_lr.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Confusion matrix Logistic Regression disimpan ke 'confusion_matrix_lr.png'")
    
    # 13. Simpan Confusion Matrix Random Forest (Tuned)
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Graduate', 'Dropout'])
    disp_rf.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix - Random Forest (Tuned)", pad=15, weight='bold')
    plt.savefig('confusion_matrix_rf.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Confusion matrix Random Forest disimpan ke 'confusion_matrix_rf.png'")
    
    # 14. Tentukan model terbaik berdasarkan F1-score (Tuned)
    results = [
        ("Logistic Regression (Tuned)", acc_lr_tuned, prec_lr_tuned, rec_lr_tuned, f1_lr_tuned, lr_grid.best_params_),
        ("Random Forest (Tuned)", acc_rf_tuned, prec_rf_tuned, rec_rf_tuned, f1_rf_tuned, rf_grid.best_params_)
    ]
    results.sort(key=lambda x: x[4], reverse=True)
    
    best = results[0]
    
    # 15. Tampilkan ringkasan ke konsol
    print("\n" + "="*60)
    print("RINGKASAN MODEL TERBAIK")
    print("="*60)
    print(f"Model:       {best[0]}")
    print(f"Accuracy:    {best[1]:.4f}")
    print(f"Precision:   {best[2]:.4f}")
    print(f"Recall:      {best[3]:.4f}")
    print(f"F1-Score:    {best[4]:.4f}")
    print(f"Best Params: {best[5]}")
    print("="*60)
    
    print("\nPerbandingan Default vs Tuned:")
    print(df_eval.to_string(index=False))

if __name__ == '__main__':
    main()