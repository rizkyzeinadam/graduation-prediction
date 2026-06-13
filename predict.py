import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
    
    # 7. Melatih Logistic Regression
    print("Melatih model Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    # 8. Melatih Random Forest
    print("Melatih model Random Forest...")
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train_scaled, y_train)
    
    # 9. Melakukan prediksi pada data test
    y_pred_lr = lr_model.predict(X_test_scaled)
    y_pred_rf = rf_model.predict(X_test_scaled)
    
    # 10. Menghitung metrik evaluasi
    # Logistic Regression
    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr)
    rec_lr = recall_score(y_test, y_pred_lr)
    f1_lr = f1_score(y_test, y_pred_lr)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    
    # Random Forest
    acc_rf = accuracy_score(y_test, y_pred_rf)
    prec_rf = precision_score(y_test, y_pred_rf)
    rec_rf = recall_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf)
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    
    # 11. Buat DataFrame hasil evaluasi
    data_eval = {
        'Model': ['Logistic Regression', 'Random Forest'],
        'Accuracy': [acc_lr, acc_rf],
        'Precision': [prec_lr, prec_rf],
        'Recall': [rec_lr, rec_rf],
        'F1-Score': [f1_lr, f1_rf]
    }
    df_eval = pd.DataFrame(data_eval)
    
    # 12. Simpan hasil evaluasi ke CSV
    df_eval.to_csv('hasil_evaluasi.csv', index=False)
    print("Hasil evaluasi berhasil disimpan ke 'hasil_evaluasi.csv'")
    
    # 13. Simpan hasil evaluasi menjadi file PNG hasil_evaluasi.png
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis('tight')
    ax.axis('off')
    
    # Format nilai desimal menjadi 4 angka di belakang koma untuk tabel gambar
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
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header baris tabel
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
            
    plt.title("Tabel Hasil Evaluasi Model", fontsize=12, pad=15, weight='bold')
    plt.savefig('hasil_evaluasi.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Tabel hasil evaluasi berhasil disimpan ke 'hasil_evaluasi.png'")
    
    # 14. Simpan Confusion Matrix Logistic Regression menjadi confusion_matrix_lr.png
    disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=['Graduate', 'Dropout'])
    disp_lr.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix - Logistic Regression", pad=15, weight='bold')
    plt.savefig('confusion_matrix_lr.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Confusion matrix Logistic Regression disimpan ke 'confusion_matrix_lr.png'")
    
    # 15. Simpan Confusion Matrix Random Forest menjadi confusion_matrix_rf.png
    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=['Graduate', 'Dropout'])
    disp_rf.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix - Random Forest", pad=15, weight='bold')
    plt.savefig('confusion_matrix_rf.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Confusion matrix Random Forest disimpan ke 'confusion_matrix_rf.png'")
    
    # 16. Tentukan model terbaik berdasarkan F1-score
    if f1_lr > f1_rf:
        best_model = "Logistic Regression"
        best_acc = acc_lr
        best_prec = prec_lr
        best_rec = rec_lr
        best_f1 = f1_lr
    else:
        best_model = "Random Forest"
        best_acc = acc_rf
        best_prec = prec_rf
        best_rec = rec_rf
        best_f1 = f1_rf
        
    # 17. Tampilkan ringkasan ke konsol
    print("\n" + "="*40)
    print(f"Model Terbaik: {best_model}")
    print(f"Accuracy: {best_acc:.4f}")
    print(f"Precision: {best_prec:.4f}")
    print(f"Recall: {best_rec:.4f}")
    print(f"F1-score: {best_f1:.4f}")
    print("="*40)

if __name__ == '__main__':
    main()
