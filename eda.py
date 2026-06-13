import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    csv_path = 'data.csv'
    
    # Cek keberadaan data.csv
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' tidak ditemukan.")
        return
        
    print("Membaca dataset untuk EDA...")
    df = pd.read_csv(csv_path, sep=';')
    
    # Bersihkan nama kolom
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
    
    # Set style seaborn
    sns.set_theme(style="whitegrid")
    
    # ==========================================
    # 1. Chart: Target Class Distribution
    # ==========================================
    print("Membuat visualisasi distribusi target...")
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(
        x='Target', 
        data=df, 
        order=df['Target'].value_counts().index,
        palette=['#2ecc71', '#e74c3c', '#3498db']
    )
    
    # Tambahkan angka di atas bar
    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height()):,}', 
            (p.get_x() + p.get_width() / 2., p.get_height()), 
            ha='center', va='bottom', 
            fontsize=10, weight='bold', 
            xytext=(0, 5), textcoords='offset points'
        )
        
    plt.title("Distribusi Status Mahasiswa (Target)", fontsize=13, weight='bold', pad=15)
    plt.xlabel("Status Akademik", fontsize=11)
    plt.ylabel("Jumlah Mahasiswa", fontsize=11)
    plt.tight_layout()
    plt.savefig('eda_target_distribution.png', dpi=300)
    plt.close()
    
    # ==========================================
    # 2. Chart: Correlation Heatmap
    # ==========================================
    print("Membuat heatmap korelasi fitur-fitur utama...")
    # Pilih fitur-fitur penting akademik, finansial, dan demografis
    key_features = [
        'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
        'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
        'Admission grade', 'Age at enrollment', 'Scholarship holder', 'Debtor',
        'Unemployment rate', 'GDP'
    ]
    
    # Filter fitur yang ada di dataframe
    available_features = [f for f in key_features if f in df.columns]
    
    plt.figure(figsize=(10, 8))
    corr_matrix = df[available_features].corr()
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=0.5,
        annot_kws={"size": 9}
    )
    
    plt.title("Heatmap Korelasi Fitur-Fitur Kunci", fontsize=13, weight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('eda_correlation_heatmap.png', dpi=300)
    plt.close()
    
    # ==========================================
    # 3. Chart: Curricular Units Approved Boxplot
    # ==========================================
    print("Membuat boxplot perbandingan unit akademik...")
    # Hapus enrolled untuk memperjelas komparasi biner kelulusan
    df_binary = df[df['Target'].isin(['Graduate', 'Dropout'])].copy()
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x='Target', 
        y='Curricular units 2nd sem (approved)', 
        data=df_binary,
        palette=['#e74c3c', '#2ecc71'] # Dropout (Red), Graduate (Green)
    )
    
    plt.title("Jumlah Unit Akademik Disetujui (Semester 2)\nGraduate vs Dropout", fontsize=13, weight='bold', pad=15)
    plt.xlabel("Status Mahasiswa", fontsize=11)
    plt.ylabel("Jumlah Unit Disetujui (Semester 2)", fontsize=11)
    plt.tight_layout()
    plt.savefig('eda_academic_performance.png', dpi=300)
    plt.close()
    
    print("Proses EDA Selesai! Seluruh gambar visualisasi disimpan.")
    print("- eda_target_distribution.png")
    print("- eda_correlation_heatmap.png")
    print("- eda_academic_performance.png")

if __name__ == '__main__':
    main()
