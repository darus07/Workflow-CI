import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Aktifkan autolog sesuai dengan ketentuan Basic Kriteria 2
mlflow.autolog()

def main():
    print("Memulai proses automated training via MLflow...")
    
    # Deteksi path secara adaptif agar tidak error di lokal maupun GitHub Actions
    csv_filename = "namadataset_preprocessing.csv"
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
    elif os.path.exists(os.path.join("MLProject", csv_filename)):
        df = pd.read_csv(os.path.join("MLProject", csv_filename))
    else:
        raise FileNotFoundError(f"Dataset {csv_filename} tidak ditemukan di folder root maupun MLProject!")

    # Silakan sesuaikan nama kolom target Anda (misal: 'target')
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model berhasil dilatih dengan Akurasi: {acc:.4f}")

if __name__ == "__main__":
    main()