import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Aktifkan autologging untuk menangkap metrik scikit-learn secara otomatis
mlflow.autolog()

def main():
    print("Memulai proses automated training via MLflow...")
    
    # Membaca dataset preprocessing lokal Anda yang berada di folder MLProject
    try:
        df = pd.read_csv("MLProject/namadataset_preprocessing.csv")
    except FileNotFoundError:
        df = pd.read_csv("namadataset_preprocessing.csv")

    # Pisahkan fitur dan target (ganti 'target' sesuai dengan nama kolom target asli Anda)
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Bungkus proses ke dalam start_run agar terbaca eksplisit oleh MLflow lokal maupun cloud
    with mlflow.start_run() as run:
        print(f"Active Run ID: {run.info.run_id}")
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"Model berhasil dilatih dengan Akurasi: {acc:.4f}")

if __name__ == "__main__":
    main()