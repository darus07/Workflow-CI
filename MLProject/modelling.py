import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Aktifkan autolog agar pelacakan diatur penuh oleh perintah 'mlflow run'
mlflow.autolog()

def main():
    print("Memulai proses automated training via MLflow...")
    
    # Membaca dataset (Pastikan file CSV ini berada di dalam folder MLProject/)
    # Jika nama file dataset Anda berbeda, silakan sesuaikan teks di bawah ini
    try:
        df = pd.read_csv("MLProject/namadataset_preprocessing.csv")
    except FileNotFoundError:
        df = pd.read_csv("namadataset_preprocessing.csv")

    # Pisahkan fitur dan target (Sesuaikan nama kolom target Anda, misal: 'target')
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pelatihan Model langsung tanpa bungkus 'with mlflow.start_run()'
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluasi singkat
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model berhasil dilatih dengan Akurasi: {acc:.4f}")

if __name__ == "__main__":
    main()