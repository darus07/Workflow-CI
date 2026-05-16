import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# HAPUS ATAU KOMENTARI baris tracking_uri lokal Anda saat di-push ke GitHub:
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Biarkan MLflow menggunakan default tracking bawaan atau environment variable
mlflow.set_experiment("Eksperimen_Breast_Cancer")
mlflow.sklearn.autolog()

def main():
    # Membaca data hasil preprocessing dari folder yang sama
    train_data = pd.read_csv("X_train_clean.csv")
    test_data = pd.read_csv("X_test_clean.csv")
    
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']
    
    with mlflow.start_run(run_name="CI_Automated_Run"):
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        print("Automated Training via MLflow Project Berhasil!")

if __name__ == "__main__":
    main()