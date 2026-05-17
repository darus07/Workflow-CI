import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Aktifkan autolog agar parameter & metrik dicatat otomatis oleh 'mlflow run'
mlflow.autolog()

def main():
    print("Memulai proses automated training via MLflow...")
    
    # 2. Memuat dataset Breast Cancer resmi (Anti-Gagal Jalur/Path)
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Eksekusi training langsung (Jangan gunakan 'with mlflow.start_run()')
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluasi model
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model berhasil dilatih dengan Akurasi: {acc:.4f}")

if __name__ == "__main__":
    main()