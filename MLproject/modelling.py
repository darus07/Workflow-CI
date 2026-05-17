import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

# Aktifkan pelacakan otomatis
mlflow.autolog()

def main():
    print("Memulai proses training model via MLflow Project...")
    
    # Load dataset bawaan agar aman dari FileNotFoundError di Ubuntu
    data = load_breast_cancer()
    
    # Inisialisasi dan latih model
    model = RandomForestClassifier(random_state=42)
    model.fit(data.data, data.target)
    
    print("Training berhasil dieksekusi!")

if __name__ == "__main__":
    main()