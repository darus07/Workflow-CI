import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    print("Memuat dataset...")
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    print("Memulai proses training model via MLflow...")
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)
        
        # Logging parameter dan metrik secara manual
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Training berhasil! Akurasi: {acc:.4f}")

if __name__ == "__main__":
    main()