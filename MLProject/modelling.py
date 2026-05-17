import mlflow

# Aktifkan autolog di luar/sebelum proses training
mlflow.autolog()

def main():
    # ... (proses data loading & preprocessing) ...
    
    # JALANKAN TRAINING LANGSUNG (Tanpa dibungkus 'with mlflow.start_run()')
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    print("Training selesai dan metrik otomatis dicatat oleh MLflow!")

if __name__ == "__main__":
    main()