import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.autolog()

def main():
    csv_filename = "namadataset_preprocessing.csv"
    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
    elif os.path.exists(os.path.join("MLProject", csv_filename)):
        df = pd.read_csv(os.path.join("MLProject", csv_filename))
    else:
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target

    target_column = 'target' if 'target' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()