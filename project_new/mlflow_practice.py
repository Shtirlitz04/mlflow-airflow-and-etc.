import pandas as pd
import numpy as np
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder 
import mlflow.sklearn
import pickle

def main():
    df = pd.read_csv('Iris.csv')
    feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

    X = df[feature_cols]
    y = df['Species']

    enc = LabelEncoder()
    y = enc.fit_transform(y.values)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    mlflow.set_experiment('Iris')

    with mlflow.start_run():
        n_estimators = 100
        max_depth = 1000
        min_samples_split = 2
        min_samples_leaf = 5
        random_state = 42
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth = max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_pred, y_test)
        mlflow.log_param("model_type", 'RandomForest')
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(
            clf,
            name="RandomForest",
            input_example=X_train.iloc[:5]
        )

        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size', len(X_test))
        with open('model_weights/model.pkl', 'wb') as f:
            pickle.dump(clf, f)
        
        print(f"Accuracy: {accuracy:.4f}")