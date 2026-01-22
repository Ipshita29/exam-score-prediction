import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from preprocessing import load_data, label_encode

def train_baseline(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Simplified loading for baseline test
    df = load_data("data/Exam_Score_Prediction.csv")
    df = df.drop("student_id", axis=1)
    # Basic encoding for baseline
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = label_encode(df, categorical_cols)
    
    X = df.drop("exam_score", axis=1)
    y = df["exam_score"]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_baseline(X_train, y_train)
    print("Baseline Linear Regression model trained.")
