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

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return mae, mse, rmse

def cross_validate(X, y):
    from sklearn.model_selection import cross_val_score
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    return -scores.mean()

if __name__ == "__main__":
    df = load_data("data/Exam_Score_Prediction.csv")
    df = df.drop("student_id", axis=1)
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = label_encode(df, categorical_cols)
    
    X = df.drop("exam_score", axis=1)
    y = df["exam_score"]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_baseline(X_train, y_train)
    mae, mse, rmse = evaluate_model(model, X_test, y_test)
    print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}")
    
    cv_mae = cross_validate(X, y)
    print(f"Cross-Validated MAE: {cv_mae:.2f}")
    
    with open('baseline_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Baseline model saved to baseline_model.pkl")
