import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from preprocessing import load_data, label_encode

def train_rf(X_train, y_train):
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    return rf

def hyperparameter_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

if __name__ == "__main__":
    df = load_data("data/Exam_Score_Prediction.csv")
    df = df.drop("student_id", axis=1)
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = label_encode(df, categorical_cols)
    
    X = df.drop("exam_score", axis=1)
    y = df["exam_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest model...")
    rf_model = train_rf(X_train, y_train)
    
    print("Performing Hyperparameter Tuning (this may take a while)...")
    best_rf, best_params = hyperparameter_tuning(X_train, y_train)
    print(f"Best Parameters: {best_params}")
    
    y_pred = best_rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Optimized RF MAE: {mae:.2f}")
    
    with open('rf_model.pkl', 'wb') as f:
        pickle.dump(best_rf, f)
    print("Random Forest model saved to rf_model.pkl")
