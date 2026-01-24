import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
from preprocessing import load_data, label_encode

def train_gbr(X_train, y_train):
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gbr.fit(X_train, y_train)
    return gbr

def plot_residuals(model, X_test, y_test):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, color='purple')
    plt.title('Residuals Distribution (Gradient Boosting)')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.savefig('residuals_dist.png')
    plt.show()

if __name__ == "__main__":
    df = load_data("data/Exam_Score_Prediction.csv")
    df = df.drop("student_id", axis=1)
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = label_encode(df, categorical_cols)
    
    X = df.drop("exam_score", axis=1)
    y = df["exam_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Gradient Boosting model...")
    gbr_model = train_gbr(X_train, y_train)
    
    plot_residuals(gbr_model, X_test, y_test)
    
    with open('final_model.pkl', 'wb') as f:
        pickle.dump(gbr_model, f)
    print("Gradient Boosting model saved as final_model.pkl")
