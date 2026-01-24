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

def plot_learning_curves(model, X, y):
    from sklearn.model_selection import learning_curve
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='neg_mean_absolute_error', train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation Score")
    plt.title('Learning Curves (Gradient Boosting)')
    plt.xlabel('Training Samples')
    plt.ylabel('MAE')
    plt.legend(loc="best")
    plt.savefig('learning_curves.png')
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
