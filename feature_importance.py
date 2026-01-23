import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from preprocessing import load_data, label_encode

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance (Random Forest)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.show()

if __name__ == "__main__":
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    df = load_data("data/Exam_Score_Prediction.csv")
    df = df.drop(["student_id", "exam_score"], axis=1)
    # Basic encoding for feature names
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = label_encode(df, categorical_cols)
    
    plot_feature_importance(model, df.columns)
