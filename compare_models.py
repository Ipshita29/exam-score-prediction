import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocessing import load_data, label_encode
from sklearn.model_selection import train_test_split

def compare_models(X_test, y_test):
    models = {
        'Baseline (LR)': 'baseline_model.pkl',
        'Random Forest': 'rf_model.pkl',
        'Gradient Boosting': 'final_model.pkl'
    }
    
    results = []
    for name, path in models.items():
        with open(path, 'rb') as f:
            model = pickle.load(f)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        results.append({'Model': name, 'MAE': mae})
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = load_data("data/Exam_Score_Prediction.csv")
    df = df.drop("student_id", axis=1)
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = label_encode(df, categorical_cols)
    X = df.drop("exam_score", axis=1)
    y = df["exam_score"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    comparison = compare_models(X_test, y_test)
    print(comparison)
