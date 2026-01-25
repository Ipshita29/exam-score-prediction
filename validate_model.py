import pickle
import pandas as pd
from preprocessing import load_data, label_encode
from sklearn.metrics import r2_score

def validate(model_path, data_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    df = load_data(data_path)
    df = df.drop("student_id", axis=1)
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = label_encode(df, categorical_cols)
    
    X = df.drop("exam_score", axis=1)
    y = df["exam_score"]
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    print(f"Model R2 Score: {r2:.4f}")
    return r2

if __name__ == "__main__":
    validate('final_model.pkl', 'data/Exam_Score_Prediction.csv')
