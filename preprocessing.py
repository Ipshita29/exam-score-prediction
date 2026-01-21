import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(filepath):
    return pd.read_csv(filepath)

def label_encode(df, columns):
    le = LabelEncoder()
    for col in columns:
        df[col] = le.fit_transform(df[col])
    return df

if __name__ == "__main__":
    df = load_data("data/Exam_Score_Prediction.csv")
    df = df.drop("student_id", axis=1)
    
    # Label encoding ordinal/binary columns
    ordinal_cols = ['internet_access', 'sleep_quality', 'facility_rating', 'exam_difficulty']
    df = label_encode(df, ordinal_cols)
    print("Columns after label encoding:")
    print(df[ordinal_cols].head())
