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
    
    # One-hot encoding nominal columns
    nominal_cols = ['gender', 'course', 'study_method']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    
    # Feature scaling numerical columns
    scaler = StandardScaler()
    num_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Simple outlier capping (99th percentile)
    for col in num_cols:
        upper_limit = df[col].quantile(0.99)
        df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X = df.drop("exam_score", axis=1)
    y = df["exam_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")
