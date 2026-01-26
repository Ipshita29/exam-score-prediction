import pickle
import argparse
import logging
from preprocessing import load_data, clean_data, label_encode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train(data_path, model_output):
    logging.info(f"Loading data from {data_path}...")
    df = load_data(data_path)
    df = clean_data(df)
    df = df.drop("student_id", axis=1)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = label_encode(df, categorical_cols)
    
    X = df.drop("exam_score", axis=1)
    y = df["exam_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_gbr(X_train, y_train)
    with open(model_output, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model trained and saved to {model_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Exam Score Prediction Model")
    parser.add_argument("--data", default="data/Exam_Score_Prediction.csv", help="Path to CSV data")
    parser.add_argument("--output", default="final_model.pkl", help="Output model path")
    args = parser.parse_args()
    train(args.data, args.output)
