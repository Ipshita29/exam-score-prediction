import pandas as pd
import pickle
import argparse
from preprocessing import load_data, label_encode

def predict(model_path, data_path, output_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    df = load_data(data_path)
    # Assume same preprocessing as training
    df = df.drop("student_id", axis=1, errors='ignore')
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = label_encode(df, categorical_cols)
    
    X = df.drop("exam_score", axis=1, errors='ignore')
    predictions = model.predict(X)
    
    output_df = pd.DataFrame({'prediction': predictions})
    output_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Exam Scores")
    parser.add_argument("--model", default="final_model.pkl", help="Path to trained model")
    parser.add_argument("--data", required=True, help="Path to input data CSV")
    parser.add_argument("--output", default="predictions.csv", help="Output predictions path")
    args = parser.parse_args()
    predict(args.model, args.data, args.output)
