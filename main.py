from preprocessing import load_data, clean_data, label_encode
from model_refinement import train_gbr
import pickle
from sklearn.model_selection import train_test_split

def main():
    print("Starting Exam Score Prediction Pipeline...")
    df = load_data("data/Exam_Score_Prediction.csv")
    df = clean_data(df)
    df = df.drop("student_id", axis=1)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = label_encode(df, categorical_cols)
    
    X = df.drop("exam_score", axis=1)
    y = df["exam_score"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = train_gbr(X_train, y_train)
    with open('final_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Pipeline completed successfully. Model saved.")

if __name__ == "__main__":
    main()
