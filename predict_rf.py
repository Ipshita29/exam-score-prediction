import pandas as pd
import pickle
import sys

def predict_rf(input_data):
    with open('rf_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    prediction = model.predict(input_data)
    return prediction

if __name__ == "__main__":
    print("Random Forest Prediction Script Ready.")
    # In practice, load CSV and pass to predict_rf
