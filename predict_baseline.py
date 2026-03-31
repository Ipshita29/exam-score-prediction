import pandas as pd
import pickle
import sys

def predict(input_data):
    with open('baseline_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    prediction = model.predict(input_data)
    return prediction

if __name__ == "__main__":
    print("Baseline Prediction Script Ready.")
