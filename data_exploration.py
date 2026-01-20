import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_explore(filepath):
    df = pd.read_csv(filepath)
    print("DataFrame Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe().T)
    return df

if __name__ == "__main__":
    df = load_and_explore("data/Exam_Score_Prediction.csv")
