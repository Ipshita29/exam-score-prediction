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

def plot_distributions(df):
    plt.figure(figsize=(12, 6))
    df['exam_score'].hist(bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Exam Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig('exam_score_dist.png')
    plt.show()

if __name__ == "__main__":
    df = load_and_explore("data/Exam_Score_Prediction.csv")
    plot_distributions(df)
