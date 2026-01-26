# Exam Score Prediction Project

This project aims to predict student exam scores based on various factors like study hours, attendance, sleep quality, and more using machine learning models.

## Current Progress
- [x] Initial Repository Setup
- [x] Data Exploration (EDA)
- [x] Data Preprocessing
- [x] Model Training (Baseline)
- [x] Advanced Modeling (Random Forest)
- [x] Model Refinement (Gradient Boosting)
- [x] Production Scripts (Train/Predict)
- [ ] Deployment Scripts

## Usage Instructions

### Training the Model
```bash
python train.py --data data/Exam_Score_Prediction.csv --output final_model.pkl
```

### Making Predictions
```bash
python predict.py --model final_model.pkl --data data/sample_input.csv --output predictions.csv
```

## Initial EDA Findings
- Exam scores follow a normal distribution centered around 50-60.
- Categorical features like 'study_method' and 'sleep_quality' show interesting variances but require encoding.
- Strong correlations observed between study hours and class attendance with the final score.

## Preprocessing & Feature Engineering
- **Label Encoding:** Applied to ordinal features like `internet_access`, `sleep_quality`, etc.
- **One-Hot Encoding:** Applied to nominal features like `gender`, `course`, and `study_method`.
- **Scaling:** Numerical features scaled using `StandardScaler`.
- **Outliers:** Handled using 99th percentile capping.
- **Splitting:** Data split 80/20 for training and testing.

## Model Comparison
| Model | MAE | RMSE | CV MAE |
| :--- | :--- | :--- | :--- |
| Linear Regression (Baseline) | 5.42| 6.81 | 5.51 |
| Random Forest (Optimized) | 4.15 | 5.23 | 4.28 |

## Feature Importance Findings
- **Study Hours:** Most significant predictor.
- **Class Attendance:** Second most important factor.
- **Exam Difficulty:** Influences score significantly.
- **Sleep Quality:** Moderate impact.

This project predicts student exam scores based on study habits, attendance, sleep quality, course type, and other lifestyle or academic factors.
The dataset contains various features such as hours of study, class attendance, sleep behavior, difficulty level.