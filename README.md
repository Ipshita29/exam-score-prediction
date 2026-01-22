# Exam Score Prediction Project

This project aims to predict student exam scores based on various factors like study hours, attendance, sleep quality, and more using machine learning models.

## Current Progress
- [x] Initial Repository Setup
- [x] Data Exploration (EDA)
- [x] Data Preprocessing
- [x] Data Preprocessing
- [x] Model Training (Baseline)
- [ ] Advanced Modeling
- [ ] Deployment Scripts

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

## Baseline Model Performance
- **Algorithm:** Linear Regression
- **Mean Absolute Error (MAE):** ~5.4
- **Root Mean Squared Error (RMSE):** ~6.8
- **Cross-Validated MAE:** ~5.5

This project predicts student exam scores based on study habits, attendance, sleep quality, course type, and other lifestyle or academic factors.
The dataset contains various features such as hours of study, class attendance, sleep behavior, difficulty level.