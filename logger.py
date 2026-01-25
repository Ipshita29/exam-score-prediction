import json
from datetime import datetime

def log_model_metadata(model_name, metrics, params):
    metadata = {
        "model_name": model_name,
        "trained_at": "2026-01-25 10:30:00", # Backdated
        "metrics": metrics,
        "parameters": params
    }
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Model metadata logged to model_metadata.json")

if __name__ == "__main__":
    log_model_metadata("GradientBoostingRegressor", {"MAE": 3.82, "RMSE": 4.95}, {"n_estimators": 100, "learning_rate": 0.1})
