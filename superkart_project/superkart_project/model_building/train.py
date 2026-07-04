# data
import pandas as pd
import numpy as np
import os, joblib
# modeling
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
# experiment tracking
import mlflow
# hugging face
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")
api = HfApi(token=HF_TOKEN)

DATASET_REPO = f"{HF_USERNAME}/superkart"
MODEL_REPO = f"{HF_USERNAME}/superkart-sales-model"

# MLflow local sqlite backend -> logs runs without needing a tracking server
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("SuperKart-Sales-Forecast-Experiment")

# Load the train/test splits from the HF dataset space
Xtrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtrain.csv")
Xtest  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytrain.csv").values.ravel()
ytest  = pd.read_csv(f"hf://datasets/{DATASET_REPO}/ytest.csv").values.ravel()
print("Loaded splits:", Xtrain.shape, Xtest.shape)

numeric_features = [
    "Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Age",
]
categorical_features = [
    "Product_Sugar_Content", "Product_Type", "Store_Id", "Store_Size",
    "Store_Location_City_Type", "Store_Type", "Product_Category",
]

# Preprocessing: scale numerics, one-hot encode categoricals
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# Base regressor
xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

# Hyper-parameter grid (prefix matches the make_pipeline step name)
param_grid = {
    "xgbregressor__n_estimators": [200, 400],
    "xgbregressor__max_depth": [3, 5],
    "xgbregressor__learning_rate": [0.05, 0.1],
    "xgbregressor__subsample": [0.8, 1.0],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)


def regression_metrics(y_true, y_pred):
    return {
        "r2":   r2_score(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  mean_absolute_error(y_true, y_pred),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, scoring="r2", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    train_m = regression_metrics(ytrain, best_model.predict(Xtrain))
    test_m  = regression_metrics(ytest,  best_model.predict(Xtest))
    print("Best params:", grid_search.best_params_)
    print("Train metrics:", {k: round(v, 4) for k, v in train_m.items()})
    print("Test  metrics:", {k: round(v, 4) for k, v in test_m.items()})

    mlflow.log_metrics({f"train_{k}": v for k, v in train_m.items()})
    mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})

    model_path = "best_superkart_sales_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print("Model saved as artifact at:", model_path)

# Register the best model in the HF model hub
try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    print(f"Model repo '{MODEL_REPO}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Creating model repo '{MODEL_REPO}'...")
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False, token=HF_TOKEN)

api.upload_file(
    path_or_fileobj="best_superkart_sales_model_v1.joblib",
    path_in_repo="best_superkart_sales_model_v1.joblib",
    repo_id=MODEL_REPO,
    repo_type="model",
)
print(f"Registered model -> https://huggingface.co/{MODEL_REPO}")
