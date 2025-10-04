# # updated model evaluation

# import numpy as np
# import pandas as pd
# import pickle
# import json
# from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
# import logging
# import mlflow
# import mlflow.sklearn
# import dagshub
# import os
# from dotenv import load_dotenv

# load_dotenv()
# # Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("DAGSHUB_PAT")
# if not dagshub_token:
#     raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "sohamfcb"
# repo_name = "mlops-mini-project"

# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# # logging configuration
# logger = logging.getLogger('model_evaluation')
# logger.setLevel('DEBUG')

# console_handler = logging.StreamHandler()
# console_handler.setLevel('DEBUG')

# file_handler = logging.FileHandler('model_evaluation_errors.log')
# file_handler.setLevel('ERROR')

# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)
# file_handler.setFormatter(formatter)

# logger.addHandler(console_handler)
# logger.addHandler(file_handler)

# def load_model(file_path: str):
#     """Load the trained model from a file."""
#     try:
#         with open(file_path, 'rb') as file:
#             model = pickle.load(file)
#         logger.debug('Model loaded from %s', file_path)
#         return model
#     except FileNotFoundError:
#         logger.error('File not found: %s', file_path)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error occurred while loading the model: %s', e)
#         raise

# def load_data(file_path: str) -> pd.DataFrame:
#     """Load data from a CSV file."""
#     try:
#         df = pd.read_csv(file_path)
#         logger.debug('Data loaded from %s', file_path)
#         return df
#     except pd.errors.ParserError as e:
#         logger.error('Failed to parse the CSV file: %s', e)
#         raise
#     except Exception as e:
#         logger.error('Unexpected error occurred while loading the data: %s', e)
#         raise

# def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
#     """Evaluate the model and return the evaluation metrics."""
#     try:
#         y_pred = clf.predict(X_test)
#         y_pred_proba = clf.predict_proba(X_test)[:, 1]

#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         auc = roc_auc_score(y_test, y_pred_proba)

#         metrics_dict = {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'auc': auc
#         }
#         logger.debug('Model evaluation metrics calculated')
#         return metrics_dict
#     except Exception as e:
#         logger.error('Error during model evaluation: %s', e)
#         raise

# def save_metrics(metrics: dict, file_path: str) -> None:
#     """Save the evaluation metrics to a JSON file."""
#     try:
#         with open(file_path, 'w') as file:
#             json.dump(metrics, file, indent=4)
#         logger.debug('Metrics saved to %s', file_path)
#     except Exception as e:
#         logger.error('Error occurred while saving the metrics: %s', e)
#         raise

# def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
#     """Save the model run ID and path to a JSON file."""
#     try:
#         model_info = {'run_id': run_id, 'model_path': model_path}
#         with open(file_path, 'w') as file:
#             json.dump(model_info, file, indent=4)
#         logger.debug('Model info saved to %s', file_path)
#     except Exception as e:
#         logger.error('Error occurred while saving the model info: %s', e)
#         raise

# def main():
#     mlflow.set_experiment("dvc-pipeline")
#     with mlflow.start_run() as run:  # Start an MLflow run
#         try:
#             clf = load_model('./models/model.pkl')
#             test_data = load_data('./data/processed/test_bow.csv')
            
#             X_test = test_data.iloc[:, :-1].values
#             y_test = test_data.iloc[:, -1].values

#             metrics = evaluate_model(clf, X_test, y_test)
            
#             save_metrics(metrics, 'reports/metrics.json')
            
#             # Log metrics to MLflow
#             for metric_name, metric_value in metrics.items():
#                 mlflow.log_metric(metric_name, metric_value)
            
#             # Log model parameters to MLflow
#             if hasattr(clf, 'get_params'):
#                 params = clf.get_params()
#                 for param_name, param_value in params.items():
#                     mlflow.log_param(param_name, param_value)
            
#             # Log model to MLflow
#             mlflow.sklearn.log_model(clf, "model")
            
#             # Save model info
#             save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
#             # Log the metrics file to MLflow
#             mlflow.log_artifact('reports/metrics.json')

#             # Log the model info file to MLflow
#             mlflow.log_artifact('reports/model_info.json')

#             # Log the evaluation errors log file to MLflow
#             mlflow.log_artifact('model_evaluation_errors.log')
#         except Exception as e:
#             logger.error('Failed to complete the model evaluation process: %s', e)
#             print(f"Error: {e}")

# if __name__ == '__main__':
#     main()


# updated model evaluation (robust + DagsHub optional)

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.sklearn
import os
from pathlib import Path
from dotenv import load_dotenv

# ---- logging (avoid duplicate handlers in repeated runs) ----
logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("model_evaluation_errors.log")
    file_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ---- env + MLflow setup ----
load_dotenv()  # loads from .env in CWD or parents
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)  # ensures .env is found even from subdirs

def _configure_mlflow():
    """
    Use DagsHub if creds exist; else fallback to local.
    DagsHub expects username=<dagshub username>, password=<token>.
    """
    dagshub_username = os.getenv("DAGSHUB_USERNAME", "sohamfcb")  # <-- your username
    dagshub_token = (
        os.getenv("DAGSHUB_TOKEN")
        or os.getenv("DAGSHUB_PAT")
        or os.getenv("MLFLOW_TRACKING_PASSWORD")
    )

    dagshub_url = "https://dagshub.com"
    repo_owner = "sohamfcb"
    repo_name = "mlops-mini-project"
    dagshub_uri = f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"

    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username   # <-- username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token      # <-- token
        mlflow.set_tracking_uri(dagshub_uri)
        logger.debug("MLflow configured for DagsHub at %s", dagshub_uri)
    else:
        local_uri = "file:./mlruns"
        mlflow.set_tracking_uri(local_uri)
        logger.warning(
            "DAGSHUB token not found. Using local MLflow tracking at %s. "
            "Set DAGSHUB_TOKEN or DAGSHUB_PAT to log to DagsHub.",
            local_uri,
        )

def _ensure_reports_dir():
    Path("reports").mkdir(parents=True, exist_ok=True)

def load_model(file_path: str):
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
        logger.debug("Model loaded from %s", file_path)
        return model
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the model: %s", e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data loaded from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise

def _get_proba_or_score(clf, X: np.ndarray) -> np.ndarray:
    """
    Return a [0,1] score for positive class.
    Uses predict_proba if available; falls back to decision_function then min-max scale.
    """
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, "decision_function"):
        raw = clf.decision_function(X).astype(float)
        # Min-max to [0,1] for AUC; if constant, fall back to zeros
        lo, hi = raw.min(), raw.max()
        if hi > lo:
            return (raw - lo) / (hi - lo)
        return np.zeros_like(raw)
    # No probabilities/scores: use preds as 0/1
    return clf.predict(X).astype(float)

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_score = _get_proba_or_score(clf, X_test)

        metrics_dict = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_test, y_score)),
        }
        logger.debug("Model evaluation metrics calculated")
        return metrics_dict
    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logger.debug("Metrics saved to %s", file_path)
    except Exception as e:
        logger.error("Error occurred while saving the metrics: %s", e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        model_info = {"run_id": run_id, "model_path": model_path}
        with open(file_path, "w") as file:
            json.dump(model_info, file, indent=4)
        logger.debug("Model info saved to %s", file_path)
    except Exception as e:
        logger.error("Error occurred while saving the model info: %s", e)
        raise

def main():
    _ensure_reports_dir()
    _configure_mlflow()

    mlflow.set_experiment("dvc-pipeline")
    with mlflow.start_run() as run:
        try:
            clf = load_model("./models/model.pkl")
            test_data = load_data("./data/processed/test_bow.csv")

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)
            save_metrics(metrics, "reports/metrics.json")

            # Log metrics to MLflow
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log params if available
            if hasattr(clf, "get_params"):
                params = clf.get_params()
                # Convert non-serializable values to strings to avoid MLflow errors
                params = {k: (str(v) if not isinstance(v, (int, float, str, bool)) else v)
                          for k, v in params.items()}
                mlflow.log_params(params)

            # Log model to MLflow
            mlflow.sklearn.log_model(clf, "model")

            # Save + log run info (use a single canonical filename)
            exp_info_path = "reports/experiment_info.json"
            save_model_info(run.info.run_id, "model", exp_info_path)

            mlflow.log_artifact("reports/metrics.json")
            mlflow.log_artifact(exp_info_path)  # <-- fixed mismatch
            mlflow.log_artifact("model_evaluation_errors.log")
        except Exception as e:
            logger.error("Failed to complete the model evaluation process: %s", e)
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
