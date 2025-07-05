import os
import sys
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from claim.logging.logger import logging
from claim.exception.exception import InsuranceClaimException
from claim.utils.artifact_utils import get_latest_train_test_paths
import joblib


class ModelEvaluator:
    def __init__(self, model_path: str):
        try:
            self.model = joblib.load(model_path)
            logging.info(f"🧠 Loaded model from {model_path}")
            self.train_path, self.test_path = get_latest_train_test_paths()
        except Exception as e:
            raise InsuranceClaimException(e, sys)

    def evaluate_model(self):
        try:
            logging.info("📥 Loading test dataset for evaluation...")
            test_df = pd.read_csv(self.test_path)

            X_test = test_df.drop("charges", axis=1)
            y_test = test_df["charges"]

            y_pred = self.model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            logging.info(f"📊 Evaluation Results → MAE: {mae:.4f}, MSE: {mse:.4f}, R2: {r2:.4f}")
            return {"mae": mae, "mse": mse, "r2": r2}
        except Exception as e:
            raise InsuranceClaimException(e, sys)


if __name__ == "__main__":
    evaluator = ModelEvaluator(model_path="models/model.pkl")
    results = evaluator.evaluate_model()
