import sys
import pandas as pd
import joblib
from claim.exception.exception import InsuranceClaimException
from claim.utils.artifact_utils import get_latest_train_test_paths
from claim.logging.logger import logging


class ModelPredictor:
    def __init__(self, model_path: str):
        try:
            self.model = joblib.load(model_path)
            logging.info(f"✅ Loaded model from {model_path}")
        except Exception as e:
            raise InsuranceClaimException(e, sys)

    def predict(self):
        try:
            train_path, _ = get_latest_train_test_paths()
            df = pd.read_csv(train_path)

            if "charges" in df.columns:
                X = df.drop("charges", axis=1)
            else:
                X = df

            predictions = self.model.predict(X)
            logging.info(f"🔮 Made predictions on latest train data")

            return predictions
        except Exception as e:
            raise InsuranceClaimException(e, sys)


if __name__ == "__main__":
    predictor = ModelPredictor(model_path="models/model.pkl")
    preds = predictor.predict()
    print("🔍 Sample Predictions:", preds[:5])
