import os
import sys
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from claim.logging.logger import logging
from claim.exception.exception import InsuranceClaimException
from claim.utils.artifact_utils import get_latest_train_test_paths


class ModelTrainer:
    def __init__(self):
        try:
            self.train_path, self.test_path = get_latest_train_test_paths()
            logging.info(f"📁 Train path: {self.train_path}, Test path: {self.test_path}")
        except Exception as e:
            raise InsuranceClaimException(e, sys)

    def load_data(self):
        try:
            logging.info("📥 Loading training and testing datasets...")
            train_df = pd.read_csv(self.train_path)
            test_df = pd.read_csv(self.test_path)
            return train_df, test_df
        except Exception as e:
            raise InsuranceClaimException(f"❌ Failed to load data: {e}", sys)

    def train_model(self):
        try:
            train_df, test_df = self.load_data()

            X_train = train_df.drop("charges", axis=1)
            y_train = train_df["charges"]

            X_test = test_df.drop("charges", axis=1)
            y_test = test_df["charges"]

            logging.info("🧠 Training Linear Regression model...")
            model = LinearRegression()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)

            logging.info(f"✅ Model trained with R2 score: {score:.4f}")
            return model, score

        except Exception as e:
            raise InsuranceClaimException(f"❌ Model training failed: {e}", sys)


if __name__ == "__main__":
    trainer = ModelTrainer()
    model, score = trainer.train_model()
