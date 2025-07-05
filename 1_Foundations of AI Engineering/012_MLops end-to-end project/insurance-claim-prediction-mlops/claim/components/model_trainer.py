import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from glob import glob
import joblib
import yaml

from claim.exception.exception import InsuranceClaimException
from claim.logging.logger import logging
from claim.entity.config_entity import ModelTrainerConfig
from claim.entity.artifact_entity import DataIngestionArtifact, ModelTrainerArtifact


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise InsuranceClaimException(e, sys)

    def load_latest_data(self) -> tuple:
        try:
            logging.info("🔍 Locating the latest train/test files...")
            latest_train_path = sorted(glob("Artifacts/*/data_ingestion/ingested/train/train.csv"))[-1]
            latest_test_path = latest_train_path.replace("train/train.csv", "test/test.csv")
            logging.info(f"📁 Train path: {latest_train_path}, Test path: {latest_test_path}")
            return pd.read_csv(latest_train_path), pd.read_csv(latest_test_path)
        except Exception as e:
            raise InsuranceClaimException(e, sys)

    def encode_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            label_cols = ["sex", "smoker", "region"]
            for col in label_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
            return df
        except Exception as e:
            raise InsuranceClaimException(e, sys)

    def train_model(self) -> ModelTrainerArtifact:
        try:
            logging.info("🧠 Training Linear Regression model...")

            train_df, test_df = self.load_latest_data()
            train_df = self.encode_dataframe(train_df)
            test_df = self.encode_dataframe(test_df)

            X_train = train_df.drop("charges", axis=1)
            y_train = train_df["charges"]

            X_test = test_df.drop("charges", axis=1)
            y_test = test_df["charges"]

            # Load model parameters from model.yaml
            with open(self.model_trainer_config.model_config_file_path, 'r') as f:
                model_config = yaml.safe_load(f)

            params = model_config.get("model_parameters", {})
            fit_intercept = params.get("fit_intercept", True)
            normalize = params.get("normalize", False)

            model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            joblib.dump(model, self.model_trainer_config.trained_model_file_path)

            logging.info(f"✅ Model training complete. R2: {r2:.4f}, RMSE: {rmse:.4f}")
            return ModelTrainerArtifact(
                model_path=self.model_trainer_config.trained_model_file_path,
                r2_score=r2,
                rmse=rmse
            )
        except Exception as e:
            raise InsuranceClaimException(e, sys)
