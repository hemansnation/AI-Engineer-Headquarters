import os
import sys
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from claim.entity.config_entity import DataValidationConfig
from claim.entity.artifact_entity import DataIngestionArtifact
from claim.entity.data_validation_artifact import DataValidationArtifact
from claim.exception.exception import InsuranceException
from claim.logger import logging
from claim.utils import read_yaml_file


def get_latest_artifact_path(base_dir="Artifacts"):
    try:
        folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
        folders.sort(reverse=True)
        latest = folders[0]
        return os.path.join(base_dir, latest)
    except Exception as e:
        raise InsuranceException(f"❌ Failed to locate latest artifact folder: {e}", sys)


class DataValidation:

    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise InsuranceException(e, sys)

    def get_schema(self) -> dict:
        try:
            return read_yaml_file(self.data_validation_config.schema_file_path)
        except Exception as e:
            raise InsuranceException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns = self.get_schema()["columns"]
            actual_columns = dataframe.columns
            return len(actual_columns) == len(expected_columns)
        except Exception as e:
            raise InsuranceException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            # Use the dynamic path if the ingestion artifact paths do not exist
            train_path = self.data_ingestion_artifact.train_file_path
            test_path = self.data_ingestion_artifact.test_file_path

            if not os.path.exists(train_path) or not os.path.exists(test_path):
                logging.warning("🔍 Static paths not found, using latest artifact folder...")
                latest_path = get_latest_artifact_path()
                train_path = os.path.join(latest_path, "train.csv")
                test_path = os.path.join(latest_path, "test.csv")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            schema = self.get_schema()
            expected_columns = list(schema["columns"].keys())

            if train_df.columns.tolist() != expected_columns:
                raise InsuranceException("Train columns do not match expected schema", sys)

            # Generate data drift report using Evidently
            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=train_df, current_data=test_df)

            report_file_path = self.data_validation_config.report_file_path
            os.makedirs(os.path.dirname(report_file_path), exist_ok=True)
            report.save_html(report_file_path)

            return DataValidationArtifact(
                schema_file_path=self.data_validation_config.schema_file_path,
                valid_train_file_path=train_path,
                valid_test_file_path=test_path,
                drift_report_file_path=report_file_path,
                validation_status=True,
                message="Data validation successful!"
            )

        except Exception as e:
            raise InsuranceException(e, sys)
