from claim.components.data_validation import DataValidation
from claim.entity.config_entity import DataValidationConfig, DataIngestionConfig
from claim.entity.artifact_entity import DataIngestionArtifact
from claim.logger import logging

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_artifact = DataIngestionArtifact(
            train_file_path="Artifacts/train.csv",
            test_file_path="Artifacts/test.csv"
        )
        self.data_validation_config = DataValidationConfig(
            schema_file_path="config/schema.yaml",
            report_file_path="Artifacts/data_drift_report.html"
        )

    def run_pipeline(self):
        logging.info("Pipeline started...")
        data_validation = DataValidation(
            data_validation_config=self.data_validation_config,
            data_ingestion_artifact=self.data_ingestion_artifact
        )
        validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"Validation Artifact: {validation_artifact}")
        logging.info("Pipeline completed.")
