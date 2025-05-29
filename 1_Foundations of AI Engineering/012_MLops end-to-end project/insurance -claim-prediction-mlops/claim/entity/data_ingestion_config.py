from datetime import datetime
import os
from claim.constants import (
    DATA_INGESTION_DIR_NAME,
    DATA_INGESTION_FEATURE_STORE,
    RAW_FILE_NAME,
    DATA_INGESTION_INGESTED_DIR,
    TRAIN_FILE_NAME,
    TEST_FILE_NAME,
    DATA_INGESTION_USER,
    DATA_INGESTION_PASSWORD,
    DATA_INGESTION_HOST_NAME,
    DATA_INGESTION_DATABASE_NAME,
    DATA_INGESTION_TABLE_NAME,
    PARAMS_PATH
)
from claim.entity.training_config import TrainingPipelineConfig
from claim.utils import read_yaml_file

class DataIngestionConfig:
    """
    Configuration class for setting up the data ingestion process.
    It initializes paths for storing raw and processed data, as well as
    database connection parameters and ingestion-specific parameters.
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initializes the DataIngestionConfig instance with paths and parameters.

        Args:
            training_pipeline_config (TrainingPipelineConfig): An instance of the TrainingPipelineConfig class.
        """

        # Base directory for data ingestion artifacts
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            DATA_INGESTION_DIR_NAME
        )

        # Path to store the raw data (feature store)
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_FEATURE_STORE,
            RAW_FILE_NAME
        )

        # Path to store the training data after splitting
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_INGESTED_DIR,
            TRAIN_FILE_NAME
        )

        # Path to store the testing data after splitting
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_INGESTED_DIR,
            TEST_FILE_NAME
        )

        # Database connection parameters
        self.db_user: str = DATA_INGESTION_USER
        self.db_password: str = DATA_INGESTION_PASSWORD
        self.db_host: str = DATA_INGESTION_HOST_NAME
        self.db_name: str = DATA_INGESTION_DATABASE_NAME
        self.db_table_name: str = DATA_INGESTION_TABLE_NAME

        # Ingestion-specific parameters loaded from a YAML configuration file
        self.ingestion_params: dict = read_yaml_file(PARAMS_PATH).get("Ingestion", {})
