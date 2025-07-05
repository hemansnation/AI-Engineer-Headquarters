from datetime import datetime
import os
from claim.constants import (
    DATA_VALIDATION_DIR_NAME,
    DATA_VALIDATION_DRIFT_REPORT_DIR,
    DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
    DATA_VALIDATION_VALID_DIR,
    TRAIN_FILE_NAME,
    TEST_FILE_NAME,
    SCHEMA_FILE_PATH  # ✅ Add this import if not already present
)
from claim.entity.training_config import TrainingPipelineConfig


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, DATA_VALIDATION_VALID_DIR)
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, TRAIN_FILE_NAME)
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, TEST_FILE_NAME)
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            DATA_VALIDATION_DRIFT_REPORT_DIR,
            DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        )

        # ✅ Add this line to define schema file path
        self.schema_file_path: str = SCHEMA_FILE_PATH
