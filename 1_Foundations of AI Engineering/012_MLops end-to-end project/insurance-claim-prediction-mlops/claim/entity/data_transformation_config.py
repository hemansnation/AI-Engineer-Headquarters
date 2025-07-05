from datetime import datetime
import os
from claim.constants import (
    DATA_TRANSFORMATION_DIR_NAME,
    DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
    TRAIN_FILE_NAME,
    TEST_FILE_NAME,
    PREPROCESSING_OBJECT_FILE_NAME
)
from claim.entity.training_config import TrainingPipelineConfig


class DataTransformationConfig:
     def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join( training_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR_NAME )
        self.transformed_train_file_path: str = os.path.join( self.data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TRAIN_FILE_NAME)
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir,  DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            TEST_FILE_NAME)
        self.transformed_object_file_path: str = os.path.join( self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,PREPROCESSING_OBJECT_FILE_NAME)