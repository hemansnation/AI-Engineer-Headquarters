from datetime import datetime
import os
from claim.constants import (
    MODEL_TRAINER_DIR_NAME,
    MODEL_TRAINER_EXPECTED_SCORE,
    MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD,
    MODEL_TRAINER_TRAINED_MODEL_DIR,
    MODEL_TRAINER_TRAINED_MODEL_NAME,
    TRAIN_FILE_NAME,
    TEST_FILE_NAME,
    MODEL_FILE_NAME,
    PREPROCESSING_OBJECT_FILE_NAME
)
from claim.entity.training_config import TrainingPipelineConfig



class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, 
            MODEL_FILE_NAME
        )
        self.expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD