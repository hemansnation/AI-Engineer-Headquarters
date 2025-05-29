from claim.entity.data_ingestion_config import DataIngestionConfig
from claim.entity.data_validation_config import DataValidationConfig
from claim.components.data_validation import DataValidation
from claim.entity.data_transformation_config import DataTransformationConfig
from claim.components.data_transformation import DataTransformation
from claim.entity.model_trainer_config import ModelTrainerConfig
from claim.components.model_trainer import ModelTrainer

from claim.components.data_ingestion import DataIngestion

from claim.entity.training_config import TrainingPipelineConfig
import sys


training_config = TrainingPipelineConfig()
data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_config)
data_ingestion=DataIngestion(data_ingestion_config=data_ingestion_config)
data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
print(data_ingestion_artifact)
print("-===========================")


data_validation_config = DataValidationConfig(training_pipeline_config=training_config)
data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                               data_validation_config=data_validation_config)
data_validation_artifact=data_validation.initiate_data_validation()
print(data_validation_artifact)
print("-===========================")

data_transformation_config = DataTransformationConfig(training_pipeline_config=training_config)
data_transformation=DataTransformation(data_validation_artifact=data_validation_artifact,
                               data_transformation_config=data_transformation_config)
data_transformation_artifact=data_transformation.initiate_data_transformation()
print(data_transformation_artifact)
print("-===========================")

model_trainer_config = ModelTrainerConfig(training_pipeline_config=training_config)
model_trainer=ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                               model_trainer_config=model_trainer_config)
model_trainer_artifact=model_trainer.initiate_model_trainer()
print(model_trainer_artifact)