import os
import sys

from claim.exception.exception import InsuranceClaimException
from claim.logging.logger import logging

from claim.components.data_ingestion import DataIngestion
from claim.components.data_validation import DataValidation
from claim.components.data_transformation import DataTransformation
from claim.components.model_trainer import ModelTrainer


from claim.entity.training_config import TrainingPipelineConfig
from claim.entity.data_ingestion_config import DataIngestionConfig
from claim.entity.data_validation_config import DataValidationConfig
from claim.entity.data_transformation_config import DataTransformationConfig
from claim.entity.model_trainer_config import ModelTrainerConfig

from claim.entity.data_ingestion_artifact import DataIngestionArtifact
from claim.entity.data_validation_artifact import DataValidationArtifact
from claim.entity.data_transformation_artifact import DataTransformationArtifact
from claim.entity.model_trainer_artifact import ModelTrainerArtifact

import sys


class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()
        
        

    def start_data_ingestion(self):
        try:
            self.data_ingestion_config=DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Start data Ingestion")
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        
        except Exception as e:
            raise InsuranceClaimException(e,sys)
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact):
        try:
            data_validation_config=DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=data_validation_config)
            logging.info("Initiate the data Validation")
            data_validation_artifact=data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise InsuranceClaimException(e,sys)
        
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,
            data_transformation_config=data_transformation_config)
            
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise InsuranceClaimException(e,sys)
        
    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config
            )

            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.model_trainer_config,
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()

            return model_trainer_artifact

        except Exception as e:
            raise InsuranceClaimException(e, sys)

        
    
    
    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
        
            
            return model_trainer_artifact
        except Exception as e:
            raise InsuranceClaimException(e,sys)
        
    
