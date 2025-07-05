import os
import yaml
from claim.constants import CONFIG_FILE_PATH, CURRENT_TIME_STAMP
from claim.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)


class Configuration:
    def __init__(self, config_file_path=CONFIG_FILE_PATH):
        with open(config_file_path, 'r') as file:
            self.config_info = yaml.safe_load(file)
        self.time_stamp = CURRENT_TIME_STAMP

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        data_ingestion_info = self.config_info['data_ingestion_config']

        return DataIngestionConfig(
            dataset_download_url=data_ingestion_info['dataset_download_url'],
            tgz_download_dir=os.path.join(data_ingestion_info['tgz_download_dir'], self.time_stamp),
            raw_data_dir=os.path.join(data_ingestion_info['raw_data_dir'], self.time_stamp),
            ingested_train_dir=os.path.join(data_ingestion_info['ingested_train_dir'], self.time_stamp),
            ingested_test_dir=os.path.join(data_ingestion_info['ingested_test_dir'], self.time_stamp),
            sqlite_path=data_ingestion_info['sqlite_path'],
            db_table_name=data_ingestion_info['db_table_name'],
            ingestion_params=data_ingestion_info['ingestion_params']
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        data_validation_info = self.config_info['data_validation_config']

        return DataValidationConfig(
            schema_file_path=data_validation_info['schema_file_path'],
            report_file_path=data_validation_info['report_file_path'],
            report_page_file_path=data_validation_info['report_page_file_path']
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        data_transformation_info = self.config_info['data_transformation_config']

        return DataTransformationConfig(
            add_bedroom_per_room=data_transformation_info['add_bedroom_per_room'],
            transformed_train_dir=os.path.join(data_transformation_info['transformed_train_dir'], self.time_stamp),
            transformed_test_dir=os.path.join(data_transformation_info['transformed_test_dir'], self.time_stamp),
            preprocessed_object_file_path=os.path.join(data_transformation_info['preprocessed_object_file_path'], self.time_stamp)
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        model_trainer_info = self.config_info['model_trainer_config']

        return ModelTrainerConfig(
            trained_model_file_path=os.path.join(model_trainer_info['trained_model_file_path'], self.time_stamp, "model.pkl"),
            base_accuracy=model_trainer_info['base_accuracy'],
            model_config_file_path=model_trainer_info['model_config_file_path']
        )

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        model_evaluation_info = self.config_info['model_evaluation_config']

        return ModelEvaluationConfig(
            model_evaluation_file_path=model_evaluation_info['model_evaluation_file_path'],
            time_stamp=self.time_stamp
        )

    def get_model_pusher_config(self) -> ModelPusherConfig:
        model_pusher_info = self.config_info['model_pusher_config']

        return ModelPusherConfig(
            export_dir_path=os.path.join(model_pusher_info['export_dir_path'], self.time_stamp)
        )
