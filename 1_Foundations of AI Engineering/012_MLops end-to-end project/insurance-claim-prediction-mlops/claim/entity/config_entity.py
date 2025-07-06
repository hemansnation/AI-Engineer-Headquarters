from dataclasses import dataclass
from typing import Dict


@dataclass
class DataIngestionConfig:
    dataset_download_url: str
    tgz_download_dir: str
    raw_data_dir: str
    ingested_train_dir: str
    ingested_test_dir: str
    sqlite_path: str
    db_table_name: str
    ingestion_params: Dict


@dataclass
class DataValidationConfig:
    schema_file_path: str
    report_file_path: str
    report_page_file_path: str


@dataclass
class DataTransformationConfig:
    add_bedroom_per_room: bool
    transformed_train_dir: str
    transformed_test_dir: str
    preprocessed_object_file_path: str


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str
    base_accuracy: float
    model_config_file_path: str


@dataclass
class ModelEvaluationConfig:
    model_evaluation_file_path: str
    time_stamp: str


@dataclass
class ModelPusherConfig:
    export_dir_path: str
