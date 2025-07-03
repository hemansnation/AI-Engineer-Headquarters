# File: claim/entity/data_ingestion_config.py

class DataIngestionConfig:
    def __init__(
        self,
        sqlite_path: str,
        db_table_name: str,
        feature_store_file_path: str,
        training_file_path: str,
        testing_file_path: str,
        ingestion_params: dict,
        ingested_train_dir: str,
        ingested_test_dir: str,
    ):
        self.sqlite_path = sqlite_path
        self.db_table_name = db_table_name
        self.feature_store_file_path = feature_store_file_path
        self.training_file_path = training_file_path
        self.testing_file_path = testing_file_path
        self.ingestion_params = ingestion_params
        self.ingested_train_dir = ingested_train_dir
        self.ingested_test_dir = ingested_test_dir
