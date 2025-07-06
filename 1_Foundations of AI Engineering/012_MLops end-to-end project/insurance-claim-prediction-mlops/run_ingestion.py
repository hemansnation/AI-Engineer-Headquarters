from claim.components.data_ingestion import DataIngestion
from claim.entity.data_ingestion_config import DataIngestionConfig
from claim.config.configuration import Configuration  # ✅ This provides training_pipeline_config

if __name__ == "__main__":
    config = Configuration()
    ingestion_config = DataIngestionConfig(config.get_training_pipeline_config())
    ingestion = DataIngestion(ingestion_config)
    ingestion.initiate_data_ingestion()
    print("✅ Data Ingestion Completed")
