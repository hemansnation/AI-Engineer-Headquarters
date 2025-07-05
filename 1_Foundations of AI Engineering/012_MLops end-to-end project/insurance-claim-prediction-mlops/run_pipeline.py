from claim.config.configuration import Configuration
from claim.components.data_ingestion import DataIngestion
from claim.components.model_trainer import ModelTrainer
from claim.logging.logger import logging
from claim.exception.exception import InsuranceClaimException
import sys

if __name__ == "__main__":
    try:
        logging.info("🚀 Starting Full ML Pipeline...")

        # Step 1: Configuration setup
        config = Configuration()

        # Step 2: Data Ingestion
        logging.info("📥 Running Data Ingestion...")
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("✅ ✅ Data Ingestion Completed")

        # Step 3: Model Training
        logging.info("🏋️‍♀️ Training model...")
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(model_trainer_config, data_ingestion_artifact)
        model_trainer_artifact = model_trainer.train_model()
        logging.info("✅ ✅ Model training completed")

        # Step 4: Wrap-up
        logging.info("📊 Evaluation complete")
        logging.info(f"📁 Model saved at: {model_trainer_artifact.model_path}")
        logging.info(f"🎯 R2 Score: {model_trainer_artifact.r2_score}")
        logging.info(f"📉 RMSE: {model_trainer_artifact.rmse}")

        print("\n✅ Pipeline executed successfully!")
        print("📁 Check 'Artifacts/' and 'models/' for outputs.")
        print("📄 Check logs/running_logs.log for details.\n")

    except InsuranceClaimException as e:
        logging.error(f"❌ Pipeline Failed: {e}")
    except Exception as e:
        logging.error(f"❌ Unexpected Error: {e}")
