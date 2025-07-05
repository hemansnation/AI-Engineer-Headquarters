import os

# === General Constants ===
from datetime import datetime
TARGET_COLUMN: str = "is_claim"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
RAW_FILE_NAME: str = "raw.csv"
ARTIFACT_DIR: str = "Artifacts"
PIPELINE_NAME: str = "Insurance_Claim_Pipeline"
PARAMS_PATH: str = "claim/params/params.yaml"

# ✅ Schema path fix — use only this one
SCHEMA_FILE_PATH: str = os.path.join("config", "schema.yaml")

ML_MODEL_PATH: str = "final_models/model.pkl"
PREPROCESSOR_MODEL_PATH: str = "final_models/preprocessor.pkl"

# === Data Ingestion Constants ===
DATA_INGESTION_HOST_NAME: str = "localhost"
DATA_INGESTION_USER: str = "root"
DATA_INGESTION_PASSWORD: str = "9062860379"
DATA_INGESTION_DATABASE_NAME: str = "insurance_db"
DATA_INGESTION_TABLE_NAME: str = "insurance_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"

# === Data Validation Constants ===
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

# === Data Transformation Constants ===
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

# === Model Training Constants ===
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.6
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05
MODEL_FILE_NAME: str = "model.pkl"
CONFIG_FILE_PATH = "config/config.yaml"
CURRENT_TIME_STAMP = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
