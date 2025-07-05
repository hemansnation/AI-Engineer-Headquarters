import os
import sys
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split

from claim.exception.exception import InsuranceClaimException
from claim.logging.logger import logging
from claim.entity.data_ingestion_config import DataIngestionConfig
from claim.entity.data_ingestion_artifact import DataIngestionArtifact


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise InsuranceClaimException(e, sys)

    def __fetch_data_as_dataframe(self) -> pd.DataFrame:
        try:
            logging.info("🔌 Connecting to SQLite and reading data from table...")

            db_path = self.data_ingestion_config.sqlite_path
            table_name = self.data_ingestion_config.db_table_name

            conn = sqlite3.connect(db_path)
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql(query, conn)
            conn.close()

            if "gender" in df.columns:
                df.rename(columns={"gender": "sex"}, inplace=True)

            logging.info("✅ Data fetched and 'gender' renamed to 'sex' if present.")
            return df

        except Exception as e:
            logging.error(f"❌ Error during SQLite fetch: {e}")
            raise InsuranceClaimException(e, sys)

    def __split_data_as_train_test(self, df: pd.DataFrame) -> DataIngestionArtifact:
        try:
            logging.info("✂️ Splitting data into train and test sets...")

            train_set, test_set = train_test_split(
                df,
                test_size=self.data_ingestion_config.ingestion_params["test_size"],
                random_state=self.data_ingestion_config.ingestion_params["random_state"]
            )

            os.makedirs(self.data_ingestion_config.ingested_train_dir, exist_ok=True)
            os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok=True)

            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir, "train.csv")
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir, "test.csv")

            train_set.to_csv(train_file_path, index=False)
            test_set.to_csv(test_file_path, index=False)

            logging.info(f"✅ Train and test files saved.")
            logging.info(f"📦 Train file path: {train_file_path}, Rows: {len(train_set)}")
            logging.info(f"📦 Test file path: {test_file_path}, Rows: {len(test_set)}")

            return DataIngestionArtifact(
                train_file_path=train_file_path,
                test_file_path=test_file_path
            )

        except Exception as e:
            raise InsuranceClaimException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("🚀 Starting data ingestion process...")
            df = self.__fetch_data_as_dataframe()
            return self.__split_data_as_train_test(df)
        except Exception as e:
            raise InsuranceClaimException(e, sys)
