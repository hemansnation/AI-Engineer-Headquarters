from claim.exception.exception import InsuranceClaimException
from claim.logging.logger import logging
from claim.entity.data_ingestion_config import DataIngestionConfig
import os
import sys
from sklearn.model_selection import train_test_split
import mysql.connector
from sqlalchemy import create_engine
import pandas as pd
from claim.entity.data_ingestion_artifact import DataIngestionArtifact



class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise InsuranceClaimException(e,sys)
        
    def __fetch_data_as_dataframe(self):

        """
        Connects to the local MySQL database and fetches data from the 'insurance_data' table
        into a Pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the fetched data.

        Raises:
            SQLAlchemyError: If there's an error connecting to the database or executing the query.
        """
        try:
            logging.info("Connecting to MySQL database and fetching data into DataFrame")
            # Define database connection parameters
            user = self.data_ingestion_config.db_user          # Replace with your MySQL username
            password = self.data_ingestion_config.db_password      # Replace with your MySQL password
            host = self.data_ingestion_config.db_host     # Replace with your MySQL host
            database = self.data_ingestion_config.db_name  # Replace with your database name

            # Construct the database URL
            db_url = f"mysql+mysqlconnector://{user}:{password}@{host}/{database}"

            # Create a SQLAlchemy engine
            engine = create_engine(db_url)

            # Define your SQL query
            query = f"SELECT * FROM {self.data_ingestion_config.db_table_name}"  # Replace with your table name

            # Execute the query and fetch data into a DataFrame
            df = pd.read_sql(query, con=engine)
            logging.info("Data fetched successfully")
            return df

        except Exception as e:
            logging.error(f"An error occurred while fetching data: {e}")
            raise InsuranceClaimException(e,sys)
    

    def __split_data_as_train_test(self,data:pd.DataFrame):
        try:
            logging.info("Splitting data into training and testing sets")
            train_data,test_data=train_test_split(data,test_size=self.data_ingestion_config.ingestion_params['split_ratio'])

            dir_path=os.path.dirname(self.data_ingestion_config.training_file_path)

            os.makedirs(dir_path,exist_ok=True)

            train_data.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_data.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)
            logging.info("Data split successfully and saved to CSV files")
        except Exception as e:
            logging.error(f"An error occurred while splitting data: {e}")
            raise InsuranceClaimException(e,sys)

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion process")
            dir_path=os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            data=self.__fetch_data_as_dataframe()
            data.to_csv(self.data_ingestion_config.feature_store_file_path)
            self.__split_data_as_train_test(data)
            logging.info("Data ingestion process completed successfully")
            return DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.training_file_path,test_file_path=self.data_ingestion_config.testing_file_path
            )
        except Exception as e:
            logging.error(f"An error occurred during data ingestion: {e}")
            raise InsuranceClaimException(e,sys)

