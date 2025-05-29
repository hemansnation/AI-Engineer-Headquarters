from claim.entity.data_ingestion_artifact import DataIngestionArtifact
from claim.entity.data_validation_artifact import DataValidationArtifact
from claim.entity.data_validation_config import DataValidationConfig
from claim.exception.exception import InsuranceClaimException 
from claim.logging.logger import logging 
from claim.constants import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os,sys
from claim.utils import read_yaml_file,write_yaml_file,read_data

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise InsuranceClaimException(e,sys)
        
        
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns=len(self._schema_config["columns"])
            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Data frame has columns:{len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                return True
            return False
        except Exception as e:
            raise InsuranceClaimException(e,sys)
        
    def detect_dataset_drift(self,existing_df,current_df,threshold=0.05)->bool:
        try:
            report={}
            for column in existing_df.columns:
                d1=existing_df[column]
                d2=current_df[column]
                check_dist=ks_2samp(d1,d2)
                if threshold<=check_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                report.update({column:{
                    "p_value":float(check_dist.pvalue),
                    "drift_status":is_found
                    
                    }})
            
            drift_report_file_path = self.data_validation_config.drift_report_file_path

            #Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,data=report)
            logging.info(f"Drift report file path: {drift_report_file_path}")
            
        except Exception as e:
            raise InsuranceClaimException(e,sys)
        
    
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path=self.data_ingestion_artifact.train_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path

            ## read the data from train and test
            train_dataframe=read_data(train_file_path)
            test_dataframe=read_data(test_file_path)
            
            ## validate number of columns

            status=self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                logging.error("Train dataframe does not contain all columns.")
                raise InsuranceClaimException(str("Train dataframe does not contain all columns."),sys)
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            print("column check status->{}".format(status))
            if not status:
                logging.error("Test dataframe does not contain all columns.") 
                raise InsuranceClaimException(str("Test dataframe does not contain all columns."),sys)  

            ## lets check datadrift
            self.detect_dataset_drift(existing_df=train_dataframe,current_df=test_dataframe)
            dir_path=os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)

            train_dataframe.to_csv(
                self.data_validation_config.valid_train_file_path, index=False, header=True

            )

            test_dataframe.to_csv(
                self.data_validation_config.valid_test_file_path, index=False, header=True
            )
            
            data_validation_artifact = DataValidationArtifact(
                valid_train_file_path=self.data_ingestion_artifact.train_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            return data_validation_artifact
        except Exception as e:
            raise InsuranceClaimException(e,sys)



