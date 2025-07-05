import sys
import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from claim.entity.data_transformation_artifact import DataTransformationArtifact
from claim.entity.data_validation_artifact import DataValidationArtifact
from claim.entity.data_transformation_config import DataTransformationConfig
from claim.exception.exception import InsuranceClaimException 
from claim.logging.logger import logging
from sklearn.preprocessing import StandardScaler
from claim.utils import save_object,read_data
from typing import List
from claim.constants import TARGET_COLUMN,PREPROCESSOR_MODEL_PATH
from imblearn.combine import SMOTETomek
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact:DataValidationArtifact=data_validation_artifact
            self.data_transformation_config:DataTransformationConfig=data_transformation_config
        except Exception as e:
            raise InsuranceClaimException(e,sys)
        
    def __handling_outliers(self,dataframe:pd.DataFrame,column:str)->pd.DataFrame:
        """
        It handles outliers in the dataframe by replacing them with the median of the column.

        Args:
          dataframe: pd.DataFrame
          columns: List[str]

        Returns:
          pd.DataFrame
        """
        try:
            logging.info(f"Handling outliers in the column: {column}")
            #handiling the outliers in the data
            Q1=dataframe[column].quantile(0.25)
            Q3=dataframe[column].quantile(0.75)
            IQR=Q3-Q1
            lower_bound=Q1-1.5*(IQR)
            upper_bound=Q3+1.5*(IQR)
            dataframe=dataframe[(dataframe[column]>=lower_bound) & (dataframe[column]<=upper_bound)]
            return dataframe
        except Exception as e:
            raise InsuranceClaimException(e,sys)
    
        
    def get_data_transformer_object(self):
        """
        It initialises a Standard Scaler object with the parameters specified.

        Args:
          cls: DataTransformation

        Returns:
          A Scaler object
        """
        logging.info(
            "Entered get_data_trnasformer_object method of Transformation class"
        )
        try:
           scaler=StandardScaler()
           return scaler
        except Exception as e:
            raise InsuranceClaimException(e,sys)
    def __process_disbalanced_cat_col(self,dataframe:pd.DataFrame,column:str)->pd.DataFrame:
        """
        It processes the categorical column by replacing the values with the mean of the column.

        Args:
          dataframe: pd.DataFrame
          columns: str

        Returns:
          pd.DataFrame
        """
        try:
            logging.info(f"Processing categorical column: {column}")
            #handiling segment categorical column
            segments=list(dataframe[column].value_counts(ascending=False)[:3].index)
            for rows in range(dataframe.shape[0]):
                if dataframe.loc[rows,column] not in segments:
                    dataframe.loc[rows,column]="Others"
            
            return dataframe
        except Exception as e:
            raise InsuranceClaimException(e,sys)

        
    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method of DataTransformation class")
        try:
            dir_path=os.path.dirname(self.data_transformation_config.transformed_train_file_path)
            os.makedirs(dir_path,exist_ok=True)


            logging.info("Starting data transformation")
            train_df=read_data(self.data_validation_artifact.valid_train_file_path)
            test_df=read_data(self.data_validation_artifact.valid_test_file_path)
            
            
            train_df=self.__handling_outliers(train_df,column="age_of_car")
            train_df=train_df.reset_index(drop=True)

            #extracting torque nm from max torque
            train_df["torque_nm"]=0
            for idx in list(train_df["max_torque"].str.split("@").index):
                train_df.loc[idx,"torque_nm"]=float(train_df.loc[idx,"max_torque"].split("@")[0].split("Nm")[0])

            #extracting torque rpm for max torque
            train_df["torque_rpm"]=0
            for idx in list(train_df["max_torque"].str.split("@").index):
                train_df.loc[idx,"torque_rpm"]=float(train_df.loc[idx,"max_torque"].split("@")[1].split("rpm")[0])

            #extracting power bhp for max power
            train_df["power_bhp"]=0
            for idx in list(train_df["max_power"].str.split("@").index):
                train_df.loc[idx,"power_bhp"]=float(train_df.loc[idx,"max_power"].split("@")[0].split("bhp")[0])

            #extract power rpm for max power
            train_df["power_rpm"]=0
            for idx in list(train_df["max_power"].str.split("@").index):
                train_df.loc[idx,"power_rpm"]=float(train_df.loc[idx,"max_power"].split("@")[1].split("rpm")[0])

            
            train_df=train_df.drop(["policy_id","max_power","max_torque","area_cluster",'engine_type','is_tpms','rear_brakes_type','steering_type','is_front_fog_lights','is_rear_window_wiper','is_rear_window_washer','is_rear_window_defogger','is_power_door_locks','is_driver_seat_height_adjustable','is_day_night_rear_view_mirror','is_ecw','is_speed_alert','population_density'],axis=1)


            train_df=self.__process_disbalanced_cat_col(train_df,column="segment")
            train_df=self.__process_disbalanced_cat_col(train_df,column="model")

            #handling the categorical columns  
            #converting all categoricals to numericals
            train_df["is_esc"]=train_df["is_esc"].map({'Yes':1,'No':0})
            train_df["is_adjustable_steering"]=train_df["is_adjustable_steering"].map({'Yes':1,'No':0})
            train_df["is_parking_sensors"]=train_df["is_parking_sensors"].map({'Yes':1,'No':0})
            train_df["is_parking_camera"]=train_df["is_parking_camera"].map({'Yes':1,'No':0})
            train_df["transmission_type"]=train_df["transmission_type"].map({'Manual':1,'Automatic':0})
            train_df["is_brake_assist"]=train_df["is_brake_assist"].map({'Yes':1,'No':0})
            train_df["is_central_locking"]=train_df["is_central_locking"].map({'Yes':1,'No':0})
            train_df["is_power_steering"]=train_df["is_power_steering"].map({'Yes':1,'No':0})
            train_df["segment"]=train_df["segment"].map({'A':1,'B2':2,'C2':3,'Others':4})
            train_df["model"]=train_df["model"].map({'M1':1,'M4':2,'M6':3,'Others':4})
            train_df["fuel_type"]=train_df["fuel_type"].map({'CNG':1,'Petrol':2,'Diesel':3})
            
            #dropping those normal non OHE cols
            train_df=train_df.drop(["segment","model","fuel_type"],axis=1)



            test_df=self.__handling_outliers(test_df,column="age_of_car")
            test_df=test_df.reset_index(drop=True)

            #extracting torque nm from max torque
            test_df["torque_nm"]=0
            for idx in list(test_df["max_torque"].str.split("@").index):
                test_df.loc[idx,"torque_nm"]=float(test_df.loc[idx,"max_torque"].split("@")[0].split("Nm")[0])

            #extracting torque rpm for max torque
            test_df["torque_rpm"]=0
            for idx in list(test_df["max_torque"].str.split("@").index):
                test_df.loc[idx,"torque_rpm"]=float(test_df.loc[idx,"max_torque"].split("@")[1].split("rpm")[0])

            #extracting power bhp for max power
            test_df["power_bhp"]=0
            for idx in list(test_df["max_power"].str.split("@").index):
                test_df.loc[idx,"power_bhp"]=float(test_df.loc[idx,"max_power"].split("@")[0].split("bhp")[0])

            #extract power rpm for max power
            test_df["power_rpm"]=0
            for idx in list(test_df["max_power"].str.split("@").index):
                test_df.loc[idx,"power_rpm"]=float(test_df.loc[idx,"max_power"].split("@")[1].split("rpm")[0])

            
            test_df=test_df.drop(["policy_id","max_power","max_torque","area_cluster",'engine_type','is_tpms','rear_brakes_type','steering_type','is_front_fog_lights','is_rear_window_wiper','is_rear_window_washer','is_rear_window_defogger','is_power_door_locks','is_driver_seat_height_adjustable','is_day_night_rear_view_mirror','is_ecw','is_speed_alert','population_density'],axis=1)


            test_df=self.__process_disbalanced_cat_col(test_df,column="segment")
            test_df=self.__process_disbalanced_cat_col(test_df,column="model")
            # test_df.to_csv("test_df.csv",index=False, header=True)
            #handling the categorical columns  
            #converting all categoricals to numericals
            test_df["is_esc"]=test_df["is_esc"].map({'Yes':1,'No':0})
            test_df["is_adjustable_steering"]=test_df["is_adjustable_steering"].map({'Yes':1,'No':0})
            test_df["is_parking_sensors"]=test_df["is_parking_sensors"].map({'Yes':1,'No':0})
            test_df["is_parking_camera"]=test_df["is_parking_camera"].map({'Yes':1,'No':0})
            test_df["transmission_type"]=test_df["transmission_type"].map({'Manual':1,'Automatic':0})
            test_df["is_brake_assist"]=test_df["is_brake_assist"].map({'Yes':1,'No':0})
            test_df["is_central_locking"]=test_df["is_central_locking"].map({'Yes':1,'No':0})
            test_df["is_power_steering"]=test_df["is_power_steering"].map({'Yes':1,'No':0})
            test_df["segment"]=test_df["segment"].map({'A':1,'B2':2,'C2':3,'Others':4})
            test_df["model"]=test_df["model"].map({'M1':1,'M4':2,'M6':3,'Others':4})
            test_df["fuel_type"]=test_df["fuel_type"].map({'CNG':1,'Petrol':2,'Diesel':3})
            
            
            #dropping those normal non OHE cols
            test_df=test_df.drop(["segment","model","fuel_type"],axis=1)

            X_train=train_df.drop([TARGET_COLUMN],axis=1)
            y_train=train_df[TARGET_COLUMN]

            smk = SMOTETomek(random_state=42)
            X_train_res,y_train_res=smk.fit_resample(X_train,y_train)

            X_test=test_df.drop([TARGET_COLUMN],axis=1)
            y_test=test_df[TARGET_COLUMN]

            preprocessor=self.get_data_transformer_object()

            preprocessor_object=preprocessor.fit(X_train_res)
            print(X_train_res.columns)
            print("=============================")
            transformed_input_train_feature=pd.concat([pd.DataFrame(preprocessor_object.transform(X_train_res),columns=X_train_res.columns),y_train_res],axis=1)

            transformed_input_test_feature =pd.concat([pd.DataFrame(preprocessor_object.transform(X_test),columns=X_test.columns),y_test],axis=1)
            
            #save numpy array data
            transformed_input_train_feature.to_csv( self.data_transformation_config.transformed_train_file_path,index=False, header=True)
            transformed_input_test_feature.to_csv( self.data_transformation_config.transformed_test_file_path,index=False, header=True)

            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object)

            save_object( PREPROCESSOR_MODEL_PATH, preprocessor_object)


            #preparing artifacts

            data_transformation_artifact=DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
            return data_transformation_artifact


            
        except Exception as e:
            raise InsuranceClaimException(e,sys)
