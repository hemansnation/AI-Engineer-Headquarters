from claim.entity.data_ingestion_artifact import DataIngestionArtifact
from claim.entity.data_validation_artifact import DataValidationArtifact
from claim.entity.data_validation_config import DataValidationConfig
from claim.entity.data_transformation_artifact import DataTransformationArtifact
from claim.entity.model_trainer_artifact import ModelTrainerArtifact
from claim.entity.model_trainer_config import ModelTrainerConfig
from claim.entity.model_trainer_artifact import ClassificationMetricArtifact
from claim.exception.exception import InsuranceClaimException 
from claim.logging.logger import logging 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score,recall_score,f1_score
import mlflow
from urllib.parse import urlparse
import pandas as pd
import os,sys
from claim.utils import read_yaml_file,write_yaml_file,read_data,save_object
from claim.constants import TARGET_COLUMN,ML_MODEL_PATH

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact
        except Exception as e:
            raise InsuranceClaimException(e,sys)
        
    def __models_runner(self,x_train:pd.DataFrame,y_train:pd.DataFrame,x_test:pd.DataFrame,y_test:pd.DataFrame):
        parameters=read_yaml_file("claim/params/params.yaml")["RandomForestClassifier"]
        model=RandomForestClassifier()
        gs = GridSearchCV(RandomForestClassifier(),parameters,cv=2)
        gs.fit(x_train,y_train)

        model.set_params(**gs.best_params_)
        model.fit(x_train,y_train)

        #model.fit(X_train, y_train)  # Train model
        save_object(file_path=self.model_trainer_config.trained_model_file_path,obj=model)
        save_object(file_path=ML_MODEL_PATH,obj=model)
        y_train_pred = model.predict(x_train)

        y_test_pred = model.predict(x_test)

        # Calculate metrics
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        logging.info(f"Precision: {train_precision}")
        logging.info(f"Recall: {train_recall}")
        logging.info(f"F1 Score: {train_f1}")

        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)

        logging.info(f"Test Precision: {test_precision}")
        logging.info(f"Test Recall: {test_recall}")
        logging.info(f"Test F1 Score: {test_f1}")

        metrics={
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        }

        return metrics,model

    def __track_model(self,model,metrics:ClassificationMetricArtifact):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        with mlflow.start_run():
            f1_score=metrics.f1_score
            precision_score=metrics.precision_score
            recall_score=metrics.recall_score

            

            mlflow.log_metric("f1_score",f1_score)
            mlflow.log_metric("precision",precision_score)
            mlflow.log_metric("recall_score",recall_score)
            mlflow.sklearn.log_model(model,"model")



    def train_model(self,x_train:pd.DataFrame,y_train:pd.DataFrame,x_test:pd.DataFrame,y_test:pd.DataFrame)->ModelTrainerArtifact:
        try:

            dir=os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(dir,exist_ok=True)


            metrics,model=self.__models_runner(x_train,y_train,x_test,y_test)

            training_metrics=ClassificationMetricArtifact(
                f1_score=metrics["train_f1"],
                precision_score=metrics["train_precision"],
                recall_score=metrics["train_recall"]
            )

            # Track the model
            self.__track_model(model,training_metrics)

            testing_metrics=ClassificationMetricArtifact(
                f1_score=metrics["test_f1"],
                precision_score=metrics["test_precision"],
                recall_score=metrics["test_recall"]
            )

            # Track the model
            self.__track_model(model,testing_metrics)

            if self.model_trainer_config.expected_accuracy>metrics["test_f1"]:


                #Creating artifact
                model_trainer_artifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,train_metric_artifact=training_metrics,test_metric_artifact=testing_metrics)

                return model_trainer_artifact
            raise InsuranceClaimException(Exception("Failed F1 Score"),sys)

        except Exception as e:
            raise InsuranceClaimException(e,sys)
        
    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            #loading training array and testing array
            train_data = read_data(train_file_path)
            test_data = read_data(test_file_path)

            x_train, y_train, x_test, y_test = (
                train_data.drop(columns=[TARGET_COLUMN],axis=1),
                train_data[TARGET_COLUMN],
                test_data.drop(columns=[TARGET_COLUMN],axis=1),
                test_data[TARGET_COLUMN],
            )

            model_trainer_artifact=self.train_model(x_train,y_train,x_test,y_test)
            return model_trainer_artifact

            
        except Exception as e:
            raise InsuranceClaimException(e,sys)