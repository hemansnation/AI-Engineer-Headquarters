# File: claim/entity/artifact_entity.py

class DataIngestionArtifact:
    def __init__(self, train_file_path: str, test_file_path: str):
        self.train_file_path = train_file_path
        self.test_file_path = test_file_path


class DataValidationArtifact:
    def __init__(
        self,
        valid_train_file_path: str,
        valid_test_file_path: str,
        drift_report_file_path: str,
        schema_file_path: str  # ✅ Added to resolve earlier pipeline error
    ):
        self.valid_train_file_path = valid_train_file_path
        self.valid_test_file_path = valid_test_file_path
        self.drift_report_file_path = drift_report_file_path
        self.schema_file_path = schema_file_path


class DataTransformationArtifact:
    def __init__(self, transformed_train_file_path: str, transformed_test_file_path: str, preprocessor_object_file_path: str):
        self.transformed_train_file_path = transformed_train_file_path
        self.transformed_test_file_path = transformed_test_file_path
        self.preprocessor_object_file_path = preprocessor_object_file_path


class ModelTrainerArtifact:
    def __init__(self, model_path: str):
        self.model_path = model_path


class ModelEvaluationArtifact:
    def __init__(self, evaluation_metrics_path: str):
        self.evaluation_metrics_path = evaluation_metrics_path


class ModelPredictionArtifact:
    def __init__(self, predictions_path: str):
        self.predictions_path = predictions_path
