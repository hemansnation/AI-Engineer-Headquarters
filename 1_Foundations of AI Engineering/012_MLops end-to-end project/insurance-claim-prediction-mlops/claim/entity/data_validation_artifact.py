from dataclasses import dataclass

@dataclass
class DataValidationArtifact:
    schema_file_path: str
    valid_train_file_path: str
    valid_test_file_path: str
    drift_report_file_path: str
    validation_status: bool
    message: str
