from dataclasses import dataclass

@dataclass
class DataValidationArtifact:
    valid_train_file_path: str
    valid_test_file_path: str
    drift_report_file_path: str