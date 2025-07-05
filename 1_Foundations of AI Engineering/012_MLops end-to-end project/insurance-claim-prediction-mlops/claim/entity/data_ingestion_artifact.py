from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """
    Data Ingestion Artifact class to store the artifacts of data ingestion.
    """

    train_file_path: str
    test_file_path: str
