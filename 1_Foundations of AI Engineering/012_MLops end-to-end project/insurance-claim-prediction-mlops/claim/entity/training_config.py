from datetime import datetime
import os
from claim.constants import PIPELINE_NAME, ARTIFACT_DIR

class TrainingPipelineConfig:
    """
    Configuration class for setting up the training pipeline.
    It initializes the pipeline name and constructs the artifact directory path
    with a timestamp to ensure uniqueness for each pipeline run.
    """

    def __init__(self):
        # Generate a timestamp in the format 'YYYY-MM-DD-HH-MM-SS'
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # Set the name of the pipeline from constants
        self.pipeline_name:str = PIPELINE_NAME

        # Set the base artifact directory name from constants
        self.artifact_name:str = ARTIFACT_DIR

        # Construct the full path to the artifact directory by combining
        # the base artifact directory with the generated timestamp
        self.artifact_dir:str = os.path.join(self.artifact_name, timestamp)
