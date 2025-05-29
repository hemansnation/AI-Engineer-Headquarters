import yaml
import os
import pickle
import sys
from claim.exception.exception import InsuranceClaimException
import pandas as pd
from claim.logging.logger import logging
def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a Python dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: Parsed contents of the YAML file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there's an error parsing the YAML file.
        Exception: For any other exceptions that may occur.
    """
    try:
        logging.info("Entered the read_yaml_file method of MainUtils class")
        # Open and read the YAML file
        with open(file_path, 'r') as file:
            logging.info(f"Reading YAML file from {file_path}")
            data = yaml.safe_load(file)
            return data

    except Exception as e:
        # Handle other exceptions and provide traceback
        logging.error(f"Error reading YAML file: {e}")
        raise InsuranceClaimException(e, sys)
    

def write_yaml_file(file_path: str, data: dict) -> None:
    """
    Writes the provided data to a YAML file at the specified path.

    Args:
        file_path (str): The path where the YAML file will be written.
        data (dict): The data to write to the YAML file.

    Raises:
        Exception: If an error occurs during the file writing process.
    """
    try:
        # Ensure the directory exists
        logging.info("Entered the write_yaml_file method")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Open the file in write mode and dump the data as YAML
        with open(file_path, 'w') as file:
            yaml.dump(data, file, default_flow_style=False)
            logging.info(f"Data written to {file_path}")
        print(f"Data successfully written to {file_path}")

    except Exception as e:
        raise InsuranceClaimException(e, sys)

def read_data(file_path: str) -> pd.DataFrame:
    """
    Reads a CSV file from the specified path into a Pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.

    Raises:
        InsuranceClaimException: If any error occurs during file reading.
    """
    try:
    
        # Attempt to read the CSV file
        logging.info("Entered the read_data method of MainUtils class")
        df = pd.read_csv(file_path)
        return df

    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        # Capture the traceback for debugging
        raise InsuranceClaimException(
            e,sys
        )



def save_object(file_path: str, obj: object) -> None:
    """
    Serializes and saves a Python object to a specified file path using pickle.

    Args:
        file_path (str): The full path (including filename) where the object will be saved.
        obj (object): The Python object to serialize and save.

    Raises:
        InsuranceClaimException: If any error occurs during the save process.
    """
    try:
        logging.info("Starting save_object function.")

        # Create the directory if it does not exist
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name, exist_ok=True)
        logging.debug(f"Ensured directory exists: {dir_name}")

        # Serialize the object and write to file in binary mode
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f"Object successfully saved at: {file_path}")

    except Exception as e:
        logging.error(f"Failed to save object at {file_path}. Reason: {e}")
        raise InsuranceClaimException(str(e), sys) from e


def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise InsuranceClaimException(e, sys) from e


