import yaml
import os
import sys
from claim.exception.exception import InsuranceClaimException

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise InsuranceClaimException(e, sys)
