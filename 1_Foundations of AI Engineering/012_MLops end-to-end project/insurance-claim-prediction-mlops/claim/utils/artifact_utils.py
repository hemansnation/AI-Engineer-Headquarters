import os
import sys
from claim.exception.exception import InsuranceClaimException
from claim.logging.logger import logging


def get_latest_artifact_folder(base_dir="Artifacts"):
    try:
        folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
        folders.sort(reverse=True)  # latest timestamp first
        latest = folders[0]
        return os.path.join(base_dir, latest)
    except Exception as e:
        raise InsuranceClaimException(e, sys)


def get_latest_train_test_paths(base_dir="Artifacts"):
    try:
        latest_folder = get_latest_artifact_folder(base_dir)

        # OPTIONAL: If train/test are inside a subfolder, fix here
        candidate_paths = [
            latest_folder,
            os.path.join(latest_folder, "ingested"),  # just in case
            os.path.join(latest_folder, "data"),      # some setups
        ]

        for path in candidate_paths:
            train_path = os.path.join(path, "train.csv")
            test_path = os.path.join(path, "test.csv")
            if os.path.exists(train_path) and os.path.exists(test_path):
                logging.info(f"✅ Using train: {train_path}, test: {test_path}")
                return train_path, test_path

        raise FileNotFoundError("train.csv or test.csv not found in any known subfolder of latest artifact.")
    except Exception as e:
        raise InsuranceClaimException(e, sys)
