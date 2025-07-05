import logging
import os
from datetime import datetime

LOG_DIR = "logs"
LOG_FILE_NAME = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE_NAME)
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="[%(asctime)s]: %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
