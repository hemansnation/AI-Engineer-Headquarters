import logging
import os
from datetime import datetime

# Generate a log file name using the current date and time
LOG_FILE_NAME = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"

# Define the path where log files will be stored
logs_path = os.path.join(os.getcwd(), "logs")

# Create the logs directory if it doesn't already exist
os.makedirs(logs_path, exist_ok=True)

# Full path to the log file
log_file_path = os.path.join(logs_path, LOG_FILE_NAME)

# Configure the logging settings
logging.basicConfig(
    filename=log_file_path,  # Log file location
    format='%(asctime)s - %(message)s - [Line: %(lineno)d] - %(name)s - %(levelname)s',  # Log format
    level=logging.INFO,  # Minimum logging level
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format in log entries
)
