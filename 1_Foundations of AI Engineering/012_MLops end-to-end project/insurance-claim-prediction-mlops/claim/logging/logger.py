import logging
import os

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Log file path
LOG_FILE = os.path.join(LOG_DIR, "running_logs.log")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s - [Line: %(lineno)d] - %(name)s - %(levelname)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),  # ensures special characters are handled in the file
        logging.StreamHandler()  # prints logs to the console
    ]
)

# Optional: create a logger instance (if you want custom naming)
logger = logging.getLogger(__name__)
