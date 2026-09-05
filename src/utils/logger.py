import logging
import logging.config
import yaml
from pathlib import Path

# Path to the logging configuration file
LOGGING_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "logging.yaml"

def setup_logger() -> logging.Logger:
    """
    Sets up the central logger for the application using the config/logging.yaml file.
    Returns a configured root logger.
    """
    if LOGGING_CONFIG_PATH.exists():
        with open(LOGGING_CONFIG_PATH, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        logging.warning("logging.yaml not found. Using basic configuration.")
    
    return logging.getLogger("aqi_predictor")

# Initialize and expose the logger instance for the whole project
logger = setup_logger()
