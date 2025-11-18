# /mylibrary/utils/logger_config.py

import logging
import os

def setup_logging():
    # Create logger
    logger = logging.getLogger('spectrumvae')
    logger.setLevel(logging.DEBUG)  # Set the logger to handle all levels (DEBUG and higher)

    # Create formatters
    detailed_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    simple_formatter = logging.Formatter('%(levelname)s - %(message)s')

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  # Set the console handler to capture all logs
    console_handler.setFormatter(simple_formatter)  # Use a simple format for console logs

    # Create file handler
    log_file = os.path.join(os.path.dirname(__file__), 'app.log')
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Set the file handler to capture all logs
    file_handler.setFormatter(detailed_formatter)  # Use a detailed format for file logs

    # Add handlers to the logger
    if not logger.hasHandlers():  # Avoid adding handlers multiple times if this function is called repeatedly
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger  # Return the configured logger

# Set up logging when the module is imported and return the logger
logger = setup_logging()

# # Use the logger directly in the script
# logger.debug("This is a debug message from app1")
# logger.info("This is an info message from app1")
# logger.warning("This is a warning message from app1")
# logger.error("This is an error message from app1")
# logger.critical("This is a critical error from app1")
