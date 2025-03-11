""" Env variables and general config file """
import logging
import inspect
import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    RESULTS = os.environ["RESULTS"]
    MAP_COUNTRIES_KEYWORDS = os.environ["MAP_COUNTRIES_KEYWORDS"]
    NUM_VACANCIES = os.environ["NUM_VACANCIES"]
    FILTER_PARAMS = os.environ["FILTER_PARAMS"]
    DATA_JOBS = os.environ["DATA_JOBS"]
    JOB_OFFERS = os.environ["JOB_OFFERS"]
    JOB_SEEKERS = os.environ["JOB_SEEKERS"]
    MODEL_ID = os.environ["MODEL_ID"]
    EMBEDDING_PATH = os.environ["EMBEDDING_PATH"]
    MATCHES = os.environ["MATCHES"]
    SIMILARITY_THRESHOLD = os.environ["SIMILARITY_THRESHOLD"]
    USERS_IDS = os.environ["USERS_IDS"]
    OUTPUT_MATCHES = os.environ["OUTPUT_MATCHES"]
    AVAILABLE_TAGS = os.environ["AVAILABLE_TAGS"]
    RETRY_DELAY_SECONDS = os.environ["RETRY_DELAY_SECONDS"]
    MAX_RETRIES = os.environ["MAX_RETRIES"]
    ROLE_WEIGHT = os.environ["ROLE_WEIGHT"]

# Custom filter to add class and method information
class ContextFilter(logging.Filter):
    def filter(self, record):
        # Get the current frame (caller of the logger)
        frame = inspect.currentframe().f_back.f_back  # 2 levels up
        # Get the class name (if called from a class method)
        record.class_name = 'N/A'
        if 'self' in frame.f_locals:
            record.class_name = frame.f_locals['self'].__class__.__name__
        # Get the method/function name
        record.method_name = frame.f_code.co_name
        return True

# Configure logging
def setup_logging():
    logger = logging.getLogger('my_app')
    logger.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter with class and method information
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(class_name)s - %(method_name)s - %(message)s'
    )
    console_handler.setFormatter(formatter)

    # Add the custom filter to the logger
    context_filter = ContextFilter()
    logger.addFilter(context_filter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

# Set up logging
setup_logging()
