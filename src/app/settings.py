import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    RESULTS = os.environ["RESULTS"]
    MAP_COUNTRIES_KEYWORDS = os.environ["MAP_COUNTRIES_KEYWORDS"]
    NUM_VACANCIES = os.environ["NUM_VACANCIES"]
    FILTER_PARAMS = os.environ["FILTER_PARAMS"]
    DATA_JOBS = os.environ["DATA_JOBS"]