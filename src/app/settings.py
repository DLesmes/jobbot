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