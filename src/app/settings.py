import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    RESULTS = os.environ["RESULTS"]
    MAP_COUNTRIES_KEYWORDS = os.environ["MAP_COUNTRIES_KEYWORDS"]
    NUM_VACANCIES = os.environ["NUM_VACANCIES"]