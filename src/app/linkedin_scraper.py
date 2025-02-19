#base
import logging
import json
import re
import unicodedata
from datetime import datetime
import pandas as pd
#env
from src.app.settings import Settings
from dotenv import load_dotenv
import os
load_dotenv()
settings = Settings()
#linkedin
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
from linkedin_jobs_scraper.filters import RelevanceFilters, TimeFilters, TypeFilters, ExperienceLevelFilters, RemoteFilters
# Change root logger level (default is WARN)
logging.basicConfig(level = logging.INFO)
#storage
from src.app.utils import save_json, open_json
jobs = []


def on_error(error):
    print('[ON_ERROR]', error)

def on_end():
    print('[ON_END]')

def linkedin_scraper():
    scraper = LinkedinScraper(
        chrome_executable_path='/usr/local/bin/chromedriver', #'/usr/bin/chromedriver', # Custom Chrome executable path (e.g. /foo/bar/bin/chromedriver) 
        chrome_options=None,  # Custom Chrome options here
        headless=True,  # Overrides headless mode only if chrome_options is None
        max_workers=1,  # How many threads will be spawned to run queries concurrently (one Chrome driver for each thread)
        slow_mo=1.3,  # Slow down the scraper to avoid 'Too many requests (429)' errors
    )
    return scraper

def pilot():
    info = f"[PILOT] Starting pilot | setting up default result file]"
    logging.info(info)
    def save_callback():
        save_json(settings.RESULTS, jobs)

    def on_data(data: EventData):

        jobs_data = {
            'vacancy_name':data.title, #title
            'company':data.company, #company
            'location':data.place, #place
            'work_modality_english':data.employment_type, #employment_type
            'seniority':data.seniority_level, #seniority_level
            'link':data.link, #link
            'job_function':data.job_function, #job_funtion
            'industries':data.industries, #industries
            'description':data.description,#description
            'apply_link':data.apply_link, #apply_link
            'publication_date':data.date,#date
            'query_keyword':f'{cargo}',
            'country':f'{country}',
            'scraping_date':datetime.now().strftime("%Y-%m-%d") #scraping_date
        }

        jobs.append(jobs_data)
        print('[seniority_level'+'-'*10, data.seniority_level)
        print('[ON_DATA]', data.title, data.company, data.date, data.link, len(data.description))
    
    scraper = linkedin_scraper()

    info = f"[PILOT] Starting pilot | setting up listeners]"
    logging.info(info)
    scraper.on(Events.DATA, on_data)
    scraper.on(Events.ERROR, on_error)
    scraper.on(Events.END, on_end)
    scraper.on(Events.END, save_callback)

    
    info = f"[PILOT] running scraping loop]"
    logging.info(info)
    map_countries_keywords = open_json(settings.MAP_COUNTRIES_KEYWORDS)
    if isinstance(map_countries_keywords, list):
        print('[1]',type(map_countries_keywords))
        print('[2]',map_countries_keywords)
        for query_keyword in map_countries_keywords:
            cargo = query_keyword["role"]
            country = query_keyword["country"]
            print(f"{'#'*10} CARGO: {cargo}\n{'#'*10}' COUNTRY: {country}")
            queries = [
                Query(
                    query=f'{cargo}',
                    options=QueryOptions(
                        locations=[country],
                        optimize=False,
                        limit=int(settings.NUM_VACANCIES),
                        filters=QueryFilters(
                            relevance=RelevanceFilters.RECENT,
                            time=TimeFilters.DAY,
                            type=[
                                TypeFilters.FULL_TIME
                            ],
                            experience=[
                                ExperienceLevelFilters.ENTRY_LEVEL,
                                ExperienceLevelFilters.ASSOCIATE,
                                ExperienceLevelFilters.MID_SENIOR
                            ]
                        )
                    )
                )
            ]
            scraper.run(queries)

if __name__== '__main__':
    pilot()