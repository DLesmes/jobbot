""" controller to manage the custom scraper process """
#custom scraper
import pandas as pd
from src.app.services.custom_scraper import customLinkedInScraper
#storage
from src.app.utils import save_json, open_json
#env
from src.app.settings import Settings
from dotenv import load_dotenv
load_dotenv()
settings = Settings()


class Searcher:
    """
    Class to manage the custom scraper process.
    It uses the customLinkedInScraper class to scrape job data from LinkedIn.
    It reads the keywords and countries from a JSON file and saves the scraped data to another JSON file.

    """
    def __init__(self):
        self.settings = settings
        self.map_countries_keywords = open_json(settings.MAP_COUNTRIES_KEYWORDS)
    
    def scrape(self):
        if isinstance(self.map_countries_keywords, list):
            print('[1]',type(self.map_countries_keywords))
            print('[2]',self.map_countries_keywords)
            for query_keyword in self.map_countries_keywords:
                role = query_keyword["role"]
                country = query_keyword["country"] # to instantiate the country on custom scraper investigate the geoid
                scraper = customLinkedInScraper(keyword=role)
                default_jobs = scraper.scrape_jobs()
                if default_jobs:
                    df_default = pd.DataFrame(default_jobs)
                    print(f"Successfully scraped {len(default_jobs)} default jobs.")
                    print(df_default.head())
                    # Save the scraped jobs to a JSON file
                    data_jobs = open_json(settings.RESULTS)
                    data_jobs.extend(default_jobs)
                    # Save the updated data to the JSON file
                    save_json(settings.RESULTS, data_jobs)
                else:
                    print("No default jobs scraped.")