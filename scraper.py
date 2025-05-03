""" main file to run te scraper """
from src.app.linkedin_scraper import pilot
#custom scraper
import pandas as pd
from src.app.custom_scraper import customLinkedInScraper
#storage
from src.app.utils import save_json, open_json
#env
from src.app.settings import Settings
from dotenv import load_dotenv
load_dotenv()
settings = Settings()

if __name__== '__main__':
    #linkedin scraper
    pilot()
    #custom scraper
    keyword_to_search = "Machine Learning"
    scraper = customLinkedInScraper(keyword=keyword_to_search)

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