""" Module to call the custom LinkedIn scraper """
import pandas as pd
from src.app.custom_scraper import customLinkedInScraper


keyword_to_search = "Machine Learning"
scraper = customLinkedInScraper(keyword=keyword_to_search)

print("\n--- Scraping Default (Max 5) ---")
default_jobs = scraper.scrape_jobs(max_jobs=5)
if default_jobs:
    df_default = pd.DataFrame(default_jobs)
    print(f"Successfully scraped {len(default_jobs)} default jobs.")
    print(df_default.head())
    # Optional: Save to CSV/JSON
    # df_default.to_csv(f"{keyword_to_search}_default_jobs.csv", index=False)
else:
    print("No default jobs scraped.")