""" main file to run the scrapers """
# linkedin scraper
from src.app.linkedin_scraper import pilot
#custom scraper
from src.app.controllers.searcher import Searcher
searcher = Searcher()

if __name__== '__main__':
    #linkedin scraper
    pilot()
    #custom scraper
    searcher.scrape()
