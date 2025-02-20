from src.app.linkedin_scraper import pilot
from src.app.services.preprocesor import Preprocesor
preprocesor = Preprocesor()

if __name__== '__main__':
    #pilot()
    preprocesor.run()
    