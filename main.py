""" main file for recommendations """
# base
import logging
from src.app.settings import setup_logging
setup_logging('Jobbot')
logger = logging.getLogger('Jobbot')
# repo imports
from src.app.controllers.seeker import Seeker
seeker = Seeker()

if __name__== '__main__':
    logger.info('running the job seeker...')
    seeker.run()
    