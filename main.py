""" main file for recommendations """
# repo imports
from src.app.controllers.seeker import Seeker
seeker = Seeker()

if __name__== '__main__':
    seeker.run()