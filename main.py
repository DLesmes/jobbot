""" main file for recommendations """
# repo imports
#from src.app.controllers.seeker import Seeker
#seeker = Seeker()
from src.app.services.mentor import Mentor
mentor = Mentor()

if __name__== '__main__':
    #seeker.run()
    mentor.save_matches()