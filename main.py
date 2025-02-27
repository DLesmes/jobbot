""" main file for recommendations """
# repo imports
from src.app.controllers.seeker import Seeker
seeker = Seeker()
from src.app.services.expirer import Expirer
expirer = Expirer()
from src.app.services.preprocesor import Preprocesor
preprocesor = Preprocesor()

if __name__== '__main__':
    #preprocesor.run()
    #expirer.update()
    seeker.run()
    