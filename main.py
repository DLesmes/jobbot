""" main file for recommendations """
# repo imports
from src.app.settings import Settings
settings = Settings()
from src.app.services.preprocesor import Preprocesor
preprocesor = Preprocesor()
#from src.app.services.embeder import Embeder
#embeder = Embeder()
from src.app.services.mentor import Mentor
mentor = Mentor()

if __name__== '__main__':
    #preprocesor.run()
    #df_embed = embeder.users()
    #print(df_embed.shape)
    #print(len(df_embed['embed'][0]))
    dict_matches = mentor.recommend()
    print(dict_matches)
    print(len(dict_matches))
    