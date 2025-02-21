""" main file for recommendations """
# repo imports
from src.app.settings import Settings
settings = Settings()
from src.app.services.preprocesor import Preprocesor
preprocesor = Preprocesor()
from src.app.services.embeder import Embeder
embeder = Embeder()

if __name__== '__main__':
    #preprocesor.run()
    df_embed = embeder.jobs()
    print(df_embed.shape)
    print(len(df_embed['embed'][0]))
    