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
from src.app.utils import Retriever, save_json
retriever = Retriever()

if __name__== '__main__':
    #preprocesor.run()
    #df_embed = embeder.users()
    #print(df_embed.shape)
    #print(len(df_embed['embed'][0]))
    #dict_matches = mentor.recommend()
    #mentor.save_matches()
    #print(dict_matches)
    #print(len(dict_matches))
    matches = retriever.get_last_matches('K7nP4qR8sT2vX5yZ9aB3cD6eF1gH4jL8m')
    print(matches)
    links = [job['link'] for job in matches]
    save_json('recos.json', links)
    