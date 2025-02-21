""" module with the pipeline to find and export matches """
# base
import ast
# repo imports
from src.app.settings import Settings
settings = Settings()
from src.app.services.preprocesor import Preprocesor
preprocesor = Preprocesor()
from src.app.services.embeder import Embeder
embeder = Embeder()
from src.app.services.mentor import Mentor
mentor = Mentor()
from src.app.utils import Retriever, save_json
retriever = Retriever()

class Seeker():
    """
    """
    def __init__(self):
        self.user_ids = ast.literal_eval(settings.USERS_IDS)
        self.output = settings.OUTPUT_MATCHES

    def run(self):
        preprocesor.run()
        _ = embeder.users()
        __ = embeder.jobs()
        mentor.save_matches()
        for user_id in self.user_ids:
            matches = retriever.get_last_matches(user_id)
            print(matches)
            links = [
                {
                    "link":job['link'],
                    "score":job['score'],
                    #"match_date":job['match_date']
                } for job in matches
            ]
            output_path = f'{self.output}/{user_id}.json'
            save_json(output_path, links)
        