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
from src.app.services.expirer import Expirer
expirer = Expirer()
from src.app.utils import (
    Retriever,
    create_job_markdown_table,
    save_markdown_to_file
)
retriever = Retriever()

class Seeker():
    """
    """
    def __init__(self):
        self.user_ids = ast.literal_eval(settings.USERS_IDS)
        self.output = settings.OUTPUT_MATCHES

    def run(self):
        preprocesor.run()
        expirer.update()
        embeder.users()
        embeder.jobs()
        mentor.save_matches()
        for user_id in self.user_ids:
            matches = retriever.get_last_matches(user_id)
            links = [
                {
                    "link":job['link'],
                    "score":job['score'],
                    "job_offer":job['vacancy_name'],
                    "publication_date":job['publication_date']
                } for job in matches
            ]
            print(links)
            recommendations = create_job_markdown_table(links)
            output_path = f'{self.output}/{user_id}.md'
            save_markdown_to_file(recommendations, output_path)
        