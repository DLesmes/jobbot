""" module with the pipeline to find and export matches """
# base
import ast
import logging
logger = logging.getLogger('Jobbot')
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
    A class to manage the pipeline for finding and exporting job matches for users.

    This class orchestrates the process of preprocessing data, expiring outdated jobs,
    embedding user and job data, finding matches, and exporting recommendations to
    markdown files. It uses various service classes (e.g., Preprocesor, Embeder, Mentor,
    Expirer, Retriever) to perform these tasks.

    Attributes:
        user_ids (list): A list of user IDs parsed from the settings.USERS_IDS configuration.
        output (str): The directory path where match recommendations are saved, sourced from
            settings.OUTPUT_MATCHES.

    Methods:
        __init__(): Initializes the Seeker instance by loading user IDs and setting the output directory.
        run(): Executes the full pipeline to find and export job matches for all users.
    """
    def __init__(self):
        """
        Initializes the Seeker instance.

        Loads user IDs from the settings configuration and sets the output directory for saving
        match recommendations.

        Raises:
            ValueError: If the USERS_IDS configuration cannot be parsed as a valid Python literal.
            SyntaxError: If there is a syntax error in the USERS_IDS configuration string.
        """
        try:
            self.user_ids = ast.literal_eval(settings.USERS_IDS)
            logger.info(f"Successfully loaded {len(self.user_ids)} user IDs")
        except (ValueError, SyntaxError) as e:
            logger.error(f"Failed to parse USERS_IDS: {str(e)}")
            raise
        self.output = settings.OUTPUT_MATCHES
        logger.info(f"Output directory set to: {self.output}")

    def run(self):
        """
        Executes the full pipeline to find and export job matches.

        This method orchestrates the following steps:
        1. Preprocesses data (currently commented out).
        2. Expires outdated jobs (currently commented out).
        3. Embeds user data (currently commented out).
        4. Embeds job data (currently commented out).
        5. Saves matches using the Mentor service.
        6. Retrieves and processes matches for each user, generating recommendations.
        7. Saves recommendations to markdown files in the output directory.

        For each user, it retrieves the latest matches, formats them into a markdown table,
        and saves the results to a file named after the user ID.

        Raises:
            Exception: If a critical error occurs during pipeline execution, it is logged and re-raised.
        """
        try:
            logger.info("Starting Seeker pipeline execution")
            
            logger.info("Starting preprocessing step")
            #preprocesor.run()
            logger.info("Preprocessing completed")
            
            logger.info("Starting expiring outdated jobs")
            #expirer.update()
            logger.info("Jobs expiration completed")
            
            logger.info("Starting user embedding")
            #embeder.users()
            logger.info("User embedding completed")
            
            logger.info("Starting job embedding")
            embeder.jobs()
            logger.info("Job embedding completed")
            
            logger.info("Saving matches through mentor service")
            mentor.save_matches()
            
            logger.info(f"Processing matches for {len(self.user_ids)} users")
            for user_id in self.user_ids:
                try:
                    logger.info(f"Processing matches for user_id: {user_id}")
                    matches = retriever.get_last_matches(user_id)
                    logger.info(f"Found {len(matches)} matches for user_id: {user_id}")
                    
                    links = [
                        {
                            "link": job['link'],
                            "score": job['score'],
                            "job_offer": job['vacancy_name'],
                            "publication_date": job['publication_date']
                        } for job in matches
                    ]
                    
                    if links:
                        logger.debug(f"Generated links for user_id {user_id}: {links}")
                    else:
                        logger.warning(f"No matches found for user_id: {user_id}")
                    
                    recommendations = create_job_markdown_table(links)
                    output_path = f'{self.output}/{user_id}.md'
                    
                    logger.info(f"Saving recommendations to {output_path}")
                    save_markdown_to_file(recommendations, output_path)
                    logger.info(f"Successfully saved recommendations for user_id: {user_id}")
                    
                except Exception as e:
                    logger.error(f"Error processing user_id {user_id}: {str(e)}", exc_info=True)
                    continue
            
            logger.info("Seeker pipeline execution completed successfully")
            
        except Exception as e:
            logger.error(f"Critical error in Seeker pipeline: {str(e)}", exc_info=True)
            raise
        