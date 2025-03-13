""" module to make the preprocesses of the data """
#base
import ast
import uuid
import shortuuid
import pandas as pd
import logging
logger = logging.getLogger('Jobbot')
# repo imports
from src.app.settings import Settings
settings = Settings()
from src.app.utils import (
    open_json,
    save_json
)


class Preprocesor:
    """
    A class to handle the ETL (Extract, Transform, Load) flow for processing scraped job data.
    
    This class manages the complete data preprocessing pipeline, including data extraction,
    augmentation, transformation, and loading of job offer data.
    
    Attributes:
        filter_params (dict): Filter parameters loaded from settings
        job_offers (str): Path to job offers data
        data_jobs (str): Path to raw job data
        gral_skills (str): Path to skills data
        namespace (uuid.UUID): UUID namespace for generating unique IDs
    """
    def __init__(self):
        """
        Initialize the Preprocesor with configured settings.
        
        Sets up the necessary attributes by loading configuration parameters
        and initializing the logger.
        """
        self.filter_params = ast.literal_eval(settings.FILTER_PARAMS)
        self.job_offers = settings.JOB_OFFERS
        self.data_jobs = settings.DATA_JOBS
        self.gral_skills = settings.SKILLS
        self.namespace = uuid.NAMESPACE_DNS
        logger.info("Preprocessor initialized with configured settings")
    
    def extract(self, path: str):
        """
        Extract data from a JSON file and convert it to a pandas DataFrame.
        
        Args:
            path (str): Path to the JSON file containing the data
            
        Returns:
            pd.DataFrame: DataFrame containing the extracted data
        """
        logger.info(f"Extracting data from {path}")
        jobs_list = open_json(path)
        df = pd.DataFrame(jobs_list)
        logger.info(f"Extracted dataframe with shape: {df.shape}")
        return df
    
    def augment(self):
        """
        Augment the raw job data with additional features and combine with existing data.
        
        This method performs several augmentation steps:
        - Generates unique job IDs
        - Determines remote work status
        - Extracts relevant skills
        - Combines with existing job offers
        - Removes duplicates and handles missing data
        
        Returns:
            pd.DataFrame: Augmented DataFrame with additional features
        """
        logger.info("Starting data augmentation process")
        df_raw = self.extract(path=self.data_jobs)
        
        logger.debug("Generating job IDs from links")
        df_raw['job_id'] = df_raw['link'].apply(
            lambda x: shortuuid.encode(
                uuid.uuid5(
                    self.namespace,
                    x
                )
            )
        )
        
        logger.debug("Determining remote work status")
        df_raw['remote'] = False
        df_raw['remote'] = (
            df_raw['description'].str.contains(
                'remote',
                case=False,
                na=False
            ) |
            df_raw['vacancy_name'].str.contains(
                'remote',
                case=False,
                na=False
            )
        )
        
        logger.debug("Extracting skills from job descriptions")
        df_skills = self.extract(self.gral_skills)
        gral_skills = df_skills.skills.to_list()
        df_raw['skills'] = df_raw['description'].apply(
            lambda x: [
                skill for skill in gral_skills
                if skill in x.lower()
            ]
        )
        
        df_preprocessed = self.extract(path=self.job_offers)
        df_concated = pd.concat([df_raw,df_preprocessed])
        
        logger.debug("Removing duplicates based on job_id")
        df_concated.drop_duplicates(
            subset=['job_id'],
            inplace=True,
            ignore_index=True
        )
        
        logger.debug("Dropping rows with missing remote status")
        df_concated.dropna(
            subset=['remote'],
            inplace=True,
            ignore_index=True
        )
        
        logger.debug("Dropping rows with missing skills")
        df_concated.dropna(
            subset=['skills'],
            inplace=True,
            ignore_index=True
        )
        
        logger.info(f"Augmented dataframe shape: {df_concated.shape}")
        return df_concated
    
    def transform(self):
        """
        Transform the augmented data by applying various cleaning and filtering operations.
        
        Transformation steps include:
        - Filtering out fake companies
        - Cleaning job links
        - Filtering for recent jobs
        - Sorting by date
        - Removing duplicates
        
        Returns:
            pd.DataFrame: Transformed DataFrame ready for loading
        """
        logger.info("Starting data transformation process")
        df = self.augment()
        
        logger.debug("Filtering out fake companies")
        df = df[~df["company"].isin(self.filter_params)].copy()
        
        logger.debug("Removing query parameters from links")
        df['link'] = df['link'].apply(lambda x: x.split('?')[0])
        
        logger.debug("Filtering for jobs from the last week")
        df['publication_date'] = pd.to_datetime(df['publication_date'])
        one_week_ago = pd.Timestamp.now() - pd.Timedelta(days=7)
        df = df[df['publication_date'] >= one_week_ago].copy()
        df['publication_date'] = df['publication_date'].dt.strftime('%Y-%m-%d')
        logger.info(f"Dataframe shape after date filtering: {df.shape}")
        
        logger.debug("Sorting by publication date (most recent first)")
        df.sort_values(
            by=['publication_date'],
            inplace=True,
            ascending=False
        )
        
        logger.debug("Removing duplicate links")
        df.drop_duplicates(
            subset=['link'],
            keep = 'first',
            inplace = True,
            ignore_index = True
        )
        logger.info(f"Dataframe shape after link deduplication: {df.shape}")
        
        logger.debug("Removing duplicate descriptions")
        df.drop_duplicates(
            subset=['description'],
            keep = 'first',
            inplace = True,
            ignore_index = True
        )
        logger.info(f"Final transformed dataframe shape: {df.shape}")
        return df
    
    def load(self):
        """
        Load the transformed data into a JSON file.
        
        Converts the DataFrame to a dictionary and saves it to the specified
        job offers file path.
        
        Returns:
            list: List of dictionaries containing the processed job records
        """
        logger.info("Starting data loading process")
        df = self.transform()
        dict_df = df.to_dict(orient='records')
        logger.info(f"Saving {len(dict_df)} job records to {self.job_offers}")
        save_json(self.job_offers, dict_df)
        return dict_df
    
    def run(self):
        """
        Execute the complete ETL pipeline.
        
        Runs the entire data processing flow from extraction to loading in sequence.
        
        Returns:
            list: List of dictionaries containing the final processed job records
        """
        logger.info("Running full ETL pipeline")
        result = self.load()
        logger.info("ETL pipeline completed successfully")
        return result
        
