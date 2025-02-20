""" module to make the preprocesses of the data """
#base
import pandas as pd
# repo imports
from settings import Settings
settings = Settings()
from src.app.utils import open_json


class preprocesor:
    """
    
    """
    def __init__(self):
        """
        """
        self.filter_params = settings.FILTER_PARAMS
    
    def extract(self, path: str = settings.DATA_JOBS):
        jobs_list = open_json(path)
        df = pd.DataFrame(jobs_list)
        return df
    
    def transform(self):
        df = self.extract
        # filter by fake companies
        df = df[~df["company"].isin(self.filter_params)].copy()
        # drop duplicates
        




