""" module to make the preprocesses of the data """
#base
import ast
import uuid
import shortuuid
import pandas as pd
# repo imports
from src.app.settings import Settings
settings = Settings()
from src.app.utils import (
    open_json,
    save_json
)


class Preprocesor:
    """
    make the ETL flow from scraped data
    """
    def __init__(self):
        """
        """
        self.filter_params = ast.literal_eval(settings.FILTER_PARAMS)
        self.job_offers = settings.JOB_OFFERS
        self.data_jobs = settings.DATA_JOBS
        self.namespace = uuid.NAMESPACE_DNS
    
    def extract(self, path: str):
        jobs_list = open_json(path)
        df = pd.DataFrame(jobs_list)
        print(df.shape)
        return df
    
    def augment(self):
        df_raw = self.extract(path=self.data_jobs)
        # setting the id
        df_raw['job_id'] = df_raw['link'].apply(
            lambda x: shortuuid.encode(
                uuid.uuid5(
                    self.namespace,
                    x
                )
            )
        )
        # setting remote
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
        df_preprocessed = self.extract(path=self.job_offers)
        df_concated = pd.concat([df_raw,df_preprocessed])
        df_concated.drop_duplicates(
            subset=['job_id'],
            inplace=True,
            ignore_index=True
        )
        df_concated.dropna(
            subset=['remote'],
            inplace=True,
            ignore_index=True
        )
        print(df_concated.shape)
        return df_concated
    
    def transform(self):
        df = self.augment()
        # filter by fake companies
        df = df[~df["company"].isin(self.filter_params)].copy()
        # droping the query params
        df['link'] = df['link'].apply(lambda x: x.split('?')[0])
        # filter by lasth week jobs
        df['publication_date'] = pd.to_datetime(df['publication_date'])
        one_week_ago = pd.Timestamp.now() - pd.Timedelta(days=7)
        df = df[df['publication_date'] >= one_week_ago].copy()
        print(df.shape)
        # most recent
        df.sort_values(
            by=['publication_date'],
            inplace=True,
            ascending=False
        )
        # drop duplicates
        df.drop_duplicates(
            subset=[
                'link'
            ],
            keep = 'first',
            inplace = True,
            ignore_index = True
        )
        print(df.shape)
        df.drop_duplicates(
            subset=[
                'description'
            ],
            keep = 'first',
            inplace = True,
            ignore_index = True
        )
        print(df.shape)
        return df
    
    def load(self):
        df = self.transform()
        dict_df = df.to_dict(orient='records')
        save_json(self.job_offers, dict_df)
        return dict_df
    
    def run(self):
        self.load()
        
