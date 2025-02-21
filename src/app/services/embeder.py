""" service to create the logic for daily emmbedings """
# base
import os
from datetime import datetime
import torch
# vector management
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# repo imports
from src.app.utils import (
    get_file_paths,
    open_json
)
from src.app.settings import Settings
settings = Settings()
from src.app.clients.clip import Clip
clip = Clip()

class Embeder():
    """
    """
    def __init__(self):
        self.root = settings.EMBEDDING_PATH
        self.job_offers = settings.JOB_OFFERS
        self.job_seekers = settings.JOB_SEEKERS
        self.embedding_path = settings.EMBEDDING_PATH
        self.today = datetime.today().strftime("%Y-%m-%d")
    
    def _get_specific_file_paths(self, specfic_file:str):
        """
        """
        files_list = get_file_paths(self.embedding_path)
        files_list = [doc for doc in files_list if doc.endswith(specfic_file)]
        dirs_list = [dir.split('/')[-2] for dir in files_list]
        return dirs_list

    def _parse_date(self, date_str: str):
        """
        Parse a date string and return None for invalid dates.
        
        Args:
            date_str (str): Date string in YYYY-MM-DD format.
        
        Returns:
            Optional[datetime]: Parsed datetime object or None if invalid.
        """
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return None

    def get_last_run(self, specfic_file:str):
        """
        Get the most recent valid date as a string in YYYY-MM-DD format.
        
        Returns:
            Optional[str]: Most recent valid date or None if no valid dates.
        """
        dirs_list = self._get_specific_file_paths(specfic_file)
        # Convert strings to datetime objects, filtering out None values
        date_objects = [dt for dt in (self._parse_date(date) for date in dirs_list) if dt is not None]
        
        # Return the most recent date or None if no valid dates
        return max(date_objects).strftime("%Y-%m-%d") if date_objects else None
    
    def users(self):
        # users
        available_users_list = open_json(self.job_seekers)
        df_available_users = pd.DataFrame(available_users_list)
        # previows jobs
        last_run = self.get_last_run('users.parquet')
        if last_run is not None:
            last_run_path = f'{self.root}{last_run}/users.parquet'
            if os.path.exists(last_run_path):
                table_last_embeds = pq.read_table(last_run_path)
                df_last_embeds = table_last_embeds.to_pandas()
            else:
                df_last_embeds = pd.DataFrame(columns=['user_id','embed'])
        else:
            df_last_embeds = pd.DataFrame(columns=['user_id','embed'])
        
        missing_embeds = list(set(df_available_users.user_id)-set(df_last_embeds.user_id))
        if len(missing_embeds) > 0:
            df_missing_embeds = df_available_users[
                df_available_users['user_id'].isin(missing_embeds)
            ][
                [
                    'user_id',
                    'skills'
                ]
            ].copy()
            df_missing_embeds['skills'] = df_missing_embeds['skills'].apply(lambda x: " ".join(x))
            embedings = clip.embed(df_missing_embeds.skills.to_list())
            df_missing_embeds['embed'] = list(embedings)
            df_missing_embeds = df_missing_embeds[['user_id','embed']].copy()
            # For NumPy arrays (GPU tensors)
            tensors = torch.stack(df_missing_embeds['embed'].tolist()).cpu()  # Move to CPU
            df_missing_embeds['embed'] = tensors.numpy().tolist()
        else:
            df_missing_embeds = pd.DataFrame(columns=['user_id','embed'])
        
        df_embeds = pd.concat([df_missing_embeds, df_last_embeds])
        df_embeds.drop_duplicates(
            subset=['user_id'],
            inplace=True,
            ignore_index=True
        )
        table_embeds = pa.Table.from_pandas(df_embeds)
        today_path = f'{self.root}{self.today}'
        if not os.path.exists(today_path):
            os.makedirs(today_path)
        today_path_file = f'{today_path}/users.parquet'
        print(f'storing file at {today_path_file}')
        pq.write_table(table_embeds, today_path_file)
        
        return df_embeds

    def jobs(self):
        # new jobs
        available_jobs_list = open_json(self.job_offers)
        df_available_jobs = pd.DataFrame(available_jobs_list)
        # previows jobs
        last_run = self.get_last_run('jobs.parquet')
        if last_run is not None:
            last_run_path = f'{self.root}{last_run}/jobs.parquet'
            table_last_embeds = pq.read_table(last_run_path)
            df_last_embeds = table_last_embeds.to_pandas()
        else:
            df_last_embeds = pd.DataFrame(columns=['job_id','embed'])
        
        missing_embeds = list(set(df_available_jobs.job_id)-set(df_last_embeds.job_id))
        if len(missing_embeds) > 0:
            df_missing_embeds = df_available_jobs[
                df_available_jobs['job_id'].isin(missing_embeds)
            ][
                [
                    'job_id',
                    'description'
                ]
            ].copy()
            embedings = clip.embed(df_missing_embeds.description.to_list())
            df_missing_embeds['embed'] = list(embedings)
            df_missing_embeds = df_missing_embeds[['job_id','embed']].copy()
            # For NumPy arrays (GPU tensors)
            tensors = torch.stack(df_embeds['embed'].tolist()).cpu()  # Move to CPU
            df_missing_embeds['embed'] = tensors.numpy().tolist()
        else:
            df_missing_embeds = pd.DataFrame(columns=['job_id','embed'])
        
        df_embeds = pd.concat([df_missing_embeds, df_last_embeds])
        df_embeds.drop_duplicates(
            subset=['job_id'],
            inplace=True,
            ignore_index=True
        )
        table_embeds = pa.Table.from_pandas(df_embeds)
        today_path = f'{self.root}{self.today}'
        if not os.path.exists(today_path):
            os.makedirs(today_path)
        today_path_file = f'{today_path}/jobs.parquet'
        print(f'storing file at {today_path_file}')
        pq.write_table(table_embeds, today_path_file)
        
        return df_embeds
    