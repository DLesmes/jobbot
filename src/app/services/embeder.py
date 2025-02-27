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
from src.app.utils import open_json
from src.app.settings import Settings
settings = Settings()
from src.app.utils import Retriever
retriever = Retriever()
from src.app.clients.clip import Clip
clip = Clip()

class Embeder():
    """
    """
    def __init__(self):
        self.root = settings.EMBEDDING_PATH
        self.job_offers = settings.JOB_OFFERS
        self.job_seekers = settings.JOB_SEEKERS
        self.today = datetime.today().strftime("%Y-%m-%d")
    
    def users(self):
        # users
        available_users_list = open_json(self.job_seekers)
        df_available_users = pd.DataFrame(available_users_list)
        # previows users
        df_last_embeds = retriever.get_last_embed('users')
        df_last_embeds = df_last_embeds[df_last_embeds['user_id'].isin(df_available_users['user_id'])].copy()
        
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

    def jobs(self):
        # new jobs
        available_jobs_list = open_json(self.job_offers)
        df_available_jobs = pd.DataFrame(available_jobs_list)
        # previows jobs
        df_last_embeds = retriever.get_last_embed('jobs')
        df_last_embeds = df_last_embeds[df_last_embeds['job_id'].isin(df_available_jobs['job_id'])].copy()
        
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
            tensors = torch.stack(df_missing_embeds['embed'].tolist()).cpu()  # Move to CPU
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
    