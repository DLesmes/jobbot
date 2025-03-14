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
            # filtering avaiabel users
            df_missing_embeds = df_available_users[
                df_available_users['user_id'].isin(missing_embeds)
            ][
                [
                    'user_id',
                    'skills',
                    'job_titles'
                ]
            ].copy()
            # role and skill embeds
            list_dict_roles_users = df_missing_embeds.to_dict(orient='records')
            list_dict_roles_embeds = [
                {
                    **roles,
                    'skills_embeds':torch.stack(list(clip.embed(roles['skills']))).cpu(),
                    'roles_embeds':torch.stack(list(clip.embed(roles['job_titles']))).cpu()
                } for roles in list_dict_roles_users
            ]
            list_dict_role_avg_embeds = [
                {
                    'user_id':roles['user_id'],
                    'avg_skill_embeds': (sum(roles['skills_embeds'])/len(roles['skills_embeds'])).numpy().tolist(),
                    'avg_role_embeds': (sum(roles['roles_embeds'])/len(roles['roles_embeds'])).numpy().tolist()
                } for roles in list_dict_roles_embeds
            ]
            df_missing_embeds = pd.DataFrame(list_dict_role_avg_embeds)
        else:
            df_missing_embeds = pd.DataFrame(columns=['user_id','avg_skill_embeds','avg_role_embeds'])
        
        df_embeds = pd.concat([df_missing_embeds, df_last_embeds])
        df_embeds.drop_duplicates(
            subset=['user_id'],
            inplace=True,
            ignore_index=True
        )
        df_embeds.dropna(inplace=True)
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
        
        missing_embeds = list(set(df_available_jobs.job_id)-set(df_last_embeds.job_id))
        if len(missing_embeds) > 0:
            df_missing_embeds = df_available_jobs[
                df_available_jobs['job_id'].isin(missing_embeds)
            ][
                [
                    'job_id',
                    'skills',
                    'vacancy_name'
                ]
            ][:5000].copy()
            print(f'Missing jobs to embed {len(df_missing_embeds)}')
            # role embedding
            df_missing_embeds['role_embeds'] = torch.stack(list(clip.embed(df_missing_embeds['vacancy_name'].to_list()))).cpu().numpy().tolist()
            df_missing_embeds = df_missing_embeds[['job_id','skills','role_embeds']].copy()
            list_dict_missing_jobs = df_missing_embeds.to_dict(orient='records')
            # skills embedding
            list_dict_roles_embeds = [
                {
                    **roles,
                    'skills_embeds':torch.stack(list(clip.embed(roles['skills']))).cpu()
                } for roles in list_dict_missing_jobs
            ]
            list_dict_missing_avg_embeds = [
                {
                    'job_id':roles['job_id'],
                    'role_embeds': roles['role_embeds'],
                    'avg_skill_embeds': (sum(roles['skills_embeds'])/len(roles['skills_embeds'])).numpy().tolist()
                } for roles in list_dict_roles_embeds
            ]
            df_missing_embeds = pd.DataFrame(list_dict_missing_avg_embeds)
        else:
            df_missing_embeds = pd.DataFrame(columns=['job_id','avg_skill_embeds','role_embeds'])
        
        df_embeds = pd.concat([df_missing_embeds, df_last_embeds])
        df_embeds.drop_duplicates(
            subset=['job_id'],
            inplace=True,
            ignore_index=True
        )
        df_embeds.dropna(inplace=True)
        table_embeds = pa.Table.from_pandas(df_embeds)
        today_path = f'{self.root}{self.today}'
        if not os.path.exists(today_path):
            os.makedirs(today_path)
        today_path_file = f'{today_path}/jobs.parquet'
        print(f'storing file at {today_path_file}')
        pq.write_table(table_embeds, today_path_file)
    