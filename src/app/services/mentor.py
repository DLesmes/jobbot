""" module to generate the recomendations """
#base
import ast
from datetime import datetime
import pandas as pd
# repo imports
from src.app.utils import (
    save_json,
    open_json,
    Retriever,
    cosine_similarity_numpy,
    is_english
)
retriever = Retriever()
from src.app.settings import Settings
settings = Settings()

class Mentor():
    """
    """
    def __init__(self):
        self.job_offers = settings.JOB_OFFERS
        self.job_seekers = settings.JOB_SEEKERS
        self.matches = settings.MATCHES
        self.filter_params = ast.literal_eval(settings.FILTER_PARAMS)
        
    def knowledge_based_filter(self, user_id):
        # user customization
        list_users = open_json(self.job_seekers)
        user = [user for user in list_users if user['user_id']==user_id]
        seniority_criteria = user[0]['seniority']
        location_criteria = user[0]['location']
        work_modality_criteria = user[0]['work_modality_english']
        remote_criteria = [ast.literal_eval(val) for val in user[0]['remote']]
        excluded_companies = self.filter_params
        english = ast.literal_eval(user[0]['english'])
        # offers
        list_jobs = open_json(self.job_offers)
        df_jobs = pd.DataFrame(list_jobs)
        df_filtered = df_jobs[
            (df_jobs["seniority"].isin(seniority_criteria)) &  # Filter by seniority
            (df_jobs["location"].isin(location_criteria)) &    # Filter by location
            (df_jobs["work_modality_english"].isin(work_modality_criteria)) &  # Filter by work modality
            (df_jobs["remote"].isin(remote_criteria)) &  # Filter by remote_criteria
            (~df_jobs["company"].isin(excluded_companies))     # Exclude specified companies
        ].copy()
        if not english:
            print(f'filtering only spanish jobs: {df_filtered.shape}')
            df_filtered['english'] = df_filtered['description'].apply(lambda x: is_english(x))
            df_filtered = df_filtered[~df_filtered['english']]
        print(f'current available jobs: {df_filtered.shape}')
        return df_filtered.job_id.to_list()

    def recommend(self):
        df_users = retriever.get_last_embed('users')
        df_jobs = retriever.get_last_embed('jobs')
        list_users = open_json(self.job_seekers)
        dict_matches = []
        for _ , row in df_users.iterrows():
            user = [user for user in list_users if user['user_id']==row['user_id']]
            knowledge_filted_job_id = self.knowledge_based_filter(row['user_id'])
            df_jobs = df_jobs[df_jobs['job_id'].isin(knowledge_filted_job_id)].copy()
            if df_jobs.shape[0]>0:
                list_dict_jobs = df_jobs.to_dict(orient='records')
                list_dict_jobs_scored = [
                    {
                        **job,
                        'skills_similarity': cosine_similarity_numpy(
                            job['embed'],
                            row['embed']
                        ),
                        'role_similarity': cosine_similarity_numpy(
                            job['role_embeds'],
                            row['avg_role_embeds']
                        )
                    }
                    for job in list_dict_jobs
                ]
                df_matches = pd.DataFrame(list_dict_jobs_scored)
                df_matches['score'] = df_matches['role_similarity']*float(user[0]['role_weight'])+df_matches['skills_similarity']*(1-float(user[0]['role_weight']))
                print(f'{'#'*10} user scores: {row['user_id']}\n',df_matches.score.value_counts)
                df_matches = df_matches[df_matches['score']>=float(user[0]['similarity_threshold'])].copy()
                df_matches['match_id'] = row['user_id']+df_matches['job_id']
                df_matches['match_date'] = datetime.today().strftime("%Y-%m-%d")
                df_matches = df_matches[['match_id','match_date','score']].copy()
                tmp_dict_matches = df_matches.to_dict(orient='records')
                dict_matches = dict_matches + tmp_dict_matches
            else:
                print(f'There are no matches for knowledge filter for the user {row['user_id']}')
        return dict_matches
    
    def save_matches(self):
        dict_matches = self.recommend()
        save_json(self.matches, dict_matches)
