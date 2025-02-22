"""util functions"""
#base
import os
import json
from datetime import datetime
# vector management
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
#repo imports
from src.app.settings import Settings
settings = Settings()


def save_json(pathfile:str, list_dicts):
    json.dump(list_dicts, open(pathfile, 'w'))
    print(f'storing file at: {pathfile}')


def open_json(pathfile:str):
    print(f'reading file at: {pathfile}')
    loaded_file = json.load(open(pathfile, 'r'))
    return loaded_file

def get_file_paths(directory):
    # List to hold paths of all files
    file_paths = []

    # Walk through directory
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths

def cosine_similarity_numpy(vec1, vec2):
    # Convert inputs to NumPy arrays if they are lists
    vec1 = np.array(vec1) if isinstance(vec1, list) else vec1
    vec2 = np.array(vec2) if isinstance(vec2, list) else vec2
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return round(dot_product / (norm_vec1 * norm_vec2),4)

def create_job_markdown_table(job_list):
    # Table header
    markdown = "# 🚀 Current Job Offers Recommendations!\n"
    markdown = "| **Job offer** | **publication_date** | **Score** |\n"
    markdown += "|---|---|---|\n"
    
    # Process each job entry
    for job in job_list:
        # Extract values from dictionary
        link = job.get('link', '')
        score = job.get('score', 0)
        job_offer = job.get('job_offer', '')
        publication_date = job.get('publication_date', '')
        
        # Format the job offer column with emoji and hyperlink
        job_offer_formatted = f"[💎 {job_offer}]({link})"
        
        # Format score with emoji
        score_formatted = f"{score}✊"
        
        # Add row to table
        markdown += f"| {job_offer_formatted} | {publication_date} | {score_formatted} |\n"
    
    return markdown

def save_markdown_to_file(markdown_content:str, filename:str):
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(markdown_content)
        print(f"Successfully saved markdown to {filename}")
    except Exception as e:
        print(f"Error saving markdown file: {e}")

class Retriever():
    """
    """
    def __init__(self):
        self.embedding_path = settings.EMBEDDING_PATH
        self.job_offers = settings.JOB_OFFERS
        self.matches = settings.MATCHES
    
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
    
    def get_last_embed(self, embed_type:str):
        """
        """
        dict_ids = {
            "users": "user_id",
            "jobs": "job_id"
        }
        # previows jobs
        last_run = self.get_last_run(f'{embed_type}.parquet')
        if last_run is not None:
            last_run_path = f'{self.embedding_path}{last_run}/{embed_type}.parquet'
            if os.path.exists(last_run_path):
                table_last_embeds = pq.read_table(last_run_path)
                df_last_embeds = table_last_embeds.to_pandas()
            else:
                df_last_embeds = pd.DataFrame(columns=[dict_ids[embed_type],'embed'])
        else:
            df_last_embeds = pd.DataFrame(columns=[dict_ids[embed_type],'embed'])
        
        return df_last_embeds
    
    def get_last_matches(self, user_id):
        dict_jobs = open_json(self.job_offers)
        df_jobs = pd.DataFrame(dict_jobs)
        dict_matches = open_json(self.matches)
        df_matches = pd.DataFrame(dict_matches)
        df_matches['user_id'] = df_matches['match_id'].apply(lambda x: str(x)[:33])
        df_matches['job_id'] = df_matches['match_id'].apply(lambda x: str(x)[33:])
        df_filtered = df_matches[df_matches['user_id']==user_id].copy()
        df_filtered.index = df_filtered.job_id
        dict_filtered = df_filtered['score'].to_dict()
        df_jobs['score'] = df_jobs['job_id'].map(dict_filtered)
        df_jobs.dropna(
            subset=['score'],
            inplace=True,
            ignore_index=True
        )
        df_jobs.sort_values(
            by=['score'],
            ascending=False,
            inplace=True
        )
        print(df_jobs.link)
        dict_jobs = df_jobs.to_dict(orient='records')
        return dict_jobs
    