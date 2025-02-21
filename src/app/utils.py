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

class Retriever():
    """
    """
    def __init__(self):
        self.embedding_path = settings.EMBEDDING_PATH
    
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
    