""" module to review the preprocessed jobs to expire them """
#base
import ast
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import numpy as np
# repo imports
from src.app.settings import Settings
settings = Settings()
from src.app.utils import (
    open_json,
    save_json
)


class Expirer:
    """
    """
    def __init__(self):
        self.job_offers = settings.JOB_OFFERS
        self.tags = ast.literal_eval(settings.AVAILABLE_TAGS)
    
    def extract(self, path: str):
        jobs_list = open_json(path)
        df = pd.DataFrame(jobs_list)
        print(df.shape)
        return df
    
    def checker(self, job: dict):
        try:
            response = requests.get(job['link'])
            if response.status_code == 404:
                print(f'### Status code job == 404: {job['link']}')
                return {
                    'job_id':job['job_id'],
                    'available':False
                }
            else:
                for tag in self.tags:
                    if tag in str(response.content):
                        print(f'### Expired job {job['link']}')
                        return {
                            'job_id':job['job_id'],
                            'available':False
                        }
                    else:
                        return {
                            'job_id':job['job_id'],
                            'available':True
                        }
                    
        except Exception as ex:
            print(f"Error when gettng the job {job['job_id']} from {job['link']} with error: {ex}")
          
    def run(self, num_processes=None, num_threads_per_process=4):
        """
        Run the job availability check using multiprocessing and multithreading.

        :param num_processes: Number of processes to use. If None, uses CPU count.
        :param num_threads_per_process: Number of threads per process.
        :return: List of availability results
        """
        # Set the number of processes
        if num_processes is None:
            num_processes = mp.cpu_count()

        # Extract data (this is done once, not parallelized)
        df_raw = self.extract(path=self.job_offers)
        if 'available' not in df_raw.columns:
            df_raw['available'] = True

        df_available = df_raw[df_raw['available'] == True].copy()
        dict_df_available = df_available[['link', 'job_id']].to_dict(orient='records')

        # Split the data into chunks for each process
        chunks = np.array_split(dict_df_available, num_processes)

        # Create a partial function with fixed arguments
        process_jobs_partial = partial(
            process_jobs,
            checker_func=self.checker,
            num_threads=num_threads_per_process
        )

        # Use ProcessPoolExecutor to parallelize across processes
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(executor.map(process_jobs_partial, chunks))

        # Flatten the results
        availability = [item for sublist in results for item in sublist]

        return availability
    
    def update(self):
        available = self.run()
        df_raw = self.extract(path=self.job_offers)
        available_dict = {job['job_id']:job['available'] for job in available}

        df_raw['available'] = df_raw.job_id.map(available_dict)
        df_updated = df_raw[df_raw['available']==True].copy()
        dict_df_updated = df_updated.to_dict(orient='records')
        print(f'final available offers {len(dict_df_updated)}')
        save_json(self.job_offers, dict_df_updated)

        
def process_jobs(jobs, checker_func, num_threads):
    """
    Process a chunk of jobs using multithreading.

    :param jobs: List of job dictionaries
    :param checker_func: Function to check job availability
    :param num_threads: Number of threads to use
    :return: List of availability results
    """
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(checker_func, jobs))
    return results        

