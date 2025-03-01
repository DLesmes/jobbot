""" module to review the preprocessed jobs to expire them """
#base
import ast
import requests
import time
import pandas as pd
# exceptions
from requests.exceptions import RequestException
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
        self.retry_delay_seconds = int(settings.RETRY_DELAY_SECONDS)
        self.max_retries = int(settings.MAX_RETRIES)
    
    def extract(self, path: str):
        jobs_list = open_json(path)
        df = pd.DataFrame(jobs_list)
        print(df.shape)
        return df
    
    def _make_request(self, url: str, retries: int = 0) -> requests.Response:
        """
        Make an HTTP GET request to the given URL with retry logic for 429 status code.
        """
        try:
            response = requests.get(url, timeout=10)  # Add timeout to avoid hanging
            if response.status_code == 429 and retries < self.max_retries:
                # Check for Retry-After header (if provided)
                retry_after = response.headers.get('Retry-After')
                wait_time = int(retry_after) if retry_after and retry_after.isdigit() else self.retry_delay_seconds
                
                print(
                    f"Received 429 status code for {url}. Retrying after {wait_time} seconds "
                    f"(attempt {retries + 1}/{self.max_retries})"
                )
                time.sleep(wait_time)
                return self._make_request(url, retries + 1)
            
            return response
        except RequestException as ex:
            print(f"Error fetching URL {url}: {ex}")
            raise

    def checker(self, job: dict) -> dict:
        """
        Check the availability of a job by making an HTTP request to the job URL.
        Returns a dictionary with job_id and availability status.
        """
        job_id = job.get('job_id')
        url = job.get('link')

        if not job_id or not url:
            print("Invalid job data: missing job_id or link")
            return {'job_id': job_id, 'available': False}

        try:
            response = self._make_request(url)

            # Handle 404 status code (Not Found)
            if response.status_code == 404:
                print(f"Job not found (404): {url}")
                return {'job_id': job_id, 'available': False}

            # Check response content for expired job tags
            response_content = str(response.content)
            if any(tag in response_content for tag in self.tags):
                print(f"Expired job detected: {url}")
                return {'job_id': job_id, 'available': False}

            # Job is available
            return {'job_id': job_id, 'available': True}

        except RequestException:
            print(f"Failed to check job {job_id} at {url}")
            return {'job_id': job_id, 'available': True}
          
    def run(self):
        """
        Run the job availability check using multiprocessing and multithreading.

        :param num_processes: Number of processes to use. If None, uses CPU count.
        :param num_threads_per_process: Number of threads per process.
        :return: List of availability results
        """

        # Extract data (this is done once, not parallelized)
        df_raw = self.extract(path=self.job_offers)
        if 'available' not in df_raw.columns:
            df_raw['available'] = True

        df_available = df_raw[df_raw['available'] != False].copy()
        dict_df_available = df_available[['link', 'job_id']].to_dict(orient='records')

        #loop
        availability = []
        for job in dict_df_available:
            result = self.checker(job)
            availability.append(result)

        return availability
    
    def update(self):
        available = self.run()
        df_raw = self.extract(path=self.job_offers)
        df_raw.reset_index(drop=True, inplace=True)
        available_dict = {job['job_id']:job['available'] for job in available}

        df_raw['available_reviewed'] = df_raw.job_id.map(available_dict)
        ind = df_raw[df_raw['available'] != False].index
        df_raw.loc[ind, 'available'] = df_raw.loc[ind, 'available_reviewed']
        df_updated = df_raw[df_raw['available']==True].copy()
        dict_df_updated = df_updated.to_dict(orient='records')
        print(f'final available offers {len(dict_df_updated)}')
        save_json(self.job_offers, dict_df_updated)      
