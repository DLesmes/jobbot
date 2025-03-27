"""util functions"""
#base
import os
import json
import re
import logging
logger = logging.getLogger('Jobbot')
from datetime import datetime
# vector management
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
# NLP
from nltk.corpus import words
import nltk
## Download the NLTK words corpus (run once)
nltk.download('words')
## Load English words into a set for faster lookup
english_words = set(words.words())
#repo imports
from src.app.settings import Settings
settings = Settings()


def save_json(pathfile: str, list_dicts):
    """
    Save a list of dictionaries to a JSON file.

    Args:
        pathfile (str): The path to the JSON file.
        list_dicts (list): A list of dictionaries to be saved.
    """
    try:
        json.dump(list_dicts, open(pathfile, 'w'))
        logger.info(f'Storing file at: {pathfile}')
    except Exception as e:
        logger.error(f'Error storing JSON file: {e}')

def open_json(pathfile: str):
    """
    Open and load a JSON file.

    Args:
        pathfile (str): The path to the JSON file.

    Returns:
        The loaded JSON data, or None if an error occurred.
    """
    try:
        logger.info(f'Reading file at: {pathfile}')
        loaded_file = json.load(open(pathfile, 'r'))
        return loaded_file
    except Exception as e:
        logger.error(f'Error reading JSON file: {e}')
        return None

def get_file_paths(directory):
    """
    Get the paths of all files in a directory and its subdirectories.

    Args:
        directory (str): The path to the directory.

    Returns:
        A list of file paths.
    """
    # List to hold paths of all files
    file_paths = []

    try:
        # Walk through directory
        for root, directories, files in os.walk(directory):
            for filename in files:
                # Join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)
    except Exception as e:
        logger.error(f'Error getting file paths: {e}')

    return file_paths

def cosine_similarity_numpy(vec1, vec2):
    """
    Calculate the cosine similarity between two vectors.

    Args:
        vec1 (list or numpy.ndarray): The first vector.
        vec2 (list or numpy.ndarray): The second vector.

    Returns:
        The cosine similarity between the two vectors, or 0 if an error occurred.
    """
    try:
        # Convert inputs to NumPy arrays if they are lists
        vec1 = np.array(vec1) if isinstance(vec1, list) else vec1
        vec2 = np.array(vec2) if isinstance(vec2, list) else vec2
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        similarity = round(dot_product / (norm_vec1 * norm_vec2), 4)
        return similarity
    except Exception as e:
        logger.error(f'Error calculating cosine similarity: {e}')
        return -1

def create_job_markdown_table(job_list):
    """
    Create a Markdown table for a list of job offers.

    Args:
        job_list (list): A list of dictionaries containing job offer information.

    Returns:
        A string containing the Markdown table, or an empty string if an error occurred.
    """
    try:
        # Table header
        markdown = "# 🚀 Latest Job Offers Recommendations!\n"
        markdown += "| 🗃️ **Job offer** | 🌡️**Score** | 🗓️ **publication_date** |\n"
        markdown += "|---|---|---|\n"

        # Process each job entry
        for job in job_list:
            # Extract values from dictionary
            link = job.get('link', '')
            score = job.get('score', 0)
            job_offer = job.get('job_offer', '')
            publication_date = job.get('publication_date', '')

            # Format the job offer column with emoji and hyperlink
            job_offer = re.sub(r"[\[\]\|\(\)]", "", job_offer)
            job_offer_formatted = f"[{job_offer}]({link})"

            # Format score with emoji
            score_formatted = f"{score * 100:.2f}%"

            # Add row to table
            markdown += f"| {job_offer_formatted} | {score_formatted} | {publication_date} |\n"

        return markdown
    except Exception as e:
        logger.error(f'Error creating job markdown table: {e}')
        return ''

def save_markdown_to_file(markdown_content: str, filename: str):
    """
    Save Markdown content to a file.

    Args:
        markdown_content (str): The Markdown content to be saved.
        filename (str): The path to the file where the Markdown content will be saved.
    """
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(markdown_content)
        logger.info(f"Successfully saved markdown to {filename}")
    except Exception as e:
        logger.error(f"Error saving markdown file: {e}")

def is_english(text, threshold=0.6):
    """
    Check if the given text is in English.
    Args:
        text (str): The input text to analyze.
        threshold (float): Minimum fraction of English words required to classify as English.
    Returns:
        bool: True if the text is likely English, False otherwise.
    """
    try:
        # Convert text to lowercase and tokenize into words
        text = text.lower()
        # Use regex to split text into words (removing punctuation)
        word_list = re.findall(r'\b\w+\b', text)

        if not word_list:
            return False  # No words found in text

        # Count how many words are in the English dictionary
        english_word_count = sum(1 for word in word_list if word in english_words)

        # Calculate the fraction of English words
        english_fraction = english_word_count / len(word_list)

        # Return True if the fraction exceeds the threshold
        return english_fraction >= threshold
    except Exception as e:
        logger.error(f'Error checking if text is English: {e}')
        return False


class Retriever:
    """
    A class for retrieving and processing job offers and user embeddings.

    Attributes:
        embedding_path (str): The path to the directory containing embeddings.
        job_offers (str): The path to the JSON file containing job offers.
        matches (str): The path to the JSON file containing user-job matches.
    """

    def __init__(self):
        self.embedding_path = settings.EMBEDDING_PATH
        self.job_offers = settings.JOB_OFFERS
        self.matches = settings.MATCHES

    def _get_specific_file_paths(self, specfic_file: str):
        """
        Get the paths of directories containing a specific file.

        Args:
            specfic_file (str): The name of the file to search for.

        Returns:
            list: A list of directory paths containing the specified file.
        """
        try:
            files_list = get_file_paths(self.embedding_path)
            files_list = [doc for doc in files_list if doc.endswith(specfic_file)]
            dirs_list = [dir.split('/')[-2] for dir in files_list]
            return dirs_list
        except Exception as e:
            logger.error(f"Error getting specific file paths: {e}")
            return []

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
            logger.warning(f"Invalid date string: {date_str}")

    def get_last_run(self, specfic_file: str):
        """
        Get the most recent valid date as a string in YYYY-MM-DD format.

        Args:
            specfic_file (str): The name of the file to search for.

        Returns:
            Optional[str]: Most recent valid date or None if no valid dates.
        """
        try:
            dirs_list = self._get_specific_file_paths(specfic_file)
            date_objects = [dt for dt in (self._parse_date(date) for date in dirs_list) if dt is not None]
            return max(date_objects).strftime("%Y-%m-%d") if date_objects else None
        except Exception as e:
            logger.error(f"Error getting last run date: {e}")
            return None

    def get_last_embed(self, embed_type: str):
        """
        Get the last embeddings for a given type (users or jobs).

        Args:
            embed_type (str): The type of embeddings to retrieve ('users' or 'jobs').

        Returns:
            pandas.DataFrame: A DataFrame containing the last embeddings.
        """
        try:
            dict_ids = {
                "users": "user_id",
                "jobs": "job_id"
            }
            last_run = self.get_last_run(f"{embed_type}.parquet")
            if last_run is not None:
                last_run_path = f"{self.embedding_path}{last_run}/{embed_type}.parquet"
                if os.path.exists(last_run_path):
                    table_last_embeds = pq.read_table(last_run_path)
                    df_last_embeds = table_last_embeds.to_pandas()
                else:
                    df_last_embeds = pd.DataFrame(columns=[dict_ids[embed_type], 'embed'])
            else:
                df_last_embeds = pd.DataFrame(columns=[dict_ids[embed_type], 'embed'])
            return df_last_embeds
        except Exception as e:
            logger.error(f"Error getting last embeddings: {e}")
            return pd.DataFrame()

    def get_last_matches(self, user_id):
        """
        Get the last job matches for a given user ID, sorted by publication date and score.

        Args:
            user_id (str): The ID of the user.

        Returns:
            list: A list of dictionaries containing job information and match scores.
        """
        try:
            dict_jobs = open_json(self.job_offers)
            df_jobs = pd.DataFrame(dict_jobs)
            dict_matches = open_json(self.matches)
            df_matches = pd.DataFrame(dict_matches)
            df_matches['user_id'] = df_matches['match_id'].apply(lambda x: str(x).split('|')[0])
            df_matches['job_id'] = df_matches['match_id'].apply(lambda x: str(x).split('|')[1])
            df_matches_user = df_matches[df_matches['user_id'] == user_id].copy()
            df_matches_user.index = df_matches_user.job_id
            dict_matches_user = df_matches_user['score'].to_dict()
            df_jobs['score'] = df_jobs['job_id'].map(dict_matches_user)
            df_jobs.dropna(
                subset=['score'],
                inplace=True,
                ignore_index=True
            )
            df_jobs.sort_values(
                by=[
                    'publication_date',
                    'score'
                ],
                ascending=[0, 0],
                inplace=True
            )
            dict_jobs = df_jobs.to_dict(orient='records')
            return dict_jobs
        except Exception as e:
            logger.error(f"Error getting last matches: {e}")
            return []
    