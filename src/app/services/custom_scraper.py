""" custom_scraper to get the joboppening form las hour on linkedin"""

# base
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional, Tuple, Any
#env
from src.app.settings import Settings
from dotenv import load_dotenv
load_dotenv()
settings = Settings()
# web scrapping
from urllib.parse import (
    urlencode,
    urlparse,
    parse_qs,
    urlunparse
)

# Configure logging for better feedback in production
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Helper Functions (Extracted from POC logic where applicable)

def parse_relative_time(time_str: str) -> str:
    """
    Parses LinkedIn's relative time string (e.g., "Hace 17 horas" or "17 hours ago")
    into an ISO format date string (YYYY-MM-DD).

    Args:
        time_str: The relative time string from LinkedIn.

    Returns:
        An estimated date string in "YYYY-MM-DD" format. Returns today's date
        if parsing fails or the time unit is unknown or irrelevant (e.g., "Promoted").
    """
    now = datetime.now()
    try:
        time_str = time_str.lower().strip()
        parts = time_str.split()

        value_str = None
        unit = None

        # Detect format and extract parts
        if "hace" in parts and len(parts) >= 3:  # Spanish format: "hace [value] [unit]"
            value_str = parts[1]
            unit = parts[2]
        elif "ago" in parts and len(parts) >= 3: # English format: "[value] [unit] ago"
            value_str = parts[0]
            unit = parts[1]
        else:
            # Handle other cases like "Just now", "Active today", "Promoted", etc.
            # These don't represent a quantifiable past duration, so default to today.
            logging.info(f"Non-standard time string detected: '{time_str}'. Defaulting to today.")
            return now.strftime("%Y-%m-%d")

        # Convert value to integer
        try:
            value = int(value_str)
        except ValueError:
            logging.warning(f"Could not parse numeric value '{value_str}' from time string: {time_str}. Defaulting to today.")
            return now.strftime("%Y-%m-%d")

        # Determine timedelta based on unit (handling plurals and both languages)
        delta = timedelta(0)
        if "hora" in unit or "hour" in unit:
            delta = timedelta(hours=value)
        elif "día" in unit or "day" in unit:
            delta = timedelta(days=value)
        elif "semana" in unit or "week" in unit:
            delta = timedelta(weeks=value)
        elif "mes" in unit or "month" in unit:
            # Approximate month as 30 days for simplicity
            delta = timedelta(days=value * 30)
        elif "minuto" in unit or "minute" in unit:
            delta = timedelta(minutes=value)
        elif "año" in unit or "year" in unit:
             # Approximate year as 365 days
             delta = timedelta(days=value * 365)
        else:
            logging.warning(f"Unknown time unit '{unit}' in '{time_str}'. Defaulting to today.")
            # delta remains timedelta(0)

        # Calculate and format the estimated post date
        post_date = now - delta
        return post_date.strftime("%Y-%m-%d")

    # Catch potential errors during parsing (e.g., unexpected format, failed int conversion)
    except (ValueError, IndexError, TypeError) as e:
        logging.warning(f"Error parsing time string '{time_str}': {e}. Defaulting to today.")
        return now.strftime("%Y-%m-%d")

def clean_job_link(link: str) -> str:
    """Removes query parameters from a LinkedIn job link."""
    parsed = urlparse(link)
    # Reconstruct the URL with only scheme, netloc, and path
    clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
    return clean_url

# Scraper Class
class customLinkedInScraper:
    """
    Scrapes LinkedIn job postings based on keywords and filters.

    Attributes:
        keyword (str): The primary job keyword for the search.
        location (str): The primary location for the search (e.g., "Colombia").
        geo_id (str): The LinkedIn geographical ID for the location.
    """

    def __init__(
            self,
            keyword: str,
            location: str = "Colombia",
            geo_id: str = "100876405",
            last_12h: str = "r43200", # Default to 12 hours
            full_time: str = "F", # Full-time
            remote: str = "2" # Remote
        ):
        """
        Initializes the customLinkedInScraper.

        Args:
            keyword (str): The job keyword (e.g., "Machine Learning").
            location (str): The location name (default: "Colombia").
            geo_id (str): LinkedIn's geoId for the location (default: "100876405" for Colombia).
        """
        if not keyword:
            raise ValueError("Keyword cannot be empty.")

        self.keyword = keyword
        self.location = location
        self.geo_id = geo_id
        self.last_12h = last_12h
        self.full_time = full_time
        self.remote = remote
        self.session = requests.Session()
        logging.info(f"customLinkedInScraper initialized for keyword='{keyword}', location='{location}'")

    def _make_request(
            self,
            url: str,
            retries: int = 3,
            delay: float = 1.0
        ) -> Optional[BeautifulSoup]:
        """
        Makes an HTTP GET request and returns a BeautifulSoup object.

        Args:
            url (str): The URL to fetch.
            retries (int): Number of times to retry on failure.
            delay (float): Delay in seconds between retries.

        Returns:
            Optional[BeautifulSoup]: A BeautifulSoup object if successful, None otherwise.
        """
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=20) # Increased timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                soup = BeautifulSoup(response.text, 'html.parser')
                # Basic check if LinkedIn might be blocking (e.g., asking for login on search page)
                if "authwall" in response.text or response.url.startswith("https://www.linkedin.com/authwall"):
                     logging.error(f"Blocked by LinkedIn (Authwall) when accessing {url}. Try using proxies or different headers/cookies.")
                     return None
                return soup
            except requests.exceptions.Timeout:
                 logging.warning(f"Timeout occurred for {url}. Retrying ({attempt + 1}/{retries})...")
            except requests.exceptions.HTTPError as e:
                logging.error(f"HTTP Error {e.response.status_code} for {url}. Attempt {attempt + 1}/{retries}.")
                if e.response.status_code in [403, 429]: # Forbidden or Too Many Requests
                     logging.error(f"Likely blocked by LinkedIn (Status: {e.response.status_code}). Consider proxies, delays, or better headers.")
                     return None # Don't retry if likely blocked
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed for {url}: {e}. Retrying ({attempt + 1}/{retries})...")

            time.sleep(delay * (attempt + 1)) # Exponential backoff

        logging.error(f"Failed to fetch URL {url} after {retries} attempts.")
        return None

    def _build_search_url(
            self,
            start: int = 0,
            filters: Optional[Dict[str, str]] = None
        ) -> str:
        """
        Constructs the LinkedIn job search URL with specified parameters and filters.

        Args:
            start (int): The starting index for job results (for pagination).
            filters (Optional[Dict[str, str]]): A dictionary of filter parameters
                                                (e.g., {'f_TPR': 'r86400'}).

        Returns:
            str: The constructed search URL.
        """
        params = {
            'keywords': self.keyword,
            'location': self.location,
            'geoId': self.geo_id,
            'start': start,
            'f_TPR': self.last_12h,      # Time Posted (e.g., r86400 for past 24h)
            'f_JT': self.full_time,       # Job Type (e.g., F for Full-time)
            'f_WT': self.remote,       # Work Type (e.g., 2 for Remote)
            # Add more base filters if needed (e.g., f_E for experience)
        }
        if filters:
            params.update(filters)

        # Remove empty filter values
        params = {k: v for k, v in params.items() if v}

        #LinkedIn often replaces spaces with '+' in keywords, let's ensure that
        if 'keywords' in params:
            params['keywords'] = params['keywords'].replace(' ', '+')

        query_string = urlencode(params)
        return f"{settings.BASE_URL}?{query_string}"

    def get_job_links(self, search_url: str) -> Tuple[List[str], int]:
        """
        Fetches the search results page and extracts all job links.
        Note: This basic version only gets links from the first page load.
              Real-world scraping often requires handling infinite scroll (Selenium/Playwright).

        Args:
            search_url (str): The URL of the LinkedIn job search results page.

        Returns:
            Tuple[List[str], int]: A list of unique job posting URLs and the
                                    total number of results reported by LinkedIn.
        """
        logging.info(f"Fetching job links from: {search_url}")
        soup = self._make_request(search_url)
        if not soup:
            return [], 0

        # --- Extract Total Job Postings ---
        total_jobs = 0
        total_jobs_element = soup.select_one('.results-context-header__job-count') # Refined selector
        if total_jobs_element:
            try:
                total_jobs_text = total_jobs_element.text.replace(',', '').strip()
                total_jobs = int(total_jobs_text)
                logging.info(f"LinkedIn reports {total_jobs} total jobs for this search.")
            except (ValueError, AttributeError) as e:
                logging.warning(f"Could not parse total job count: {e}")
        else:
            logging.warning("Could not find total job count element.")

        # --- Extract Job Links ---
        job_links = set() # Use a set to avoid duplicates
        link_elements = soup.select('ul.jobs-search__results-list li a.base-card__full-link') # More specific selector
        if not link_elements:
             link_elements = soup.select('a.base-card__full-link') # Fallback to POC selector
             if not link_elements:
                  logging.warning("Could not find job link elements using primary or fallback selectors.")


        for link_tag in link_elements:
            if 'href' in link_tag.attrs:
                raw_link = link_tag['href']
                # Basic validation to ensure it's a job link
                if "/jobs/view/" in raw_link:
                    clean_link = clean_job_link(raw_link)
                    job_links.add(clean_link)
                else:
                    logging.debug(f"Skipping non-job link: {raw_link}")


        logging.info(f"Found {len(job_links)} unique job links on the initial page.")

        # Placeholder for pagination/infinite scroll handling

        return list(job_links), total_jobs

    def _parse_job_details(
            self,
            soup: BeautifulSoup,
            job_url: str
        ) -> Optional[Dict[str, Any]]:
        """
        Parses the HTML soup of a job detail page to extract relevant information.

        Args:
            soup (BeautifulSoup): The parsed HTML of the job detail page.
            job_url (str): The URL of the job page being parsed.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the parsed job details,
                                      or None if essential information is missing.
        """
        details = {}

        # Title
        try:
            details['title'] = soup.select_one('h1.top-card-layout__title').text.strip()
        except AttributeError:
            logging.warning(f"Could not find job title for {job_url}")
            return None # Title is essential

        # Company
        try:
            details['company'] = soup.select_one('a.topcard__org-name-link').text.strip()
        except AttributeError:
            details['company'] = None
            logging.warning(f"Could not find company name for {job_url}")

        # Location - This can be tricky, might need refinement based on variations
        try:
            # Attempt 1: Specific bullet point often used for location
            location_span = soup.select_one('span.topcard__flavor--bullet')
            if location_span:
                 details['location'] = location_span.text.strip()
            else:
                 # Attempt 2: Find the container and get the second flavor text
                 subline = soup.select_one('h4.top-card-layout__second-subline')
                 if subline:
                      flavors = subline.select('span.topcard__flavor')
                      if len(flavors) > 1:
                           # Often, the second span.topcard__flavor contains the location
                           # Need to be careful not to grab the date/applicant count here
                           potential_loc = flavors[1].text.strip()
                           # Basic check to avoid grabbing 'X applicants' or relative date
                           if "solicit" not in potential_loc.lower() and "hace" not in potential_loc.lower():
                                details['location'] = potential_loc
                           else: # Fallback if second flavor is not location
                                details['location'] = flavors[0].text.strip() # Maybe it's just the company name span again?
                      elif flavors: # Only one flavor span (might be company name)
                            details['location'] = flavors[0].text.strip() # Less ideal, might just be company
                      else:
                           details['location'] = None
                 else:
                      details['location'] = None

            if not details.get('location'): # If still None after attempts
                 logging.warning(f"Could not determine specific location format for {job_url}")

        except Exception as e: # Catch broader exceptions during complex selection
            details['location'] = None
            logging.warning(f"Error parsing location for {job_url}: {e}")


        # Date
        try:
            time_tag = soup.select_one('span.posted-time-ago__text')
            if time_tag:
                details['date'] = parse_relative_time(time_tag.text)
            else:
                 details['date'] = datetime.now().strftime("%Y-%m-%d") # Default to today if not found
                 logging.warning(f"Could not find posting date for {job_url}")
        except AttributeError:
            details['date'] = datetime.now().strftime("%Y-%m-%d")
            logging.warning(f"Error accessing text of posting date element for {job_url}")


        # Description
        try:
            desc_div = soup.select_one('div.show-more-less-html__markup')
            details['description'] = desc_div.text.strip() if desc_div else None
        except AttributeError:
            details['description'] = None
            logging.warning(f"Could not find description for {job_url}")

        # Job Criteria (Seniority, Employment Type, Function, Industries)
        details['seniority_level'] = None
        details['employment_type'] = None
        details['job_function'] = None
        details['industries'] = None

        criteria_items = soup.select('li.description__job-criteria-item')
        if not criteria_items:
             # Fallback: Look for alternative structure if the primary one fails
             criteria_container = soup.select_one('.job-details-jobs-unified-top-card__job-insight')
             if criteria_container and ' · ' in criteria_container.text:
                  # Very basic split, assumes order: Seniority · Type
                  parts = criteria_container.text.split('·')
                  if len(parts) >= 2:
                       details['seniority_level'] = parts[0].strip()
                       details['employment_type'] = parts[1].strip()
                       logging.info(f"Used fallback criteria parsing for {job_url}")


        for item in criteria_items:
            try:
                header = item.select_one('h3.description__job-criteria-subheader').text.strip().lower()
                text = item.select_one('span.description__job-criteria-text').text.strip()

                if 'nivel de antigüedad' in header or 'seniority level' in header:
                    details['seniority_level'] = text
                elif 'tipo de empleo' in header or 'employment type' in header:
                    details['employment_type'] = text
                elif 'función laboral' in header or 'job function' in header:
                    details['job_function'] = text
                elif 'sectores' in header or 'industries' in header:
                    details['industries'] = text
            except AttributeError:
                logging.debug(f"Could not parse a criteria item for {job_url}")
                continue # Skip this item if structure is unexpected

        return details

    def scrape_single_job(self, job_url: str) -> Optional[Dict[str, Any]]:
        """
        Scrapes the details for a single job posting URL.

        Args:
            job_url (str): The URL of the job posting.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with the scraped job data
                                      in the specified format, or None if scraping fails.
        """
        logging.info(f"Scraping details for: {job_url}")
        soup = self._make_request(job_url)
        if not soup:
            logging.error(f"Failed to get soup object for job URL: {job_url}")
            return None

        parsed_data = self._parse_job_details(soup, job_url)
        if not parsed_data:
            logging.error(f"Failed to parse essential details for job URL: {job_url}")
            return None

        # Format the data according to the requested structure
        job_data = {
            'vacancy_name': parsed_data.get('title'),
            'company': parsed_data.get('company'),
            'location': parsed_data.get('location'),
            'work_modality_english': parsed_data.get('employment_type'), # Field name requested
            'seniority': parsed_data.get('seniority_level'),
            'link': job_url,
            'job_function': parsed_data.get('job_function'),
            'industries': parsed_data.get('industries'),
            'description': parsed_data.get('description'),
            'apply_link': parsed_data.get('apply_link'),
            'publication_date': parsed_data.get('date'),
            'query_keyword': self.keyword, # Use the keyword the scraper was initialized with
            'country': self.location,     # Use the location the scraper was initialized with
            'scraping_date': datetime.now().strftime("%Y-%m-%d")
        }
        return job_data

    def scrape_jobs(
            self,
            filters: Optional[Dict[str, str]] = None,
            max_jobs: Optional[int] = None,
            delay_between_jobs: float = 2.0
        ) -> List[Dict[str, Any]]:
        """
        Performs the complete scraping process: finds job links and scrapes details.

        Args:
            filters (Optional[Dict[str, str]]): LinkedIn filter parameters (e.g., {'f_TPR': 'r86400'}).
                                                 If None, uses default search.
            max_jobs (Optional[int]): The maximum number of job details to scrape.
                                      If None, scrapes all found links (from the first page).
            delay_between_jobs (float): Seconds to wait between scraping job detail pages.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing data for one job posting.
        """
        start_url = self._build_search_url(filters=filters)
        job_links, total_found = self.get_job_links(start_url)
        logging.info(f"Found {total_found} total jobs, scraping details for {len(job_links)} job links...")
        
        if not job_links:
            logging.warning("No job links found. Exiting.")
            return []

        all_job_data = []
        links_to_scrape = job_links[:max_jobs] if max_jobs is not None else job_links
        logging.info(f"Starting detail scraping for {len(links_to_scrape)} job links...")

        for i, link in enumerate(links_to_scrape):
            job_details = self.scrape_single_job(link)
            if job_details:
                all_job_data.append(job_details)

            # Add delay to be respectful to the server
            if i < len(links_to_scrape) - 1:
                 logging.debug(f"Waiting {delay_between_jobs}s before next job...")
                 time.sleep(delay_between_jobs)

        logging.info(f"Finished scraping. Collected details for {len(all_job_data)} jobs.")
        return all_job_data
