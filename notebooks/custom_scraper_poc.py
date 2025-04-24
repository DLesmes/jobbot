#%%
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
#%% 
link_keyword_and_location = 'https://www.linkedin.com/jobs/search?keywords=Machine%20Learning&location=Colombia&geoId=100876405&position=1&pageNum=0'
link_ultimas24h = 'https://www.linkedin.com/jobs/search?keywords=Machine%20Learning&location=Colombia&geoId=100876405&f_TPR=r86400&f_JT=F&position=1&pageNum=0'
link_jornada_completa = 'https://www.linkedin.com/jobs/search?keywords=Machine%20Learning&location=Colombia&geoId=100876405&f_TPR=r86400&position=1&pageNum=0'
link_remoto = 'https://www.linkedin.com/jobs/search?keywords=Machine%20Learning&location=Colombia&geoId=100876405&f_TPR=r86400&f_WT=2&position=1&pageNum=0'
url = "https://www.linkedin.com/jobs/search?keywords=Machine%2BLearning&location=Colombia&geoId=100876405&f_TPR=r86400&currentJobId=4214881225&position=27&pageNum=0"
res = requests.get(url)
soup = BeautifulSoup(res.text, 'html.parser')
soup
# %%
# Extract the total number of job postings indicated in the page
total_job_postings = soup.select_one('h1>span').text
total_job_postings = total_job_postings.replace(',', '')
total_job_postings = int(total_job_postings)
total_job_postings
#%%
# Extract the job postings
job_postings = soup.select('a.base-card__full-link')
job_links = [job['href'] for job in job_postings if 'href' in job.attrs]
len(job_links)
#%%
job_link = job_links[0].split('?')[0]
job_link
# %%
res_job = requests.get(job_link)
soup_job = BeautifulSoup(res_job.text, 'html.parser')
soup_job
# %%
soup_job.text.strip()
# %%
# Extract the job description
job_description = soup_job.select_one('div.show-more-less-html__markup').text
job_description.strip()
# %%
# Extract the job title
job_title = soup_job.select_one('h1').text
job_title.strip()

# %%
# Extract the company name
company_name = soup_job.select_one('a.topcard__org-name-link').text
company_name.strip()

# %%
# Extract the location
location = soup_job.select_one('span.topcard__flavor').text
location.strip()
# %%
# Extract the job posting date
job_posting_date = soup_job.select_one('time').text
time_expresion = job_posting_date.strip().split(' ')
hours = [int(i) for i in time_expresion if i.isdigit()]
job_posting_date = datetime.now() - timedelta(hours=hours[0])
str(job_posting_date).split(' ')[0]
# %%
# Extract the seniority level
seniority_level = soup_job.select_one('span.description__job-criteria-text').text
seniority_level.strip()
# %%
# Extract the job function
job_function = soup_job.select_one('span.description__job-criteria-text').text
job_function.strip()

# %%
# Extract the industries
industries = soup_job.select_one('span.description__job-criteria-text').text
industries.strip()

# %%
