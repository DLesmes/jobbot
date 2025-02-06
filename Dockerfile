from joyzoursky/python-chromedriver:3.9

COPY . .
RUN pip install -r ./requirements.txt

ENV FILE="linkedin/dict_smart_query_keyword.json"

COPY data/data_jobs.json /mnt/
CMD ["python", "main.py"]