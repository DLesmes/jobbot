FROM joyzoursky/python-chromedriver:3.9
WORKDIR /src/app
# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libffi-dev \
    libssl-dev

COPY . .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r ./requirements.txt

COPY data/data_jobs.json /mnt/
COPY data/dict_smart_query_keyword.json /mnt/
CMD ["python", "main.py"]