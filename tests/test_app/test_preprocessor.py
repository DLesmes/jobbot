import pytest
import json
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from src.app.services.preprocesor import Preprocesor
from src.app.settings import Settings
import uuid

@pytest.fixture
def temp_test_dir(tmp_path):
    """Fixture to create and clean up a temporary test directory."""
    test_dir = tmp_path / "test_preprocessor"
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Cleanup after tests
    if test_dir.exists():
        shutil.rmtree(test_dir)

@pytest.fixture
def preprocessor(temp_test_dir):
    """Fixture to create a Preprocesor instance with temporary paths."""
    # Create temporary files
    data_jobs = temp_test_dir / "data_jobs.json"
    job_offers = temp_test_dir / "job_offers.json"
    skills = temp_test_dir / "skills.json"
    
    # Create sample data
    sample_jobs = [
        {
            "link": "https://example.com/job1",
            "description": "Python Developer with Machine Learning experience",
            "vacancy_name": "Senior Python Developer",
            "company": "Tech Corp",
            "publication_date": (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        },
        {
            "link": "https://example.com/job2",
            "description": "Remote Java Developer position",
            "vacancy_name": "Java Developer",
            "company": "Remote Inc",
            "publication_date": (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        }
    ]
    
    sample_skills = {
        "skills": ["python", "java", "machine learning", "remote"]
    }
    
    # Save sample data
    with open(data_jobs, 'w') as f:
        json.dump(sample_jobs, f)
    with open(job_offers, 'w') as f:
        json.dump([], f)
    with open(skills, 'w') as f:
        json.dump(sample_skills, f)
    
    # Create preprocessor with temporary paths
    preprocessor = Preprocesor()
    preprocessor.data_jobs = str(data_jobs)
    preprocessor.job_offers = str(job_offers)
    preprocessor.gral_skills = str(skills)
    
    return preprocessor

def test_preprocessor_initialization(preprocessor):
    """Test that Preprocesor initializes correctly."""
    assert isinstance(preprocessor.filter_params, list)
    assert isinstance(preprocessor.namespace, uuid.UUID)  # Changed from int to UUID
    assert preprocessor.job_offers is not None
    assert preprocessor.data_jobs is not None
    assert preprocessor.gral_skills is not None

def test_preprocessor_extract(preprocessor, temp_test_dir):
    """Test data extraction functionality."""
    # Test successful extraction
    df = preprocessor.extract(preprocessor.data_jobs)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ['link', 'description', 'vacancy_name', 'company', 'publication_date']
    
    # Test file not found - should return empty DataFrame
    result = preprocessor.extract("nonexistent_file.json")
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0

def test_preprocessor_augment(preprocessor):
    """Test data augmentation functionality."""
    df = preprocessor.augment()
    
    # Check augmented columns
    assert 'job_id' in df.columns
    assert 'remote' in df.columns
    assert 'skills' in df.columns
    
    # Verify job_id generation
    assert df['job_id'].nunique() == len(df)
    
    # Verify remote detection
    assert df.loc[df['description'].str.contains('remote', case=False), 'remote'].all()
    
    # Verify skills extraction
    assert all(isinstance(skills, list) for skills in df['skills'])
    assert any('python' in skills for skills in df['skills'])

def test_preprocessor_transform(preprocessor):
    """Test data transformation functionality."""
    df = preprocessor.transform()
    
    # Verify date filtering
    one_week_ago = pd.Timestamp.now() - pd.Timedelta(days=7)
    assert all(pd.to_datetime(date) >= one_week_ago for date in df['publication_date'])
    
    # Verify link cleaning
    assert all('?' not in link for link in df['link'])
    
    # Verify sorting
    dates = pd.to_datetime(df['publication_date'])
    assert dates.is_monotonic_decreasing
    
    # Verify no duplicates
    assert not df.duplicated(subset=['link']).any()
    assert not df.duplicated(subset=['description']).any()

def test_preprocessor_load(preprocessor):
    """Test data loading functionality."""
    result = preprocessor.load()
    
    # Verify result format
    assert isinstance(result, list)
    assert all(isinstance(record, dict) for record in result)
    
    # Verify file was created
    assert Path(preprocessor.job_offers).exists()
    
    # Verify file content
    with open(preprocessor.job_offers, 'r') as f:
        loaded_data = json.load(f)
        assert len(loaded_data) == len(result)

def test_preprocessor_run(preprocessor):
    """Test the complete ETL pipeline."""
    result = preprocessor.run()
    
    # Verify final result
    assert isinstance(result, list)
    assert all(isinstance(record, dict) for record in result)
    
    # Verify output file
    assert Path(preprocessor.job_offers).exists()
    with open(preprocessor.job_offers, 'r') as f:
        loaded_data = json.load(f)
        assert len(loaded_data) == len(result)

def test_preprocessor_empty_data(preprocessor, temp_test_dir):
    """Test handling of empty data."""
    # Create empty files with required structure but empty data
    empty_jobs = temp_test_dir / "empty_jobs.json"
    empty_data = [
        {
            "link": "",
            "description": "",
            "vacancy_name": "",
            "company": "",
            "publication_date": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')  # Set date to 30 days ago
        }
    ]
    with open(empty_jobs, 'w') as f:
        json.dump(empty_data, f)
    
    preprocessor.data_jobs = str(empty_jobs)
    
    # Should not raise any errors
    result = preprocessor.run()
    assert isinstance(result, list)
    assert len(result) == 0  # After filtering, should be empty
