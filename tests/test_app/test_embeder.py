import pytest
import torch
import pandas as pd
import os
import shutil
from datetime import datetime
from src.app.services.embeder import Embeder
from src.app.settings import Settings

@pytest.fixture
def embeder():
    """Fixture to create an Embeder instance for testing."""
    return Embeder()

@pytest.fixture
def sample_job_data():
    """Fixture to provide sample job data for testing."""
    return [
        {
            "job_id": "1",
            "skills": ["Python", "Machine Learning", "Data Analysis"],
            "vacancy_name": "Data Scientist"
        },
        {
            "job_id": "2",
            "skills": ["Java", "Spring", "Microservices"],
            "vacancy_name": "Backend Developer"
        }
    ]

@pytest.fixture
def sample_user_data():
    """Fixture to provide sample user data for testing."""
    return [
        {
            "user_id": "1",
            "skills": ["Python", "Data Science", "SQL"],
            "job_titles": ["Data Analyst", "Data Scientist"]
        },
        {
            "user_id": "2",
            "skills": ["Java", "Spring", "REST"],
            "job_titles": ["Backend Developer", "Software Engineer"]
        }
    ]

@pytest.fixture
def temp_test_dir(tmp_path):
    """Fixture to create and clean up a temporary test directory."""
    # Create a unique test directory
    test_dir = tmp_path / "test_embeder"
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Cleanup after tests
    if test_dir.exists():
        shutil.rmtree(test_dir)

def test_embeder_initialization(embeder):
    """Test that Embeder initializes correctly."""
    assert embeder.root == Settings.EMBEDDING_PATH
    assert embeder.job_offers == Settings.JOB_OFFERS
    assert embeder.job_seekers == Settings.JOB_SEEKERS
    assert embeder.today == datetime.today().strftime("%Y-%m-%d")
    assert embeder.embedder is not None

def test_embeder_jobs_embedding(embeder, sample_job_data, temp_test_dir):
    """Test job embedding generation and storage."""
    # Store original paths
    original_root = embeder.root
    original_job_offers = embeder.job_offers
    
    try:
        # Set temporary paths for testing
        embeder.root = str(temp_test_dir) + "/"  # Add trailing slash to match behavior
        embeder.job_offers = str(temp_test_dir / "test_jobs.json")
        
        # Create test data file
        pd.DataFrame(sample_job_data).to_json(embeder.job_offers, orient='records')
        
        # Test embedding generation
        embeder.jobs()
        
        # Verify that embeddings were created
        today_path = temp_test_dir / embeder.today
        assert (today_path / 'jobs.parquet').exists()
        
    finally:
        # Restore original paths
        embeder.root = original_root
        embeder.job_offers = original_job_offers

def test_embeder_users_embedding(embeder, sample_user_data, temp_test_dir):
    """Test user embedding generation and storage."""
    # Store original paths
    original_root = embeder.root
    original_job_seekers = embeder.job_seekers
    
    try:
        # Set temporary paths for testing
        embeder.root = str(temp_test_dir) + "/"  # Add trailing slash to match behavior
        embeder.job_seekers = str(temp_test_dir / "test_users.json")
        
        # Create test data file
        pd.DataFrame(sample_user_data).to_json(embeder.job_seekers, orient='records')
        
        # Test embedding generation
        embeder.users()
        
        # Verify that embeddings were created
        today_path = temp_test_dir / embeder.today
        assert (today_path / 'users.parquet').exists()
        
    finally:
        # Restore original paths
        embeder.root = original_root
        embeder.job_seekers = original_job_seekers

def test_embeder_embedding_dimensions(embeder, sample_job_data):
    """Test that generated embeddings have correct dimensions."""
    # Test job embeddings
    job_embeddings = embeder.embedder.embed(sample_job_data[0]['skills'])
    assert isinstance(job_embeddings, torch.Tensor)
    assert job_embeddings.shape[1] == 768  # Assuming 768 dimensions

def test_embeder_empty_data(embeder, temp_test_dir):
    """Test handling of empty data."""
    # Store original paths
    original_root = embeder.root
    original_job_offers = embeder.job_offers
    original_job_seekers = embeder.job_seekers
    
    try:
        # Set temporary paths for testing
        embeder.root = str(temp_test_dir) + "/"  # Add trailing slash to match behavior
        embeder.job_offers = str(temp_test_dir / "empty_jobs.json")
        embeder.job_seekers = str(temp_test_dir / "empty_users.json")
        
        # Create empty test files
        pd.DataFrame().to_json(embeder.job_offers, orient='records')
        pd.DataFrame().to_json(embeder.job_seekers, orient='records')
        
        # Should not raise any errors
        embeder.jobs()
        embeder.users()
        
    finally:
        # Restore original paths
        embeder.root = original_root
        embeder.job_offers = original_job_offers
        embeder.job_seekers = original_job_seekers
