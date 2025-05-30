import pytest
import json
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from src.app.services.mentor import Mentor
from src.app.settings import Settings

@pytest.fixture
def temp_test_dir(tmp_path):
    """Fixture to create and clean up a temporary test directory."""
    test_dir = tmp_path / "test_mentor"
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Cleanup after tests
    if test_dir.exists():
        shutil.rmtree(test_dir)

@pytest.fixture
def mentor(temp_test_dir):
    """Fixture to create a Mentor instance with temporary paths."""
    # Create temporary files
    job_offers = temp_test_dir / "job_offers.json"
    job_seekers = temp_test_dir / "job_seekers.json"
    matches = temp_test_dir / "matches.json"
    
    # Create sample job offers data
    sample_jobs = [
        {
            "job_id": "job1",
            "seniority": "Senior",
            "location": "Bogota",
            "work_modality_english": "Full-time",
            "remote": True,
            "company": "Tech Corp",
            "description": "Python Developer position",
            "avg_skill_embeds": [0.1, 0.2, 0.3],
            "role_embeds": [0.4, 0.5, 0.6]
        },
        {
            "job_id": "job2",
            "seniority": "Junior",
            "location": "Medellin",
            "work_modality_english": "Part-time",
            "remote": False,
            "company": "Startup Inc",
            "description": "Java Developer position",
            "avg_skill_embeds": [0.2, 0.3, 0.4],
            "role_embeds": [0.5, 0.6, 0.7]
        }
    ]
    
    # Create sample job seekers data
    sample_seekers = [
        {
            "user_id": "user1",
            "seniority": ["Senior"],
            "location": ["Bogota"],
            "work_modality_english": ["Full-time"],
            "remote": ["True"],
            "english": "True",
            "role_weight": "0.7",
            "similarity_threshold": "0.5"
        }
    ]
    
    # Save sample data
    with open(job_offers, 'w') as f:
        json.dump(sample_jobs, f)
    with open(job_seekers, 'w') as f:
        json.dump(sample_seekers, f)
    with open(matches, 'w') as f:
        json.dump([], f)
    
    # Create mentor with temporary paths
    mentor = Mentor()
    mentor.job_offers = str(job_offers)
    mentor.job_seekers = str(job_seekers)
    mentor.matches = str(matches)
    
    return mentor

def test_mentor_initialization(mentor):
    """Test that Mentor initializes correctly."""
    assert isinstance(mentor.filter_params, list)
    assert mentor.job_offers is not None
    assert mentor.job_seekers is not None
    assert mentor.matches is not None

def test_knowledge_based_filter(mentor):
    """Test knowledge-based filtering functionality."""
    # Test filtering for user1
    filtered_jobs = mentor.knowledge_based_filter("user1")
    assert isinstance(filtered_jobs, list)
    assert len(filtered_jobs) > 0
    
    # Test filtering with non-existent user
    filtered_jobs = mentor.knowledge_based_filter("nonexistent_user")
    assert filtered_jobs is None

def test_knowledge_based_filter_english(mentor, temp_test_dir):
    """Test knowledge-based filtering with English language preference."""
    # Create a user that only wants Spanish jobs
    spanish_user = {
        "user_id": "spanish_user",
        "seniority": ["Senior"],
        "location": ["Bogota"],
        "work_modality_english": ["Full-time"],
        "remote": ["True"],
        "english": "False",
        "role_weight": "0.7",
        "similarity_threshold": "0.5"
    }
    
    # Update job seekers file
    with open(mentor.job_seekers, 'r') as f:
        users = json.load(f)
    users.append(spanish_user)
    with open(mentor.job_seekers, 'w') as f:
        json.dump(users, f)
    
    filtered_jobs = mentor.knowledge_based_filter("spanish_user")
    assert isinstance(filtered_jobs, list)

def test_recommend(mentor):
    """Test recommendation generation."""
    # Mock the retriever's get_last_embed method
    with pytest.MonkeyPatch.context() as m:
        # Mock user embeddings
        user_embeddings = pd.DataFrame({
            'user_id': ['user1'],
            'avg_skill_embeds': [[0.1, 0.2, 0.3]],
            'avg_role_embeds': [[0.4, 0.5, 0.6]]
        })
        
        # Mock job embeddings
        job_embeddings = pd.DataFrame({
            'job_id': ['job1'],
            'avg_skill_embeds': [[0.1, 0.2, 0.3]],
            'role_embeds': [[0.4, 0.5, 0.6]]
        })
        
        m.setattr("src.app.services.mentor.retriever.get_last_embed", 
                 lambda x: user_embeddings if x == 'users' else job_embeddings)
        
        recommendations = mentor.recommend()
        assert isinstance(recommendations, list)
        if recommendations:  # If there are any matches
            assert all(isinstance(match, dict) for match in recommendations)
            assert all('match_id' in match for match in recommendations)
            assert all('match_date' in match for match in recommendations)
            assert all('score' in match for match in recommendations)

def test_run(mentor):
    """Test the complete recommendation process."""
    # Mock the retriever's get_last_embed method
    with pytest.MonkeyPatch.context() as m:
        # Mock user embeddings
        user_embeddings = pd.DataFrame({
            'user_id': ['user1'],
            'avg_skill_embeds': [[0.1, 0.2, 0.3]],
            'avg_role_embeds': [[0.4, 0.5, 0.6]]
        })
        
        # Mock job embeddings
        job_embeddings = pd.DataFrame({
            'job_id': ['job1'],
            'avg_skill_embeds': [[0.1, 0.2, 0.3]],
            'role_embeds': [[0.4, 0.5, 0.6]]
        })
        
        m.setattr("src.app.services.mentor.retriever.get_last_embed", 
                 lambda x: user_embeddings if x == 'users' else job_embeddings)
        
        mentor.run()
        
        # Verify matches file was created and updated
        assert Path(mentor.matches).exists()
        with open(mentor.matches, 'r') as f:
            matches = json.load(f)
            assert isinstance(matches, list)

def test_empty_data(mentor, temp_test_dir):
    """Test handling of empty data."""
    # Create empty files
    empty_jobs = temp_test_dir / "empty_jobs.json"
    empty_seekers = temp_test_dir / "empty_seekers.json"
    
    with open(empty_jobs, 'w') as f:
        json.dump([], f)
    with open(empty_seekers, 'w') as f:
        json.dump([], f)
    
    mentor.job_offers = str(empty_jobs)
    mentor.job_seekers = str(empty_seekers)
    
    # Should not raise any errors
    result = mentor.run()
    assert result is None
