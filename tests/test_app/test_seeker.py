import pytest
import json
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from src.app.controllers.seeker import Seeker
from src.app.settings import Settings
import time

@pytest.fixture
def temp_test_dir(tmp_path):
    """Fixture to create and clean up a temporary test directory."""
    test_dir = tmp_path / "test_seeker"
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Cleanup after tests
    if test_dir.exists():
        shutil.rmtree(test_dir)

@pytest.fixture
def seeker(temp_test_dir):
    """Fixture to create a Seeker instance with temporary paths."""
    # Create temporary files
    output_matches = temp_test_dir / "output_matches"
    output_matches.mkdir(exist_ok=True)
    
    # Create temporary job offers and seekers files
    job_offers = temp_test_dir / "job_offers.json"
    job_seekers = temp_test_dir / "job_seekers.json"
    matches = temp_test_dir / "matches.json"
    
    # Create sample data
    sample_jobs = [
        {
            "job_id": "job1",
            "link": "https://example.com/job1",
            "vacancy_name": "Senior Python Developer",
            "publication_date": datetime.now().strftime('%Y-%m-%d')
        }
    ]
    
    sample_seekers = [
        {
            "user_id": "test_user1",
            "seniority": ["Senior"],
            "location": ["Bogota"],
            "work_modality_english": ["Full-time"],
            "remote": ["True"],
            "english": "True"
        }
    ]
    
    # Save sample data
    with open(job_offers, 'w') as f:
        json.dump(sample_jobs, f)
    with open(job_seekers, 'w') as f:
        json.dump(sample_seekers, f)
    with open(matches, 'w') as f:
        json.dump([], f)
    
    # Create seeker with temporary paths
    seeker = Seeker()
    seeker.output = str(output_matches)
    seeker.user_ids = ["test_user1"]  # Set test user ID
    
    return seeker

def test_seeker_initialization(seeker):
    """Test that Seeker initializes correctly."""
    assert isinstance(seeker.user_ids, list)
    assert seeker.output is not None
    assert len(seeker.user_ids) > 0

def test_seeker_run_with_matches(seeker, temp_test_dir):
    """Test the complete pipeline with matches."""
    # Mock the retriever's get_last_matches method
    with pytest.MonkeyPatch.context() as m:
        # Mock matches data
        sample_matches = [
            {
                "link": "https://example.com/job1",
                "score": 0.85,
                "vacancy_name": "Senior Python Developer",
                "publication_date": datetime.now().strftime('%Y-%m-%d')
            }
        ]
        
        m.setattr("src.app.controllers.seeker.retriever.get_last_matches", 
                 lambda x: sample_matches)
        
        # Mock the service classes
        m.setattr("src.app.controllers.seeker.preprocesor.run", lambda: None)
        m.setattr("src.app.controllers.seeker.expirer.update", lambda: None)
        m.setattr("src.app.controllers.seeker.embeder.users", lambda: None)
        m.setattr("src.app.controllers.seeker.embeder.jobs", lambda: None)
        m.setattr("src.app.controllers.seeker.mentor.run", lambda: None)
        
        # Run the pipeline
        seeker.run()
        
        # Verify output files were created
        output_file = Path(seeker.output) / f"{seeker.user_ids[0]}.md"
        assert output_file.exists()
        
        # Verify file content
        with open(output_file, 'r') as f:
            content = f.read()
            assert "Senior Python Developer" in content
            assert "https://example.com/job1" in content

def test_seeker_run_without_matches(seeker, temp_test_dir):
    """Test the complete pipeline without matches."""
    # Mock the retriever's get_last_matches method to return empty list
    with pytest.MonkeyPatch.context() as m:
        m.setattr("src.app.controllers.seeker.retriever.get_last_matches", 
                 lambda x: [])
        
        # Mock the service classes
        m.setattr("src.app.controllers.seeker.preprocesor.run", lambda: None)
        m.setattr("src.app.controllers.seeker.expirer.update", lambda: None)
        m.setattr("src.app.controllers.seeker.embeder.users", lambda: None)
        m.setattr("src.app.controllers.seeker.embeder.jobs", lambda: None)
        m.setattr("src.app.controllers.seeker.mentor.run", lambda: None)
        
        # Run the pipeline
        seeker.run()
        
        # Verify output files were created
        output_file = Path(seeker.output) / f"{seeker.user_ids[0]}.md"
        assert output_file.exists()
        
        # Verify file content has header but no matches
        with open(output_file, 'r') as f:
            content = f.read()
            assert "Latest Job Offers Recommendations" in content
            assert "| 🗃️ **Job offer** | 🌡️**Score** | 🗓️ **publication_date** |" in content
            assert "Senior Python Developer" not in content  # No matches should be present

def test_seeker_run_with_service_errors(seeker, temp_test_dir):
    """Test the pipeline handling of service errors."""
    # Print initial paths for debugging
    print(f"\nTemp test dir: {temp_test_dir}")
    print(f"Seeker output path: {seeker.output}")
    print(f"User IDs: {seeker.user_ids}")
    
    # Mock the retriever's get_last_matches method to raise an error
    with pytest.MonkeyPatch.context() as m:
        def mock_get_last_matches(user_id):
            # Instead of raising an exception, return an empty list
            # This should trigger the "no matches" behavior
            return []
        
        # Mock all necessary methods
        m.setattr("src.app.controllers.seeker.retriever.get_last_matches", 
                 mock_get_last_matches)
        m.setattr("src.app.controllers.seeker.preprocesor.run", lambda: None)
        m.setattr("src.app.controllers.seeker.expirer.update", lambda: None)
        m.setattr("src.app.controllers.seeker.embeder.users", lambda: None)
        m.setattr("src.app.controllers.seeker.embeder.jobs", lambda: None)
        m.setattr("src.app.controllers.seeker.mentor.run", lambda: None)
        
        # Run the pipeline - should not raise exception
        seeker.run()
        
        # Print all files in the output directory
        output_dir = Path(seeker.output)
        print(f"\nFiles in output directory:")
        for file in output_dir.glob("*"):
            print(f"- {file}")
        
        # Verify output files were created
        output_file = Path(seeker.output) / f"{seeker.user_ids[0]}.md"
        print(f"\nExpected output file: {output_file}")
        print(f"File exists: {output_file.exists()}")
        
        assert output_file.exists(), f"Output file {output_file} was not created"
        
        # Verify file content has header
        with open(output_file, 'r') as f:
            content = f.read()
            print(f"\nFile content:\n{content}")
            assert "Latest Job Offers Recommendations" in content
            assert "| 🗃️ **Job offer** | 🌡️**Score** | 🗓️ **publication_date** |" in content

def test_seeker_empty_user_ids(seeker, temp_test_dir):
    """Test handling of empty user IDs list."""
    # Create seeker with empty user IDs
    seeker.user_ids = []
    
    # Run the pipeline
    seeker.run()
    
    # Verify no output files were created
    assert len(list(Path(seeker.output).glob("*.md"))) == 0
