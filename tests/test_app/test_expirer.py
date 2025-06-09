import pytest
import json
import pandas as pd
import ast
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.app.services.expirer import Expirer
from src.app.settings import Settings

@pytest.fixture
def expirer():
    """Fixture to create an Expirer instance for testing."""
    return Expirer()

@pytest.fixture
def temp_test_dir(tmp_path):
    """Fixture to create and clean up a temporary test directory."""
    test_dir = tmp_path / "test_expirer"
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Cleanup after tests
    if test_dir.exists():
        shutil.rmtree(test_dir)

@pytest.fixture
def sample_job_data(temp_test_dir, job_links):
    """Fixture to create sample job data for testing."""
    data = [
        {
            "job_id": "1",
            "link": job_links['available'] or "https://www.linkedin.com/jobs/view/test-available",
            "title": "Software Engineer",
            "available": True
        },
        {
            "job_id": "2",
            "link": job_links['expired'] or "https://www.linkedin.com/jobs/view/test-expired",
            "title": "Data Scientist",
            "available": True
        },
        {
            "job_id": "3",
            "link": "https://www.linkedin.com/jobs/view/test-broken",
            "title": "Product Manager",
            "available": True
        }
    ]
    file_path = temp_test_dir / "test_jobs.json"
    with open(file_path, 'w') as f:
        json.dump(data, f)
    return file_path

def test_expirer_initialization(expirer):
    """Test that Expirer initializes correctly."""
    assert expirer.job_offers == Settings.JOB_OFFERS
    assert expirer.tags == ast.literal_eval(Settings.AVAILABLE_TAGS)
    assert isinstance(expirer.retry_delay_seconds, int)
    assert isinstance(expirer.max_retries, int)

def test_expirer_extract(expirer, sample_job_data):
    """Test data extraction functionality."""
    # Store original path
    original_job_offers = expirer.job_offers
    
    try:
        # Set temporary path for testing
        expirer.job_offers = str(sample_job_data)
        
        # Test successful extraction
        df = expirer.extract(str(sample_job_data))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # Updated to match actual data length
        assert list(df.columns) == ['job_id', 'link', 'title', 'available']

        # Test file not found - the method returns empty DataFrame on error
        result = expirer.extract("nonexistent_file.json")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0  # Should be empty DataFrame
    finally:
        # Restore original path
        expirer.job_offers = original_job_offers

def test_expirer_make_request(expirer):
    """Test HTTP request functionality with retry logic."""
    with patch('requests.get') as mock_get:
        # Test successful request
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"Job is available"
        mock_get.return_value = mock_response
        
        response = expirer._make_request("https://www.linkedin.com/jobs/view/test")
        assert response.status_code == 200
        assert mock_get.call_count == 1  # Only one call for successful request

        # Reset mock for next test
        mock_get.reset_mock()
        
        # Test 429 with retry
        mock_response.status_code = 429
        mock_response.headers = {'Retry-After': '1'}
        mock_get.side_effect = [mock_response, MagicMock(status_code=200)]
        
        response = expirer._make_request("https://www.linkedin.com/jobs/view/test")
        assert response.status_code == 200
        assert mock_get.call_count == 2  # Two calls due to retry

def test_expirer_checker(expirer, job_links):
    """Test job availability checker."""
    with patch('requests.get') as mock_get:
        # Test available job
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"Job is available"
        mock_get.return_value = mock_response
        
        result = expirer.checker({
            'job_id': '1',
            'link': job_links['available'] or 'https://www.linkedin.com/jobs/view/test'
        })
        assert result['available'] is True

        # Test expired job (404)
        mock_response.status_code = 404
        result = expirer.checker({
            'job_id': '2',
            'link': job_links['expired'] or 'https://www.linkedin.com/jobs/view/test'
        })
        assert result['available'] is False

        # Test expired job (with tag)
        mock_response.status_code = 200
        # Use the first tag from Settings.AVAILABLE_TAGS
        available_tags = ast.literal_eval(Settings.AVAILABLE_TAGS)
        mock_response.content = available_tags[0].encode()  # Use actual tag from settings
        result = expirer.checker({
            'job_id': '3',
            'link': 'https://www.linkedin.com/jobs/view/test'
        })
        assert result['available'] is False

def test_expirer_run(expirer, sample_job_data):
    """Test the main run method."""
    # Store original path
    original_job_offers = expirer.job_offers
    
    try:
        # Set temporary path for testing
        expirer.job_offers = str(sample_job_data)
        
        with patch.object(expirer, 'checker') as mock_checker:
            mock_checker.return_value = {'job_id': '1', 'available': True}
            
            results = expirer.run()
            assert isinstance(results, list)
            assert len(results) == 3  # Updated to match actual data length
            assert all(isinstance(r, dict) for r in results)
            assert all('job_id' in r and 'available' in r for r in results)
    finally:
        # Restore original path
        expirer.job_offers = original_job_offers

def test_expirer_update(expirer, sample_job_data):
    """Test the update method."""
    # Store original path
    original_job_offers = expirer.job_offers
    
    try:
        # Set temporary path for testing
        expirer.job_offers = str(sample_job_data)
        
        with patch.object(expirer, 'run') as mock_run:
            mock_run.return_value = [
                {'job_id': '1', 'available': True},
                {'job_id': '2', 'available': False},
                {'job_id': '3', 'available': True}
            ]
            
            expirer.update()
            
            # Verify the updated file
            with open(sample_job_data, 'r') as f:
                updated_data = json.load(f)
                assert len(updated_data) == 2  # Only available jobs should remain
                assert all(job['available'] for job in updated_data)
    finally:
        # Restore original path
        expirer.job_offers = original_job_offers

def test_expirer_empty_data(expirer, temp_test_dir):
    """Test handling of empty data."""
    # Store original paths
    original_job_offers = expirer.job_offers
    
    try:
        # Create empty test file
        empty_file = temp_test_dir / "empty_jobs.json"
        with open(empty_file, 'w') as f:
            json.dump([], f)
        
        # Set temporary path for testing
        expirer.job_offers = str(empty_file)
        
        # Should not raise any errors
        expirer.run()
        expirer.update()
    finally:
        # Restore original path
        expirer.job_offers = original_job_offers

def test_expirer_checker_with_real_links(expirer, job_links):
    """Test job availability checker with real links if provided."""
    if job_links['available']:
        with patch('requests.get') as mock_get:
            # Mock successful response for available job
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = b"Job is available"
            mock_get.return_value = mock_response
            
            result = expirer.checker({
                'job_id': '1',
                'link': job_links['available']
            })
            assert result['available'] is True, f"Job at {job_links['available']} should be available"

    if job_links['expired']:
        with patch('requests.get') as mock_get:
            # Mock 404 response for expired job
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response
            
            result = expirer.checker({
                'job_id': '2',
                'link': job_links['expired']
            })
            assert result['available'] is False, f"Job at {job_links['expired']} should be expired"

    # Test broken link
    result = expirer.checker({
        'job_id': '3',
        'link': "https://www.linkedin.com/jobs/view/test-broken"
    })
    assert result['available'] is False  # Should be False for broken link
