import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--available-job-link",
        action="store",
        default=None,
        help="URL of an available job posting"
    )
    parser.addoption(
        "--expired-job-link",
        action="store",
        default=None,
        help="URL of an expired job posting"
    )

@pytest.fixture
def job_links(request):
    """Fixture to get job links from command line arguments."""
    return {
        'available': request.config.getoption("--available-job-link"),
        'expired': request.config.getoption("--expired-job-link")
    }
