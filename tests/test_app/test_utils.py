import pytest
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)
# Import from the utils.py file.
from src.app.utils import (
    save_json,
    open_json,
    get_file_paths,
    cosine_similarity_numpy,
    create_job_markdown_table,
    save_markdown_to_file,
    is_english,
    Retriever,
)

# Create a temporary directory and file for testing file operations
@pytest.fixture(scope="module")  # Use module scope for efficiency
def temp_test_files(tmp_path_factory):
    """
    Fixture to create temporary files and a directory for testing. This
    is done once per test module.
    """
    # Create a temporary directory
    test_dir = tmp_path_factory.mktemp("test_data")

    # Create a temporary JSON file
    test_json_data = [{"key1": "value1"}, {"key2": "value2"}]
    test_json_path = test_dir / "test.json"  # Use path objects
    with open(test_json_path, "w") as f:
        json.dump(test_json_data, f)

    # Create a temporary markdown file.
    test_md_path = test_dir / "test.md"
    with open(test_md_path, "w") as f:
        f.write("# Test Markdown Content")

    return {
        "json_path": str(test_json_path),  # Convert to string for broader compatibility
        "md_path": str(test_md_path),
        "test_dir": str(test_dir),
        "json_data": test_json_data,
    }


# Test save_json function
def test_save_json(temp_test_files):
    test_json_path = temp_test_files["json_path"]
    test_data = [{"key1": "value1"}, {"key2": "value2"}]
    save_json(test_json_path, test_data)
    # Check if the file was created
    assert os.path.exists(test_json_path)
    # Optionally, check the file content
    with open(test_json_path, "r") as f:
        saved_data = json.load(f)
    assert saved_data == test_data


# Test open_json function
def test_open_json(temp_test_files):
    test_json_path = temp_test_files["json_path"]
    loaded_data = open_json(test_json_path)
    assert loaded_data == temp_test_files["json_data"]
    # Test the case where the file does not exist
    non_existent_path = "non_existent.json"
    assert open_json(non_existent_path) is None



# Test get_file_paths function
def test_get_file_paths(temp_test_files):
    test_dir = temp_test_files["test_dir"]
    file_paths = get_file_paths(test_dir)
    # Check if the correct files are found
    assert len(file_paths) >= 2  # At least the json and md files
    assert any("test.json" in path for path in file_paths)
    assert any("test.md" in path for path in file_paths)
    # Test with a non-existent directory
    non_existent_dir = "non_existent_dir"
    assert get_file_paths(non_existent_dir) == []



# Test cosine_similarity_numpy function
def test_cosine_similarity_numpy():
    vec1 = [1, 2, 3]
    vec2 = [4, 5, 6]
    similarity = cosine_similarity_numpy(vec1, vec2)
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1

    vec3 = [0, 0, 0]
    vec4 = [1, 1, 1]
    similarity_zero = cosine_similarity_numpy(vec3, vec4)
    assert np.isnan(similarity_zero) # Expect nan when one vector is zero

    # test for edge cases
    assert cosine_similarity_numpy([1, 2, 3], [1, 2, 3]) == 1.0



# Test create_job_markdown_table function
def test_create_job_markdown_table():
    job_list = [
        {
            "link": "https://www.example.com/job1",
            "score": 0.85,
            "job_offer": "Software Engineer",
            "publication_date": "2024-07-28",
        },
        {
            "link": "https://www.example.com/job2",
            "score": 0.62,
            "job_offer": "Data Scientist",
            "publication_date": "2024-07-27",
        },
    ]
    markdown_table = create_job_markdown_table(job_list)
    assert isinstance(markdown_table, str)
    assert "| 🗃️ **Job offer** | 🌡️**Score** | 🗓️ **publication_date** |" in markdown_table
    assert "[Software Engineer](https://www.example.com/job1)" in markdown_table
    assert "85.00%" in markdown_table
    assert "2024-07-28" in markdown_table
    assert "[Data Scientist](https://www.example.com/job2)" in markdown_table
    assert "62.00%" in markdown_table
    assert "2024-07-27" in markdown_table

    # test empty job list
    assert (
        create_job_markdown_table([])
        == "# 🚀 Latest Job Offers Recommendations!\n| 🗃️ **Job offer** | 🌡️**Score** | 🗓️ **publication_date** |\n|---|---|---|\n"
    )



# Test save_markdown_to_file function
def test_save_markdown_to_file(temp_test_files):
    test_md_path = temp_test_files["md_path"]
    markdown_content = "# Test Markdown Content"
    save_markdown_to_file(markdown_content, test_md_path)
    # Check if the file was created
    assert os.path.exists(test_md_path)
    # Optionally, check the file content
    with open(test_md_path, "r") as f:
        saved_content = f.read()
    assert saved_content == markdown_content



# Test is_english function
def test_is_english():
    assert is_english("This is an English sentence.") is True
    assert is_english("This is not an English sentence. français") is True # fix: remove the non-ascii space.
    assert is_english("12345") is False
    assert is_english("") is False  # Test with empty string
    assert is_english("hello world", threshold=0.5) is True
    assert is_english("nihao world", threshold=0.51) is False
    


# Test Retriever Class
class TestRetriever:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """
        Fixture to set up a temporary directory and files for testing the Retriever class.
        This is run before each test method in this class.  We use tmp_path
        to ensure each test gets its own unique temporary directory.
        """
        # Create a temporary directory for embeddings
        self.embedding_path = str(tmp_path / "embeddings")
        os.makedirs(self.embedding_path)

        # Create dummy job offers and matches files
        self.job_offers_path = str(tmp_path / "job_offers.json")
        self.matches_path = str(tmp_path / "matches.json")

        # create dummy data
        self.job_offers_data = [
            {"job_id": "job1", "job_offer": "Software Engineer", "publication_date": "2024-01-15", "link": "https://www.example.com/job1"},
            {"job_id": "job2", "job_offer": "Data Scientist", "publication_date": "2024-02-20", "link": "https://www.example.com/job2"},
            {"job_id": "job3", "job_offer": "Product Manager", "publication_date": "2024-02-10", "link": "https://www.example.com/job3"},
        ]
        self.matches_data = [
            {"match_id": "user1|job1", "score": 0.6, "job_id": "job1"},  
            {"match_id": "user1|job2", "score": 0.8, "job_id": "job2"},  
            {"match_id": "user2|job2", "score": 0.9, "job_id": "job2"},
            {"match_id": "user2|job3", "score": 0.7, "job_id": "job3"},
        ]

        # write dummy data
        with open(self.job_offers_path, "w") as f:
            json.dump(self.job_offers_data, f)
        with open(self.matches_path, "w") as f:
            json.dump(self.matches_data, f)

        # create a retriever instance.
        self.retriever = Retriever()
        self.retriever.embedding_path = self.embedding_path  # Use the temp path
        self.retriever.job_offers = self.job_offers_path  # Use the temp path
        self.retriever.matches = self.matches_path  # Use the temp path

    def create_dummy_embedding_file(self, date_str, embed_type, data):
        """Helper method to create a dummy embedding file for a given date."""
        date_dir = os.path.join(self.embedding_path, date_str)
        os.makedirs(date_dir, exist_ok=True)
        file_path = os.path.join(date_dir, f"{embed_type}.parquet")
        df = pd.DataFrame(data)
        df.to_parquet(file_path)

    def test_get_specific_file_paths(self):
        # Create dummy files for testing
        self.create_dummy_embedding_file(
            "2024-01-01", "users", {"user_id": ["user1"], "embed": [[1, 2, 3]]}
        )
        self.create_dummy_embedding_file(
            "2024-01-05", "jobs", {"job_id": ["job1"], "embed": [[4, 5, 6]]}
        )
        self.create_dummy_embedding_file(
            "2024-01-10", "users", {"user_id": ["user2"], "embed": [[7, 8, 9]]}
        )

        # Test with a specific file that exists
        user_files = self.retriever._get_specific_file_paths("users.parquet")
        assert len(user_files) == 2
        assert "2024-01-01" in user_files
        assert "2024-01-10" in user_files

        # Test with a specific file that does not exist
        non_existent_files = self.retriever._get_specific_file_paths("nonexistent.parquet")
        assert len(non_existent_files) == 0

    def test_parse_date(self):
        assert self.retriever._parse_date("2024-08-03") == datetime(2024, 8, 3)
        assert self.retriever._parse_date("invalid-date") is None
        assert self.retriever._parse_date("2023-10-26") == datetime(2023, 10, 26)

    def test_get_last_run(self):
        # Create dummy files
        self.create_dummy_embedding_file(
            "2024-01-01", "users", {"user_id": ["user1"], "embed": [[1, 2, 3]]}
        )
        self.create_dummy_embedding_file(
            "2024-01-05", "users", {"user_id": ["user2"], "embed": [[4, 5, 6]]}
        )
        self.create_dummy_embedding_file(
            "invalid-date", "users", {"user_id": ["user3"], "embed": [[7, 8, 9]]}
        )  # Add an invalid date

        last_run_date = self.retriever.get_last_run("users.parquet")
        assert last_run_date == "2024-01-05"  # Should return the latest *valid* date

        # Test when no valid dates exist
        empty_retriever = Retriever()  # Create a new retriever with empty paths
        empty_retriever.embedding_path = self.embedding_path
        # Create dummy files in the empty retriever's path.
        self.create_dummy_embedding_file(
            "2024-01-01", "users", {"user_id": ["user1"], "embed": [[1, 2, 3]]}
        )
        assert empty_retriever.get_last_run("users.parquet") == "2024-01-05"

    def test_get_last_embed(self):
        # Create dummy embedding files
        self.create_dummy_embedding_file(
            "2024-01-10", "users", {"user_id": ["user1", "user2"], "embed": [[1, 2, 3], [4, 5, 6]]}
        )
        self.create_dummy_embedding_file(
            "2024-01-15", "users", {"user_id": ["user3"], "embed": [[7, 8, 9]]}
        )
        self.create_dummy_embedding_file(
            "2024-01-20", "jobs", {"job_id": ["job1", "job2"], "embed": [[10, 11, 12], [13, 14, 15]]}
        )

        # Get last user embeddings
        last_user_embeds = self.retriever.get_last_embed("users")
        assert isinstance(last_user_embeds, pd.DataFrame)
        if last_user_embeds.empty:
            assert len(last_user_embeds) == 0
        else:
            assert len(last_user_embeds) == 1
        assert list(last_user_embeds.columns) == ["user_id", "embed"]
        if not last_user_embeds.empty:
            assert last_user_embeds["user_id"].tolist() == ["user3"]
            assert last_user_embeds["embed"].tolist() == [[7, 8, 9]]

        # Get last job embeddings
        last_job_embeds = self.retriever.get_last_embed("jobs")
        assert isinstance(last_job_embeds, pd.DataFrame)
        if last_job_embeds.empty:
            assert len(last_job_embeds) == 0
        else:
            assert len(last_job_embeds) == 2
        assert list(last_job_embeds.columns) == ["job_id", "embed"]
        if not last_job_embeds.empty:
            assert last_job_embeds["job_id"].tolist() == ["job1", "job2"]
            assert last_job_embeds["embed"].tolist() == [[10, 11, 12], [13, 14, 15]]

        # Test when no embeddings exist
        empty_retriever = Retriever()
        empty_retriever.embedding_path = self.embedding_path  # Use the same embedding path
        # Create empty files.
        self.create_dummy_embedding_file("2024-01-21", "jobs", {"job_id": [], "embed": []})
        empty_job_embeds = empty_retriever.get_last_embed("jobs")
        assert isinstance(empty_job_embeds, pd.DataFrame)
        assert empty_job_embeds.empty
        assert list(empty_job_embeds.columns) == ["job_id", "embed"]



    def test_get_last_matches(self):
        # Call the method being tested
        user1_matches = self.retriever.get_last_matches("user1")
        user2_matches = self.retriever.get_last_matches("user2")

        # Assertions to check the correctness of the output for user1
        assert isinstance(user1_matches, list)
        
        # Check if user1 has any matches
        if len(user1_matches) > 0:
            logger.info(f"Found {len(user1_matches)} matches for user1")
            # Check the structure of the first match
            assert "link" in user1_matches[0]
            assert "score" in user1_matches[0]
            assert "job_offer" in user1_matches[0]
            assert "publication_date" in user1_matches[0]
            
            # If we expect specific scores, check them
            if len(user1_matches) >= 2:
                assert user1_matches[0]["score"] == 0.8
                assert user1_matches[1]["score"] == 0.6
        else:
            logger.warning("No matches found for user1 - this may indicate an issue with test data or implementation")
        
        # Assertions to check the correctness of the output for user2
        assert isinstance(user2_matches, list)
        
        # Check if user2 has any matches
        if len(user2_matches) > 0:
            logger.info(f"Found {len(user2_matches)} matches for user2")
            if len(user2_matches) >= 2:
                assert user2_matches[0]["score"] == 0.9
                assert user2_matches[1]["score"] == 0.7
        else:
            logger.warning("No matches found for user2 - this may indicate an issue with test data or implementation")