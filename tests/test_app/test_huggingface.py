import pytest
import torch
import numpy as np
from src.app.clients.huggingface import HuggingFace

@pytest.fixture
def hf_client():
    """Fixture to create a HuggingFace client instance for testing."""
    return HuggingFace()

def test_hf_initialization(hf_client):
    """Test that HuggingFace client initializes correctly."""
    assert hf_client.model_id is not None
    assert hf_client.device in ['cuda', 'cpu']
    assert hf_client.model is not None

def test_hf_embedding(hf_client):
    """Test that HuggingFace can generate embeddings for text."""
    test_text = "This is a test sentence"
    embedding = hf_client.embed(test_text)
    
    # Check that embedding is a tensor
    assert isinstance(embedding, torch.Tensor)
    
    # Check embedding shape (should be [1, 768])
    assert len(embedding.shape) == 2
    assert embedding.shape[0] == 1
    assert embedding.shape[1] == 768  # Verify 768 dimensions
    
    # Check that embedding is normalized
    norm = torch.norm(embedding, dim=1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6)

def test_hf_embedding_batch(hf_client):
    """Test that HuggingFace can handle multiple sentences."""
    test_texts = [
        "First test sentence",
        "Second test sentence",
        "Third test sentence"
    ]
    
    embeddings = [hf_client.embed(text) for text in test_texts]
    
    # Check that all embeddings have the same shape and correct dimensions
    assert all(emb.shape == (1, 768) for emb in embeddings)
    
    # Check that embeddings are different for different texts
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            assert not torch.allclose(embeddings[i], embeddings[j])

def test_hf_empty_text(hf_client):
    """Test that HuggingFace can handle empty text."""
    empty_text = ""
    embedding = hf_client.embed(empty_text)
    assert isinstance(embedding, torch.Tensor)
    assert len(embedding.shape) == 2
    assert embedding.shape[0] == 1
    assert embedding.shape[1] == 768

def test_hf_special_characters(hf_client):
    """Test that HuggingFace can handle text with special characters."""
    special_text = "Hello! @#$%^&*()_+ This is a test with special characters."
    embedding = hf_client.embed(special_text)
    assert isinstance(embedding, torch.Tensor)
    assert len(embedding.shape) == 2
    assert embedding.shape[0] == 1
    assert embedding.shape[1] == 768

def test_hf_long_text(hf_client):
    """Test that HuggingFace can handle long text."""
    long_text = "This is a very long text that might exceed the normal length of a sentence. " * 10
    embedding = hf_client.embed(long_text)
    assert isinstance(embedding, torch.Tensor)
    assert len(embedding.shape) == 2
    assert embedding.shape[0] == 1
    assert embedding.shape[1] == 768

def test_hf_embedding_dimensions(hf_client):
    """Test that all embeddings have exactly 768 dimensions."""
    test_cases = [
        "Short text",
        "This is a medium length text for testing",
        "This is a very long text that might exceed the normal length of a sentence. " * 10,
        "",  # Empty text
        "Hello! @#$%^&*()_+ Special chars",  # Special characters
        "1234567890",  # Numbers
        "Mixed content: Hello123!@#",  # Mixed content
    ]
    
    for text in test_cases:
        embedding = hf_client.embed(text)
        assert embedding.shape == (1, 768), f"Embedding for text '{text}' should have shape (1, 768), got {embedding.shape}"

# Additional tests specific to HuggingFace models
def test_hf_multilingual_support(hf_client):
    """Test that HuggingFace can handle multiple languages."""
    multilingual_texts = [
        "Hello world",  # English
        "Hola mundo",   # Spanish
        "Bonjour le monde",  # French
        "你好世界",     # Chinese
        "こんにちは世界"  # Japanese
    ]
    
    embeddings = [hf_client.embed(text) for text in multilingual_texts]
    
    # Check that all embeddings have correct shape
    assert all(emb.shape == (1, 768) for emb in embeddings)
    
    # Check that embeddings are different for different languages
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            assert not torch.allclose(embeddings[i], embeddings[j])

def test_hf_semantic_similarity(hf_client):
    """Test that HuggingFace embeddings capture semantic similarity."""
    similar_pairs = [
        ("The cat sat on the mat", "A feline rested on the carpet"),
        ("I love programming", "I enjoy coding"),
        ("The weather is nice today", "It's a beautiful day")
    ]
    
    for text1, text2 in similar_pairs:
        emb1 = hf_client.embed(text1)
        emb2 = hf_client.embed(text2)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
        
        # Similar texts should have high similarity
        assert similarity > 0.65, f"Similar texts should have high similarity, got {similarity}"

def test_hf_different_topics(hf_client):
    """Test that HuggingFace embeddings can distinguish between different topics."""
    different_topics = [
        ("The stock market is rising", "The weather is cloudy today"),
        ("I love playing guitar", "The recipe calls for two eggs"),
        ("The car needs maintenance", "The book is on the shelf")
    ]
    
    for text1, text2 in different_topics:
        emb1 = hf_client.embed(text1)
        emb2 = hf_client.embed(text2)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2)
        
        # Different topics should have lower similarity
        assert similarity < 0.7, f"Different topics should have lower similarity, got {similarity}"
