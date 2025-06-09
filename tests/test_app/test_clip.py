import pytest
import torch
import numpy as np
from src.app.clients.clip import Clip

@pytest.fixture
def clip_client():
    """Fixture to create a CLIP client instance for testing."""
    return Clip()

def test_clip_initialization(clip_client):
    """Test that CLIP client initializes correctly."""
    assert clip_client.model_id is not None
    assert clip_client.device in ['cuda', 'cpu']
    assert clip_client.model is not None
    assert clip_client.preprocess is not None
    assert clip_client.tokenizer is not None

def test_clip_embedding(clip_client):
    """Test that CLIP can generate embeddings for text."""
    test_text = "This is a test sentence"
    embedding = clip_client.embed(test_text)
    
    # Check that embedding is a tensor
    assert isinstance(embedding, torch.Tensor)
    
    # Check embedding shape (should be [1, 768])
    assert len(embedding.shape) == 2
    assert embedding.shape[0] == 1
    assert embedding.shape[1] == 768  # Verify 768 dimensions
    
    # Check that embedding is normalized
    norm = torch.norm(embedding, dim=1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6)

def test_clip_embedding_batch(clip_client):
    """Test that CLIP can handle multiple sentences."""
    test_texts = [
        "First test sentence",
        "Second test sentence",
        "Third test sentence"
    ]
    
    embeddings = [clip_client.embed(text) for text in test_texts]
    
    # Check that all embeddings have the same shape and correct dimensions
    assert all(emb.shape == (1, 768) for emb in embeddings)  # Verify 768 dimensions for all
    
    # Check that embeddings are different for different texts
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            assert not torch.allclose(embeddings[i], embeddings[j])

def test_clip_empty_text(clip_client):
    """Test that CLIP can handle empty text."""
    empty_text = ""
    embedding = clip_client.embed(empty_text)
    assert isinstance(embedding, torch.Tensor)
    assert len(embedding.shape) == 2
    assert embedding.shape[0] == 1
    assert embedding.shape[1] == 768  # Verify 768 dimensions

def test_clip_special_characters(clip_client):
    """Test that CLIP can handle text with special characters."""
    special_text = "Hello! @#$%^&*()_+ This is a test with special characters."
    embedding = clip_client.embed(special_text)
    assert isinstance(embedding, torch.Tensor)
    assert len(embedding.shape) == 2
    assert embedding.shape[0] == 1
    assert embedding.shape[1] == 768  # Verify 768 dimensions

def test_clip_long_text(clip_client):
    """Test that CLIP can handle long text."""
    long_text = "This is a very long text that might exceed the normal length of a sentence. " * 10
    embedding = clip_client.embed(long_text)
    assert isinstance(embedding, torch.Tensor)
    assert len(embedding.shape) == 2
    assert embedding.shape[0] == 1
    assert embedding.shape[1] == 768  # Verify 768 dimensions

def test_clip_embedding_dimensions(clip_client):
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
        embedding = clip_client.embed(text)
        assert embedding.shape == (1, 768), f"Embedding for text '{text}' should have shape (1, 768), got {embedding.shape}"

# New tests for multilingual support, semantic similarity, and different topics
def test_clip_multilingual_support(clip_client):
    """Test that CLIP can handle multiple languages."""
    multilingual_texts = [
        "Hello world",  # English
        "Hola mundo",   # Spanish
        "Bonjour le monde",  # French
        "你好世界",     # Chinese
        "こんにちは世界"  # Japanese
    ]
    
    embeddings = [clip_client.embed(text) for text in multilingual_texts]
    
    # Check that all embeddings have correct shape
    assert all(emb.shape == (1, 768) for emb in embeddings)
    
    # Check that embeddings are different for different languages
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            assert not torch.allclose(embeddings[i], embeddings[j])

def test_clip_semantic_similarity(clip_client):
    """Test that CLIP embeddings capture semantic similarity."""
    similar_pairs = [
        ("The cat sat on the mat", "A feline rested on the carpet"),
        ("I love programming", "I enjoy coding"),
        ("The weather is nice today", "It's a beautiful day")
    ]
    
    for text1, text2 in similar_pairs:
        emb1 = clip_client.embed(text1)
        emb2 = clip_client.embed(text2)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)
        
        # Similar texts should have high similarity
        assert similarity > 0.7, f"Similar texts should have high similarity, got {similarity}"

def test_clip_different_topics(clip_client):
    """Test that CLIP embeddings can distinguish between different topics."""
    different_topics = [
        ("The stock market is rising", "The weather is cloudy today"),
        ("I love playing guitar", "The recipe calls for two eggs"),
        ("The car needs maintenance", "The book is on the shelf")
    ]
    
    for text1, text2 in different_topics:
        emb1 = clip_client.embed(text1)
        emb2 = clip_client.embed(text2)
        
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)
        
        # Different topics should have lower similarity
        assert similarity < 0.8, f"Different topics should have lower similarity, got {similarity}"
