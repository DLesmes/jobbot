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
    
    # Check embedding shape (should be [1, embedding_dim])
    assert len(embedding.shape) == 2
    assert embedding.shape[0] == 1
    
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
    
    # Check that all embeddings have the same shape
    assert all(emb.shape == embeddings[0].shape for emb in embeddings)
    
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

def test_clip_special_characters(clip_client):
    """Test that CLIP can handle text with special characters."""
    special_text = "Hello! @#$%^&*()_+ This is a test with special characters."
    embedding = clip_client.embed(special_text)
    assert isinstance(embedding, torch.Tensor)
    assert len(embedding.shape) == 2
    assert embedding.shape[0] == 1

def test_clip_long_text(clip_client):
    """Test that CLIP can handle long text."""
    long_text = "This is a very long text that might exceed the normal length of a sentence. " * 10
    embedding = clip_client.embed(long_text)
    assert isinstance(embedding, torch.Tensor)
    assert len(embedding.shape) == 2
    assert embedding.shape[0] == 1
