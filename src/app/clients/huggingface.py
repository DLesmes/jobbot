""" embeddings using HuggingFace models """
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from src.app.settings import Settings

settings = Settings()

class HuggingFace():
    """
    A class for generating embeddings using HuggingFace models.
    Compatible with the CLIP interface for seamless integration.
    """
    def __init__(self):
        self.model_id = settings.HUGGINGFACE_MODEL_ID
        # Alternative options:
        # "sentence-transformers/all-MiniLM-L6-v2"  # 384d model
        # "sentence-transformers/all-mpnet-base-v2"  # 768d model
        # "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  # 768d model, multilingual
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize the HuggingFace embeddings
        self.model = HuggingFaceEmbeddings(
            model_name=self.model_id,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}  # This ensures normalized embeddings
        )

    def embed(self, txt: str):
        """
        Generate embeddings for a single text or list of texts.
        
        Args:
            txt (str or list): Input text or list of texts to embed
            
        Returns:
            torch.Tensor: Normalized embeddings with shape [1, 768] or [n, 768] for list input
        """
        # Handle both single string and list inputs
        if isinstance(txt, str):
            txt = [txt]
            
        # Generate embeddings
        embeddings = self.model.embed_documents(txt)
        
        # Convert to torch tensor and ensure correct shape
        embeddings_tensor = torch.tensor(embeddings, device=self.device)
        
        # For single text input, ensure shape is [1, 768]
        if isinstance(txt, str):
            # First squeeze to remove any extra dimensions
            embeddings_tensor = embeddings_tensor.squeeze()
            # Then add batch dimension if needed
            if len(embeddings_tensor.shape) == 1:
                embeddings_tensor = embeddings_tensor.unsqueeze(0)
            
        return embeddings_tensor
