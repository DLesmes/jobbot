""" embedds for open clip models """
#embeds
import torch
import torch.nn.functional as F
import open_clip
# repo imports
from src.app.settings import Settings
settings = Settings()

class Clip():
    """
    """
    def __init__(self):
        self.model_id = settings.MODEL_ID
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            self.model_id,
            device=self.device,
            precision='fp16'
        )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_id)

    def embed(self, txt: str):
        text_input = self.tokenizer(txt).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_text(text_input)
        
        embedding = embedding/embedding.norm(dim=1, keepdim=True)
        return embedding
    