import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class QuestionEncoder(nn.Module):
    """
    Encodes questions into embeddings using a small transformer model.
    Default: sentence-transformers/all-MiniLM-L6-v2 (384 dim output)
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", output_dim=384):
        super(QuestionEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.output_dim = output_dim
        
        # Freeze the encoder (optional, can be unfrozen for fine-tuning)
        for param in self.model.parameters():
            param.requires_grad = False
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, questions):
        """
        Args:
            questions: List of question strings
        Returns:
            Tensor of shape [batch_size, output_dim]
        """
        encoded = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**encoded)
        
        embeddings = self.mean_pooling(outputs, encoded['attention_mask'])
        return embeddings
    
    def to(self, device):
        self.model = self.model.to(device)
        return super().to(device)

