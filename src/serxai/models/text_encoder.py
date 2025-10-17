"""Text encoder wrapper (HuggingFace BERT/RoBERTa) placeholder"""
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, backbone="roberta-base", proj_dim=512):
        super().__init__()
        # actual HF model will be loaded in the real implementation
        self.backbone_name = backbone
        self.proj = nn.Linear(768, proj_dim)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # returns token embeddings placeholder
        batch = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        import torch
        dummy = torch.zeros(batch, seq_len, 768)
        return self.proj(dummy)
