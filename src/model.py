import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel
import numpy as np

class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class EnhancedDistilBERT(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 2,
                 adapter_size: int = 64, lora_rank: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        config = self.distilbert.config
        hidden_size = config.hidden_size
        num_layers = config.n_layers
        
        for param in self.distilbert.parameters():
            param.requires_grad = False
        
        for name, param in self.distilbert.named_parameters():
            if 'LayerNorm' in name:
                param.requires_grad = True
        
        self.lora_layers = nn.ModuleList([
            LoRALayer(hidden_size, hidden_size, rank=lora_rank)
            for _ in range(num_layers)
        ])
        
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, adapter_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(adapter_size, hidden_size),
                nn.LayerNorm(hidden_size)
            ) for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        self.contrastive_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128)
        )
    
    def forward(self, input_ids, attention_mask, return_contrastive=False):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states
        x = hidden_states[-1]
        
        for i, (lora, adapter) in enumerate(zip(self.lora_layers, self.adapters)):
            layer_hidden = hidden_states[min(i + 1, len(hidden_states) - 1)]
            x_lora = lora(layer_hidden)
            x_adapter = adapter(layer_hidden)
            x = layer_hidden + x_lora + x_adapter
        
        pooled = x[:, 0]
        logits = self.classifier(pooled)
        
        if return_contrastive:
            contrastive_features = self.contrastive_head(pooled)
            return logits, contrastive_features
        
        return logits