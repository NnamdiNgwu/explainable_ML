import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input."""
    
    def __init__(self, d_model, max_len=100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class MultiHeadRiskAttention(nn.Module):
    """Risk-aware attention that focuses on anomalous patterns."""
    
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.risk_scorer = nn.Linear(d_model, 1)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Standard self-attention
        attn_out, attn_weights = self.attention(query=x, key=x, value=x)
        
        # Residual connection
        x = self.layer_norm(x + attn_out)
        
        # Risk scoring for each timestep
        risk_scores = torch.sigmoid(self.risk_scorer(x))
        
        # Weight attention by risk scores
        risk_weighted = x * risk_scores
        
        return risk_weighted, attn_weights

class CybersecurityTransformer(nn.Module):
    """Advanced Transformer for cybersecurity sequence modeling."""
    
    def __init__(self, cont_dim, cat_dims, cat_emb_dims, num_classes=3, 
                 d_model=64, nhead=8, num_layers=2, max_len=50, dropout=0.1):
        super().__init__()
        
        self.cont_dim = cont_dim
        self.cat_dims = cat_dims
        self.cat_emb_dims = cat_emb_dims
        self.d_model = d_model
        
        # Embedding layers for categorical features
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, emb_dim) 
            for cat_dim, emb_dim in zip(cat_dims, cat_emb_dims)
        ])
        
        # Input projection
        total_dim = cont_dim + sum(cat_emb_dims)
        self.input_projection = nn.Linear(total_dim, d_model)
        
        # Positional encoding for temporal awareness
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Risk-aware attention mechanism
        self.risk_attention = MultiHeadRiskAttention(d_model, nhead)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            # nn.BatchNorm1d(d_model // 2), #
            nn.ReLU(),
            nn.Dropout(dropout * 1.5), #
            #nn.Dropout(dropout * 1.5), #
            nn.Linear(d_model // 2, d_model // 4), #
            # nn.BatchNorm1d(d_model // 4), #
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes) #
        )
        
    #     # Add weight decay regularization term
    #     self.weight_decay = 1e-4
    
    # def get_l2_loss(self):
    #     """Calculate L2 regularization loss for all parameters."""
    #     l2_loss = 0.0
    #     for param in self.parameters():
    #         l2_loss += torch.norm(param, p=2)
    #     return self.l2_lamba * l2_loss

        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)
            
    def forward(self, cont_features, cat_high=None, cat_low=None):
        batch_size, seq_len = cont_features.shape[:2]
        
        # Handle categorical features
        cat_embeds = []
        
        if cat_high is not None and len(self.cat_embeddings) > 0:
            # Embed high cardinality categorical features
            for i in range(cat_high.shape[-1]):
                if i < len(self.cat_embeddings):
                    cat_feat = cat_high[..., i]
                    # Clamp to valid embedding range
                    cat_feat = torch.clamp(cat_feat, 0, self.cat_dims[i] - 1)
                    cat_embeds.append(self.cat_embeddings[i](cat_feat))
        
        if cat_low is not None and len(self.cat_embeddings) > cat_high.shape[-1]:
            # Embed low cardinality categorical features
            for i in range(cat_low.shape[-1]):
                emb_idx = cat_high.shape[-1] + i
                if emb_idx < len(self.cat_embeddings):
                    cat_feat = cat_low[..., i]
                    # Clamp to valid embedding range
                    cat_feat = torch.clamp(cat_feat, 0, self.cat_dims[emb_idx] - 1)
                    cat_embeds.append(self.cat_embeddings[emb_idx](cat_feat))
        
        # Concatenate all features
        if cat_embeds:
            x = torch.cat([cont_features] + cat_embeds, dim=-1)
        else:
            x = cont_features
            
        # Project to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer
        x = self.transformer(x)
        
        # Risk-aware attention
        x, attention_weights = self.risk_attention(x)
        
        # Global pooling - focus on most recent events
        # Create attention weights that favor recent timesteps
        time_weights = torch.softmax(
            torch.arange(seq_len, dtype=torch.float, device=x.device), 
            dim=0
        )
        
        # Apply time-weighted pooling
        x = (x * time_weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits

def build_cybersecurity_transformer_from_maps(embed_maps: dict, continuous_dim: int, num_classes: int = 3):
    """Build transformer model from embedding maps."""
    
    # Extract categorical dimensions and embedding dimensions
    cat_dims = []
    cat_emb_dims = []
    
    for feature_name, vocab_map in embed_maps.items():
        vocab_size = len(vocab_map)
        cat_dims.append(vocab_size)
        # Embedding dimension: min(50, sqrt(vocab_size))
        emb_dim = min(50, max(1, int(math.sqrt(vocab_size))))
        cat_emb_dims.append(emb_dim)
    
    print(f"Building transformer: cont_dim={continuous_dim}, cat_dims={cat_dims}, cat_emb_dims={cat_emb_dims}")
    
    model = CybersecurityTransformer(
        cont_dim=continuous_dim,
        cat_dims=cat_dims,
        cat_emb_dims=cat_emb_dims,
        num_classes=num_classes,
        d_model= 64, #128,  # Use 64 for short sequences
        nhead=8, #8, #4,      # Increased heads for finer interactions
        num_layers=4, #4, #2, # Depth sufficient for days-apart events
        max_len=50,
        dropout=0.3 #0.3 #0.1 # Sronger regularization to curb overfitting
    )
    
    return model