"""in addition to Learnable parametric time-decay pooling αt = softmax(w t + b) combined with risk gating
An auxiliary explanation head to predict “rationale tokens” (multitask)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional

torch.set_float32_matmul_precision('high')
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer input."""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # store as (max_len, 1, d_model); will be transpose in forward
        pe = pe.unsqueeze(1) # [max_len, 1, d_model]
        self.register_buffer('pe', pe) # no grad
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        # x shape: [B, L, D] (batch_first)
        # pe shape: [max_len, 1, d_model]
        seq_len = x.size(1)
        return x + self.pe[:seq_len].transpose(0, 1)  # [B, L, D] + [D, 1, max_len] -> [B, L, D]


class RelativePositionBias(nn.Module):
    """
    Learned relative position bias (Shaw et al., 2018-style).
    Produces an additive attention bias matrix B ∈ R[L, L], where B[i,j] depends on (j - i).
    """
    def __init__(self, max_len: int, num_buckets: Optional[int] = None):
        super().__init__()
        self.max_len = max_len
        # Simple form: one embedding per relative distance in [-K, K]
        self.K = max_len - 1
        self.num_embeddings = 2 * self.K + 1
        self.rel_bias = nn.Embedding(self.num_embeddings, 1)  # scalar bias per distance
        # Optional bucketing (kept simple: not used by default)
        self.num_buckets = num_buckets
    
    def forward(self, seq_len: int, device=None) -> torch.Tensor:
        # positions: [L]
        pos = torch.arange(seq_len, device=device)
        # relative distances d_ij = j - i in [-K, K]
        rel = pos[None, :] - pos[:, None]  # [L, L]
        rel = torch.clamp(rel, -self.K, self.K)
        idx = rel + self.K  # map to [0, 2K]
        bias = self.rel_bias(idx).squeeze(-1)  # [L, L]
        return bias

class MultiHeadRiskAttention(nn.Module):
    """Risk-aware attention that focuses on anomalous patterns."""
    
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.risk_scorer = nn.Linear(d_model, 1)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x, attn_bias: Optional[torch.Tensor] = None):
        # Standard self-attention with optional additive bias (as attn_mask)
        attn_kwargs = {}
        if attn_bias is not None:
            # PyTorch MHA expects attn_mask added to attention logits before softmax
            # Shape must be [L, L] for batch_first=True
            attn_kwargs["attn_mask"] = attn_bias

        attn_out, attn_weights = self.attention(query=x, key=x, value=x, **attn_kwargs)
        
        # Residual connection
        x = self.layer_norm(x + attn_out)
        
        # Risk scoring for each timestep
        risk_scores = torch.sigmoid(self.risk_scorer(x))  # [B, L, 1]
        
        # Weight attention by risk scores
        risk_weighted = x * risk_scores
        
        return risk_weighted, attn_weights, risk_scores  # return risk scores for pooling

class CybersecurityTransformer(nn.Module):
    """Advanced Transformer for cybersecurity sequence modeling."""
    def __init__(self, cont_dim, cat_dims, cat_emb_dims, num_classes=3, 
                 d_model=64, nhead=8, num_layers=4, max_len=50, dropout=0.3,
                 explanation_vocab_size: int = 0, explanation_dropout: float = 0.1,
                 use_relative_positions: bool = True, use_parametric_decay: bool = True,
                 causal: bool = False, no_self: bool =  False):
        super().__init__()
        self.cont_dim = cont_dim
        self.cat_dims = cat_dims
        self.cat_emb_dims = cat_emb_dims
        self.d_model = d_model
        self.max_len = max_len
        self.use_relative_positions = use_relative_positions
        self.use_parametric_decay = use_parametric_decay
        self.explanation_vocab_size = explanation_vocab_size
        self.causal = causal
        self.no_self = no_self
        # Embedding layers for categorical features
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, emb_dim) 
            for cat_dim, emb_dim in zip(cat_dims, cat_emb_dims)
        ])
        # Input projection
        total_dim = cont_dim + sum(cat_emb_dims)
        self.input_projection = nn.Linear(total_dim, d_model)
        
        # Positional encoding for temporal awareness (kept for fallback; maybe bypasses if relative bias is used)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Relative position bias module
        self.rel_pos_bias = RelativePositionBias(max_len) if use_relative_positions else None
        
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
        
        # Risk-aware attention mechanism (supports attn bias)
        self.risk_attention = MultiHeadRiskAttention(d_model, nhead)

        # parametric time decay α_t = softmax(w t + b)
        if use_parametric_decay:
            self.time_w = nn.Parameter(torch.tensor(0.05)) # small positive init
            self.time_b = nn.Parameter(torch.tensor(0.0))  # can be negative to allow decay
        else:
            self.register_parameter('time_w', None)
            self.register_parameter('time_b', None)

        # Optional gating network for pooling (complements risk scorer)
        self.pooling_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.BatchNorm1d(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(d_model // 2, d_model // 4),
            nn.BatchNorm1d(d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, num_classes)
        )

        # Auxiliary explanation head (multitask): predict rationale tokens
        if explanation_vocab_size and explanation_vocab_size > 0:
            self.rationale_head = nn.Sequential(
                nn.Dropout(explanation_dropout),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, explanation_vocab_size) # logits over vocab (multi-label or softmax)
            )
        else:
            self.rationale_head = None
        
        # Initialize weights
        self.apply(self._init_weights)

        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.xavier_uniform_(module.weight)
    
    def _compose_attn_mask(self, rel_bias, seq_len, device):
        """
        Combine relative bias with structural mask:
        - causal: forbid attending to future tokens (upper triangle = -inf)
        - no_self: optionally suppress self-attention diagonal
        """
        if rel_bias is None:
            mask = torch.zeros(seq_len, seq_len, device=device)
        else:
            mask = rel_bias.clone()
        if self.causal:
            future = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            mask = mask.masked_fill(future, float('-inf'))
        if self.no_self:
            diag = torch.eye(seq_len, device=device).bool()
            mask = mask.masked_fill(diag, float('-inf'))
        return mask
    def forward(self, cont_features, cat_high=None, cat_low=None, return_explanations: bool = True):
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
        
        # Add positional encoding (kept for stability; relative bias affects attention scores)
        x = self.pos_encoding(x)

        # Compute relative position bias (shared across encoder layers)
        # attn_bias = None
        # if self.use_relative_positions:
        #     attn_bias = self.rel_pos_bias(seq_len, device=x.device) # [L, L]
        
        # # Apply transformer (with additive bias mask if available)
        # x = self.transformer(x, mask=attn_bias)

        # # Risk-aware attention (also receive the same bias)
        # x, attention_weights, risk_scores = self.risk_attention(x, attn_bias)
        
        rel_bias = self.rel_pos_bias(seq_len, device=x.device) if self.use_relative_positions else None
        attn_mask = self._compose_attn_mask(rel_bias, seq_len, x.device)
        x = self.transformer(x, mask=attn_mask)
        # For risk attention pass only finite (non -inf) part; reuse attn_mask
        x, attention_weights, risk_scores = self.risk_attention(x, attn_mask)

        # Parametric time decay weights α_t = softmax(w t + b)
        positions = torch.arange(seq_len, dtype=torch.float, device=x.device)  # [L]
        if self.use_parametric_decay:
            time_logits = self.time_w * positions + self.time_b  # [L]
            time_weights = torch.softmax(time_logits, dim=0)  # [L]
        else:
            # Default monotonic prior
            time_weights = torch.softmax(positions, dim=0)
        
        # Gate pooling: combine time weights with learned gates (risk_scores and pool_gate)
        pool_gates = self.pooling_gate(x).squeeze(-1)  # [B, L]
        risk_scores_flat = risk_scores.squeeze(-1)  # [B, L]
        # Combine: elementwise product then renormalize
        combined_weights = pool_gates * risk_scores_flat * time_weights.unsqueeze(0)  # [B, L]
        combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + 1e-8)

        # Weighted sum pooling
        # x_pooled = torch.einsum('bl,bld->bd', combined_weights, x)  # [B, H]
        #combined_weights: [B, L], x: [B, L, D]
        x_pooled = (x * combined_weights.unsqueeze(-1)).sum(dim=1) # [B, D]

        # Classification
        logits = self.classifier(x_pooled)

        # Optional auxiliary rationale logits
        out = {
            "logits": logits,
            "attention_weights": attention_weights,
            "time_weights": time_weights,
            "risk_scores": risk_scores,
            "combined_weights": combined_weights
        }

        if self.rationale_head is not None and return_explanations:
            rationale_logits = self.rationale_head(x_pooled) # [B, V]
            out["rationale_logits"] = rationale_logits
        
        return out  # returns logits, attention weights, time weights, risk scores, combined weights, rationale logits if available

        # # Global pooling - focus on most recent events
        # # Create attention weights that favor recent timesteps
        # time_weights = torch.softmax(
        #     torch.arange(seq_len, dtype=torch.float, device=x.device), 
        #     dim=0
        # )
        
        # # Apply time-weighted pooling
        # x = (x * time_weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        
        # # Classification
        # logits = self.classifier(x)
        
        # return logits

def build_cybersecurity_transformer_from_maps(embed_maps: dict,
                                             continuous_dim: int,
                                             num_classes: int,
                                             explanation_vocab_size: int = 0,
                                             assert_min_classes: int=2,
                                             causal: bool = False,
                                             no_self: bool = False):
    """Build transformer model from embedding maps."""
    if num_classes < assert_min_classes:
        raise ValueError(f"num_classes must be at least {assert_min_classes}, got {num_classes}; check label generation.")
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
        num_layers=4, #2, # Depth sufficient for days-apart events
        max_len=50,
        dropout=0.3, #0.1 # Sronger regularization to curb overfitting
        explanation_vocab_size=explanation_vocab_size,
        use_relative_positions=True,  # Use relative position bias by default
        use_parametric_decay=True,  # Use parametric time decay by default
        causal=False,  # Causal attention to respect sequence order
        no_self=False  # Allow self-attention to focus on all tokens
    )
    
    return model