"""Encoders for different model types."""
import torch
import numpy as np
import pandas as pd
from flask import current_app
from typing import Dict, Tuple
import logging


def encode_tabular(event: Dict) -> np.ndarray:
    """Return *preprocessed* 1D vector ready for the RF estimator."""
    models = current_app.ml_models

    feature_lists  = models['feature_lists']
    features_order = models['feature_names']  # raw feature order
    preprocessor   = models['preprocessor']

    # Build raw-row DataFrame with correct dtypes
    cont = feature_lists['CONTINUOUS_USED']
    boo  = feature_lists['BOOLEAN_USED']
    cats = feature_lists['HIGH_CAT_USED'] + feature_lists['LOW_CAT_USED']

    row = {}
    for f in features_order:
        v = event.get(f, None)
        if f in cont:
            try:
                row[f] = float(v) if v is not None else 0.0
            except (ValueError, TypeError):
                row[f] = 0.0
        elif f in boo:
            row[f] = (v.lower() in ('true','1','yes','on')) if isinstance(v, str) else bool(v)
        elif f in cats:
            row[f] = str(v) if v is not None else "unknown"
        else:
            try:
                row[f] = float(v) if v is not None else 0.0
            except (ValueError, TypeError):
                row[f] = 0.0

    X_df = pd.DataFrame([row], columns=features_order)

    # Transform with the fitted preprocessor (ColumnTransformer)
    X_proc = preprocessor.transform(X_df)
    if hasattr(X_proc, "toarray"):  # sparse
        X_proc = X_proc.toarray()

    return X_proc[0]  # 1D vector for a single row


def encode_sequence_semantic(event: Dict, feature_lists: dict, embed_maps: dict = None, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode single event as sequence for transformer compatibility."""

    if device is None or isinstance(device, dict):
        device = 'cpu'

    if feature_lists is None or embed_maps is None:
        try:
            models = current_app.ml_models
            feature_lists = feature_lists or models['feature_lists']
            embed_maps = embed_maps or models['embed_maps']
            device = device or models.get( 'device', 'cpu')
        except RuntimeError:
            # Outside Flask context - parameters must be provided
            if feature_lists is None or embed_maps is None:
                raise ValueError("feature_lists and embed_maps must be provided when outside Flask context")
        
    
    # Sequence parameters (match training)
    seq_len = 10  # Reasonable length for inference
    
    # Continuous + Boolean features
    cont_features = feature_lists["CONTINUOUS_USED"] + feature_lists["BOOLEAN_USED"]
    cont_values = [event.get(c, 0.0) for c in cont_features]
    
    # Create sequence: latest event + padding
    cont_seq = [[0.0] * len(cont_values)] * (seq_len - 1) + [cont_values]  # Pad with zeros, end with actual event
    cont = torch.tensor([cont_seq], dtype=torch.float32, device=device)  # [1, seq_len, cont_dim]
    
    # High-cardinality categorical
    high_cat_values = [embed_maps[c].get(str(event.get(c, '')), 0) for c in feature_lists["HIGH_CAT_USED"]]
    high_cat_seq = [[0] * len(high_cat_values)] * (seq_len - 1) + [high_cat_values]
    cat_high = torch.tensor([high_cat_seq], dtype=torch.long, device=device)  # [1, seq_len, high_cat_dim]
    
    # Low-cardinality categorical
    low_cat_values = [embed_maps[c].get(str(event.get(c, '')), 0) for c in feature_lists["LOW_CAT_USED"]]
    low_cat_seq = [[0] * len(low_cat_values)] * (seq_len - 1) + [low_cat_values]
    cat_low = torch.tensor([low_cat_seq], dtype=torch.long, device=device)  # [1, seq_len, low_cat_dim]
    try:
        return cont.to(device), cat_high.to(device), cat_low.to(device)
    except Exception as e:
        logging.warning(f"Device conversion failed, using CPU: {e}")
        return cont.to('cpu'), cat_high.to('cpu'), cat_low.to('cpu')

def encode_sequence_flat_for_shap(event: Dict, feature_lists=None, embed_maps=None) -> torch.Tensor:
    """Convert event to flat tensor for SHAP compatibility."""
    # try to get flask context first, 
    if feature_lists is None or embed_maps is None:
        try:
            models = current_app.ml_models
            feature_lists = models['feature_lists']
            embed_maps = models['embed_maps']
        except RuntimeError:
             # Outside Flask context - parameters must be provided
            if feature_lists is None or embed_maps is None:
                raise ValueError("feature_lists and embed_maps must be provided when outside Flask context")


    arr = []
    # Continuous and boolean features
    for c in feature_lists["CONTINUOUS_USED"] + feature_lists["BOOLEAN_USED"]:
        arr.append(float(event.get(c, 0.0)))
    
    # High-cardinality categorical features
    for c in feature_lists["HIGH_CAT_USED"]:
        arr.append(float(embed_maps[c].get(str(event.get(c, '')), 0)))
    
    # Low-cardinality categorical features
    for c in feature_lists["LOW_CAT_USED"]:
        arr.append(float(embed_maps[c].get(str(event.get(c, '')), 0)))
    
    # return np.array([arr], dtype=np.float32)
    return torch.tensor([arr], dtype=torch.float32)


def transformer_tensors_to_flat(cont: torch.Tensor, cat_high: torch.Tensor, cat_low: torch.Tensor, feature_lists: Dict) -> np.ndarray:
    """Convert transformer tensors back to flat array for feature mapping."""
    # Extract last timestep (actual event, not padding)
    cont_values = cont[0, -1, :].cpu().numpy()  # Last timestep
    cat_high_values = cat_high[0, -1, :].cpu().numpy()
    cat_low_values = cat_low[0, -1, :].cpu().numpy()
    
    # Combine all features in the same order as feature_names
    all_values = np.concatenate([cont_values, cat_high_values, cat_low_values])
    return all_values

def dict_to_transformer_tensors(event_dict, feature_lists, embed_maps, device='cpu'):
    """Convert event dictionary to transformer input tensors."""
    cont, cat_high, cat_low = encode_sequence_semantic(
        event_dict,
        feature_lists,
        embed_maps,
        device)
    return cont.to(device), cat_high.to(device), cat_low.to(device)