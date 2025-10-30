"""cascade.py: RF→Transformer cascade with two thresholds (tau, tau2)."""
import torch
import numpy as np
import logging
from typing import Dict
from flask import current_app
from .encoders import encode_tabular, encode_sequence_semantic

def decide_with_tau2(transformer_probs: np.ndarray, tau2: float) -> (int, float):
    """
    Applies tau2 to decide final label after escalation.
    - Binary (C=2): positive if P[1] >= tau2 else 0.
    - Multiclass (C>=3, class 0 = Low/Benign): if (1 - P[0]) >= tau2 -> argmax over classes 1..C-1; else 0.
    Returns: (predicted_class_index, gating_confidence_used_for_tau2)
    """
    C = transformer_probs.shape[0]
    if C == 2:
        p_pos = float(transformer_probs[1])
        return (1 if p_pos >= tau2 else 0, p_pos)
    else:
        p_not_low = float(1.0 - transformer_probs[0])
        if p_not_low >= tau2:
            rest_idx = 1 + int(transformer_probs[1:].argmax())
            return (rest_idx, p_not_low)
        else:
            return (0, p_not_low)

def rf_predict_proba(rf_model, X: np.ndarray) -> np.ndarray:
    """Get RF prediction probabilities."""
    return rf_model.predict_proba(X.reshape(1, -1))[0]

def transformer_predict_proba(model, cont: torch.Tensor, cat_high: torch.Tensor, cat_low: torch.Tensor, device: str) -> np.ndarray:
    """Transformer prediction probabilities."""
    model.eval()
    
    with torch.no_grad():
        # Ensure tensors are on correct device
        logging.info(f"Transformer input shapes - cont: {cont.shape}, cat_high: {cat_high.shape}, cat_low: {cat_low.shape}")
        
        cont = cont.to(device)
        cat_high = cat_high.to(device)  
        cat_low = cat_low.to(device)
        try:
            # Forward pass
            # logits = model(cont, cat_high, cat_low)
            # probs = torch.softmax(logits, dim=1)
            output = model(cont, cat_high, cat_low)
            logging.info(f"Model output type: {type(output)}")

            # handle tuple output (logits, attention_weights)
            if isinstance(output, tuple):
                logging.info(f"Output is tuple with {len(output)} elements")
                logits = output[0]
                logging.info(f"Logits shape: {logits.shape}")
            else:
                logits = output
                logging.info(f"Logits shape: {logits.shape}")
            
            probs = torch.softmax(logits, dim=1)
            
            return probs.cpu().numpy()[0]
        except Exception as e:
            logging.error(f"Error in transformer forward pass: {e}")
            logging.error(f"cont type: {type(cont)}, shape: {cont.shape}")
            logging.error(f"cat_high type: {type(cat_high)}, shape: {cat_high.shape}")
            logging.error(f"cat_low type: {type(cat_low)}, shape: {cat_low.shape}")
            raise


def cascade_predict(event: Dict) -> Dict:
    """
    RF→Transformer cascade with two thresholds:
      if max(P_RF) < tau: return Benign (no escalation)
      else: escalate; final label via Transformer and tau2
    """
    models = current_app.ml_models
    tau  = models["tau"]
    tau2 = models["tau2"]

    # ---------------- Stage 1: RF gate ----------------
    try:
        X_tab = encode_tabular(event)
        rf_proba = rf_predict_proba(models["rf"], X_tab)           # shape [C_rf]
        rf_conf  = float(rf_proba.max())
        rf_pred  = int(rf_proba.argmax())

        # Gate: confident-benign → stop early
        if rf_conf < tau:
            # Return explicit Benign/Low (index 0 assumed benign class)
            return {
                "prediction": 0,
                "confidence": 1.0 - rf_conf,  # confidence of "benign" gate (informative)
                "model_used": "RF-gate",
                "probabilities": rf_proba.tolist(),
                "tau": tau,
                "tau2": tau2,
                "escalated": False,
                "gate_reason": "max(P_RF) < tau → Benign (no escalation)"
            }
        # else: escalate to Transformer
    except Exception as e:
        # If RF fails, log and fall through to Transformer as a safety net
        logging.warning(f"RF stage failed, escalating to Transformer: {e}")

    # ---------------- Stage 2: Transformer + tau2 ----------------
    try:
        cont, cat_high, cat_low = encode_sequence_semantic(
            event,
            models["feature_lists"],
            models["embed_maps"],
            models.get("device", "cpu")
        )
        trans_probs = transformer_predict_proba(
            models["transformer"],
            cont, cat_high, cat_low,
            models.get("device", "cpu")
        )  # np.ndarray [C_trans]

        final_pred, gate_conf = decide_with_tau2(trans_probs, tau2)
        return {
            "prediction": int(final_pred),
            "confidence": float(gate_conf),
            "model_used": "CybersecurityTransformer",
            "probabilities": trans_probs.tolist(),
            "tau": tau,
            "tau2": tau2,
            "escalated": True,
            "gate_reason": "max(P_RF) ≥ tau → escalated; tau2 applied to Transformer output"
        }
    except Exception as e:
        logging.error(f"Transformer stage failed: {e}")
        # As last resort, degrade gracefully with RF argmax (already computed if RF succeeded)
        if "rf_proba" in locals():
            return {
                "prediction": int(rf_pred),
                "confidence": float(rf_conf),
                "model_used": "RandomForest (fallback)",
                "probabilities": rf_proba.tolist(),
                "tau": tau,
                "tau2": tau2,
                "escalated": False,
                "gate_reason": "Transformer failure; returned RF argmax"
            }
        raise RuntimeError(f"Both models failed: {e}")
