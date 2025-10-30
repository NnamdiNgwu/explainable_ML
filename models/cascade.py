#!/usr/bin/env python3
"""cascade.py - build, tune, and serialize the RF -> Transformer cascade

The cascade follows the rule:
    if max(P_RF) >= τ  -> use RF class
    else              -> fall through to Cybersecurity Transformer class

• Finds the ROC knee on validation data to choose τ (default 0.60).
• Persists the threshold plus model paths in `cascade_config.json`.
• Provides `cascade_predict` for downstream serving code.

The cascade follows the rule:
    if max(P_RF) < τ   -> return Benign immediately (no Transformer)
    else               -> escalate to Cybersecurity Transformer and decide via τ₂

• Finds τ on validation (PR–F1 with RF max prob).
• Optionally finds τ₂ on RF-escalated validation (PR–F1 with Transformer prob).
• Persists thresholds plus model paths in `cascade_config.json`.
• Provides helpers for downstream serving.

Run after both `train_rf.py` and `train_cybersecurity_transformer.py` have produced
   • rf_model.joblib
   •  embedding_maps.json
   python -m src.models.cascade --data_dir data_processed

"""

from __future__ import annotations
import argparse, json, pathlib, collections
import numpy as np, joblib, torch
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split
from models.safe_smote import SafeSMOTE
from models.to_be_submitted.cybersecurity_transformer3 import build_cybersecurity_transformer_from_maps


# ---------------------------- load artefacts ----------------------------
def load_models(data_dir: pathlib.Path, device: str):
    rf = joblib.load(data_dir / "rf_model.joblib")

    embed_maps = json.loads((data_dir / "embedding_maps.json").read_text())
    # cont_dim = 6 # post_burst, destination_entropy, hour, megabytes_sen, first_time_destination, after_hours
    feature_lists = json.loads((data_dir / "feature_lists.json").read_text())
    cont_dim = len(feature_lists["CONTINUOUS_USED"]) + len(feature_lists["BOOLEAN_USED"])

    transformer = build_cybersecurity_transformer_from_maps(embed_maps, 
                                                            continuous_dim=cont_dim, 
                                                            num_classes=3)
    state_dict = torch.load(data_dir / "cybersecurity_transformer.pt", map_location=device)

    # Remove _orig_mod. prefix from keys if present
    if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '') if key.startswith('_orig_mod.') else key
            new_state_dict[new_key] = value
        state_dict = new_state_dict

    transformer.load_state_dict(state_dict)

    transformer.to(device).eval()
    return rf, transformer, embed_maps 

# ---------------------------- inference helpers ----------------------------
def rf_predict_proba(rf,x_tab_row):
    return rf.predict_proba(x_tab_row.reshape(1, -1))[0] # return np array (3,)


def transformer_predict_proba(transformer, seq_test, device, cont_dim, high_cat_dim, low_cat_dim):
    """Return P(High) for a 3-class transformer. Accepts (cont, cat) or (cont, cat_high, cat_low)."""
    transformer.to(device).eval()
    probs = []

    with torch.no_grad():
        if isinstance(seq_test, tuple):
            if len(seq_test) == 3:
                # Already split: (cont, cat_high, cat_low)
                cont_test, cat_high, cat_low = seq_test
                N = len(cont_test)
                for i in range(N):
                    cont_seq = cont_test[i:i+1].to(device)
                    cat_high_seq = cat_high[i:i+1].long().to(device)
                    cat_low_seq  = cat_low[i:i+1].long().to(device)

                    logits = transformer(cont_seq, cat_high_seq, cat_low_seq)
                    p = torch.softmax(logits, dim=1)[0]
                    probs.append(float(p[2]))  # High = index 2
            elif len(seq_test) == 2:
                # Unsplit: (cont, cat); split using dims
                cont_test, cat_all = seq_test
                N = len(cont_test)
                for i in range(N):
                    cont_seq = cont_test[i:i+1].to(device)
                    cat_slice = cat_all[i:i+1]
                    cat_high_seq = cat_slice[:, :, :high_cat_dim].long().to(device)
                    cat_low_seq  = cat_slice[:, :, high_cat_dim:].long().to(device)

                    logits = transformer(cont_seq, cat_high_seq, cat_low_seq)
                    p = torch.softmax(logits, dim=1)[0]
                    probs.append(float(p[2]))  # High = index 2
            else:
                raise ValueError(f"Unexpected seq_test tuple length: {len(seq_test)}")
        else:
            # If someone passes a raw tensor, require caller to split outside; be explicit.
            raise TypeError("seq_test must be a tuple: (cont, cat) or (cont, cat_high, cat_low)")
    return np.array(probs)



# ---------------------------- tune τ on validation set ----------------------------

def tune_threshold(rf, transformer, X_val_tab, seq_val, y_val, device, cont_dim, high_cat_dim, low_cat_dim ):#, positive_class):
    rf_max, pred_rf = [], []
    
    for x_tab in X_val_tab: #s in zip(X_val_tab, seq_val):
        p_rf = rf_predict_proba(rf, x_tab)
        rf_max.append(p_rf.max())
        pred_rf.append(p_rf)

    # Use RF confidence only - threshold  will define split
    precisions, recalls, thresholds = precision_recall_curve(
        (y_val == 2).astype(int), np.array(rf_max)) # y_val==2 , 2 is positive_class=2 which is the critical class
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = f1_scores.argmax()
    return  thresholds[best_idx]    # τ 

def tune_tau2(transformer, seq_val, y_val, escalated_mask, device, cont_dim, high_cat_dim, low_cat_dim):
    """Tune τ2 on RF-escalated subset using transformer probabilities."""
    if escalated_mask.sum() < 5:
        return 0.5  # fallback default

    p_trans = transformer_predict_proba(transformer, seq_val, device, cont_dim, high_cat_dim, low_cat_dim)
    p_trans = np.asarray(p_trans)

    # Align to escalated indices if needed
    if len(p_trans) != len(y_val):
        # If probabilities were computed only for escalated items, assume they are aligned already
        y_use = y_val[escalated_mask]
        p_use = p_trans
    else:
        y_use = y_val[escalated_mask]
        p_use = p_trans[escalated_mask]

    uniq = np.unique(y_val)
    pos_label = 2 if 2 in uniq else (1 if 1 in uniq else int(uniq.max()))
    y_bin = (y_use == pos_label).astype(int)

    precisions, recalls, thresholds = precision_recall_curve(y_bin, p_use)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    if len(thresholds) == 0:
        return 0.5
    best_idx = f1_scores.argmax()
    return float(thresholds[best_idx])  # τ2


# ---------------------------- main --------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=pathlib.Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val_frac", type=float, default=0.1,
                        help="fraction of train set reserved for threshold tuning")
    # parser.add_argument("--positive_class_name", type=str, default="critical",
    # help="Class name to treat as positive for threshold tuning (e.g., 'critical')")
    args = parser.parse_args()
   

   # load feature lists to ensure correct dimensions for tensors splitting before passing to LSTM
    feature_lists = json.loads((args.data_dir / "feature_lists.json").read_text())
    cont_dim = len(feature_lists["CONTINUOUS_USED"]) + len(feature_lists["BOOLEAN_USED"])
    high_cat_dim = len(feature_lists["HIGH_CAT_USED"])
    low_cat_dim = len(feature_lists["LOW_CAT_USED"])

    # load feature arrays -------------------------------------
    X_tab = np.load(args.data_dir / "X_train_tab.npy", allow_pickle=True)
    y_tab = np.load(args.data_dir / "y_train.npy", allow_pickle=True)

    seq_data = torch.load(args.data_dir / "seq_train.pt")

    # Stratified split to ensure all classes are represented in validation
    val_frac = args.val_frac
    min_val_size = len(np.unique(y_tab))  # Ensure at least one sample per class in validation
    val_size = max(min_val_size, int(len(X_tab) * val_frac))
    test_size = val_size / len(X_tab)  # proportion of the dataset

     # Stratified validation split
    if len(np.unique(y_tab)) > 1 and len(y_tab) > 1:
        indices = np.arange(len(X_tab))
        train_idx, val_idx = train_test_split(
            indices, test_size=test_size, stratify=y_tab, random_state=42
        )

        X_train_tab, X_val_tab = X_tab[train_idx], X_tab[val_idx]
        y_train, y_val = y_tab[train_idx], y_tab[val_idx]
        # seq_train_split, seq_val = seq_train[train_idx], seq_train[val_idx]
    else:
        # Fallback for tiny or single-class datasets
        val_size = max(1, int(len(X_tab) * val_frac))
        # X_val_tab, y_val, seq_val = X_tab[:val_size], y_tab[:val_size], seq_train[:val_size]
        val_idx = np.arange(val_size)
        X_val_tab, y_val = X_tab[:val_size], y_tab[:val_size]


    if isinstance(seq_data, tuple):
        print("Using tuple format data")
        cont_train, cat_train = seq_data
        # PRESERVE TUPLE FORMAT - NO CONVERSION
        cont_val = cont_train[val_idx]
        cat_high_val = cat_train[val_idx, :, :high_cat_dim].long()
        cat_low_val = cat_train[val_idx, :, high_cat_dim:].long()
        
        # MAINTAIN TUPLE INTERFACE THROUGHOUT
        seq_val = (cont_val, cat_high_val, cat_low_val)

    # Check class distribution in validation set
    class_counts = collections.Counter(y_val)
    print("Validation set class distribution:", dict(class_counts))
    print("Unique classes in y_val:", np.unique(y_val))

    rf, transformer, _ = load_models(args.data_dir, args.device)
    tau = tune_threshold(rf,transformer, X_val_tab, seq_val, y_val, args.device, cont_dim, high_cat_dim, low_cat_dim)
    print(f"Optimal threshold τ (F1) ≈ {tau:.2f}")

    # Compute τ2 on RF-escalated subset
    rf_max = np.array([rf_predict_proba(rf, x).max() for x in X_val_tab])
    escalated_mask = rf_max >= tau
    tau2 = tune_tau2(transformer, seq_val, y_val, escalated_mask, args.device, cont_dim, high_cat_dim, low_cat_dim)
    print(f"Optimal transformer threshold τ2 (F1 on escalated) ≈ {tau2:.2f}")

    cfg = {
        "rf_model_path"   : str((args.data_dir / "rf_model.joblib").resolve()),
        # "lstm_model_path" : str((args.data_dir / "lstm_attention.pt").resolve()),
        "transformer_model_path": str((args.data_dir / "cybersecurity_transformer.pt").resolve()),
        "tau"              : float(tau),
        "tau2"             : float(tau2),
        "cont_dim"         : cont_dim,
        "high_cat_dim"     : high_cat_dim,
        "low_cat_dim"      : low_cat_dim
    }
    (args.data_dir / "cascade_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Saved cascade_config.json")

if __name__ == "__main__":
    main()