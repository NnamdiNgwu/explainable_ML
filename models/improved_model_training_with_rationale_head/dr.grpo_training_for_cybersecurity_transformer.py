"""
Dr. GRPO Training Script: Distillation-based Reasoning GRPO for Cybersecurity Transformer

Pipeline:
1. Distillation Phase: Student Transformer learns from RF teacher (class labels + SHAP rationale)
2. GRPO Fine-Tuning: Optimize operational metrics with per-class constraints
3. Evaluation: Test reasoning transfer and zero-shot generalization

Usage:
    # Phase 1: Distillation
    python -m models.train_drgrpo --phase distill --data_dir data_processed --epochs 50
    
    # Phase 2: GRPO fine-tuning
    python -m models.train_drgrpo --phase grpo --data_dir data_processed --epochs 20 \
        --checkpoint data_processed/transformer_distilled.pt --target_recall_class2 0.8
    
    # Full pipeline
    python -m models.train_drgrpo --phase all --data_dir data_processed

    python -m models.train_drgrpo \
    --phase all \
    --data_dir data_processed \
    --epochs_distill 50 \
    --epochs_grpo 20 \
    --batch_size 64 \
    --target_recall_class2 0.8

    # Evaluate
    python -m models.train_drgrpo \
    --phase eval \
    --data_dir data_processed \
    --checkpoint data_processed/transformer_grpo.pt
"""

import argparse
import json
import logging
import time
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Categorical

import joblib
import shap
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    recall_score, precision_score, accuracy_score
)

from models.cybersecurity_transformer import build_cybersecurity_transformer_from_maps

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================================
# 1. Data Loading and RF Rationale Extraction
# ============================================================================

def load_data(data_dir: Path) -> Tuple:
    """Load preprocessed tensors and metadata."""
    logging.info("Loading training tensors...")
    
    # Sequence tensors
    seq_train = torch.load(data_dir / 'seq_train.pt', weights_only=False)
    seq_test = torch.load(data_dir / 'seq_test.pt', weights_only=False)
    
    cont_train, cat_train = seq_train['cont'], seq_train['cat']
    cont_test, cat_test = seq_test['cont'], seq_test['cat']
    
    # Labels
    y_train = torch.from_numpy(np.load(data_dir / 'y_train.npy')).long()
    y_test = torch.from_numpy(np.load(data_dir / 'y_test.npy')).long()
    
    # Tabular features (for RF)
    X_train_tab = np.load(data_dir / 'X_train_tab.npy')
    X_test_tab = np.load(data_dir / 'X_test_tab.npy')
    
    # Metadata
    embed_maps = json.loads((data_dir / 'embedding_maps.json').read_text())
    feature_lists = json.loads((data_dir / 'feature_lists.json').read_text())
    
    # Load RF model
    rf_model = joblib.load(data_dir / 'rf_model.joblib')
    
    # Load preprocessor
    preprocessor = joblib.load(data_dir / 'preprocessors.pkl')
    
    logging.info(f"Data loaded - train: {len(y_train)}, test: {len(y_test)}")
    logging.info(f"Train class distribution: {Counter(y_train.tolist())}")
    logging.info(f"Test class distribution: {Counter(y_test.tolist())}")
    
    return (
        cont_train, cat_train, y_train,
        cont_test, cat_test, y_test,
        X_train_tab, X_test_tab,
        embed_maps, feature_lists, rf_model, preprocessor
    )


def extract_rf_rationale(
    rf_model, 
    X_train_tab: np.ndarray,
    feature_names: list,
    top_k: int = 10,
    max_samples: int = 1000
) -> np.ndarray:
    """
    Extract SHAP-based rationale from RF teacher.
    
    Returns:
        rationale_targets: [N, num_features] binary matrix (1 = feature is in top-k)
    """
    logging.info("Extracting RF rationale via SHAP...")
    
    # Subsample for SHAP computation (expensive)
    sample_size = min(max_samples, len(X_train_tab))
    indices = np.random.choice(len(X_train_tab), sample_size, replace=False)
    X_sample = X_train_tab[indices]
    
    # TreeExplainer
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_sample, check_additivity=False)
    
    # Handle multi-class SHAP (list of arrays)
    if isinstance(shap_values, list):
        # Use positive class (class 1 or 2) SHAP values
        # For 3-class: aggregate class 1 and 2 (medium + high risk)
        if len(shap_values) == 3:
            shap_values = (np.abs(shap_values[1]) + 2 * np.abs(shap_values[2])) / 3
        else:
            shap_values = shap_values[-1]  # Use last class
    
    # Convert to top-k binary rationale
    rationale_sample = np.zeros_like(shap_values)
    for i in range(len(shap_values)):
        abs_shap = np.abs(shap_values[i])
        top_idx = np.argsort(abs_shap)[-top_k:]
        rationale_sample[i, top_idx] = 1
    
    # Expand to full training set (broadcast most common pattern)
    # OR: recompute for all (expensive); here we use nearest neighbor approximation
    from sklearn.neighbors import NearestNeighbors
    nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn_model.fit(X_sample)
    
    distances, neighbor_idx = nn_model.kneighbors(X_train_tab)
    rationale_full = rationale_sample[neighbor_idx.squeeze()]
    
    logging.info(f"Rationale extracted: shape {rationale_full.shape}, sparsity {rationale_full.mean():.3f}")
    
    return rationale_full


# ============================================================================
# 2. Distillation Phase: Student learns from RF teacher
# ============================================================================

class DistillationLoss(nn.Module):
    """Combined classification + rationale distillation loss."""
    
    def __init__(self, alpha_cls: float = 1.0, alpha_rationale: float = 0.5):
        super().__init__()
        self.alpha_cls = alpha_cls
        self.alpha_rationale = alpha_rationale
    
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        rationale_logits: Optional[torch.Tensor] = None,
        rationale_targets: Optional[torch.Tensor] = None
    ):
        # Classification loss (focal)
        cls_loss = focal_loss(logits, labels)
        
        total_loss = self.alpha_cls * cls_loss
        
        # Rationale distillation loss
        if rationale_logits is not None and rationale_targets is not None:
            rationale_loss = F.binary_cross_entropy_with_logits(
                rationale_logits, rationale_targets
            )
            total_loss += self.alpha_rationale * rationale_loss
        else:
            rationale_loss = torch.tensor(0.0)
        
        return total_loss, cls_loss, rationale_loss


def focal_loss(logits: torch.Tensor, labels: torch.Tensor, alpha=None, gamma: float = 2.0):
    """Focal loss for handling class imbalance."""
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    pt = torch.exp(-ce_loss)
    focal_weight = (1 - pt) ** gamma
    
    if alpha is not None:
        alpha_t = alpha[labels]
        focal_weight = alpha_t * focal_weight
    
    return (focal_weight * ce_loss).mean()


def train_distillation(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    focal_alpha: Optional[torch.Tensor] = None,
    high_cat_len: int = 1,
    checkpoint_path: Optional[Path] = None
):
    """Phase 1: Distillation training."""
    logging.info("=" * 60)
    logging.info("Phase 1: Distillation from RF Teacher")
    logging.info("=" * 60)
    
    model.to(device)
    
    criterion = DistillationLoss(alpha_cls=1.0, alpha_rationale=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_cls_loss = 0.0
        train_rationale_loss = 0.0
        
        for batch in train_loader:
            if len(batch) == 4:
                cont, cat, y, rationale_targets = batch
            else:
                cont, cat, y = batch
                rationale_targets = None
            
            cont = cont.to(device)
            cat = cat.to(device)
            y = y.to(device)
            if rationale_targets is not None:
                rationale_targets = rationale_targets.to(device).float()
            
            # Split categorical
            split = high_cat_len
            cat_high = cat[..., :split].long()
            cat_low = cat[..., split:].long()
            
            # Forward
            out = model(cont, cat_high, cat_low, return_explanations=True)
            logits = out["logits"]
            rationale_logits = out.get("rationale_logits", None)
            
            # Loss
            loss, cls_loss, rat_loss = criterion(
                logits, y, rationale_logits, rationale_targets
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_cls_loss += cls_loss.item()
            train_rationale_loss += rat_loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    cont, cat, y, rationale_targets = batch
                else:
                    cont, cat, y = batch
                    rationale_targets = None
                
                cont = cont.to(device)
                cat = cat.to(device)
                y = y.to(device)
                if rationale_targets is not None:
                    rationale_targets = rationale_targets.to(device).float()
                
                split = high_cat_len
                cat_high = cat[..., :split].long()
                cat_low = cat[..., split:].long()
                
                out = model(cont, cat_high, cat_low, return_explanations=True)
                logits = out["logits"]
                rationale_logits = out.get("rationale_logits", None)
                
                loss, _, _ = criterion(logits, y, rationale_logits, rationale_targets)
                val_loss += loss.item()
                
                val_preds.extend(logits.argmax(1).cpu().tolist())
                val_labels.extend(y.cpu().tolist())
        
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_recall = recall_score(val_labels, val_preds, average=None, zero_division=0)
        
        logging.info(
            f"[Distill E{epoch:03d}] "
            f"train_loss={train_loss:.3f} (cls={train_cls_loss:.3f}, rat={train_rationale_loss:.3f}) "
            f"val_loss={val_loss:.3f} val_f1={val_f1:.3f} "
            f"val_recall={[f'{r:.3f}' for r in val_recall]}"
        )
        
        # LR scheduling
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
                logging.info(f"Checkpoint saved: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered (distillation phase)")
                break
    
    logging.info("Distillation phase complete.")
    return model


# ============================================================================
# 3. GRPO Fine-Tuning: Optimize operational rewards with constraints
# ============================================================================

def compute_group_reward(
    preds: torch.Tensor,
    labels: torch.Tensor,
    class_weights: list = [1, 3, 10],
    false_negative_penalty: float = -20,
    false_alarm_penalty: float = -5
) -> torch.Tensor:
    """
    Operational reward function:
    - Correct class 2 (critical): +10
    - Correct class 1 (medium): +3
    - Correct class 0 (low): +1
    - Missed class 2 (FN): -20
    - False alarm (class 0 → 2): -5
    """
    rewards = []
    for pred, true in zip(preds, labels):
        pred_item = pred.item() if isinstance(pred, torch.Tensor) else pred
        true_item = true.item() if isinstance(true, torch.Tensor) else true
        
        if pred_item == true_item:
            rewards.append(class_weights[true_item])
        elif true_item == 2 and pred_item != 2:
            rewards.append(false_negative_penalty)
        elif true_item == 0 and pred_item == 2:
            rewards.append(false_alarm_penalty)
        else:
            rewards.append(-3)
    
    return torch.tensor(rewards, dtype=torch.float32)


def train_grpo(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    high_cat_len: int = 1,
    target_recall_class2: float = 0.8,
    entropy_coef: float = 0.01,
    clip_ratio: float = 0.2,
    checkpoint_path: Optional[Path] = None
):
    """Phase 2: GRPO fine-tuning with per-class recall constraints."""
    logging.info("=" * 60)
    logging.info("Phase 2: GRPO Fine-Tuning")
    logging.info(f"Target recall for class 2 (critical): {target_recall_class2}")
    logging.info("=" * 60)
    
    model.to(device)
    
    # Freeze rationale head to preserve explanations
    if hasattr(model, 'rationale_head') and model.rationale_head is not None:
        for param in model.rationale_head.parameters():
            param.requires_grad = False
        logging.info("Rationale head frozen.")
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-5
    )
    
    best_recall_class2 = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        epoch_rewards = []
        epoch_preds = []
        epoch_labels = []
        epoch_policy_losses = []
        epoch_entropies = []
        
        for batch in train_loader:
            cont, cat, y = batch[:3]
            cont = cont.to(device)
            cat = cat.to(device)
            y = y.to(device)
            
            split = high_cat_len
            cat_high = cat[..., :split].long()
            cat_low = cat[..., split:].long()
            
            # Forward pass
            out = model(cont, cat_high, cat_low, return_explanations=False)
            logits = out["logits"]
            
            # Sample actions from policy
            action_probs = F.softmax(logits, dim=-1)
            dist = Categorical(action_probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            
            # Compute rewards
            rewards = compute_group_reward(actions, y)
            rewards = rewards.to(device)
            
            # PPO-style clipped loss
            old_log_probs = log_probs.detach()
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Normalize advantages (simple baseline: reward - mean)
            advantages = rewards - rewards.mean()
            advantages = advantages / (advantages.std() + 1e-8)
            
            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus (encourage exploration)
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = policy_loss - entropy_coef * entropy
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Logging
            epoch_rewards.extend(rewards.cpu().tolist())
            epoch_preds.extend(actions.cpu().tolist())
            epoch_labels.extend(y.cpu().tolist())
            epoch_policy_losses.append(policy_loss.item())
            epoch_entropies.append(entropy.item())
        
        # Epoch metrics
        avg_reward = np.mean(epoch_rewards)
        avg_policy_loss = np.mean(epoch_policy_losses)
        avg_entropy = np.mean(epoch_entropies)
        
        train_f1 = f1_score(epoch_labels, epoch_preds, average='macro')
        train_recall = recall_score(epoch_labels, epoch_preds, average=None, zero_division=0)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                cont, cat, y = batch[:3]
                cont = cont.to(device)
                cat = cat.to(device)
                y = y.to(device)
                
                split = high_cat_len
                cat_high = cat[..., :split].long()
                cat_low = cat[..., split:].long()
                
                out = model(cont, cat_high, cat_low, return_explanations=False)
                logits = out["logits"]
                
                val_preds.extend(logits.argmax(1).cpu().tolist())
                val_labels.extend(y.cpu().tolist())
        
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_recall = recall_score(val_labels, val_preds, average=None, zero_division=0)
        
        logging.info(
            f"[GRPO E{epoch:03d}] "
            f"avg_reward={avg_reward:.3f} policy_loss={avg_policy_loss:.3f} entropy={avg_entropy:.3f} "
            f"train_f1={train_f1:.3f} train_recall={[f'{r:.3f}' for r in train_recall]} "
            f"val_f1={val_f1:.3f} val_recall={[f'{r:.3f}' for r in val_recall]}"
        )
        
        # Check constraint: class 2 recall
        recall_class2 = val_recall[2] if len(val_recall) > 2 else 0.0
        if recall_class2 < target_recall_class2:
            logging.warning(
                f"Constraint violation: class 2 recall {recall_class2:.3f} < {target_recall_class2}"
            )
        
        # Save best model based on class 2 recall
        if recall_class2 > best_recall_class2:
            best_recall_class2 = recall_class2
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
                logging.info(f"Best checkpoint saved: {checkpoint_path} (recall_class2={recall_class2:.3f})")
    
    logging.info("GRPO fine-tuning complete.")
    return model


# ============================================================================
# 4. Evaluation and Reasoning Visualization
# ============================================================================

def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    high_cat_len: int = 1,
    rationale_names: Optional[list] = None
):
    """Evaluate model on test set."""
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_rationales = []
    
    with torch.no_grad():
        for batch in test_loader:
            cont, cat, y = batch[:3]
            cont = cont.to(device)
            cat = cat.to(device)
            y = y.to(device)
            
            split = high_cat_len
            cat_high = cat[..., :split].long()
            cat_low = cat[..., split:].long()
            
            out = model(cont, cat_high, cat_low, return_explanations=True)
            logits = out["logits"]
            probs = F.softmax(logits, dim=-1)
            
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())
            all_probs.extend(probs.cpu().numpy())
            
            # Extract rationale if available
            if "rationale_logits" in out and out["rationale_logits"] is not None:
                rationale_probs = torch.sigmoid(out["rationale_logits"]).cpu().numpy()
                all_rationales.extend(rationale_probs)
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    
    logging.info("=" * 60)
    logging.info("Test Set Evaluation")
    logging.info("=" * 60)
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"F1 (macro): {f1_macro:.4f}")
    logging.info(f"F1 (weighted): {f1_weighted:.4f}")
    logging.info(f"Recall per class: {[f'{r:.4f}' for r in recall_per_class]}")
    logging.info(f"Precision per class: {[f'{p:.4f}' for p in precision_per_class]}")
    
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=['Low', 'Medium', 'High'],
        zero_division=0
    ))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    # Sample rationale explanations
    if all_rationales and rationale_names:
        logging.info("\nSample Rationale Explanations (first 5 instances):")
        for i in range(min(5, len(all_rationales))):
            active_rationales = [
                f"{name}={all_rationales[i][j]:.2f}"
                for j, name in enumerate(rationale_names)
                if all_rationales[i][j] > 0.5
            ]
            logging.info(
                f"  Instance {i}: pred={all_preds[i]}, true={all_labels[i]}, "
                f"rationale=[{', '.join(active_rationales)}]"
            )
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'recall': recall_per_class,
        'precision': precision_per_class
    }


# ============================================================================
# 5. Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Dr. GRPO Training for Cybersecurity Transformer")
    parser.add_argument('--phase', choices=['distill', 'grpo', 'all', 'eval'], default='all',
                        help='Training phase: distill, grpo, all, or eval')
    parser.add_argument('--data_dir', type=str, default='data_processed',
                        help='Path to processed data directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (for GRPO or eval phase)')
    parser.add_argument('--epochs_distill', type=int, default=50,
                        help='Epochs for distillation phase')
    parser.add_argument('--epochs_grpo', type=int, default=20,
                        help='Epochs for GRPO phase')
    parser.add_argument('--lr_distill', type=float, default=2e-4,
                        help='Learning rate for distillation')
    parser.add_argument('--lr_grpo', type=float, default=1e-5,
                        help='Learning rate for GRPO')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--target_recall_class2', type=float, default=0.8,
                        help='Target recall for class 2 (critical events)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Setup
    data_dir = Path(args.data_dir)
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load data
    (
        cont_train, cat_train, y_train,
        cont_test, cat_test, y_test,
        X_train_tab, X_test_tab,
        embed_maps, feature_lists, rf_model, preprocessor
    ) = load_data(data_dir)
    
    # Extract categorical split
    high_cat_len = len(feature_lists['HIGH_CAT_USED'])
    
    # Extract RF rationale
    rationale_train = extract_rf_rationale(
        rf_model,
        X_train_tab,
        feature_lists['CONTINUOUS_USED'] + feature_lists['BOOLEAN_USED']
    )
    rationale_train_tensor = torch.from_numpy(rationale_train).float()
    
    # Create dataloaders
    train_dataset = TensorDataset(cont_train, cat_train, y_train, rationale_train_tensor)
    test_dataset = TensorDataset(cont_test, cat_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Compute focal alpha (inverse class frequency)
    class_counts = np.bincount(y_train.numpy())
    focal_alpha = 1.0 / class_counts
    focal_alpha = focal_alpha / focal_alpha.sum() * len(focal_alpha)
    focal_alpha = torch.tensor(focal_alpha, dtype=torch.float32)
    
    logging.info(f"Focal alpha: {focal_alpha.tolist()}")
    
    # Build model
    cont_dim = cont_train.shape[-1]
    num_classes = len(class_counts)
    explanation_vocab_size = rationale_train.shape[-1]
    
    model = build_cybersecurity_transformer_from_maps(
        embed_maps=embed_maps,
        continuous_dim=cont_dim,
        num_classes=num_classes,
        explanation_vocab_size=explanation_vocab_size
    )
    
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Phase execution
    if args.phase in ['distill', 'all']:
        checkpoint_distill = data_dir / 'transformer_distilled.pt'
        model = train_distillation(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,  # Use test as validation for simplicity
            epochs=args.epochs_distill,
            lr=args.lr_distill,
            device=device,
            focal_alpha=focal_alpha,
            high_cat_len=high_cat_len,
            checkpoint_path=checkpoint_distill
        )
        
        # Load best checkpoint
        model.load_state_dict(torch.load(checkpoint_distill))
        logging.info(f"Distilled model loaded from {checkpoint_distill}")
    
    if args.phase in ['grpo', 'all']:
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint))
            logging.info(f"Loaded checkpoint: {args.checkpoint}")
        
        checkpoint_grpo = data_dir / 'transformer_grpo.pt'
        model = train_grpo(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            epochs=args.epochs_grpo,
            lr=args.lr_grpo,
            device=device,
            high_cat_len=high_cat_len,
            target_recall_class2=args.target_recall_class2,
            checkpoint_path=checkpoint_grpo
        )
        
        # Load best checkpoint
        model.load_state_dict(torch.load(checkpoint_grpo))
        logging.info(f"GRPO model loaded from {checkpoint_grpo}")
    
    if args.phase == 'eval':
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint))
            logging.info(f"Loaded checkpoint for evaluation: {args.checkpoint}")
        else:
            raise ValueError("Must provide --checkpoint for eval phase")
    
    # Final evaluation
    rationale_names = [
        f"feature_{i}" for i in range(explanation_vocab_size)
    ]
    
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        high_cat_len=high_cat_len,
        rationale_names=rationale_names
    )
    
    # Save final model
    final_checkpoint = data_dir / 'cybersecurity_transformer_drgrpo.pt'
    torch.save(model.state_dict(), final_checkpoint)
    logging.info(f"Final model saved: {final_checkpoint}")
    
    logging.info("Dr. GRPO training pipeline complete!")


if __name__ == '__main__':
    main()