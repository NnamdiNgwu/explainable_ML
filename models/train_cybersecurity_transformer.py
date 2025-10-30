"""train_transformer.py - Train Cybersecurity Transformer model.

Usage:
------
python -m src.models.train_cyber_transformer \
    --data_dir data_processed \
    --epochs 50 --batch 32 --lr 0.001
"""

import argparse
import pathlib
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score, classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from models.cybersecurity_transformer import build_cybersecurity_transformer_from_maps
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
# from torch.utils.data import WeightedRandomSampler

# def build_class_balanced_sampler(y: torch.Tensor) -> WeightedRandomSampler:
#     """Create a sampler with inverse-frequency weights for class balance."""
#     classes, counts = torch.unique(y, return_counts=True)
#     freq = {int(c): float(cnt) for c, cnt in zip(classes, counts)}
#     weights = torch.tensor([1.0 / freq[int(label)] for label in y], dtype=torch.float)
#     return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        # self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss =  F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = -self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()  # Use mean reduction
    
class EarlyStopping:
    """Early stopping based on routine event performance"""
    def __init__(self, patience =10, min_delta=0.01, routine_threshold=0.4):
        self.patience = patience
        self.min_delta = min_delta
        self.routine_threshold = routine_threshold
        self.best_score = float('inf')
        self.counter = 0
        self.best_weight = None
    
    def __call__(self, model, val_loss, routine_score):
        # Combined score: validation loss + penalty for high routine score
        routine_penalty = max(0, routine_score - self.routine_threshold) * 2.0
        combined_score = val_loss + routine_penalty

        if combined_score < self.best_score - self.min_delta:
            self.best_score = combined_score
            self.counter = 0
            self.best_weight = model.state_dict().copy()
            return False  # Not stopping
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.best_weight is not None:
                    model.load_state_dict(self.best_weight)
                print(f"Early stopping triggered after {self.counter} epochs with validation loss {val_loss :.4f} ")
                print(f"Early stopping triggered after {self.counter} epochs with routine score {routine_score:.4f}")
                return True
        return False  # Not stopping


def validate_routine_event(model, val_loader, device):
    """Validate model on routine events extracted from validation set."""
    model.eval()
    routine_predictions = []
    
    with torch.no_grad():
        # Take first few batches as "routine" events (assuming they're mixed)
        batch_count = 0
        for cont_val, cat_val, y_val in val_loader:
            if batch_count >= 3:  # Only check first 3 batches
                break
                
            cat_high = cat_val[..., :2].long().to(device)
            cat_low = cat_val[..., 2:].long().to(device)
            cont_val = cont_val.to(device)
            
            logits = model(cont_val, cat_high, cat_low)
            probs = torch.softmax(logits, dim=1)
            
            # Filter for samples that should be "normal" (class 0)
            normal_mask = y_val == 0
            if normal_mask.sum() > 0:
                normal_probs = probs[normal_mask.to(device)]
                routine_predictions.extend(normal_probs[:, 1].cpu().numpy())  # Risk probabilities
            
            batch_count += 1

        return np.mean(routine_predictions) if routine_predictions else 0.0
            

def train_transformer(model, train_loader, val_loader, epochs, lr, device, num_classes=3):
    """Train the cybersecurity transformer."""
    
    model.to(device)

    # Use Focal Loss for class imbalance
    criterion = FocalLoss(alpha=0.5, gamma=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Use class weights to handle imbalance
    # class_weights = torch.tensor([1.0, 12.0]).to(device)  # Weight classes based on rarity
    # criterion = nn.CrossEntropyLoss(weight=class_weights)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    early_stopping = EarlyStopping(patience=15, min_delta=0.01, routine_threshold=0.3)
    
    history = []
    best_f1 = 0.0
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        for batch_idx, (cont_batch, cat_batch, y_batch) in enumerate(train_loader):
            # Split categorical features
            cat_high = cat_batch[..., :2].long()  # First 2 categorical features
            cat_low = cat_batch[..., 2:].long()   # Remaining categorical features
            
            cont_batch = cont_batch.to(device)
            cat_high = cat_high.to(device)
            cat_low = cat_low.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(cont_batch, cat_high, cat_low)
            loss = criterion(logits, y_batch)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_predictions.extend(logits.argmax(1).cpu().numpy())
            train_targets.extend(y_batch.cpu().numpy())

        # scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for cont_val, cat_val, y_val in val_loader:
                cat_high = cat_val[..., :2].long()
                cat_low = cat_val[..., 2:].long()
                
                cont_val = cont_val.to(device)
                cat_high = cat_high.to(device)
                cat_low = cat_low.to(device)
                y_val = y_val.to(device)
                
                logits = model(cont_val, cat_high, cat_low)
                loss = criterion(logits, y_val)
                
                val_loss += loss.item()
                val_predictions.extend(logits.argmax(1).cpu().numpy())
                val_targets.extend(y_val.cpu().numpy())
        
        # Calculate metrics
        train_f1 = f1_score(train_targets, train_predictions, average='macro')
        val_f1 = f1_score(val_targets, val_predictions, average='macro')
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Routine event validation (every 3 epochs)
        routine_score = 0.0
        if epoch % 3 == 0:
            routine_score = validate_routine_event(model, val_loader, device)
            print(f"Routine event validation score: {routine_score:.4f}")
        
        # Learning rate scheduling
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            # For ReduceLROnPlateau, step with validation loss
            scheduler.step(avg_val_loss)
        else:
            scheduler.step() # For CosineAnnealingLR
        
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'routine_score': routine_score if  epoch % 3 == 0 else None,
            'learning_rate': scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else lr
        })
        
        if val_f1 > best_f1:
            best_f1 = val_f1
        
        print(f"Epoch {epoch:02d}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
              f"train_f1={train_f1:.4f}, val_f1={val_f1:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
        
        # Check early stopping
        if epoch % 3 == 0 and early_stopping(model, avg_val_loss, routine_score):

            break
    
    return history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=pathlib.Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', default='cpu')
    # parser.add_argument("--expl_vocab", type=int, default=0)
    args = parser.parse_args()
    
    # device = torch.device(args.device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load data
        print("Loading training data...")
        seq_train = torch.load(args.data_dir / 'seq_train.pt')
        seq_test = torch.load(args.data_dir / 'seq_test.pt')
        y_train = torch.load(args.data_dir / 'y_train_t.pt')
        y_test = torch.load(args.data_dir / 'y_test_t.pt')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return
    
    # Handle tuple format
    if isinstance(seq_train, tuple):
        cont_train, cat_train = seq_train
        cont_test, cat_test = seq_test
    else:
        # If single tensor, we need to split it
        with open(args.data_dir / 'feature_lists.json', 'r') as f:
            feature_lists = json.load(f)
        
        cont_dim = len(feature_lists['CONTINUOUS_USED']) + len(feature_lists['BOOLEAN_USED'])
        
        cont_train = seq_train[..., :cont_dim]
        cat_train = seq_train[..., cont_dim:]
        cont_test = seq_test[..., :cont_dim]
        cat_test = seq_test[..., cont_dim:]
    
    print(f"Data shapes - cont_train: {cont_train.shape}, cat_train: {cat_train.shape}")
    print(f"Label distribution - train: {np.bincount(y_train)}, test: {np.bincount(y_test)}")
    
    # Load embedding maps
    with open(args.data_dir / 'embedding_maps.json', 'r') as f:
        embed_maps = json.load(f)
    
    with open(args.data_dir / 'feature_lists.json', 'r') as f:
        feature_lists = json.load(f)
    
    continuous_dim = len(feature_lists['CONTINUOUS_USED']) + len(feature_lists['BOOLEAN_USED'])
    
    # Build model
    print("Building transformer model...")
    model = build_cybersecurity_transformer_from_maps(embed_maps, continuous_dim, num_classes=3)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = torch.compile(model)  # Compile model for better performance if available
    
    # Create data loaders
    train_dataset = TensorDataset(cont_train, cat_train, y_train)
    test_dataset = TensorDataset(cont_test, cat_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(test_dataset, batch_size=args.batch)
    
    # Train model
    print(f"Training transformer for {args.epochs} epochs...")
    start_time = time.time()
    
    history = train_transformer(model, train_loader, val_loader, args.epochs, args.lr, device)
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed/60:.1f} minutes")

    # Final evaluation
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for cont_test_batch, cat_test_batch, y_test_batch in val_loader:
            cat_high = cat_test_batch[..., :2].long().to(device)
            cat_low = cat_test_batch[..., 2:].long().to(device)
            cont_test_batch = cont_test_batch.to(device)
            
            logits = model(cont_test_batch, cat_high, cat_low)
            test_predictions.extend(logits.argmax(1).cpu().numpy())
            test_targets.extend(y_test_batch.numpy())
            
    
    # Calculate final metrics
    test_accuracy = accuracy_score(test_targets, test_predictions)
    test_f1 = f1_score(test_targets, test_predictions, average='macro')
    
    print(f"\nFinal Test Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(test_targets, test_predictions, 
                              target_names=['Low','Medium','High'])) #['Minor', 'Major', 'Critical']))
    
    # Save model and history
    torch.save(model.state_dict(), args.data_dir / 'cybersecurity_transformer.pt')
    history_df = pd.DataFrame(history)
    history_df.to_csv(args.data_dir / 'transformer_history.csv', index=False)
    
    print(f"Model saved to: {args.data_dir / 'cybersecurity_transformer.pt'}")
    print(f"Training history saved to: {args.data_dir / 'transformer_history.csv'}")

    # Enhanced plotting with multiple metrics
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Training and Validation Loss
    plt.subplot(2, 3, 1)
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', marker='o')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: F1 Scores
    plt.subplot(2, 3, 2)
    plt.plot(history_df['epoch'], history_df['train_f1'], label='Train F1', marker='o')
    plt.plot(history_df['epoch'], history_df['val_f1'], label='Validation F1', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Routine Event Risk Scores (if available)
    if 'routine_score' in history_df.columns and history_df['routine_score'].notna().any():
        routine_scores = history_df[history_df['routine_score'].notna()]
        plt.subplot(2, 3, 3)
        plt.plot(routine_scores['epoch'], routine_scores['routine_score'], 
                label='Routine Event Risk', marker='d', color='red')
        plt.axhline(y=0.3, color='orange', linestyle='--', label='Risk Threshold')
        plt.xlabel('Epoch')
        plt.ylabel('Avg Risk Score')
        plt.title('Routine Event Risk Monitoring')
        plt.legend()
        plt.grid(True)
    
    # Plot 4: Loss Difference (Overfitting indicator)
    plt.subplot(2, 3, 4)
    loss_diff = history_df['val_loss'] - history_df['train_loss']
    plt.plot(history_df['epoch'], loss_diff, label='Val Loss - Train Loss', marker='x', color='purple')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.title('Overfitting Indicator')
    plt.legend()
    plt.grid(True)
    
    # Plot 5: Learning Rate (if using scheduler)
    plt.subplot(2, 3, 5)
    # Note: You'd need to track LR in history for this to work
    plt.text(0.5, 0.5, 'Learning Rate\nTracking\n(Not implemented)', 
             ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Learning Rate Schedule')
    
    # Plot 6: Model Performance Summary
    plt.subplot(2, 3, 6)
    final_metrics = [
        f"Final Test Accuracy: {test_accuracy:.3f}",
        f"Final Test F1: {test_f1:.3f}",
        f"Best Val F1: {max(history_df['val_f1']):.3f}",
        f"Training Epochs: {len(history_df)}",
        f"Final Train Loss: {history_df['train_loss'].iloc[-1]:.3f}",
        f"Final Val Loss: {history_df['val_loss'].iloc[-1]:.3f}"
    ]
    
    plt.text(0.1, 0.9, '\n'.join(final_metrics), 
             ha='left', va='top', transform=plt.gca().transAxes,
             fontfamily='monospace', fontsize=10)
    plt.title('Training Summary')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(args.data_dir / 'training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training analysis plot saved to: {args.data_dir / 'training_analysis.png'}")

     # Prepare calibration data from validation set
    cal_predictions = []
    cal_targets = []
    
    model.eval()
    with torch.no_grad():
        for cont_val, cat_val, y_val in val_loader:
            cat_high = cat_val[..., :2].long().to(device)
            cat_low = cat_val[..., 2:].long().to(device)
            cont_val = cont_val.to(device)
            
            logits = model(cont_val, cat_high, cat_low)
            probs = torch.softmax(logits, dim=1)
            
            cal_predictions.append(probs.cpu().numpy())
            cal_targets.append(y_val.numpy())
    
    cal_predictions = np.vstack(cal_predictions)
    cal_targets = np.hstack(cal_targets)

    # Apply calibration using CalibratedClassifierCV
    print("Applying probability calibration...")
    class PyTorchWrapper(BaseEstimator, ClassifierMixin):
        def __init__(self, predictions):
            self.predictions = predictions
            self.classes_ = np.array([0, 1])
            
        def fit(self, X, y):
            return self
            
        def predict_proba(self, X):
            return self.predictions[X.astype(int)]
    
    # Create wrapper and calibrate
    wrapper = PyTorchWrapper(cal_predictions)
    indices = np.arange(len(cal_predictions))
    
    calibrated_model = CalibratedClassifierCV(wrapper, method='sigmoid', cv=3)
    calibrated_model.fit(indices, cal_targets)
    
    # Save calibrated model
    import joblib
    joblib.dump(calibrated_model, args.data_dir / 'cybersecurity_transformer_calibrated.pkl')
    
    print("✅ Training completed with regularization, early stopping, and calibration!")
    print(f"Original model: {args.data_dir / 'cybersecurity_transformer.pt'}")
    print(f"Calibrated model: {args.data_dir / 'cybersecurity_transformer_calibrated.pkl'}")
    

if __name__ == '__main__':
    main()