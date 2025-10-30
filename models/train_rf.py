#!/usr/bin/env python3
"""
train_rf.py - Train Random-Forest baseline and save rf_model.joblib.

Reads tabular feature metrices created by `make_features.py` and and trains a 
SMOTE-balanced RandomForest with (optionally) a small hyper-parameter grid.
Outputs
 - rf_model.joblib               - fitted sklearn model
 - cv_results.csv                - (optional) grid-search scores

 Usage ::
    python -m src.models.train_rf \
           --data_dir data_processed \
           --n_jobs          -1 \
           --grid             2   # 0 = skip grid-search

Docker ::
The repo Dockerfile installs scikit-learn, imbalanced-learn, and joblib.
When inside the container, run the same command as above.
"""
from __future__ import annotations
import argparse, json, pathlib, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import BaseEstimator
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.metrics import roc_curve, auc, RocCurveDisplay, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from models.safe_smote import SafeSMOTE

# --------------------------- CLI -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Random Forest on tabular data.")
    parser.add_argument("--data_dir", type=pathlib.Path, required=True,
                        help="Directory containing X_train_tab.npy, y_train.npy, preprocessors.pkl")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="CPU cores for RF & grid-search (-1 = all cores)")
    parser.add_argument("--grid", type=int, default=0,
                        help="Run small hyper-parameter grid, 0 = fixed params")
    return parser.parse_args()


# --------------------------- prepare data -------------------------------
def load_tabular(data_dir: pathlib.Path):
    X_train = np.load(data_dir / "X_train_tab.npy", allow_pickle=True)
    X_test  = np.load(data_dir / "X_test_tab.npy", allow_pickle=True)
    y_train = np.load(data_dir / "y_train.npy", allow_pickle=True)

    y_test  = np.load(data_dir / "y_test.npy", allow_pickle=True) # not used in training, but for evaluation
    # for i in range(X_train.shape[1]):
    #     print(f"Col {i} unique:", np.unique(X_train[:, i]))

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Ensure numeric type
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)

    assert not np.isnan(X_train).any(), "NaNs remain in X_train!"
    assert not np.isnan(X_test).any(), "NaNs remain in X_test!"
    assert not np.isnan(y_train).any(), "NaNs remain in y_train!"
    assert not np.isnan(y_test).any(), "NaNs remain in y_test!"

    for i in range(X_train.shape[1]):
        print(f"Col {i} unique (as str):", set(str(v) for v in X_train[:, i]))
    

    print("NaNs in X_train:", np.isnan(X_train.astype(float)).sum())
    print("NaNs in y_train:", np.isnan(y_train.astype(float)).sum())
    print("Any NaNs in X_train?", np.any(np.isnan(X_train.astype(float))))
    print("Any NaNs in y_train?", np.any(np.isnan(y_train.astype(float))))

    print("x_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    # SMOTENC: mark categorical indices (booleans+low+high one-hots at the *end* of matrix)
    preprocessors = joblib.load(data_dir / "preprocessors.pkl")
    pre = preprocessors['pre']
    ohe = preprocessors['onehot']
    ordenc = preprocessors['ordinal']
    # number of numerical+bool cols is len(cont)+ len(bool)
    num_bool_dim = len(pre.transformers_[0][2]) + len(pre.transformers_[1][2])
    cat_mask     = [False] * num_bool_dim + [True] * (X_train.shape[1] - num_bool_dim)
    return X_train, y_train, X_test, y_test, cat_mask


def build_pipeline(cat_mask, n_jobs):
    """Leakage-free pipeline that adapts SMOTE per CV fold."""

    # Use the top-level SafeSMOTE class
    smote = SafeSMOTE(categorical_features=cat_mask, random_state=42, k_neighbors=5)
    rf = RandomForestClassifier(random_state=42, n_jobs=n_jobs, 
                                n_estimators=200,
                                class_weight={0:1, 1:6, 2:12})
                                # class_weight={0:1, 1:12})
    
    pipeline = Pipeline([
        ("smote", smote),
        ("clf", rf)
    ])
    
    return pipeline

def validate_no_leakage(X_train, X_test, y_train, y_test):
    """Check for obvious signs of data leakage."""
    
    print("🔍 Checking for data leakage...")
    
    # Check 1: Identical samples
    train_test_overlap = 0
    for i, train_sample in enumerate(X_train[:1000]):  # Sample check
        if any(np.array_equal(train_sample, test_sample) for test_sample in X_test):
            train_test_overlap += 1
    
    if train_test_overlap > 0:
        print(f"❌ Found {train_test_overlap} identical samples in train/test")
    
    # Check 2: Suspicious feature statistics
    train_means = np.mean(X_train, axis=0)
    test_means = np.mean(X_test, axis=0)
    
    # Features should have different means due to temporal split
    mean_diff = np.abs(train_means - test_means)
    suspiciously_similar = np.sum(mean_diff < 0.01)
    
    if suspiciously_similar > len(train_means) * 0.8:
        print(f"⚠️  {suspiciously_similar}/{len(train_means)} features have very similar train/test means")
        print("   This might indicate preprocessing leakage")
    
    # Check 3: Class distribution
    train_dist = np.bincount(y_train) / len(y_train)
    test_dist = np.bincount(y_test) / len(y_test)
    print(f"📊 Train class distribution: {train_dist}")
    print(f"📊 Test class distribution: {test_dist}")
    
    print("✅ Leakage validation complete")


# ----------------------------- main --------------------------------
def main():
    args = parse_args()
    X_train, y_train, X_test, y_test, cat_mask = load_tabular(args.data_dir)

    validate_no_leakage(X_train, X_test, y_train, y_test)

    pipe = build_pipeline(cat_mask, args.n_jobs)#, y_train)
    # --- Cross-validation ---
    print("=== Stratified 5-Fold Cross-Validation ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring='f1_macro', n_jobs=-1)
    print("F1 Macro scores for each fold:", scores)
    print("Mean F1 Macro:", np.mean(scores))
    print("Std F1 Macro:", np.std(scores))


    if args.grid:
        # Small grid-search for hyper-parameters
        param_grid = {
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [12, 16],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__min_samples_split': [2, 4],
            'clf__max_features': ['sqrt', 'log2'],
            'clf__class_weight': [{0:1, 1:6, 2:12}, "balanced"],
        }
        min_class_count = np.bincount(y_train).min() #
        cv = StratifiedKFold(n_splits=min(5, min_class_count), shuffle=True, random_state=42)
        grid_search = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=args.n_jobs,
                                   scoring='f1_macro', verbose=2, refit=True)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        pd.DataFrame(grid_search.cv_results_).to_csv(args.data_dir / "cv_results.csv", index=False)
        print(f"Best params: {grid_search.best_params_}")
        print(f"Best CV F1 Macro: {grid_search.best_score_:.4f}")
    else:
        print("NaNs in X_train:", np.isnan(X_train).sum())
        # Fixed hyper-parameters
        best_model = pipe.fit(X_train, y_train)

    joblib.dump(best_model, args.data_dir / "rf_model.joblib")
    print(f"[✓] Saved RandomForest model to {args.data_dir / 'rf_model.joblib'}")
    print(f"[✓] Model score: {best_model.score(X_train, y_train):.4f} (train)")
    # print(f"[✓] Model score: {best_model.score(X_train, y_train):.4f} (test)")
    # print(f"[✓] Model score: {best_model.score(X_train, y_train):.4f} (validation)")
    print(f"[✓] Model score: {best_model.score(X_test, y_test):.4f} (test)")

    # --- Add this block to check class balance and detailed metrics ---
    y_pred = best_model.predict(X_test)
    print("\nClassification report (test set):")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix (test set):")
    print(confusion_matrix(y_test, y_pred))


    # === Multiclass ROC / AUC ===
    if hasattr(best_model, "predict_proba"):
        rf_est = best_model.named_steps['clf'] if hasattr(best_model, "named_steps") else best_model
        classes = rf_est.classes_
        proba = best_model.predict_proba(X_test)
        # Binarize
        y_bin = label_binarize(y_test, classes=classes)
        # Compute per-class ROC
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, roc_auc_score
        fig, ax = plt.subplots(figsize=(6,5))
        for i, c in enumerate(classes):
            fpr_c, tpr_c, _ = roc_curve(y_bin[:, i], proba[:, i])
            ax.plot(fpr_c, tpr_c, lw=1.5, label=f"Class {c} ROC (AUC={roc_auc_score(y_bin[:, i], proba[:, i]):.3f})")
        # Macro AUC
        macro_auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        ax.plot([0,1],[0,1],'--', color='gray', lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title(f"Multiclass ROC (macro AUC={macro_auc:.3f})")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path = args.data_dir / "rf_roc_multiclass.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[✓] Multiclass ROC saved to {out_path}")
        # If truly binary, also save simplified binary curve
        if len(classes) == 2:
            fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
            bin_auc = roc_auc_score(y_test, proba[:, 1])
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC={bin_auc:.3f}')
            ax2.plot([0,1],[0,1],'--', color='navy', lw=1)
            ax2.set_xlim(0,1); ax2.set_ylim(0,1.05)
            ax2.set_xlabel("False Positive Rate"); ax2.set_ylabel("True Positive Rate")
            ax2.set_title("Binary ROC")
            ax2.legend()
            fig2.tight_layout()
            out_bin = args.data_dir / "rf_roc_binary.png"
            fig2.savefig(out_bin, dpi=150)
            plt.close(fig2)
            print(f"[✓] Binary ROC saved to {out_bin}")
    else:
        print("Model does not support predict_proba; ROC skipped.")
if __name__ == "__main__":
    main()
   
