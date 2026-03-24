"""
Enhanced feature engineering with anomaly detection and ground truth integration.
"""
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from src.utils.ground_truth_labeling import GroundTruthLabeler
from src.utils.enhanced_business_rules import EnhancedBusinessRulesConfig
import logging

logger = logging.getLogger(__name__)

# Add near top (after imports) if not already imported
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

def _fallback_behavioral(df, is_training, state):
    from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
    skip = {'label','user'}
    # Candidate numeric (non-datetime) cols
    candidates = []
    for c in df.columns:
        if c in skip: continue
        s = df[c]
        if is_datetime64_any_dtype(s): continue
        if is_numeric_dtype(s):
            candidates.append(c)

    if is_training:
        if not candidates:
            feature_names = []
            X = np.zeros((len(df), 1))
        else:
            X_df = df[candidates].apply(lambda col: col.astype('float64')).fillna(0)
            var = X_df.var(axis=0)
            feature_names = [c for c,v in var.items() if v > 0]
            if not feature_names:
                feature_names = []
                X = np.zeros((len(df), 1))
            else:
                X = X_df[feature_names].values
        state['behavioral_features'] = feature_names
    else:
        feature_names = state.get('behavioral_features', [])
        if not feature_names:
            X = np.zeros((len(df), 1))
        else:
            # Build aligned matrix
            cols_present = {c for c in df.columns}
            build_cols = {}
            for c in feature_names:
                if c in cols_present:
                    build_cols[c] = df[c].astype('float64').fillna(0)
                else:
                    build_cols[c] = pd.Series(0.0, index=df.index)
            X = pd.DataFrame(build_cols)[feature_names].values

    # Fit / transform scaler
    if 'scaler' not in state:
        state['scaler'] = StandardScaler()
    if is_training:
        Xs = state['scaler'].fit_transform(X)
        iso = IsolationForest(n_estimators=100, random_state=42, contamination=0.02)
        scores = -iso.fit_predict(Xs)
        state['iso'] = iso
    else:
        Xs = state['scaler'].transform(X)
        iso = state.get('iso')
        scores = -iso.predict(Xs) if iso is not None else np.zeros(len(df))

    user_counts = df['user'].value_counts() if 'user' in df.columns else None
    cluster_score = df['user'].map(user_counts.rdiv(user_counts.max())) if user_counts is not None else 0
    return {
        'user_anomaly_score': (scores - scores.min()) / (scores.ptp() + 1e-9),
        'cluster_anomaly_score': cluster_score.fillna(0).values if hasattr(cluster_score,'values') else np.zeros(len(df))
    }

def _simple_temporal(df):
    return (df['hour'].isin([0,1,2,3,4])).astype(float) if 'hour' in df.columns else 0.0

def _simple_volume(df):
    col = 'megabytes_sent'
    v = df[col].fillna(0)
    thr = v.quantile(0.99)
    return (v > thr).astype(float)

def _simple_network(df):
    if 'destination_domain' in df.columns:
        freq = df['destination_domain'].value_counts()
        rare = df['destination_domain'].map(freq < freq.quantile(0.05)).astype(float)
        return rare
    return 0.0

def add_anomaly_detection_features(data: pd.DataFrame, is_training: bool = True, labeler=None) -> pd.DataFrame:
    logger.info("Adding anomaly detection features...")
    if labeler is None:
        labeler = GroundTruthLabeler()

    # Shared state for training vs test
    if not hasattr(labeler, '_anomaly_state'):
        labeler._anomaly_state = {}

    # Behavioral
    if hasattr(labeler, 'detect_behavioral_anomalies'):
        anomaly_scores = labeler.detect_behavioral_anomalies(data, is_training)
    else:
        anomaly_scores = _fallback_behavioral(data, is_training, labeler._anomaly_state)

    data['user_anomaly_score'] = anomaly_scores['user_anomaly_score']
    data['cluster_anomaly_score'] = anomaly_scores['cluster_anomaly_score']

    # Temporal
    if hasattr(labeler, 'detect_temporal_anomalies'):
        data['temporal_anomaly_score'] = labeler.detect_temporal_anomalies(data)
    else:
        data['temporal_anomaly_score'] = _simple_temporal(data)

    # Volume
    if hasattr(labeler, 'detect_volume_anomalies'):
        data['volume_anomaly_score'] = labeler.detect_volume_anomalies(data)
    else:
        data['volume_anomaly_score'] = _simple_volume(data)

    # Network
    if hasattr(labeler, 'detect_network_anomalies'):
        data['network_anomaly_score'] = labeler.detect_network_anomalies(data)
    else:
        data['network_anomaly_score'] = _simple_network(data)

    data['composite_anomaly_score'] = (
        0.3 * data['user_anomaly_score'] +
        0.2 * data['cluster_anomaly_score'] +
        0.2 * data['temporal_anomaly_score'] +
        0.15 * data['volume_anomaly_score'] +
        0.15 * data['network_anomaly_score']
    )
    return data
def add_ground_truth_features(data: pd.DataFrame, decoy_file_path: str = None) -> pd.DataFrame:
    """Add ground truth-based features."""
    
    logger.info("Adding ground truth features...")
    
    labeler = GroundTruthLabeler(decoy_file_path)
    
    # Decoy interaction features
    data['decoy_risk_score'] = labeler.identify_decoy_interactions(data)
    data['is_decoy_interaction'] = (data['decoy_risk_score'] > 0.5).astype(int)
    
    # Time since last decoy interaction
    data['days_since_decoy'] = 0  # Placeholder - implement based on your temporal logic
    
    return data

def add_pattern_discovery_features(data: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """Add features that help discover new patterns beyond rules."""
    
    logger.info("Adding pattern discovery features...")
    
    # Cross-feature interactions
    data['size_hour_interaction'] = data['megabytes_sent'].fillna(0) * data['hour']
    data['entropy_burst_interaction'] = data['destination_entropy'].fillna(0) * data['post_burst'].fillna(0)
    data['weekend_afterhours_interaction'] = data['is_weekend'].astype(int) * data['after_hours'].astype(int)
    
    # Sequence-based features
    data = data.sort_values(['date'])
    data['upload_velocity'] = data['megabytes_sent'].fillna(0).rolling(window=5, min_periods=1).sum()
    def rolling_domain_switching_rate(domains, window=10):
        result = []
        for i in range(len(domains)):
            start = max(0, i - window + 1)
            window_slice = domains[start:i+1]
            unique_count = len(set(window_slice))
            result.append(unique_count / len(window_slice))
        return result

    data['domain_switching_rate'] = rolling_domain_switching_rate(data['destination_domain'].astype(str).tolist(), window=10)
    
    # Frequency-based features
    data['rare_hour_flag'] = data['hour'].map(
        data['hour'].value_counts().apply(lambda x: 1 if x < data['hour'].value_counts().quantile(0.1) else 0)
    )
    
    return data