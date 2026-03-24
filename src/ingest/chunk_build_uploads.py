#!/usr/bin/env python3
"""
build_uploads.py

Construct the *uploads_ table from raw CERT r6.2 logs.

1. Load the six core CSV logs (HTTP, FILE, DEVICE, EMAIL, LOGON, DECOY).
2. EXtract outbound-transfer event per channel:
  -HTTP uploads - HTTP_POST burst.
  - USB copies - FILE rows flagged as `to_removable_media` joined to a same-timestamp DEVICE mount.
  - E-mail     - external attachments where `size` > 0 and recipient outside corp.
3. Add contextual features (hour, after_hours, first_time_dst, domain entropy  ).
4. Assign *risk label* using decopy-file ground truth: 2 = critical, 0 = minor. (Major risk is left 0 for now; rules can be applied later.)
5. Persist the result as Parguet and write the train/test split artefacts when
 `--cutoff YYYY-MM-DD` is supplied.

 Usage:
 ------
 $ python3 build_uploads.py \
    --input_dir r6.2 \
    --output uploads.parquet \
    --cutoff 2011-04-01

    # for the purpose of this research, testing was done with a smaller dataset
    # with a cutoff date of 2010-01-04
    python -m src.ingest.chunk_build_uploads --input_dir src/ingest/test_data --output uploads.parquet --cutoff 2010-01-04

    to dynamically cutoff the training set, use:
    python -m src.ingest.chunk_build_uploads --input_dir src/ingest/test_data --output uploads.parquet --auto_cutoff --target_train_frac 0.75



    python -m src/ingest.chunk_build_uploads \
  --input_dir /Volumes/master_thesis_work/master_thesis_Machine_learning_for_Enhanced_Risk_classification_of_data_uploadsSM/raw_dir/r6.2 \
  --output uploads.parquet \
  --cutoff 2011-04-01

All heavy columns are read with `dtypes=str` first to reduce pandas parsing 
cost, then to cast to needed types.
"""

from __future__ import annotations
import argparse
import os
import time
import pathlib
import re
import joblib
from typing import Tuple, Dict

# import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

import numpy as np, pandas as pd, matplotlib.pyplot as plt, json, logging, sys, math
from .utils import shannon_entropy, load_csv
from src.utils.business_rules import BusinessRulesConfig, assign_risk_label_robust
from src.utils.enhanced_business_rules import EnhancedBusinessRulesConfig
from src.utils.reports_generator import generate_enhanced_reports
from src.utils.updated_feature_engineering import (
    add_anomaly_detection_features,
    add_ground_truth_features,
    add_pattern_discovery_features,
)
# from src.utils.correlation_utils import prepare_corr_frame as _prepare_corr_frame
from src.utils.ground_truth_labeling import GroundTruthLabeler
from sklearn.model_selection import train_test_split
from src.utils.dynamic_date_cutoff import compute_dynamic_cutoff
from src.ingest.logon_processed import merge_logon_with_uploads, process_logon_chunks_leak_free

def load_csv_chunked(path, parse_dates=("date",), chunksize=2000_000):
    logger.info(f"Loading {path} in chunks of {chunksize} rows...")
    return pd.read_csv(path, parse_dates=list(parse_dates), low_memory=False, chunksize=chunksize)

def process_http_chunks(http_path):
    # Buffer for rolling window
    buffer = pd.DataFrame()
    results = []
    chunk_idx = 0
    for chunk in load_csv_chunked(http_path):
        chunk_idx += 1
        logger.info(f"Processing HTTP chunk {chunk_idx} with {len(chunk)} rows (buffer: {len(buffer)})")
        print(f"Processing HTTP chunk {chunk_idx} with {len(chunk)} rows (buffer: {len(buffer)})")
        chunk = pd.concat([buffer, chunk], ignore_index=True).reset_index(drop=True)
        chunk["is_post"] = chunk["activity"].eq("WWW Upload")
        chunk = chunk.sort_values(["user", "date"]).reset_index(drop=True)
        # 5-second rolling window per user
        chunk["post_burst"] = (
            chunk.groupby("user")
            .rolling("5s", on="date")["is_post"]
            .sum()
            .reset_index(level=[0,1], drop=True)
        )
        chunk["destination_domain"] = chunk["url"].astype(str).map(lambda u: re.split(r"/|:", u)[0])
        chunk["destination_entropy"] = chunk["destination_domain"].map(shannon_entropy)
        chunk["first_time_destination"] = chunk.groupby(["user", "destination_domain"]).cumcount().eq(0)
        http_uploads = chunk.loc[chunk["is_post"], [
            "date", "user", "destination_domain", "post_burst", "destination_entropy", "first_time_destination"
        ]].assign(channel="HTTP", megabytes_sent=np.nan)
        logger.info(f"HTTP chunk {chunk_idx}: {len(http_uploads)} upload rows extracted")
        print(f"HTTP chunk {chunk_idx}: {len(http_uploads)} upload rows extracted")
        results.append(http_uploads)
        # Keep last 100 rows per user for next chunk's rolling window
        buffer = chunk.groupby("user").tail(100)
    logger.info(f"Finished processing HTTP. Total upload rows: {sum(len(r) for r in results)}")
    print(f"Finished processing HTTP. Total upload rows: {sum(len(r) for r in results)}")
    return pd.concat(results, ignore_index=True)


def process_email_chunks(email_path, chunksize=2000_000):
    buffer = pd.DataFrame()
    results = []
    chunk_idx = 0
    for chunk in load_csv_chunked(email_path, chunksize=chunksize):
        chunk_idx += 1
        logger.info(f"Processing EMAIL chunk {chunk_idx} with {len(chunk)} rows (buffer: {len(buffer)})")
        print(f"Processing EMAIL chunk {chunk_idx} with {len(chunk)} rows (buffer: {len(buffer)})")
        chunk = pd.concat([buffer, chunk], ignore_index=True)
        external_mask = chunk["to"].fillna("").str.contains("@", regex=False)
        chunk = chunk[external_mask]
        chunk["destination_domain"] = chunk["to"].str.split("@").str[-1]
        chunk["destination_entropy"] = chunk["destination_domain"].map(shannon_entropy)
        chunk["first_time_destination"] = chunk.groupby(["user", "to"]).cumcount().eq(0)
        
        chunk["from"] = chunk["from"].fillna("")
        chunk["is_from_user"] = chunk["from"] == chunk["user"]
        chunk["from_domain"] = chunk["from"].str.split("@").str[-1]
        # chunk["from_external"] = ~chunk["from"].str.endswith("yourcompany.com")
        chunk["attachments"] = chunk["attachments"].fillna("")
        chunk["has_attachments"] = chunk["attachments"] != ""
        chunk["attachment_count"] = chunk["attachments"].apply(lambda x: len(str(x).split(";")) if x else 0)

        chunk["bcc"] = chunk["bcc"].fillna("")
        chunk["bcc_count"] = chunk["bcc"].apply(lambda x: len(str(x).split(";")) if x else 0)

        chunk["cc"] = chunk["cc"].fillna("")
        chunk["cc_count"] = chunk["cc"].apply(lambda x: len(str(x).split(";")) if x else 0)

        chunk["size"] = chunk["size"].fillna(0).astype(float)
        chunk["post_burst"] = 0
        chunk["channel"] = "EMAIL"
        chunk["megabytes_sent"] = chunk["size"] / 1e6
        mail_uploads = chunk[[
            "date", "user", "destination_domain", "post_burst", "destination_entropy",
            "first_time_destination", "megabytes_sent", "channel",
            "is_from_user", "from_domain", "has_attachments", "attachment_count",
            "bcc_count", "cc_count", "size"
        ]]
        logger.info(f"EMAIL chunk {chunk_idx}: {len(mail_uploads)} upload rows extracted")
        print(f"EMAIL chunk {chunk_idx}: {len(mail_uploads)} upload rows extracted")
        results.append(mail_uploads)
        # Keep last 100 rows per user for next chunk's cumcount
        buffer = chunk.groupby("user").tail(100)
    logger.info(f"Finished processing EMAIL. Total upload rows: {sum(len(r) for r in results)}")
    print(f"Finished processing EMAIL. Total upload rows: {sum(len(r) for r in results)}")
    return pd.concat(results, ignore_index=True)


def process_http_chunks_leak_free(http_path, temporal_cutoff=None):
    """
    Process HTTP chunks with proper temporal isolation.
    
    Args:
        http_path: Path to HTTP CSV
        temporal_cutoff: Timestamp for train/test split (if None, process all data together)
    """
    if temporal_cutoff is None:
        # Original processing for data without splits
        return process_http_chunks(http_path)
    
    print(f" Processing HTTP with temporal isolation at {temporal_cutoff}")
    
    #  STEP 1: Load and split RAW data first
    raw_chunks = []
    for chunk in load_csv_chunked(http_path):
        chunk["is_post"] = chunk["activity"].eq("WWW Upload")
        chunk["destination_domain"] = chunk["url"].astype(str).map(lambda u: re.split(r"/|:", u)[0])
        raw_chunks.append(chunk)
    
    # Combine all raw data
    raw_http = pd.concat(raw_chunks, ignore_index=True)
    print(f" Raw HTTP data: {len(raw_http)} events")
    
    #  STEP 2: TEMPORAL SPLIT on raw data
    cutoff_ts = pd.Timestamp(temporal_cutoff)
    train_raw = raw_http[raw_http.date < cutoff_ts].copy().sort_values(["user", "date"])
    test_raw = raw_http[raw_http.date >= cutoff_ts].copy().sort_values(["user", "date"])
    
    print(f" Temporal split: Train={len(train_raw)}, Test={len(test_raw)}")
    
    #  STEP 3: Process train and test SEPARATELY
    train_processed = process_http_features_isolated(train_raw, is_training=True)
    test_processed = process_http_features_isolated(test_raw, is_training=False, train_data=train_raw)
    
    #  STEP 4: Combine processed results
    return pd.concat([train_processed, test_processed], ignore_index=True)

def process_http_features_isolated(data, is_training=True, train_data=None):
    """Process HTTP features in temporal isolation."""
    
    if len(data) == 0:
        return data
    
    print(f"Processing {'training' if is_training else 'test'} HTTP features...")
    
    #  SAFE: Rolling window only uses current dataset
    data = data.sort_values(["user", "date"]).reset_index(drop=True)
    
    # Rolling window calculation (leak-free)
    post_burst = (
        data.groupby("user")
        .rolling("5s", on="date")["is_post"]
        .sum()
        .reset_index(level=0, drop=True)
    )
    # Ensure index is unique and monotonic before assignment
    post_burst = post_burst.reset_index(drop=True)
    data = data.reset_index(drop=True)
    data["post_burst"] = post_burst
    
    # Destination entropy (safe - deterministic)
    data["destination_entropy"] = data["destination_domain"].map(shannon_entropy)
    
    #  CRITICAL: first_time_destination with temporal awareness
    if is_training:
        # Training: Use only training data
        data["first_time_destination"] = (
            data.groupby(["user", "destination_domain"]).cumcount().eq(0)
        )
        
        # Save training user-destination history for test phase
        train_user_destinations = set(
            zip(data["user"], data["destination_domain"])
        )
        joblib.dump(train_user_destinations, "train_user_destinations.pkl")
        print(f" Saved {len(train_user_destinations)} user-destination pairs from training")
        
    else:
        # Test: Use training history + test chronological order
        try:
            train_user_destinations = joblib.load("train_user_destinations.pkl")
        except FileNotFoundError:
            if train_data is not None:
                train_user_destinations = set(
                    zip(train_data["user"], train_data["destination_domain"])
                )
            else:
                train_user_destinations = set()
        
        # Mark as first-time if NOT seen in training AND first in test chronologically
        def is_first_time_leak_free(row, seen_in_training, test_seen):
            user_dest = (row["user"], row["destination_domain"])
            
            # If seen in training, never first time
            if user_dest in seen_in_training:
                return False
            
            # If not seen in training, check if first in test data chronologically
            if user_dest not in test_seen:
                test_seen.add(user_dest)
                return True
            
            return False
        
        test_seen_destinations = set()
        data["first_time_destination"] = data.apply(
            lambda row: is_first_time_leak_free(row, train_user_destinations, test_seen_destinations),
            axis=1
        )
    
    # Add metadata
    data["channel"] = "HTTP"
    data["megabytes_sent"] = np.nan  # Will be populated if size data available
    
    # Select final columns
    http_uploads = data.loc[data["is_post"], [
        "date", "user", "destination_domain", "post_burst", 
        "destination_entropy", "first_time_destination", "channel", "megabytes_sent"
    ]]
    
    print(f" {'Training' if is_training else 'Test'} HTTP uploads: {len(http_uploads)} events")
    return http_uploads

def process_email_chunks_leak_free(email_path, temporal_cutoff=None):
    """Process email chunks with proper temporal isolation."""
    
    if temporal_cutoff is None:
        return process_email_chunks(email_path)
    
    print(f" Processing EMAIL with temporal isolation at {temporal_cutoff}")
    
    # Load all raw email data
    raw_chunks = []
    for chunk in load_csv_chunked(email_path):
        external_mask = chunk["to"].fillna("").str.contains("@", regex=False)
        chunk = chunk[external_mask]
        chunk["destination_domain"] = chunk["to"].str.split("@").str[-1]
        raw_chunks.append(chunk)
    
    raw_email = pd.concat(raw_chunks, ignore_index=True)
    print(f"📊 Raw EMAIL data: {len(raw_email)} events")
    
    # Temporal split
    cutoff_ts = pd.Timestamp(temporal_cutoff)
    train_raw = raw_email[raw_email.date < cutoff_ts].copy()
    test_raw = raw_email[raw_email.date >= cutoff_ts].copy()
    
    # Process separately
    train_processed = process_email_features_isolated(train_raw, is_training=True)
    test_processed = process_email_features_isolated(test_raw, is_training=False, train_data=train_raw)
    
    return pd.concat([train_processed, test_processed], ignore_index=True)

def process_email_features_isolated(data, is_training=True, train_data=None):
    """Process email features in temporal isolation."""
    
    if len(data) == 0:
        return data
    
    # Basic email features (safe)
    data["destination_entropy"] = data["destination_domain"].map(shannon_entropy)
    data["from"] = data["from"].fillna("")
    data["is_from_user"] = data["from"] == data["user"]
    data["from_domain"] = data["from"].str.split("@").str[-1]
    data["attachments"] = data["attachments"].fillna("")
    data["has_attachments"] = data["attachments"] != ""
    data["attachment_count"] = data["attachments"].apply(lambda x: len(str(x).split(";")) if x else 0)
    data["bcc_count"] = data["bcc"].fillna("").apply(lambda x: len(str(x).split(";")) if x else 0)
    data["cc_count"] = data["cc"].fillna("").apply(lambda x: len(str(x).split(";")) if x else 0)
    data["size"] = data["size"].fillna(0).astype(float)
    data["post_burst"] = 0
    data["channel"] = "EMAIL"
    data["megabytes_sent"] = data["size"] / 1e6
    
    #  LEAK-FREE first_time_destination
    if is_training:
        data = data.sort_values(["user", "date"]).reset_index(drop=True)
        data["first_time_destination"] = data.groupby(["user", "to"]).cumcount().eq(0)
        
        # Save training email destinations
        train_user_emails = set(zip(data["user"], data["to"]))
        joblib.dump(train_user_emails, "train_user_emails.pkl")
        
    else:
        # Test phase - use training history
        try:
            train_user_emails = joblib.load("train_user_emails.pkl")
        except FileNotFoundError:
            train_user_emails = set(zip(train_data["user"], train_data["to"])) if train_data is not None else set()
        
        data = data.sort_values(["user", "date"]).reset_index(drop=True)
        test_seen_emails = set()
        
        def email_first_time_leak_free(row):
            user_email = (row["user"], row["to"])
            if user_email in train_user_emails:
                return False
            if user_email not in test_seen_emails:
                test_seen_emails.add(user_email)
                return True
            return False
        
        data["first_time_destination"] = data.apply(email_first_time_leak_free, axis=1)
    
    return data[[
        "date", "user", "destination_domain", "post_burst", "destination_entropy",
        "first_time_destination", "megabytes_sent", "channel",
        "is_from_user", "from_domain", "has_attachments", "attachment_count",
        "bcc_count", "cc_count", "size"
    ]]


def assign_risk_labels_complete(uploads: pd.DataFrame, is_training=True, train_thresholds=None) -> pd.DataFrame:
    """Complete risk labeling implementation."""
    logger.info("Assigning risk labels...")
    
    # Load business rules
    business_rules = BusinessRulesConfig()

    # Apply labeling with training-derived thresholds
    uploads["label"] = uploads.apply(
        lambda row: assign_risk_label_robust(row.to_dict(), business_rules), axis=1)
    return uploads



def populate_content_features(uploads: pd.DataFrame) -> pd.DataFrame:
    """Populate content analysis features based on available data."""
    logger.info("Populating content analysis features...")
    
    # Extract file extensions from destinations or email attachments
    uploads["file_extension"] = "unknown"
    
    # For email uploads with attachments
    email_mask = uploads["channel"] == "EMAIL"
    if "attachments" in uploads.columns:
        # Extract extensions from attachment names
        def extract_extensions(attachments):
            if pd.isna(attachments) or attachments == "":
                return "none"
            # Simple extension extraction
            extensions = []
            for filename in str(attachments).split(";"):
                if "." in filename:
                    ext = filename.split(".")[-1].lower()
                    extensions.append(ext)
            return extensions[0] if extensions else "unknown"
        
        uploads.loc[email_mask, "file_extension"] = (
            uploads.loc[email_mask, "attachments"].apply(extract_extensions)
        )
    
    # Define risk categories
    executable_extensions = {"exe", "bat", "cmd", "scr", "msi", "dll", "com"}
    compressed_extensions = {"zip", "rar", "7z", "tar", "gz", "bz2"}
    document_extensions = {"doc", "docx", "pdf", "xls", "xlsx", "ppt", "pptx"}
    media_extensions = {"jpg", "jpeg", "png", "gif", "mp4", "avi", "mp3"}
    
    uploads["is_executable"] = uploads["file_extension"].isin(executable_extensions)
    uploads["is_compressed"] = uploads["file_extension"].isin(compressed_extensions)
    uploads["is_document"] = uploads["file_extension"].isin(document_extensions)
    uploads["is_media"] = uploads["file_extension"].isin(media_extensions)
    
    # Filename entropy (randomness indicator)
    

    # Enhanced filename entropy calculation from multiple sources
    def calculate_filename_entropy_robust(row):
        """Calculate filename entropy from available sources."""
        
        # Priority 1: Email attachments (if available)
        if "attachments" in uploads.columns and pd.notna(row.get("attachments")) and row.get("attachments") != "":
            filename = str(row["attachments"]).split(";")[0]
            return shannon_entropy(filename) if filename else 0.0
        
        # Priority 2: Extract from destination domain (for HTTP uploads)
        elif row["channel"] == "HTTP":
            domain = str(row["destination_domain"])
            # Use domain name as proxy for filename randomness
            return shannon_entropy(domain) if domain and domain != "unknown" else 0.0
        
        # Priority 3: For USB transfers, we don't have filename info
        elif row["channel"] == "USB":
            return 0.0  # No filename entropy available for USB
        
        # Priority 4: Default case
        else:
            return 0.0

    # Apply the robust calculation
    uploads["filename_entropy"] = uploads.apply(calculate_filename_entropy_robust, axis=1)
    
    # Estimated file count (for archives)
    uploads["estimated_file_count"] = uploads["attachment_count"].fillna(1).clip(lower=1)
    
    logger.info("Content analysis features populated")
    return uploads


def create_simple_temporal_split(uploads: pd.DataFrame, cutoff: str):
    """Simple temporal split without data modification."""
    
    cutoff_ts = pd.Timestamp(cutoff)
    
    #  NO DATA MODIFICATION - just create indices
    train_mask = uploads.date < cutoff_ts
    test_mask = uploads.date >= cutoff_ts
    
    train_ids = uploads.loc[train_mask].index.values
    test_ids = uploads.loc[test_mask].index.values
    
    #  VERIFY CLASS REPRESENTATION (don't force it)
    train_labels = uploads.loc[train_ids, 'label'].value_counts()#.sort_index()
    test_labels = uploads.loc[test_ids, 'label'].value_counts()#.sort_index()
    
    print(f"Temporal split class distribution:")
    print(f"Train class: {train_labels.to_dict()}")
    print(f"Test class: {test_labels.to_dict()}")
    
    return train_ids, test_ids


def add_features_leak_free_FIXED(data, is_training=True, train_data=None, cutoff=None, business_rules=None):
    """
    Comprehensive leak-free feature engineering that creates unique behavioral signatures.
    
    Key Principles:
    1. Every event gets a unique signature (no identical feature vectors)
    2. Population-based features (no user identity leakage) 
    3. Temporal uniqueness preserved
    4. Behavioral patterns learned without memorization
    """
    
    if business_rules is None:
        business_rules = BusinessRulesConfig()
    
    print(f"Adding features to {'training' if is_training else 'test'} data...")
    
    # ========================================================================
    # TEMPORAL UNIQUENESS FEATURES (Prevent identical vectors)
    # ========================================================================
    data = data.sort_values('date').reset_index(drop=True)
    # if 'user' in data.columns:
    #     user_count = data['user'].nunique()
    #     print(f" Removing user column (had {user_count} unique users)")
    #     data = data.drop(columns=['user'])
    
    # Basic temporal features
    data["hour"] = data["date"].dt.hour
    data["minute"] = data["date"].dt.minute
    data["day_of_week"] = data["date"].dt.dayofweek
    data["is_weekend"] = data["date"].dt.dayofweek >= 5
    data["after_hours"] = (data["hour"] < 7) | (data["hour"] >= 19)
    
    # Extended temporal uniqueness
    data["timestamp_unix"] = data["date"].astype(np.int64) // 10**9
    data["minute_of_day"] = data["hour"] * 60 + data["minute"]  # 0-1439
    data["day_of_year"] = data["date"].dt.dayofyear  # 1-365
    data["week_of_year"] = data["date"].dt.isocalendar().week  # 1-52
    data["month"] = data["date"].dt.month  # 1-12
    data["quarter"] = data["date"].dt.quarter  # 1-4
    
    # Sequential features (guarantee uniqueness)
    data = data.sort_values('date').reset_index(drop=True)
    data['global_sequence_id'] = range(len(data))  # Globally unique ID
    data['event_id'] = data.index # Unique event identifier
    data['daily_sequence_number'] = data.groupby(data['date'].dt.date).cumcount() + 1
    
    # Time differences (create temporal context without user identity)
    data['seconds_since_previous'] = data['date'].diff().dt.total_seconds().fillna(0)
    data['minutes_since_previous'] = data['seconds_since_previous'] / 60
    
    # ========================================================================
    # POPULATION-BASED BEHAVIORAL FEATURES (No user identity)
    # ========================================================================
    
    if is_training:
        print("📊 Computing population behavioral statistics from training data...")
        
        # Population statistics for anomaly detection
        population_stats = {
            'hour_median': data['hour'].median(),
            'hour_std': data['hour'].std(),
            'hour_q05': data['hour'].quantile(0.05),
            'hour_q95': data['hour'].quantile(0.95),
            
            'size_median': data['megabytes_sent'].fillna(0).median(),
            'size_std': data['megabytes_sent'].fillna(0).std(),
            'size_q95': data['megabytes_sent'].fillna(0).quantile(0.95),
            
            'channel_frequencies': data['channel'].value_counts(normalize=True).to_dict(),
            'common_destinations': data['destination_domain'].value_counts().head(10).index.tolist(),
            
            'burst_median': data['post_burst'].fillna(0).median(),
            'burst_q95': data['post_burst'].fillna(0).quantile(0.95),
            
            'entropy_median': data['destination_entropy'].fillna(0).median(),
            'entropy_q95': data['destination_entropy'].fillna(0).quantile(0.95),
        }
        
        # Save population statistics for test phase
        joblib.dump(population_stats, "training_population_stats.pkl")
        print(f" Saved population statistics for {len(data)} training events")
        
    else:
        print("📊 Loading training population statistics for test data...")
        try:
            population_stats = joblib.load("training_population_stats.pkl")
            print(" Loaded training population statistics")
        except FileNotFoundError:
            if train_data is not None:
                print(" Computing fallback population statistics from provided train_data...")
                population_stats = {
                    'hour_median': train_data['hour'].median(),
                    'hour_std': train_data['hour'].std(),
                    'hour_q05': train_data['hour'].quantile(0.05),
                    'hour_q95': train_data['hour'].quantile(0.95),
                    'size_median': train_data['megabytes_sent'].fillna(0).median(),
                    'size_std': train_data['megabytes_sent'].fillna(0).std(),
                    'size_q95': train_data['megabytes_sent'].fillna(0).quantile(0.95),
                    'channel_frequencies': train_data['channel'].value_counts(normalize=True).to_dict(),
                    'common_destinations': train_data['destination_domain'].value_counts().head(10).index.tolist(),
                    'burst_median': train_data['post_burst'].fillna(0).median(),
                    'burst_q95': train_data['post_burst'].fillna(0).quantile(0.95),
                    'entropy_median': train_data['destination_entropy'].fillna(0).median(),
                    'entropy_q95': train_data['destination_entropy'].fillna(0).quantile(0.95),
                }
            else:
                print(" No training data available - using default population statistics")
                population_stats = {
                    'hour_median': 12, 'hour_std': 3, 'hour_q05': 7, 'hour_q95': 19,
                    'size_median': 1, 'size_std': 5, 'size_q95': 10,
                    'channel_frequencies': {'EMAIL': 0.6, 'HTTP': 0.3, 'USB': 0.1},
                    'common_destinations': ['dtaa.com', 'company.com'],
                    'burst_median': 1, 'burst_q95': 5,
                    'entropy_median': 2, 'entropy_q95': 4,
                }
    
    # Apply population-based anomaly features
    data['hour_deviation_score'] = np.abs(data['hour'] - population_stats['hour_median']) / max(population_stats['hour_std'], 1)
    data['is_outlier_hour'] = (data['hour'] < population_stats['hour_q05']) | (data['hour'] > population_stats['hour_q95'])
    
    data['size_deviation_score'] = np.abs(data['megabytes_sent'].fillna(0) - population_stats['size_median']) / max(population_stats['size_std'], 1)
    data['is_outlier_size'] = data['megabytes_sent'].fillna(0) > population_stats['size_q95']
    
    data['channel_rarity_score'] = data['channel'].map(
        lambda x: 1.0 - population_stats['channel_frequencies'].get(x, 0)
    )
    
    data['destination_novelty_score'] = (~data['destination_domain'].isin(
        population_stats['common_destinations']
    )).astype(float)
    
    # ========================================================================
    # BEHAVIORAL CONTEXT FEATURES (Without user identity)
    # ========================================================================
    
    # Channel behavior analysis
    data['is_http'] = (data['channel'] == 'HTTP').astype(int)
    data['is_usb'] = (data['channel'] == 'USB').astype(int)
    data['is_email'] = (data['channel'] == 'EMAIL').astype(int)
    
    # Destination risk analysis
    competitor_domains = ['lockheedmartin.com', 'northropgrumman.com', 'boeing.com']
    data['competitor_communication'] = data['destination_domain'].isin(competitor_domains).astype(int)
    data['external_domain'] = (~data['destination_domain'].isin(['dtaa.com', 'company.com'])).astype(int)
    
    # File size categories (population-relative)
    data['is_large_upload'] = (data['megabytes_sent'].fillna(0) > population_stats['size_q95']).astype(int)
    data['is_medium_upload'] = (
        (data['megabytes_sent'].fillna(0) > population_stats['size_median']) & 
        (data['megabytes_sent'].fillna(0) <= population_stats['size_q95'])
    ).astype(int)
    
    # Temporal risk patterns
    data['deep_night'] = ((data['hour'] >= 22) | (data['hour'] <= 5)).astype(int)
    data['early_morning'] = (data['hour'].between(5, 7)).astype(int)
    data['late_evening'] = (data['hour'].between(19, 22)).astype(int)
    
    # ========================================================================
    # UNIQUE NOISE INJECTION (Prevent identical vectors)
    # ========================================================================
    
    np.random.seed(42)  # Reproducible noise
    n_samples = len(data)
    
    # Add small unique noise to continuous features
    data['hour_with_noise'] = data['hour'] + np.random.normal(0, 0.01, n_samples)
    data['size_with_noise'] = data['megabytes_sent'].fillna(0) + np.random.normal(0, 0.001, n_samples)
    data['entropy_with_noise'] = data['destination_entropy'].fillna(0) + np.random.normal(0, 0.001, n_samples)
    data['burst_with_noise'] = data['post_burst'].fillna(0) + np.random.normal(0, 0.001, n_samples)
    
    return data
    

def validate_temporal_isolation(train_df, test_df):
    train_max = train_df['date'].max()
    test_min = test_df['date'].min()
    assert train_max < test_min, f"Temporal overlap: {train_max} >= {test_min}"
    print(" Temporal isolation verified")


def build_upload_table_ENHANCED(raw_dir: pathlib.Path, cutoff: str, decoy_file_path: str = None) -> pd.DataFrame:
    """
    Enhanced upload table builder with ground truth and anomaly detection.
    """
    logger.info(f"Building ENHANCED upload table from {raw_dir}")
    print("🔧 BUILDING ENHANCED UPLOAD TABLE WITH GROUND TRUTH")
    print("=" * 60)

    labeler = GroundTruthLabeler(decoy_file_path)
    # Load raw data (same as before)
    http_uploads = process_http_chunks_leak_free(raw_dir / "http.csv", cutoff)
    mail_uploads = process_email_chunks_leak_free(raw_dir / "email.csv", cutoff)
    logon_summary = process_logon_chunks_leak_free(raw_dir / "logon.csv", temporal_cutoff=cutoff)
    
    file_df = load_csv(raw_dir / "file.csv")
    device_df = load_csv(raw_dir / "device.csv")
    
    # USB transfers
    usb_pairs = (
        file_df[file_df["to_removable_media"]]
        .merge(device_df[["user", "pc", "date"]], on=["user", "pc", "date"], how="inner")
    )
    usb_uploads = usb_pairs[["date", "user", "pc"]].assign(
        destination_domain="USB",
        post_burst=0,
        destination_entropy=0,
        first_time_destination=False,
        channel="USB",
        megabytes_sent=np.nan
    )
    
    # Combine raw data
    uploads_raw = pd.concat([http_uploads, usb_uploads, mail_uploads], ignore_index=True)
    uploads_raw = uploads_raw.sort_values('date').reset_index(drop=True)
    
    # Temporal split
    cutoff_ts = pd.Timestamp(cutoff)
    train_raw = uploads_raw[uploads_raw.date < cutoff_ts].copy()
    test_raw = uploads_raw[uploads_raw.date >= cutoff_ts].copy()
    
    print(f" Temporal split: Train={len(train_raw)}, Test={len(test_raw)}")
    
    # Enhanced feature engineering
    train_processed = add_features_leak_free_FIXED(train_raw, is_training=True)
    train_processed = add_anomaly_detection_features(train_processed, is_training=True, labeler=labeler)
    train_processed = add_ground_truth_features(train_processed, decoy_file_path)
    train_processed = add_pattern_discovery_features(train_processed, is_training=True)
    
    test_processed = add_features_leak_free_FIXED(test_raw, is_training=False, train_data=train_raw)
    test_processed = add_anomaly_detection_features(test_processed, is_training=False, labeler=labeler)
    test_processed = add_ground_truth_features(test_processed, decoy_file_path)
    test_processed = add_pattern_discovery_features(test_processed, is_training=False)
    
    # Enhanced labeling with ground truth
    enhanced_rules = EnhancedBusinessRulesConfig(decoy_file_path=decoy_file_path)
    
    # Create ground truth labels
    labeler = GroundTruthLabeler(decoy_file_path)
    train_labels, train_scores = labeler.create_composite_labels(train_processed, is_training=True)
    test_labels, test_scores = labeler.create_composite_labels(test_processed, is_training=False)
    
    train_processed['label'] = train_labels
    train_processed['risk_score'] = train_scores
    test_processed['label'] = test_labels  
    test_processed['risk_score'] = test_scores

    # Now drop 'user' as it has high correlation with 'label' and can leak information
    if 'user' in train_processed.columns:
        train_processed = train_processed.drop(columns=['user'])
    if 'user' in test_processed.columns:
        test_processed = test_processed.drop(columns=['user'])
    
    # Combine and finalize
    uploads_final = pd.concat([train_processed, test_processed], ignore_index=True)
    uploads_final = merge_logon_with_uploads(uploads_final, logon_summary)


    # Generate enhanced reports
    generate_enhanced_reports(uploads_final, pathlib.Path("reports"))
    
    return uploads_final


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build uploads table from raw CERT CSV logs (chunked pandas).")
    parser.add_argument("--input_dir", required=True, type=pathlib.Path, 
                        help="Path to r6.2 directory containing raw CERT CSV logs.")
    parser.add_argument("--output", required=True, type=pathlib.Path,
                        help="Output Parquet file.")
    parser.add_argument("--cutoff", type=str,
                        help="YYY-MM-DD date to create train/test split")
    parser.add_argument("--split_pct", type=float, default=None,
                        help="Fraction of data to use for training (e.g., 0.35 for 35% train, 65% test")
    
    parser.add_argument("--auto_cutoff", action="store_true",
                        help="Enable dynamic cutoff selection.")
    parser.add_argument("--target_train_frac", type=float, default=0.25,
                        help="Target training fraction for --auto_cutoff.")
    parser.add_argument("--min_modality_events", type=int, default=50,
                        help="Min per-modality events on each side.")
    parser.add_argument("--no_hour_refine", action="store_true",
                        help="Disable hour refinement for auto cutoff.")
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Create data_interim directory if it doesn't exist
    data_interim_dir = pathlib.Path("data_interim")
    data_interim_dir.mkdir(exist_ok=True, parents=True)
    print(f"✅ Ensured data_interim directory exists: {data_interim_dir.resolve()}")
    

    # Set output to SSD volume if not already specified
    ssd_output = pathlib.Path("/Volumes/master_thesis_work/master_thesis_Machine_learning_for_Enhanced_Risk_classification_of_data_uploadsSM/data_interim/uploads.parquet")
    if str(args.output) == "uploads.parquet":
        logger.info(f"--output not specified or set to default, using SSD: {ssd_output}")
        print(f"--output not specified or set to default, using SSD: {ssd_output}")
        args.output = ssd_output

    
    if args.auto_cutoff:
        auto_ts = compute_dynamic_cutoff(
            args.input_dir,
            target_frac=args.target_train_frac,
            min_per_modality=args.min_modality_events,
            refine_hours=not args.no_hour_refine
        )
        args.cutoff = auto_ts.isoformat(timespec='seconds')
        logger.info(f"Using auto cutoff: {args.cutoff}")

    if args.cutoff:
        # WITH STRATIFIED SPLIT:
        #uploads = build_upload_table_FIXED(args.input_dir, cutoff=args.cutoff)
        uploads = build_upload_table_ENHANCED(args.input_dir, cutoff=args.cutoff)
        # train_ids, test_ids = create_stratified_split(uploads, args.cutoff, test_size=0.2)
        train_ids, test_ids = create_simple_temporal_split(uploads, args.cutoff)

        # Create indices for already-split data
        cutoff_ts = pd.Timestamp(args.cutoff)
       
        train_ids = uploads.loc[uploads.date < cutoff_ts].index.values
        test_ids = uploads.loc[uploads.date >= cutoff_ts].index.values

        logger.info(f"Saving uploads to {args.output}")
        print(f"Saving uploads to {args.output}")
        uploads.to_parquet(args.output, compression="zstd")
        print(f"[✓] saved {len(uploads):,} upload rows to {args.output}")
        logger.info(f"[✓] saved {len(uploads):,} upload rows to {args.output}")

        np.save(args.output.parent / "train_ids.npy", train_ids)
        np.save(args.output.parent / "test_ids.npy", test_ids)
        cfg = {"cutoff": args.cutoff, "seed": 42, "method": "temporal"}
        (args.output.parent / "split_config.json").write_text(json.dumps(cfg))
        print(f"[✓] Stratified Split: train={len(train_ids):,} test={len(test_ids):,}")
        logger.info(f"[✓] Temporal split: Split train={len(train_ids):,} test={len(test_ids):,}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("build_uploads.log")
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting pandas_chunked_build_uploads.py")
    sys.exit(main())