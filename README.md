# Cross Architecture Explanation Transfer

![Cross Architecture Explanation Transfer](https://github.com/NnamdiNgwu/explainable/blob/main/image/TXM.jpg)
![Cross Architecture Explanation Transfer](https://github.com/NnamdiNgwu/explainable/blob/main/image/batch.png)

## Overview

Breaking deep neural network with TXM outputs.

### Cascade SIEM: RF→Transformer Risk Classification with Cross-Model Explanations (TXM)

Production-minded research code for a two-stage SIEM pipeline:

- **Random Forest (RF)**: Fast gating on tabular events
- **Transformer**: Sequence-aware escalation, optionally distilled from RF
- **TXM (Transformer eXplanation Mapper)**: Consistent, cross-model explanations
- **Tunable cascade thresholds**: τ (RF gate) and τ₂ (Transformer gate)

## Key Highlights

### Cascade Decision Rule

The cascade classification works as follows:
- If `max P_RF(x) < τ`: Classify as **Benign**
- If `max P_RF(x) ≥ τ AND P_Trans(X) ≥ τ₂`: Classify as **Malicious**
- Otherwise: Classify as **Benign**

### Features

- **TXM**: Deterministic RF→Transformer attribution mapping with probability-ratio scaling; optional Integrated Gradients overlay for audits
- **RF→Transformer Knowledge Distillation (KD)**: Lift sequence model performance without architecture bloat; escalation-aware loss
- **Fidelity metrics**: For TXM (served & offline): sign_fidelity, rank_fidelity@k, prob_monotonicity
- **SOC-ready**: Plain-language rationales, latency/throughput instrumentation, explicit gates for tuning alert volume

## Setup and Installation

### 1. Create and activate a virtual environment

```bash
cd /explainable_architectural_transfer/src
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3. Install dependencies

```bash
cd /explainable_architectural_transfer
pip install -r requirements.txt
```

## Training the Cascade Model

Run the cascade training pipeline to generate model configurations:

```bash
python -m src.models.cascade --data_dir data_processed
# This writes cascade_config.json with tau, tau2, model paths, and dimensions
```

## Running the API Server

### Start the Flask development server

```bash
python -m flask --app src.serving.app run
```

### Alternative: Run directly for testing

```bash
python3 -m src.serving.app
```

## API Usage

### Making Predictions

Send a POST request to the prediction endpoint with event data:

```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "post_burst": 5,                          # Number of events in burst
    "destination_entropy": 6.2,               # Entropy of destination addresses
    "hour": 23,                               # Hour of day (0-23)
    "timestamp_unix": 1716203600,             # Unix timestamp
    "seconds_since_previous": 5,              # Time since last event
    "size": 25.8,                             # Event size in MB
    
    "megabytes_sent": 25.8,                   # Total MB sent
    "uploads_last_24h": 12,                   # Upload count in last 24 hours
    "user_upload_count": 3,                   # User-specific upload count
    "user_mean_upload_size": 18.5,            # User average upload size
    "user_std_upload_size": 8.2,              # User upload size std deviation
    "user_unique_destinations": 8,            # Number of unique destinations
    "user_destination_count": 12,             # Total destination count
    "attachment_count": 3,                    # Number of attachments
    "bcc_count": 2,                           # BCC recipient count
    "cc_count": 0,                            # CC recipient count
    
    "temporal_anomaly_score": 0.9,            # Time-based anomaly score
    "volume_anomaly_score": 0.8,              # Volume-based anomaly score
    "network_anomaly_score": 0.7,             # Network-based anomaly score
    "composite_anomaly_score": 0.95,          # Overall anomaly score
    "destination_novelty_score": 0.85,        # Novelty of destination
    "decoy_risk_score": 0.6,                  # Decoy detection risk score
    "days_since_decoy": 2,                    # Days since last decoy interaction
    "size_hour_interaction": 0.9,             # Interaction: size × hour
    "entropy_burst_interaction": 0.8,         # Interaction: entropy × burst
    "upload_velocity": 0.95,                  # Upload speed metric
    "domain_switching_rate": 0.7,             # Rate of domain changes
    
    "first_time_destination": true,           # First time accessing destination
    "after_hours": true,                      # Event occurred after hours
    "is_large_upload": true,                  # Upload exceeds size threshold
    "rare_hour_flag": true,                   # Activity in unusual hour
    "has_attachments": true,                  # Event has attachments
    "is_from_user": true,                     # Originates from user account
    "is_outlier_hour": true,                  # Hour is statistical outlier
    "is_outlier_size": true,                  # Size is statistical outlier
    "is_usb": true,                           # USB device involved
    "is_weekend": true,                       # Event occurred on weekend
    
    "destination_domain": "suspicious.net",   # Target domain name
    "user": "threat_user",                    # User identifier
    "channel": "USB",                         # Communication channel
    "from_domain": "company.com"              # Origin domain
  }'
```

### Response Format

The API returns a JSON response with:
- **prediction**: Classification result (Benign/Malicious)
- **rf_probability**: Random Forest confidence score
- **transformer_probability**: Transformer confidence score (if escalated)
- **explanation**: TXM-generated explanation of the decision
- **feature_importance**: Top contributing features

## Cascade Tuning

To tune the cascade thresholds τ and τ₂:

1. Adjust the RF gate threshold (τ) to control initial filtering
2. Adjust the Transformer gate threshold (τ₂) for escalation sensitivity
3. Monitor the trade-off between false positives and detection rate

## Project Structure

```
explainable_ML/
├── src/
│   ├── features/          # Feature engineering modules
│   ├── ingest/            # Data ingestion pipeline
│   ├── models/            # Model training and cascade logic
│   ├── serving/           # Flask API server
│   ├── evaluation/        # Model evaluation metrics
│   └── utils/             # Utility functions
├── models/                # Saved model artifacts
├── test_events/           # Sample test events
└── requirements.txt       # Python dependencies
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here]
