
# Cross Architecture Explanation Transfer

![Cross Architecture Explanation Transfer](/explainable_architectural_transfer/image)


Breaking deep neural network with TXM outputs.


       
  
Cascade SIEM: RF→Transformer Risk Classification with Cross-Model Explanations (TXM)
Production-minded research code for a two-stage SIEM pipeline:
Random Forest (RF) for fast gating on tabular events
Transformer for sequence-aware escalation, optionally distilled from RF
TXM (Transformer eXplanation Mapper) for consistent, cross-model explanations
Tunable cascade thresholds 
τ
τ (RF gate) and 
τ
2
τ 
2
​	
  (Transformer gate)


Highlights
Cascade decision rule (as served):
Classify
(
x
,
X
)
=
{
Benign
,
max
⁡
P
R
F
(
x
)
<
τ
,
Malicious
,
max
⁡
P
R
F
(
x
)
≥
τ
∧
P
T
r
a
n
s
(
X
)
≥
τ
2
,
Benign
,
otherwise.
 
Classify(x,X)= 
⎩
⎨
⎧
​	
  
Benign,
Malicious,
Benign,
​	
  
maxP 
RF
​	
 (x)<τ,
maxP 
RF
​	
 (x)≥τ∧P 
Trans
​	
 (X)≥τ 
2
​	
 ,
otherwise.
​	
 
TXM: deterministic RF→Transformer attribution mapping with probability-ratio scaling; optional Integrated Gradients overlay for audits.
RF→Transformer Knowledge Distillation (KD): lift sequence model performance without architecture bloat; escalation-aware loss.
Fidelity metrics for TXM (served & offline): sign_fidelity, rank_fidelity@k, prob_monotonicity.
SOC-ready: plain-language rationales, latency/throughput instrumentation, explicit gates for tuning alert volume.

Cascade Tuning and Serving
Tune 
τ
τ and 
τ
2
τ 
2
​	
 
python -m src.models.cascade --data_dir data_processed
# writes cascade_config.json with tau, tau2, model paths, dims

Cascade Tuning and Serving
Tune 
τ
τ and 
τ
2
τ 
2
​	

cd /explainable_architectural_transfer/src
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

cd /explainable_architectural_transfer
pip install -r requirements.txt

python -m flask --app src.serving.app run

# To run a test
 curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"post_burst": 5,
  "destination_entropy": 6.2,
  "hour": 23,
  "timestamp_unix": 1716203600,
  "seconds_since_previous": 5,
  "size": 25.8,

  "megabytes_sent": 25.8,
  "uploads_last_24h": 12,
  "user_upload_count": 3,
  "user_mean_upload_size": 18.5,
  "user_std_upload_size": 8.2,
  "user_unique_destinations": 8,
  "user_destination_count": 12,
  "attachment_count": 3,
  "bcc_count": 2,
  "cc_count": 0,

  "temporal_anomaly_score": 0.9,
  "volume_anomaly_score": 0.8,
  "network_anomaly_score": 0.7,
  "composite_anomaly_score": 0.95,
  "destination_novelty_score": 0.85,
  "decoy_risk_score": 0.6,
  "days_since_decoy": 2,
  "size_hour_interaction": 0.9,
  "entropy_burst_interaction": 0.8,
  "upload_velocity": 0.95,
  "domain_switching_rate": 0.7,

  "first_time_destination": true,
  "after_hours": true,
  "is_large_upload": true,
  "rare_hour_flag": true,
  "has_attachments": true,
  "is_from_user": true,
  "is_outlier_hour": true,
  "is_outlier_size": true,
  "is_usb": true,
  "is_weekend": true,

  "destination_domain": "suspicious.net",
  "user": "threat_user",
  "channel": "USB",
  "from_domain": "company.com"
}'