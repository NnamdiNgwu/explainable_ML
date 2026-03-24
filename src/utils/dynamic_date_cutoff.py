import pandas as pd
import pathlib, json, logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class CutoffResult:
    cutoff: pd.Timestamp
    train_frac: float
    per_modality_counts: dict
    score: float

def _read_timestamp_series(path: pathlib.Path, ts_col: str, modality: str, parse_fmt=None, nrows=None):
    if not path.exists():
        return pd.DataFrame(columns=['timestamp','modality'])
    try:
        df = pd.read_csv(path, usecols=[ts_col], nrows=nrows)
    except ValueError:
        return pd.DataFrame(columns=['timestamp','modality'])
    if parse_fmt:
        df['timestamp'] = pd.to_datetime(df[ts_col], format=parse_fmt, errors='coerce')
    else:
        df['timestamp'] = pd.to_datetime(df[ts_col], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df['modality'] = modality
    return df[['timestamp','modality']]

def _gather_raw_timestamps(raw_dir: pathlib.Path) -> pd.DataFrame:
    specs = [
        ('http.csv',   'date'),
        ('email.csv',  'date'),
        ('logon.csv',  'date'),
        ('file.csv',   'date'),
        ('device.csv', 'date'),
    ]
    frames = [_read_timestamp_series(raw_dir / f, col, f.split('.')[0]) for f, col in specs]
    if not frames:
        return pd.DataFrame(columns=['timestamp','modality'])
    all_ts = pd.concat(frames, ignore_index=True)
    return all_ts.sort_values('timestamp')

def _evaluate_cutoffs(all_ts: pd.DataFrame,
                      target_frac: float = 0.65,
                      min_per_modality: int = 50,
                      refine_hours: bool = True) -> CutoffResult:
    if all_ts.empty:
        raise ValueError("No timestamps found for dynamic cutoff.")
    days = all_ts['timestamp'].dt.normalize().unique()
    if len(days) < 3:
        raise ValueError("Not enough distinct days for temporal split.")
    day_candidates = days[1:-1]

    results = []
    modalities = all_ts['modality'].unique()
    for d in day_candidates:
        mask = all_ts['timestamp'] < d
        tf = mask.mean()
        if tf == 0 or tf == 1:
            continue
        per_mod_train = all_ts[mask].groupby('modality').size().to_dict()
        per_mod_test  = all_ts[~mask].groupby('modality').size().to_dict()
        if any(per_mod_train.get(m,0) < min_per_modality or per_mod_test.get(m,0) < min_per_modality for m in modalities):
            continue
        size_dev = abs(tf - target_frac)
        results.append(CutoffResult(pd.Timestamp(d), tf, {'train': per_mod_train,'test': per_mod_test}, size_dev))

    if not results:
        # Relax requirement once
        if min_per_modality > 10:
            return _evaluate_cutoffs(all_ts, target_frac, max(10, min_per_modality//2), refine_hours)
        raise ValueError("No viable day cutoffs after relaxation.")

    best_day = min(results, key=lambda r: r.score)
    if not refine_hours:
        return best_day

    # Hour refinement
    day_rows = all_ts[all_ts['timestamp'].dt.normalize() == best_day.cutoff]
    hour_points = sorted(day_rows['timestamp'].dt.floor('H').unique())
    hour_results = []
    for h in hour_points:
        cutoff_ts = pd.Timestamp(h)
        mask = all_ts['timestamp'] < cutoff_ts
        if mask.sum()==0 or (~mask).sum()==0:
            continue
        per_mod_train = all_ts[mask].groupby('modality').size().to_dict()
        per_mod_test  = all_ts[~mask].groupby('modality').size().to_dict()
        if any(per_mod_train.get(m,0) < min_per_modality or per_mod_test.get(m,0) < min_per_modality for m in modalities):
            continue
        tf = mask.mean()
        size_dev = abs(tf - target_frac)
        hour_results.append(CutoffResult(cutoff_ts, tf, {'train': per_mod_train,'test': per_mod_test}, size_dev))
    if hour_results:
        refined = min(hour_results, key=lambda r: r.score)
        if refined.score < best_day.score:
            return refined
    return best_day

def compute_dynamic_cutoff(raw_dir: pathlib.Path,
                           target_frac: float = 0.65,
                           min_per_modality: int = 50,
                           refine_hours: bool = True) -> pd.Timestamp:
    all_ts = _gather_raw_timestamps(raw_dir)
    result = _evaluate_cutoffs(all_ts, target_frac, min_per_modality, refine_hours)
    meta = {
        "cutoff": str(result.cutoff),
        "train_frac": result.train_frac,
        "target_frac": target_frac,
        "per_modality_counts": result.per_modality_counts,
    }
    out_dir = pathlib.Path("data_interim")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "auto_cutoff_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"[auto_cutoff] cutoff={result.cutoff} train_frac={result.train_frac:.4f}")
    return result.cutoff