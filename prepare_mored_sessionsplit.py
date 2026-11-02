#!/usr/bin/env python3
"""Prepare leakage-aware MORED V3 using whole original recording sessions.

Chronological assignment (based on raw timestamp-gap session IDs):
  train: sessions 0-4
  val:   session 5
  test:  sessions 6-9

This is a separate experiment and never overwrites Corrected_V2 outputs.
No interpolation/fill is performed. Minute means require >=10 genuine valid
5-second observations; windows remain entirely inside a single continuous run.
"""
from __future__ import annotations
import argparse, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

LOOKBACK, FORECAST, STRIDE = 1440, 1440, 60
MIN_RAW = 10
EXPECTED_SESSIONS = {
    "train": [0, 1, 2, 3, 4],
    "val": [5],
    "test": [6, 7, 8, 9],
}
EXPECTED_WINDOWS = {"train": 245, "val": 119, "test": 359}
EXPECTED_PER_SESSION = {0: 0, 1: 197, 2: 0, 3: 46, 4: 2, 5: 119, 6: 140, 7: 157, 8: 27, 9: 35}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=root / "Data/MORED/raw/APT6.csv")
    p.add_argument("--output", type=Path, default=root / "Data/MORED/Corrected_V3_SessionSplit")
    p.add_argument("--allow-noncanonical-source", action="store_true")
    return p.parse_args()


def load_raw(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Original source not found: {path}")
    print(f"Reading raw MORED source: {path}")
    df = pd.read_csv(path, usecols=["timestamp", "real_power"], dtype={"timestamp": "string"}, low_memory=False)
    df["datetime"] = pd.to_datetime(df["timestamp"].str.strip(), format="%d/%m/%Y %H:%M:%S", errors="raise")
    df["real_power"] = pd.to_numeric(df["real_power"], errors="coerce")
    df = df[["datetime", "real_power"]].sort_values("datetime").reset_index(drop=True)
    if df.empty or df.datetime.isna().any() or df.datetime.duplicated().any():
        raise ValueError("Empty source, invalid timestamps, or duplicate timestamps detected")
    neg = int((df.real_power < 0).sum())
    df.loc[df.real_power < 0, "real_power"] = np.nan
    gap = df.datetime.diff() > pd.Timedelta(seconds=60)
    df["raw_segment_id"] = gap.fillna(False).cumsum().astype(np.int32)
    info = {
        "raw_rows": int(len(df)), "negative_raw_readings_invalidated": neg,
        "raw_segment_count": int(df.raw_segment_id.nunique()),
        "start_timestamp": str(df.datetime.min()), "end_timestamp": str(df.datetime.max()),
        "segment_rule": "new recording session when consecutive raw timestamps differ by more than 60 seconds",
    }
    return df, info


def aggregate_minutes(raw: pd.DataFrame):
    pieces = []
    bins_total = 0
    for sid, s in raw.groupby("raw_segment_id", sort=True):
        minute = s.set_index("datetime").real_power.resample("1min").agg(consumption="mean", raw_samples="count")
        minute["raw_segment_id"] = int(sid)
        minute.index.name = "datetime"
        bins_total += len(minute)
        pieces.append(minute.reset_index())
    grid = pd.concat(pieces, ignore_index=True).sort_values(["datetime", "raw_segment_id"]).reset_index(drop=True)
    grid["raw_samples"] = grid.raw_samples.astype(np.int16)
    retained = grid.loc[grid.raw_samples >= MIN_RAW].copy().sort_values("datetime").reset_index(drop=True)
    if retained.consumption.isna().any() or (retained.raw_samples > 12).any():
        raise RuntimeError("Retained raw minute aggregation contains invalid values or >12 observations/minute")
    retained.insert(0, "global_minute_row", np.arange(len(retained), dtype=np.int64))
    retained["continuity_run_id"] = (
        retained.raw_segment_id.ne(retained.raw_segment_id.shift())
        | retained.datetime.diff().ne(pd.Timedelta(minutes=1))
    ).cumsum().astype(np.int32) - 1
    info = {
        "minute_grid_bins": int(bins_total),
        "rejected_minute_bins_below_10_genuine_readings": int((grid.raw_samples < MIN_RAW).sum()),
        "retained_minute_rows": int(len(retained)),
        "retained_sample_count_distribution": {str(int(k)): int(v) for k, v in retained.raw_samples.value_counts().sort_index().items()},
        "continuity_runs_after_missing_minute_breaks": int(retained.continuity_run_id.nunique()),
        "minimum_genuine_raw_samples_per_minute": MIN_RAW,
        "interpolation_used": False,
        "forward_fill_used": False,
        "back_fill_used": False,
    }
    return retained, info


def make_session_frame(clean: pd.DataFrame, split: str) -> pd.DataFrame:
    ids = EXPECTED_SESSIONS[split]
    frame = clean.loc[clean.raw_segment_id.isin(ids)].copy().sort_values("datetime").reset_index(drop=True)
    observed_ids = sorted(frame.raw_segment_id.unique().tolist())
    if observed_ids != ids:
        raise RuntimeError(f"{split}: expected raw session IDs {ids}, found {observed_ids}")
    # A new split-local segment at every source-session boundary and every missing minute.
    frame["split_segment_id"] = (
        frame.raw_segment_id.ne(frame.raw_segment_id.shift())
        | frame.datetime.diff().ne(pd.Timedelta(minutes=1))
    ).cumsum().astype(np.int32) - 1
    frame["split"] = split
    return frame


def create_windows(frame: pd.DataFrame, split: str, scaler: MinMaxScaler, out: Path):
    xs, ys, xraws, yraws = [], [], [], []
    starts_global, input_start, input_end, target_start, target_end = [], [], [], [], []
    manifest = []
    for seg_id, s in frame.groupby("split_segment_id", sort=True):
        s = s.sort_values("datetime").reset_index(drop=True)
        if len(s) > 1 and not s.datetime.diff().dropna().eq(pd.Timedelta(minutes=1)).all():
            raise RuntimeError(f"{split} segment {seg_id} is not minute-contiguous")
        v = s.scaled_consumption.to_numpy(np.float32)
        raw_v = s.consumption.to_numpy(np.float32)
        t = s.datetime.to_numpy(dtype="datetime64[ns]")
        global_rows = s.global_minute_row.to_numpy(np.int64)
        raw_counts = s.raw_samples.to_numpy(np.int16)
        total = LOOKBACK + FORECAST
        if len(s) < total:
            continue
        for i in range(0, len(s) - total + 1, STRIDE):
            end = i + total
            x, y = v[i:i+LOOKBACK], v[i+LOOKBACK:end]
            xr, yr = raw_v[i:i+LOOKBACK], raw_v[i+LOOKBACK:end]
            if not all(np.isfinite(a).all() for a in (x, y, xr, yr)):
                raise RuntimeError("Non-finite data in accepted window")
            ins = pd.Timestamp(t[i]).strftime("%Y-%m-%d %H:%M:%S")
            ine = pd.Timestamp(t[i+LOOKBACK-1]).strftime("%Y-%m-%d %H:%M:%S")
            tas = pd.Timestamp(t[i+LOOKBACK]).strftime("%Y-%m-%d %H:%M:%S")
            tae = pd.Timestamp(t[end-1]).strftime("%Y-%m-%d %H:%M:%S")
            xs.append(x[:, None]); ys.append(y[:, None]); xraws.append(xr); yraws.append(yr)
            starts_global.append(int(global_rows[i]))
            input_start.append(ins); input_end.append(ine); target_start.append(tas); target_end.append(tae)
            manifest.append({
                "split": split, "candidate_window": len(manifest), "accepted": True,
                "split_segment_id": int(seg_id), "raw_segment_id": int(s.raw_segment_id.iloc[0]),
                "global_start_minute_row": int(global_rows[i]), "input_start": ins, "input_end": ine,
                "target_start": tas, "target_end": tae, "lookback_minutes": LOOKBACK,
                "forecast_minutes": FORECAST, "stride_minutes": STRIDE,
                "minimum_raw_samples_in_48h_window": int(raw_counts[i:end].min()),
                "all_minutes_meet_raw_sample_threshold": bool((raw_counts[i:end] >= MIN_RAW).all()),
            })
    n = len(xs)
    def stack(items, shape, dtype=np.float32):
        return np.stack(items).astype(dtype) if items else np.empty(shape, dtype=dtype)
    np.savez_compressed(
        out / f"{split}_windows.npz",
        X=stack(xs, (0, LOOKBACK, 1)), y=stack(ys, (0, FORECAST, 1)),
        X_raw=stack(xraws, (0, LOOKBACK)), y_raw=stack(yraws, (0, FORECAST)),
        starts_global=np.asarray(starts_global, dtype=np.int64),
        input_start=np.asarray(input_start, dtype="U19"), input_end=np.asarray(input_end, dtype="U19"),
        target_start=np.asarray(target_start, dtype="U19"), target_end=np.asarray(target_end, dtype="U19"),
    )
    summary = {
        "split": split, "raw_session_ids": ";".join(map(str, EXPECTED_SESSIONS[split])),
        "rows": int(len(frame)), "start_timestamp": str(frame.datetime.min()), "end_timestamp": str(frame.datetime.max()),
        "split_local_continuity_segments": int(frame.split_segment_id.nunique()),
        "raw_recording_sessions": int(frame.raw_segment_id.nunique()),
        "candidate_complete_windows": n, "accepted_windows": n,
        "lookback_minutes": LOOKBACK, "forecast_minutes": FORECAST, "stride_minutes": STRIDE,
        "minimum_raw_samples_per_minute": MIN_RAW, "imputation_used": False,
        "window_file": str(out / f"{split}_windows.npz"),
    }
    return summary, manifest


def main():
    args = parse_args(); raw_path = args.input.expanduser().resolve(); out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw, raw_info = load_raw(raw_path)
    if raw_info["raw_rows"] != 1_346_393 or raw_info["raw_segment_count"] != 10:
        if not args.allow_noncanonical_source:
            raise RuntimeError(f"Canonical source mismatch: rows={raw_info['raw_rows']}, sessions={raw_info['raw_segment_count']}")
    clean, agg_info = aggregate_minutes(raw)
    if len(clean) != 112_260:
        if not args.allow_noncanonical_source:
            raise RuntimeError(f"Expected 112260 retained minute means, got {len(clean)}")
    frames = {s: make_session_frame(clean, s) for s in ("train", "val", "test")}
    # Every original recording session must be assigned exactly once.
    assigned = [x for values in EXPECTED_SESSIONS.values() for x in values]
    if sorted(assigned) != list(range(10)) or len(set(assigned)) != 10:
        raise RuntimeError("Session allocation is not a disjoint, complete partition of source sessions")
    # Enforce strict chronology across partitions.
    if not (frames["train"].datetime.max() < frames["val"].datetime.min() and frames["val"].datetime.max() < frames["test"].datetime.min()):
        raise RuntimeError("Session split is not strictly chronological")
    print("Fitting MinMaxScaler on genuine retained training-session minutes only...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(frames["train"].consumption.to_numpy(np.float64).reshape(-1, 1))
    for name, frame in frames.items():
        frame["scaled_consumption"] = scaler.transform(frame.consumption.to_numpy(np.float64).reshape(-1, 1)).astype(np.float32).ravel()
        frame.to_csv(out / f"{name}.csv", index=False)
    clean.to_csv(out / "mored_clean.csv", index=False)
    with (out / "scaler.pkl").open("wb") as f: pickle.dump(scaler, f)
    summaries, manifest_rows = [], []
    for name, frame in frames.items():
        summ, rows = create_windows(frame, name, scaler, out)
        summaries.append(summ); manifest_rows.extend(rows)
        print(f"{name.upper():5s}: sessions={EXPECTED_SESSIONS[name]}, rows={len(frame):,}, continuous_runs={summ['split_local_continuity_segments']}, windows={summ['accepted_windows']}")
    actual = {x["split"]: x["accepted_windows"] for x in summaries}
    manifest_df = pd.DataFrame(manifest_rows)
    per_session = {sid: int((manifest_df["raw_segment_id"] == sid).sum()) for sid in range(10)}
    if per_session != EXPECTED_PER_SESSION:
        raise RuntimeError(f"Per-session window audit mismatch: expected {EXPECTED_PER_SESSION}, observed {per_session}")
    if actual != EXPECTED_WINDOWS:
        raise RuntimeError(f"Expected windows {EXPECTED_WINDOWS}, observed {actual}; inspect source before training")
    pd.DataFrame(summaries).to_csv(out / "split_summary.csv", index=False)
    manifest_df.to_csv(out / "window_manifest.csv", index=False)
    np.savez_compressed(out / "minute_observation_audit.npz",
        datetime=clean.datetime.to_numpy(dtype="datetime64[ns]").astype(str),
        consumption=clean.consumption.to_numpy(np.float64), raw_samples=clean.raw_samples.to_numpy(np.int16),
        raw_segment_id=clean.raw_segment_id.to_numpy(np.int32), continuity_run_id=clean.continuity_run_id.to_numpy(np.int32))
    metadata = {
        "experiment": "mored_corrected_v3_session_split_no_interpolation_raw_count_filter",
        "dataset": "MORED Premises 6 / APT6.csv", "source_file": str(raw_path), "source_sha256": sha256_file(raw_path),
        "raw_data": raw_info, "minute_aggregation": {**agg_info, "missing_sample_policy": "no interpolation, forward fill, or backward fill; minute means use only genuine valid raw observations"},
        "split_policy": "chronological whole-original-recording-session split; no original raw session is divided across train/val/test",
        "session_assignment": EXPECTED_SESSIONS,
        "split_chronology": {s: {"start": str(f.datetime.min()), "end": str(f.datetime.max())} for s, f in frames.items()},
        "continuity_policy": "break at each raw recording-session boundary and every missing retained minute; no windows cross a break",
        "scaler": {"type": "sklearn MinMaxScaler", "feature_range": [0,1], "fit_data": "all genuine retained training-session minute means only", "data_min": float(scaler.data_min_[0]), "data_max": float(scaler.data_max_[0]), "n_samples_seen": int(scaler.n_samples_seen_)},
        "window_config": {"lookback_minutes": LOOKBACK, "forecast_minutes": FORECAST, "stride_minutes": STRIDE, "minimum_raw_samples_per_minute": MIN_RAW},
        "actual_splits": summaries, "old_corrected_v2_outputs_overwritten": False,
    }
    (out / "preprocessing_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nMORED V3 SESSION-BASED PREPROCESSING COMPLETE")
    print(f"Source SHA256: {metadata['source_sha256']}")
    print(f"Total windows: {sum(actual.values())}; test raw recording sessions: {len(EXPECTED_SESSIONS['test'])}")
    print(f"Windows by original raw session: {per_session}")
    print(f"No interpolation: YES | Scaler training-only: YES | Old V2 untouched: YES")
    print(f"Output: {out}")

if __name__ == "__main__":
    try: main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr); raise
