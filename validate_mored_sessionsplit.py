#!/usr/bin/env python3
"""Independent integrity validator for MORED V3 whole-recording-session splits."""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data/MORED/Corrected_V3_SessionSplit"
EXPECTED = {"train": 245, "val": 119, "test": 359}
SESSION_IDS = {"train": [0,1,2,3,4], "val": [5], "test": [6,7,8,9]}


def check(ok, msg):
    if not ok: raise RuntimeError("FAIL: " + msg)
    print("PASS:", msg)


def main():
    meta = json.loads((DATA/"preprocessing_metadata.json").read_text())
    check(meta.get("experiment") == "mored_corrected_v3_session_split_no_interpolation_raw_count_filter", "metadata identifies V3 session split")
    check(meta.get("session_assignment") == SESSION_IDS, "complete predeclared session allocation")
    check(meta.get("minute_aggregation",{}).get("interpolation_used") is False, "metadata states no interpolation")
    split_frames = {}
    expected_shapes = {"train": (245,1440), "val": (119,1440), "test": (359,1440)}
    for split in ("train","val","test"):
        frame = pd.read_csv(DATA/f"{split}.csv", parse_dates=["datetime"])
        split_frames[split] = frame
        observed = sorted(frame.raw_segment_id.astype(int).unique().tolist())
        check(observed == SESSION_IDS[split], f"{split}: only assigned raw recording sessions {observed}")
        check(frame.consumption.notna().all() and np.isfinite(frame.consumption).all(), f"{split}: finite genuine minute means")
        check((frame.raw_samples >= 10).all(), f"{split}: every minute has at least 10 genuine raw observations")
        with np.load(DATA/f"{split}_windows.npz", allow_pickle=False) as z:
            d={k:z[k].copy() for k in z.files}
        n, h = expected_shapes[split]
        check(d["X"].shape == (n,h,1) and d["y"].shape == (n,h,1), f"{split}: scaled windows have expected shape")
        check(d["X_raw"].shape == (n,h) and d["y_raw"].shape == (n,h), f"{split}: raw windows have expected shape")
        for key in ("X","y","X_raw","y_raw"):
            check(np.isfinite(d[key]).all(), f"{split}: {key} finite")
        starts = pd.to_datetime(d["input_start"]).to_numpy(dtype="datetime64[ns]")
        ends = pd.to_datetime(d["input_end"]).to_numpy(dtype="datetime64[ns]")
        tstarts = pd.to_datetime(d["target_start"]).to_numpy(dtype="datetime64[ns]")
        tends = pd.to_datetime(d["target_end"]).to_numpy(dtype="datetime64[ns]")
        check(np.all(ends-starts == np.timedelta64(1439,"m")), f"{split}: each 24h input is continuous")
        check(np.all(tends-tstarts == np.timedelta64(1439,"m")), f"{split}: each 24h target is continuous")
        check(np.all(tstarts-ends == np.timedelta64(1,"m")), f"{split}: target immediately follows input")
        man = pd.read_csv(DATA/"window_manifest.csv")
        ms = man[man.split.eq(split)]
        check(len(ms)==n, f"{split}: manifest has {n} windows")
        check(ms.raw_segment_id.astype(int).isin(SESSION_IDS[split]).all(), f"{split}: every window belongs to an assigned original session")
        check((ms.minimum_raw_samples_in_48h_window >= 10).all(), f"{split}: all 48h windows meet raw-observation threshold")
    # Verify ordered holdout partitions; no timestamp range overlaps.
    tr, va, te = (split_frames[x] for x in ("train","val","test"))
    check(tr.datetime.max() < va.datetime.min() and va.datetime.max() < te.datetime.min(), "strict chronological train < validation < test")
    # All ten source recording sessions are assigned exactly once.
    all_assigned = [i for vals in SESSION_IDS.values() for i in vals]
    check(sorted(all_assigned)==list(range(10)) and len(set(all_assigned))==10, "all 10 original raw sessions assigned once")
    # Scaler must be fitted to all genuine training-minute rows only.
    with (DATA/"scaler.pkl").open("rb") as f: scaler=pickle.load(f)
    train_values=tr.consumption.to_numpy(dtype=np.float64)
    check(int(scaler.n_samples_seen_)==len(train_values), "scaler fit sample count equals training-minute rows only")
    check(np.isclose(float(scaler.data_min_[0]),train_values.min()) and np.isclose(float(scaler.data_max_[0]),train_values.max()), "scaler bounds match training minutes only")
    for split,frame in (("train",tr),("val",va),("test",te)):
        with np.load(DATA/f"{split}_windows.npz",allow_pickle=False) as z: d={k:z[k].copy() for k in z.files}
        ex=scaler.transform(d["X_raw"].reshape(-1,1)).reshape(d["X"].shape)
        ey=scaler.transform(d["y_raw"].reshape(-1,1)).reshape(d["y"].shape)
        check(np.allclose(d["X"],ex,atol=2e-6) and np.allclose(d["y"],ey,atol=2e-6), f"{split}: scaled windows reproduce with training-only scaler")
    test_manifest=pd.read_csv(DATA/"window_manifest.csv")
    test_manifest=test_manifest[test_manifest.split.eq("test")]
    test_clusters=sorted(test_manifest.raw_segment_id.astype(int).unique().tolist())
    check(test_clusters == [6,7,8,9], "test has exactly four original recording-session clusters")
    print("\nMORED V3 SESSION-SPLIT VALIDATION: ALL CHECKS PASSED")
    print("Windows: train=245, validation=119, test=359")
    print(f"Independent test recording sessions: {test_clusters}")
    print("Important: there are 4 raw recording-session clusters and 7 within-session continuity runs producing windows.")
    print("All model training/evaluation must be rerun on this new split; V2 scores remain a separate experiment.")

if __name__ == "__main__":
    try: main()
    except Exception as e:
        print(f"VALIDATION FAILED: {e}",file=sys.stderr); raise
