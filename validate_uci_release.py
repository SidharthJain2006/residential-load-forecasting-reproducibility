#!/usr/bin/env python3
"""Independent structural validator for the published corrected UCI window bundle."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data" / "UCI" / "processed"
EXPECTED = {"train": (903, 1440), "val": (189, 1440), "test": (192, 1440)}

def main():
    meta = json.loads((DATA / "preprocessing_metadata.json").read_text())
    assert meta["raw_rows"] == 2_075_259
    assert "No interpolation" in meta["missing_value_policy"]
    scaler = joblib.load(DATA / "scaler.joblib")
    manifest = pd.read_csv(DATA / "window_manifest.csv")
    summary = pd.read_csv(DATA / "split_summary.csv")
    for split, shape in EXPECTED.items():
        arr = np.load(DATA / f"{split}_windows.npz", allow_pickle=False)
        for key in ("X", "y", "X_raw", "y_raw"):
            assert key in arr.files, f"{split}: missing {key}"
            assert np.isfinite(arr[key]).all(), f"{split}: non-finite {key}"
        assert arr["X_raw"].shape == shape and arr["y_raw"].shape == shape
        assert arr["X"].shape[:2] == shape and arr["y"].shape[:2] == shape
        assert np.allclose(arr["X"].reshape(-1), scaler.transform(arr["X_raw"].reshape(-1,1)).reshape(-1), atol=2e-6)
        assert np.allclose(arr["y"].reshape(-1), scaler.transform(arr["y_raw"].reshape(-1,1)).reshape(-1), atol=2e-6)
        ts = pd.to_datetime(arr["input_end"])
        target = pd.to_datetime(arr["target_start"])
        assert np.all((target - ts).total_seconds() == 60), f"{split}: target does not immediately follow input"
        assert len(arr["starts_global"]) == shape[0]
        row = summary.loc[summary["split"] == split].iloc[0]
        assert int(row["accepted_windows"]) == shape[0]
        mr = manifest[(manifest["split"] == split) & (manifest["accepted"].astype(bool))]
        assert len(mr) == shape[0]
        print(f"[PASS] {split}: {shape[0]} finite, correctly scaled, chronological 48-hour windows")
    assert int(meta["scaler"]["training_samples"]) == 1_444_337
    assert float(meta["scaler"]["training_minimum"]) == 0.076
    assert float(meta["scaler"]["training_maximum"]) == 11.122
    print("[PASS] UCI metadata, training-only scaler values, split manifest, and window counts")
    print("UCI corrected processed-data validation: ALL CHECKS PASSED")

if __name__ == "__main__": main()
