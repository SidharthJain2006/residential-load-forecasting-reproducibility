#!/usr/bin/env python3
"""Load leakage-aware UCI windows exported by prepare_uci.py.

Returns arrays in the shapes expected by the existing custom PyTorch scripts:
X: (n_windows, 1440), y: (n_windows, 1440), timestamps: (n_windows, 1440).
The saved NPZ files retain a trailing singleton channel; this loader squeezes it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

LOOKBACK = 1440
FORECAST = 1440
EXPECTED_COUNTS = {"train": 903, "val": 189, "test": 192}


def _load_one_split(data_dir: Path, name: str, all_timestamps: np.ndarray) -> dict[str, Any]:
    path = data_dir / f"{name}_windows.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing processed split: {path}")

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "X", "y", "X_raw", "y_raw", "starts_global",
            "input_start", "input_end", "target_start", "target_end"
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"{path.name} is missing keys: {sorted(missing)}")

        X = np.asarray(archive["X"], dtype=np.float32)
        y = np.asarray(archive["y"], dtype=np.float32)
        X_raw = np.asarray(archive["X_raw"], dtype=np.float32)
        y_raw = np.asarray(archive["y_raw"], dtype=np.float32)
        starts = np.asarray(archive["starts_global"], dtype=np.int64)
        input_start = np.asarray(archive["input_start"])
        input_end = np.asarray(archive["input_end"])
        target_start = np.asarray(archive["target_start"])
        target_end = np.asarray(archive["target_end"])

    n = EXPECTED_COUNTS[name]
    if X.shape != (n, LOOKBACK, 1):
        raise ValueError(f"{name}: unexpected X shape {X.shape}")
    if y.shape != (n, FORECAST, 1):
        raise ValueError(f"{name}: unexpected y shape {y.shape}")
    if X_raw.shape != (n, LOOKBACK) or y_raw.shape != (n, FORECAST):
        raise ValueError(f"{name}: unexpected raw X/y shapes {X_raw.shape}, {y_raw.shape}")
    if starts.shape != (n,):
        raise ValueError(f"{name}: unexpected starts shape {starts.shape}")
    if not all(np.isfinite(a).all() for a in (X, y, X_raw, y_raw)):
        raise ValueError(f"{name}: non-finite values found in saved windows")
    if np.any(starts < 0) or np.any(starts + LOOKBACK > len(all_timestamps)):
        raise ValueError(f"{name}: window start index is outside the timestamp array")

    # Reconstruct every input timestamp from the original full timeline.
    # This preserves compatibility with the old calendar-feature code.
    time_windows = np.stack([
        all_timestamps[s:s + LOOKBACK] for s in starts
    ]).astype("datetime64[ns]")

    # Confirm that the reconstructed boundary timestamps match the manifest
    # fields written into the NPZ file.
    if not np.array_equal(time_windows[:, 0].astype("datetime64[s]"), input_start.astype("datetime64[s]")):
        raise ValueError(f"{name}: input_start values do not match the raw timeline")
    if not np.array_equal(time_windows[:, -1].astype("datetime64[s]"), input_end.astype("datetime64[s]")):
        raise ValueError(f"{name}: input_end values do not match the raw timeline")
    if not np.array_equal(all_timestamps[starts + LOOKBACK].astype("datetime64[s]"), target_start.astype("datetime64[s]")):
        raise ValueError(f"{name}: target_start values do not match the raw timeline")
    if not np.array_equal(all_timestamps[starts + LOOKBACK + FORECAST - 1].astype("datetime64[s]"), target_end.astype("datetime64[s]")):
        raise ValueError(f"{name}: target_end values do not match the raw timeline")

    return {
        "X": X[:, :, 0],
        "y": y[:, :, 0],
        "X_raw": X_raw,
        "y_raw": y_raw,
        "starts_global": starts,
        "timestamps": time_windows,
        "input_start": input_start,
        "input_end": input_end,
        "target_start": target_start,
        "target_end": target_end,
    }


def load_uci_windows(project_root: str | Path | None = None) -> dict[str, Any]:
    """Load all clean UCI splits, the original scaler, and timestamp metadata."""
    root = Path(project_root).resolve() if project_root else Path(__file__).resolve().parent
    data_dir = root / "Data" / "UCI" / "processed"

    mask_path = data_dir / "raw_observation_mask.npz"
    if not mask_path.is_file():
        raise FileNotFoundError(f"Missing observation mask: {mask_path}")
    with np.load(mask_path, allow_pickle=False) as mask_archive:
        observed = np.asarray(mask_archive["observed"], dtype=np.uint8)
        all_timestamps = np.asarray(mask_archive["timestamps"]).astype("datetime64[ns]")

    if len(observed) != len(all_timestamps):
        raise ValueError("Observation mask and timestamp lengths differ")
    if len(all_timestamps) < LOOKBACK + FORECAST:
        raise ValueError("Raw timestamp sequence is unexpectedly short")

    scaler_path = data_dir / "scaler.joblib"
    scaler = joblib.load(scaler_path)

    result: dict[str, Any] = {
        "root": root,
        "data_dir": data_dir,
        "scaler": scaler,
        "observed_mask": observed,
        "all_timestamps": all_timestamps,
    }

    for split in ("train", "val", "test"):
        result[split] = _load_one_split(data_dir, split, all_timestamps)

    # Ensure the stored windows only refer to observed source readings.
    for split in ("train", "val", "test"):
        starts = result[split]["starts_global"]
        for start in starts:
            if not observed[start:start + LOOKBACK + FORECAST].all():
                raise ValueError(f"{split}: a saved window includes a missing original reading")

    print("Loaded corrected UCI windows:")
    for split in ("train", "val", "test"):
        item = result[split]
        print(f"  {split:5s} X={item['X'].shape}, y={item['y'].shape}, timestamps={item['timestamps'].shape}")
    print("Scaler loaded from corrected training-only preprocessing.")
    return result


def inverse_array(array: np.ndarray, scaler: Any) -> np.ndarray:
    """Invert MinMax scaling while preserving the original array shape."""
    values = np.asarray(array)
    original_shape = values.shape
    return scaler.inverse_transform(values.reshape(-1, 1)).reshape(original_shape)


if __name__ == "__main__":
    data = load_uci_windows()
    print("\nUCI window loader smoke test PASSED.")
