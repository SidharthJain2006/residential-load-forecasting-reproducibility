#!/usr/bin/env python3
"""Prepare leakage-aware UCI Household Power Consumption windows.

This script deliberately does NOT interpolate missing readings. It preserves the
original observation mask, makes chronological 70/15/15 splits, fits a MinMax
scaler on observed training-split readings only, and exports only windows whose
entire lookback + forecast interval contains genuine observations.

Default project paths:
  Input : Data/UCI/raw/household_power_consumption.txt
  Output: Data/UCI/processed/

Example:
  python src/prepare_uci.py
  python src/prepare_uci.py --input "Data/UCI/raw/household_power_consumption 3.txt"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


EXPECTED_RAW_ROWS = 2_075_259
EXPECTED_ACCEPTED_WINDOWS = {"train": 903, "val": 189, "test": 192}
EXPECTED_CANDIDATE_WINDOWS = {"train": 1007, "val": 215, "test": 215}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create leakage-aware UCI forecasting windows without imputation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Data/UCI/raw/household_power_consumption.txt"),
        help="Path to the original semicolon-delimited UCI .txt file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Data/UCI/processed"),
        help="Directory where processed windows and audit files are written.",
    )
    parser.add_argument("--lookback", type=int, default=1440, help="Input length in minutes.")
    parser.add_argument("--forecast", type=int, default=1440, help="Forecast length in minutes.")
    parser.add_argument("--stride", type=int, default=1440, help="Window stride in minutes.")
    parser.add_argument(
        "--allow-noncanonical-size",
        action="store_true",
        help="Do not stop if the raw row count differs from the audited UCI source.",
    )
    return parser.parse_args()


def load_raw(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot find the raw UCI file: {path}\n"
            "Place the original household_power_consumption .txt file in "
            "Data/UCI/raw/ or pass its path with --input."
        )

    print(f"[1/6] Reading raw file: {path}")
    # UCI source is semicolon-delimited and uses '?' for missing measurements.
    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?"],
        keep_default_na=True,
        usecols=["Date", "Time", "Global_active_power"],
        dtype={"Date": "string", "Time": "string"},
        low_memory=False,
    )

    df["timestamp"] = pd.to_datetime(
        df["Date"].str.strip() + " " + df["Time"].str.strip(),
        format="%d/%m/%Y %H:%M:%S",
        errors="raise",
    )
    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"], errors="coerce"
    )
    df = df[["timestamp", "Global_active_power"]].copy()

    if df.empty:
        raise ValueError("The raw UCI file contains no data rows.")
    if df["timestamp"].isna().any():
        raise ValueError("Some timestamps could not be parsed.")
    if df["timestamp"].duplicated().any():
        examples = df.loc[df["timestamp"].duplicated(), "timestamp"].head().tolist()
        raise ValueError(f"Duplicate timestamps found, for example: {examples}")
    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError("Timestamps are not in chronological order. Do not sort silently;")

    deltas = df["timestamp"].diff().dropna()
    expected_delta = pd.Timedelta(minutes=1)
    irregular = deltas.ne(expected_delta)
    if irregular.any():
        examples = deltas[irregular].head(5).tolist()
        raise ValueError(
            "The raw file is not a continuous one-minute grid. Stop and inspect the "
            f"timestamp gaps before building windows. Examples: {examples}"
        )

    return df


def chronological_bounds(n_rows: int) -> dict[str, tuple[int, int]]:
    # Match the audited chronological split: 70%, 15%, 15% by row count.
    train_end = int(0.70 * n_rows)
    val_size = round(0.15 * n_rows)
    val_end = train_end + val_size
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n_rows),
    }


def timestamp_string(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def process_split(
    split_name: str,
    start: int,
    end: int,
    values: np.ndarray,
    timestamps: np.ndarray,
    scaler: MinMaxScaler,
    lookback: int,
    forecast: int,
    stride: int,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split_values = values[start:end]
    split_times = timestamps[start:end]
    window_length = lookback + forecast

    if len(split_values) < window_length:
        raise ValueError(f"{split_name} split is shorter than one full window.")

    candidate_starts = np.arange(
        0, len(split_values) - window_length + 1, stride, dtype=np.int64
    )
    observed = np.isfinite(split_values)
    missing_prefix = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(~observed, dtype=np.int64)]
    )
    missing_per_window = (
        missing_prefix[candidate_starts + window_length]
        - missing_prefix[candidate_starts]
    )
    accepted_mask = missing_per_window == 0
    accepted_starts = candidate_starts[accepted_mask]

    # A single accepted window consists only of raw, originally observed values.
    if len(accepted_starts):
        x_raw = np.stack(
            [split_values[s : s + lookback] for s in accepted_starts]
        ).astype(np.float32)
        y_raw = np.stack(
            [split_values[s + lookback : s + window_length] for s in accepted_starts]
        ).astype(np.float32)
        x_scaled = scaler.transform(x_raw.reshape(-1, 1)).reshape(
            len(accepted_starts), lookback, 1
        ).astype(np.float32)
        y_scaled = scaler.transform(y_raw.reshape(-1, 1)).reshape(
            len(accepted_starts), forecast, 1
        ).astype(np.float32)
    else:
        x_raw = np.empty((0, lookback), dtype=np.float32)
        y_raw = np.empty((0, forecast), dtype=np.float32)
        x_scaled = np.empty((0, lookback, 1), dtype=np.float32)
        y_scaled = np.empty((0, forecast, 1), dtype=np.float32)

    manifest_rows: list[dict[str, Any]] = []
    accepted_index = 0
    for candidate_index, (local_start, missing_count, keep) in enumerate(
        zip(candidate_starts, missing_per_window, accepted_mask)
    ):
        global_start = start + int(local_start)
        row = {
            "split": split_name,
            "candidate_window_index": int(candidate_index),
            "accepted": bool(keep),
            "accepted_window_index": accepted_index if keep else None,
            "missing_points_in_48h_window": int(missing_count),
            "global_start_row": global_start,
            "input_start": timestamp_string(pd.Timestamp(split_times[local_start])),
            "input_end": timestamp_string(
                pd.Timestamp(split_times[local_start + lookback - 1])
            ),
            "target_start": timestamp_string(
                pd.Timestamp(split_times[local_start + lookback])
            ),
            "target_end": timestamp_string(
                pd.Timestamp(split_times[local_start + window_length - 1])
            ),
        }
        manifest_rows.append(row)
        if keep:
            accepted_index += 1

    split_file = output_dir / f"{split_name}_windows.npz"
    np.savez_compressed(
        split_file,
        X=x_scaled,
        y=y_scaled,
        X_raw=x_raw,
        y_raw=y_raw,
        starts_global=(start + accepted_starts).astype(np.int64),
        input_start=np.array(
            [timestamp_string(pd.Timestamp(split_times[s])) for s in accepted_starts],
            dtype="U19",
        ),
        input_end=np.array(
            [timestamp_string(pd.Timestamp(split_times[s + lookback - 1])) for s in accepted_starts],
            dtype="U19",
        ),
        target_start=np.array(
            [timestamp_string(pd.Timestamp(split_times[s + lookback])) for s in accepted_starts],
            dtype="U19",
        ),
        target_end=np.array(
            [timestamp_string(pd.Timestamp(split_times[s + window_length - 1])) for s in accepted_starts],
            dtype="U19",
        ),
    )

    summary = {
        "split": split_name,
        "start_row_in_raw_file": int(start),
        "end_row_exclusive_in_raw_file": int(end),
        "rows": int(end - start),
        "start_timestamp": timestamp_string(pd.Timestamp(split_times[0])),
        "end_timestamp": timestamp_string(pd.Timestamp(split_times[-1])),
        "missing_raw_readings": int((~observed).sum()),
        "observed_raw_readings": int(observed.sum()),
        "candidate_windows": int(len(candidate_starts)),
        "accepted_complete_windows": int(len(accepted_starts)),
        "rejected_windows_with_missing_points": int((~accepted_mask).sum()),
        "lookback_minutes": int(lookback),
        "forecast_minutes": int(forecast),
        "stride_minutes": int(stride),
        "imputed_values_created": 0,
        "window_file": str(split_file),
    }
    return summary, manifest_rows


def main() -> int:
    args = parse_args()
    if min(args.lookback, args.forecast, args.stride) <= 0:
        raise ValueError("lookback, forecast and stride must all be positive integers.")

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(args.input)
    n_rows = len(df)
    if n_rows != EXPECTED_RAW_ROWS:
        message = (
            f"Raw row count is {n_rows:,}; the audited source has "
            f"{EXPECTED_RAW_ROWS:,} rows."
        )
        if not args.allow_noncanonical_size:
            raise ValueError(message + " Use --allow-noncanonical-size only after confirming the source version.")
        print(f"WARNING: {message}")

    values = df["Global_active_power"].to_numpy(dtype=np.float64)
    timestamps = df["timestamp"].to_numpy()
    observed = np.isfinite(values)

    print("[2/6] Raw-data audit")
    print(f"  Rows: {n_rows:,}")
    print(f"  Observed Global_active_power: {int(observed.sum()):,}")
    print(f"  Missing Global_active_power: {int((~observed).sum()):,} ({(~observed).mean() * 100:.4f}%)")
    print(f"  Time range: {timestamp_string(pd.Timestamp(timestamps[0]))} to {timestamp_string(pd.Timestamp(timestamps[-1]))}")

    bounds = chronological_bounds(n_rows)
    train_start, train_end = bounds["train"]
    train_values = values[train_start:train_end]
    train_observed_values = train_values[np.isfinite(train_values)]
    if train_observed_values.size == 0:
        raise ValueError("No observed training values are available to fit the scaler.")

    print("[3/6] Fitting MinMaxScaler on observed training-split readings only")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_observed_values.reshape(-1, 1))
    joblib.dump(scaler, output_dir / "scaler.joblib")

    print("[4/6] Creating complete, non-imputed windows")
    summaries: list[dict[str, Any]] = []
    all_manifest_rows: list[dict[str, Any]] = []
    for split_name in ("train", "val", "test"):
        start, end = bounds[split_name]
        summary, manifest = process_split(
            split_name=split_name,
            start=start,
            end=end,
            values=values,
            timestamps=timestamps,
            scaler=scaler,
            lookback=args.lookback,
            forecast=args.forecast,
            stride=args.stride,
            output_dir=output_dir,
        )
        summaries.append(summary)
        all_manifest_rows.extend(manifest)

    summary_df = pd.DataFrame(summaries)
    manifest_df = pd.DataFrame(all_manifest_rows)
    summary_df.to_csv(output_dir / "split_summary.csv", index=False)
    manifest_df.to_csv(output_dir / "window_manifest.csv", index=False)
    np.savez_compressed(
        output_dir / "raw_observation_mask.npz",
        observed=observed.astype(np.uint8),
        timestamps=timestamps.astype("datetime64[ns]").astype(str),
    )

    metadata: dict[str, Any] = {
        "source_file": str(args.input),
        "raw_rows": int(n_rows),
        "raw_start_timestamp": timestamp_string(pd.Timestamp(timestamps[0])),
        "raw_end_timestamp": timestamp_string(pd.Timestamp(timestamps[-1])),
        "timestamp_frequency": "1 minute",
        "target_column": "Global_active_power",
        "missing_value_policy": "No interpolation; preserve missingness and reject any window containing a missing point.",
        "split_policy": "Chronological 70/15/15 by row count; no shuffling.",
        "split_bounds_zero_based_half_open": {
            name: {"start": int(a), "end_exclusive": int(b)}
            for name, (a, b) in bounds.items()
        },
        "scaler": {
            "type": "sklearn.preprocessing.MinMaxScaler",
            "feature_range": [0, 1],
            "fit_data": "Observed Global_active_power values in the training split only.",
            "data_min": float(scaler.data_min_[0]),
            "data_max": float(scaler.data_max_[0]),
            "n_samples_seen": int(scaler.n_samples_seen_),
        },
        "window_config": {
            "lookback_minutes": int(args.lookback),
            "forecast_minutes": int(args.forecast),
            "stride_minutes": int(args.stride),
            "require_all_input_and_target_points_observed": True,
        },
        "expected_audit_window_counts_for_canonical_source": EXPECTED_ACCEPTED_WINDOWS,
        "actual_splits": summaries,
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
    }
    with (output_dir / "preprocessing_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("[5/6] Verifying expected window counts")
    for row in summaries:
        name = row["split"]
        actual = row["accepted_complete_windows"]
        expected = EXPECTED_ACCEPTED_WINDOWS[name]
        candidates_expected = EXPECTED_CANDIDATE_WINDOWS[name]
        print(
            f"  {name:5s}: {actual:4d} accepted / {row['candidate_windows']:4d} candidates "
            f"(audit expectation: {expected} / {candidates_expected})"
        )
        if n_rows == EXPECTED_RAW_ROWS and (
            actual != expected or row["candidate_windows"] != candidates_expected
        ):
            raise RuntimeError(
                f"Window-count audit mismatch for {name}: got {actual}/"
                f"{row['candidate_windows']}, expected {expected}/{candidates_expected}."
            )

    # Final invariants: exported windows must be finite and dimensions exact.
    for name in ("train", "val", "test"):
        with np.load(output_dir / f"{name}_windows.npz") as data:
            if not np.isfinite(data["X"]).all() or not np.isfinite(data["y"]).all():
                raise RuntimeError(f"Non-finite values found in exported {name} windows.")
            if data["X"].shape[1:] != (args.lookback, 1):
                raise RuntimeError(f"Unexpected X shape for {name}: {data['X'].shape}")
            if data["y"].shape[1:] != (args.forecast, 1):
                raise RuntimeError(f"Unexpected y shape for {name}: {data['y'].shape}")

    print("[6/6] Completed successfully")
    print(f"  Output directory: {output_dir.resolve()}")
    print("  Files created:")
    for file_path in sorted(output_dir.iterdir()):
        if file_path.is_file():
            print(f"    - {file_path.name} ({file_path.stat().st_size:,} bytes)")
    print("\nImportant: These are the corrected clean windows. Do not use the old UCI split CSVs for the new experiments.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
