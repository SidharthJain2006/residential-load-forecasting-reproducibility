#!/usr/bin/env python3
"""Independent integrity audit for all six corrected MORED model outputs.

Run from the project folder, or place this script in the project root and run:
    python audit_mored_predictions_corrected.py

The audit verifies every prediction against the canonical corrected test targets,
checks the saved target timestamps, recomputes reported metrics, and writes an
all-model audit CSV plus a concise summary. It does not retrain any model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data" / "MORED" / "Corrected_V3_SessionSplit"
RESULTS_ROOT = ROOT / "Results" / "MORED"
AUDIT_DIR = RESULTS_ROOT / "Corrected_V3_SessionSplit_Audit"
SEEDS = (0, 1, 2, 3, 4)
METRICS = ("RMSE", "MAE", "MAPE", "R2", "StdR")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_npy(path: Path) -> np.ndarray:
    require(path.is_file(), f"Missing required file: {path}")
    return np.load(path, allow_pickle=False)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    require(yt.shape == yp.shape, f"Metric shape mismatch: {yt.shape} vs {yp.shape}")
    require(np.isfinite(yt).all() and np.isfinite(yp).all(), "Non-finite values in metric arrays")
    nz = np.abs(yt) > 1e-8
    return {
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE": float(mean_absolute_error(yt, yp)),
        "MAPE": float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])) * 100.0),
        "R2": float(r2_score(yt, yp)),
        "StdR": float(np.std(yp) / (np.std(yt) + 1e-12)),
    }


def read_saved_timestamps(path: Path) -> np.ndarray:
    arr = load_npy(path)
    try:
        return arr.astype("datetime64[ns]")
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not parse target timestamps in {path}: {exc}") from exc


def check_truth_file(path: Path, canonical_y: np.ndarray, label: str) -> None:
    stored = load_npy(path)
    require(stored.shape == canonical_y.shape,
            f"{label}: stored y_true shape {stored.shape} != canonical {canonical_y.shape}")
    require(np.array_equal(stored, canonical_y),
            f"{label}: stored y_true does not exactly match corrected test_windows.npz y_raw")


def compare_metrics(saved_row: pd.Series, got: dict[str, float], label: str) -> None:
    for metric in METRICS:
        require(metric in saved_row.index, f"{label}: saved results row has no {metric} column")
        old = float(saved_row[metric])
        new = got[metric]
        require(np.isfinite(old) and abs(old - new) <= max(1e-6, abs(new) * 1e-7),
                f"{label}: {metric} differs from saved metric (saved={old:.10g}, recomputed={new:.10g})")


def main() -> int:
    metadata_path = DATA_DIR / "preprocessing_metadata.json"
    test_path = DATA_DIR / "test_windows.npz"
    require(metadata_path.is_file(), f"Missing corrected preprocessing metadata: {metadata_path}")
    require(test_path.is_file(), f"Missing corrected test windows: {test_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    require(metadata.get("experiment") == "mored_corrected_v3_session_split_no_interpolation_raw_count_filter",
            "Unexpected MORED preprocessing experiment; refusing to audit legacy data")
    missing_policy = str(metadata.get("minute_aggregation", {}).get("missing_sample_policy", "")).lower()
    require("no interpolation" in missing_policy,
            "Preprocessing metadata does not explicitly confirm the no-interpolation policy")

    with np.load(test_path, allow_pickle=False) as z:
        test = {key: z[key].copy() for key in z.files}
    for key in ("y_raw", "target_start"):
        require(key in test, f"Corrected test NPZ is missing {key}")
    y_true = np.asarray(test["y_raw"], dtype=np.float32)
    if y_true.ndim == 3 and y_true.shape[-1] == 1:
        y_true = y_true[..., 0]
    require(y_true.shape == (359, 1440), f"Expected canonical y_true shape (359,1440), got {y_true.shape}")
    require(np.isfinite(y_true).all(), "Canonical corrected targets contain non-finite values")
    target_starts = pd.to_datetime(test["target_start"]).to_numpy(dtype="datetime64[ns]")
    canonical_times = target_starts[:, None] + np.arange(y_true.shape[1]).astype("timedelta64[m]")[None, :]

    records: list[dict[str, Any]] = []
    checks: list[str] = []

    # Five-seed neural/custom-PyTorch models.
    model_specs = [
        ("VanillaTransformer", "Corrected_V3_SessionSplit_Vanilla_DAT", "VanillaTransformer", "mored_corrected_results.csv", "VanillaTransformer"),
        ("DAT", "Corrected_V3_SessionSplit_Vanilla_DAT", "DAT", "mored_corrected_results.csv", "DAT"),
        ("LSTM", "Corrected_V3_SessionSplit_LSTM", "LSTM", "lstm_corrected_results.csv", "LSTM"),
        ("DLinear", "Corrected_V3_SessionSplit_DLinear", "DLinear", "dlinear_corrected_results.csv", "DLinear"),
        ("PatchTST-style", "Corrected_V3_SessionSplit_PatchTST", "PatchTST", "patchtst_corrected_results.csv", "PatchTST-style"),
    ]
    for display_name, folder, file_prefix, results_csv, csv_model_name in model_specs:
        result_dir = RESULTS_ROOT / folder
        results_path = result_dir / results_csv
        require(results_path.is_file(), f"Missing {display_name} results CSV: {results_path}")
        saved_results = pd.read_csv(results_path)
        require("Seed" in saved_results.columns and "Model" in saved_results.columns,
                f"{display_name}: results CSV lacks Model/Seed columns")
        subset = saved_results[saved_results["Model"].astype(str) == csv_model_name]
        require(set(subset["Seed"].astype(int)) == set(SEEDS),
                f"{display_name}: expected result rows for seeds {SEEDS}, found {subset['Seed'].tolist()}")

        truth_candidates = [result_dir / "y_true_raw.npy"]
        truth_file = next((p for p in truth_candidates if p.is_file()), None)
        require(truth_file is not None, f"{display_name}: missing saved y_true_raw.npy")
        check_truth_file(truth_file, y_true, display_name)
        time_file = result_dir / "test_target_timestamps.npy"
        saved_times = read_saved_timestamps(time_file)
        require(saved_times.shape == canonical_times.shape and np.array_equal(saved_times, canonical_times),
                f"{display_name}: saved target timestamps differ from corrected test windows")

        for seed in SEEDS:
            pred_path = result_dir / f"{file_prefix}_seed{seed}_predictions.npy"
            pred = load_npy(pred_path)
            require(pred.shape == y_true.shape,
                    f"{display_name} seed {seed}: prediction shape {pred.shape} != {y_true.shape}")
            require(np.isfinite(pred).all(), f"{display_name} seed {seed}: non-finite predictions")
            got = metrics(y_true, pred)
            saved_row = subset[subset["Seed"].astype(int) == seed].iloc[0]
            compare_metrics(saved_row, got, f"{display_name} seed {seed}")
            records.append({"Model": display_name, "Seed": seed, **got, "Prediction_file": pred_path.name})
        checks.append(f"{display_name}: 5 predictions, y_true, timestamps and metrics PASS")

    # Single zero-shot TimesFM run.
    tf_dir = RESULTS_ROOT / "Corrected_V3_SessionSplit_TimesFM25"
    tf_pred = load_npy(tf_dir / "TimesFM25_predictions_raw.npy")
    require(tf_pred.shape == y_true.shape, f"TimesFM: prediction shape {tf_pred.shape} != {y_true.shape}")
    require(np.isfinite(tf_pred).all(), "TimesFM: non-finite predictions")
    check_truth_file(tf_dir / "TimesFM25_y_true_raw.npy", y_true, "TimesFM")
    tf_times = read_saved_timestamps(tf_dir / "TimesFM25_target_timestamps.npy")
    require(tf_times.shape == canonical_times.shape and np.array_equal(tf_times, canonical_times),
            "TimesFM: saved target timestamps differ from corrected test windows")
    tf_metrics = metrics(y_true, tf_pred)
    tf_saved = pd.read_csv(tf_dir / "TimesFM25_metrics.csv")
    require(len(tf_saved) == 1, "TimesFM: expected one saved zero-shot metrics row")
    compare_metrics(tf_saved.iloc[0], tf_metrics, "TimesFM 2.5 zero-shot")
    records.append({"Model": "TimesFM 2.5 zero-shot", "Seed": "zero-shot", **tf_metrics,
                    "Prediction_file": "TimesFM25_predictions_raw.npy"})
    checks.append("TimesFM 2.5: prediction, y_true, timestamps and metrics PASS")

    audit = pd.DataFrame(records)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_DIR / "mored_prediction_integrity_audit.csv", index=False)
    summary_rows = []
    for model_name, group in audit.groupby("Model", sort=False):
        row: dict[str, Any] = {"Model": model_name}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1)) if len(group) > 1 else np.nan
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(AUDIT_DIR / "mored_prediction_integrity_summary.csv", index=False)
    report = {
        "status": "PASS",
        "dataset": "MORED Premises 6 APT6, corrected V3 whole-session no-interpolation",
        "canonical_test_targets_shape": list(y_true.shape),
        "test_windows": int(y_true.shape[0]),
        "forecast_minutes_per_window": int(y_true.shape[1]),
        "models_audited": 6,
        "trained_seeds_per_model": 5,
        "checks": checks,
        "outputs": [
            "mored_prediction_integrity_audit.csv",
            "mored_prediction_integrity_summary.csv",
        ],
    }
    (AUDIT_DIR / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nMORED CORRECTED-V3 SESSION-SPLIT PREDICTION INTEGRITY AUDIT: ALL CHECKS PASSED")
    print(f"Canonical test targets: {y_true.shape}; minute values={y_true.size:,}")
    for item in checks:
        print("  PASS:", item)
    print("\nRecomputed metric summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nAudit files saved to:", AUDIT_DIR)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"AUDIT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
