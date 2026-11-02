#!/usr/bin/env python3
"""Audit the corrected UCI predictions across all completed model runs.

This script does NOT train models or change existing results. It:
- compares each model's saved ground-truth array to the canonical clean test targets;
- checks prediction dimensions and finite values;
- verifies saved target timestamps (full per-minute timestamps or target starts);
- recomputes RMSE, MAE, MAPE, R2, and StdR from saved predictions;
- compares recomputed metrics with the model result CSVs when available;
- writes a consolidated audit report to Results/UCI/Corrected_V2_Audit/.

Run from the DAT_Reproducible_V2 project root:
    python audit_uci_predictions.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "Data" / "UCI" / "processed"
OUTPUT = ROOT / "Results" / "UCI" / "Corrected_V2_Audit"
LOOKBACK = 1440
FORECAST = 1440
N_TEST = 192
SEEDS = (0, 1, 2, 3, 4)
METRIC_NAMES = ("RMSE", "MAE", "MAPE", "R2", "StdR")

# Directory, prediction filename pattern, result CSV, and ground-truth filename.
MODELS = {
    "Vanilla Transformer": {
        "directory": "Corrected_V2_1_VanillaTransformer",
        "prediction_pattern": "VanillaTransformer_seed{seed}_predictions.npy",
        "results_csv": "vanilla_transformer_corrected_v2_1_results.csv",
        "truth_file": "y_true_raw.npy",
        "seeded": True,
    },
    "DAT": {
        "directory": "Corrected_V2_DAT",
        "prediction_pattern": "DAT_seed{seed}_predictions.npy",
        "results_csv": "dat_corrected_results.csv",
        "truth_file": "y_true_raw.npy",
        "seeded": True,
    },
    "LSTM": {
        "directory": "Corrected_V2_LSTM",
        "prediction_pattern": "LSTM_seed{seed}_predictions.npy",
        "results_csv": "lstm_corrected_results.csv",
        "truth_file": "y_true_raw.npy",
        "seeded": True,
    },
    "DLinear": {
        "directory": "Corrected_V2_DLinear",
        "prediction_pattern": "DLinear_seed{seed}_predictions.npy",
        "results_csv": "dlinear_corrected_results.csv",
        "truth_file": "y_true_raw.npy",
        "seeded": True,
    },
    "PatchTST-style": {
        "directory": "Corrected_V2_PatchTST",
        "prediction_pattern": "PatchTST_seed{seed}_predictions.npy",
        "results_csv": "patchtst_corrected_results.csv",
        "truth_file": "y_true_raw.npy",
        "seeded": True,
    },
    "TimesFM 2.5 zero-shot": {
        "directory": "Corrected_V2_TimesFM25",
        "prediction_pattern": "timesfm25_predictions.npy",
        "results_csv": "timesfm25_metrics.csv",
        "truth_file": "y_true_raw.npy",
        "seeded": False,
    },
}


def log_pass(message: str) -> None:
    print(f"[PASS] {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
    log_pass(message)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(f"Metric arrays differ: {yt.shape} vs {yp.shape}")
    if not np.isfinite(yt).all() or not np.isfinite(yp).all():
        raise ValueError("Metric arrays contain non-finite values")

    error = yp - yt
    rmse = float(np.sqrt(np.mean(error ** 2)))
    mae = float(np.mean(np.abs(error)))
    nonzero = np.abs(yt) > 1e-8
    mape = float(np.mean(np.abs(error[nonzero] / yt[nonzero])) * 100.0)
    denominator = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - np.sum(error ** 2) / denominator) if denominator > 0 else float("nan")
    stdr = float(np.std(yp) / np.std(yt)) if np.std(yt) > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "StdR": stdr}


def load_canonical_test() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test_path = PROCESSED / "test_windows.npz"
    mask_path = PROCESSED / "raw_observation_mask.npz"
    require(test_path.is_file(), f"Canonical clean test windows found: {test_path}")
    require(mask_path.is_file(), f"Original timestamp/missingness mask found: {mask_path}")

    with np.load(test_path, allow_pickle=False) as z:
        y_true = np.asarray(z["y_raw"], dtype=np.float32)
        starts = np.asarray(z["starts_global"], dtype=np.int64)

    with np.load(mask_path, allow_pickle=False) as z:
        all_ts = np.asarray(z["timestamps"]).astype("datetime64[ns]")
        observed = np.asarray(z["observed"], dtype=np.uint8).astype(bool)

    require(y_true.shape == (N_TEST, FORECAST), f"Canonical test targets have shape {y_true.shape}")
    require(starts.shape == (N_TEST,), f"Canonical test start indices have shape {starts.shape}")
    require(len(all_ts) == len(observed), "Raw timestamps and observation mask have equal length")

    for idx, start in enumerate(starts):
        start = int(start)
        require_window_ok = observed[start:start + LOOKBACK + FORECAST].all()
        if not require_window_ok:
            raise RuntimeError(f"Canonical test window {idx} contains a missing raw observation")

    target_timestamps = np.stack([
        all_ts[int(s) + LOOKBACK:int(s) + LOOKBACK + FORECAST]
        for s in starts
    ]).astype("datetime64[ns]")
    require(target_timestamps.shape == (N_TEST, FORECAST), "Canonical target timestamps have shape (192, 1440)")
    log_pass("All canonical test windows contain only originally observed readings")
    return y_true, starts, target_timestamps


def compare_saved_metrics(csv_path: Path, seed: int | str, metrics: dict[str, float]) -> dict[str, Any]:
    if not csv_path.is_file():
        return {"saved_metrics_csv": "missing", "saved_metrics_match": None}
    df = pd.read_csv(csv_path)
    if "Seed" not in df.columns:
        return {"saved_metrics_csv": "present_but_no_Seed_column", "saved_metrics_match": None}

    if isinstance(seed, str):
        selected = df[df["Seed"].astype(str).str.lower() == seed.lower()]
    else:
        numeric_seed = pd.to_numeric(df["Seed"], errors="coerce")
        selected = df[numeric_seed == seed]

    if selected.empty:
        return {"saved_metrics_csv": "present_but_seed_row_missing", "saved_metrics_match": None}

    row = selected.iloc[0]
    missing_cols = [m for m in METRIC_NAMES if m not in df.columns]
    if missing_cols:
        return {"saved_metrics_csv": f"present_but_missing_columns:{','.join(missing_cols)}", "saved_metrics_match": None}

    deltas = {m: float(abs(float(row[m]) - metrics[m])) for m in METRIC_NAMES}
    matches = all(delta <= 5e-5 for delta in deltas.values())
    return {
        "saved_metrics_csv": "present",
        "saved_metrics_match": bool(matches),
        "saved_metric_abs_differences": deltas,
    }


def verify_timestamps(model_name: str, folder: Path, canonical_ts: np.ndarray) -> str:
    candidates = [
        "test_target_timestamps.npy",
        "test_target_start_timestamps.npy",
    ]
    path = next((folder / name for name in candidates if (folder / name).is_file()), None)
    if path is None:
        return "no timestamp file present"

    saved = np.load(path, allow_pickle=False)
    saved_dt = saved.astype("datetime64[ns]")

    if saved_dt.shape == canonical_ts.shape:
        if not np.array_equal(saved_dt, canonical_ts):
            raise RuntimeError(f"{model_name}: full per-minute target timestamps do not match canonical UCI timestamps")
        log_pass(f"{model_name}: full per-minute target timestamps match canonical test timestamps")
        return "full 192x1440 timestamps verified"

    expected_starts = canonical_ts[:, 0]
    if saved_dt.shape == (N_TEST,):
        if not np.array_equal(saved_dt, expected_starts):
            raise RuntimeError(f"{model_name}: target-start timestamps do not match canonical UCI timestamps")
        log_pass(f"{model_name}: target-start timestamps match canonical test timestamps")
        return "192 target-start timestamps verified"

    raise RuntimeError(f"{model_name}: unrecognized timestamp shape {saved_dt.shape} in {path.name}")


def main() -> int:
    print("\n====================================================")
    print("CORRECTED UCI CROSS-MODEL PREDICTION AUDIT")
    print("====================================================")
    print(f"Project root: {ROOT}")

    canonical_y, test_starts, canonical_ts = load_canonical_test()
    per_run_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    model_aggregate_rows: list[dict[str, Any]] = []
    truth_reference: np.ndarray | None = None

    for model_name, spec in MODELS.items():
        folder = ROOT / "Results" / "UCI" / spec["directory"]
        print(f"\n--- {model_name} ---")
        require(folder.is_dir(), f"Result folder exists: {spec['directory']}")

        config_path = folder / "run_config.json"
        require(config_path.is_file(), f"{model_name}: run_config.json exists")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if spec["seeded"]:
            require(
                config.get("seeds") == list(SEEDS),
                f"{model_name}: configuration records seeds 0-4",
            )
            require(
                int(config.get("max_optimizer_steps_per_seed", -1)) == 300,
                f"{model_name}: configuration records 300 optimizer steps per seed",
            )
        else:
            require(
                config.get("model_id") == "google/timesfm-2.5-200m-pytorch",
                "TimesFM 2.5: expected pretrained checkpoint is recorded",
            )
            require(
                config.get("full_evaluation_completed") is True,
                "TimesFM 2.5: full zero-shot evaluation is recorded as completed",
            )

        truth_path = folder / spec["truth_file"]
        require(truth_path.is_file(), f"{model_name}: saved y_true_raw.npy exists")
        saved_truth = np.load(truth_path, allow_pickle=False)
        require(saved_truth.shape == canonical_y.shape, f"{model_name}: saved ground truth shape is {saved_truth.shape}")
        truth_equal = bool(np.array_equal(saved_truth, canonical_y))
        require(truth_equal, f"{model_name}: saved ground truth exactly matches canonical clean UCI targets")

        if truth_reference is None:
            truth_reference = saved_truth.copy()
        else:
            require(np.array_equal(saved_truth, truth_reference), f"{model_name}: ground truth exactly matches every other model")

        timestamp_status = verify_timestamps(model_name, folder, canonical_ts)
        metrics_csv = folder / spec["results_csv"]
        require(metrics_csv.is_file(), f"{model_name}: saved metrics CSV exists")
        expected_seeds = SEEDS if spec["seeded"] else ("zero-shot",)
        model_metrics: list[dict[str, float]] = []

        for seed in expected_seeds:
            prediction_name = spec["prediction_pattern"].format(seed=seed)
            pred_path = folder / prediction_name
            require(pred_path.is_file(), f"{model_name} seed {seed}: prediction file exists")
            pred = np.load(pred_path, allow_pickle=False)
            require(pred.shape == canonical_y.shape, f"{model_name} seed {seed}: prediction shape is {pred.shape}")
            require(np.isfinite(pred).all(), f"{model_name} seed {seed}: all predictions are finite")

            metrics = compute_metrics(canonical_y, pred)
            model_metrics.append(metrics)
            saved_check = compare_saved_metrics(metrics_csv, seed, metrics)
            require(
                saved_check.get("saved_metrics_match") is True,
                f"{model_name} seed {seed}: recomputed metrics match saved CSV (tolerance 5e-5); "
                f"details={saved_check}",
            )

            row: dict[str, Any] = {"Model": model_name, "Seed": seed, **metrics}
            row["prediction_file"] = prediction_name
            row["metrics_csv_check"] = saved_check.get("saved_metrics_match")
            per_run_rows.append(row)
            audit_rows.append({
                "Model": model_name,
                "Seed": seed,
                "truth_matches_canonical": truth_equal,
                "prediction_shape": str(tuple(pred.shape)),
                "prediction_finite": bool(np.isfinite(pred).all()),
                "timestamp_check": timestamp_status,
                "saved_metrics_csv": saved_check.get("saved_metrics_csv"),
                "saved_metrics_match_recomputed": saved_check.get("saved_metrics_match"),
            })

        if spec["seeded"]:
            for metric in METRIC_NAMES:
                vals = np.asarray([r[metric] for r in model_metrics], dtype=float)
                model_aggregate_rows.append({
                    "Model": model_name,
                    "Metric": metric,
                    "Mean": float(np.mean(vals)),
                    "SD_sample": float(np.std(vals, ddof=1)),
                    "N_runs": int(len(vals)),
                })
        else:
            for metric in METRIC_NAMES:
                vals = np.asarray([r[metric] for r in model_metrics], dtype=float)
                model_aggregate_rows.append({
                    "Model": model_name,
                    "Metric": metric,
                    "Mean": float(np.mean(vals)),
                    "SD_sample": float("nan"),
                    "N_runs": 1,
                })

        if model_name == "TimesFM 2.5 zero-shot":
            quantile_path = folder / "timesfm25_quantiles.npy"
            require(quantile_path.is_file(), "TimesFM 2.5: saved quantile forecasts exist")
            quantiles = np.load(quantile_path, allow_pickle=False)
            require(
                quantiles.shape == (N_TEST, FORECAST, 10),
                f"TimesFM 2.5: quantile forecast shape is {quantiles.shape}",
            )
            require(np.isfinite(quantiles).all(), "TimesFM 2.5: all saved quantile forecasts are finite")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    per_run_df = pd.DataFrame(per_run_rows)
    audit_df = pd.DataFrame(audit_rows)
    aggregate_df = pd.DataFrame(model_aggregate_rows)

    per_run_df.to_csv(OUTPUT / "recomputed_per_run_metrics.csv", index=False)
    audit_df.to_csv(OUTPUT / "prediction_integrity_audit.csv", index=False)
    aggregate_df.to_csv(OUTPUT / "recomputed_model_summary_long.csv", index=False)

    # Human-readable wide summary for quick comparison.
    wide_rows: list[dict[str, Any]] = []
    for model_name in MODELS:
        subset = aggregate_df[aggregate_df["Model"] == model_name]
        row: dict[str, Any] = {"Model": model_name}
        for _, metric_row in subset.iterrows():
            metric = metric_row["Metric"]
            row[f"{metric}_mean"] = metric_row["Mean"]
            row[f"{metric}_sd"] = metric_row["SD_sample"]
        wide_rows.append(row)
    pd.DataFrame(wide_rows).to_csv(OUTPUT / "recomputed_model_summary_wide.csv", index=False)

    report = {
        "audit": "corrected_uci_cross_model_prediction_integrity",
        "test_windows": N_TEST,
        "forecast_minutes": FORECAST,
        "canonical_truth_shape": list(canonical_y.shape),
        "models_audited": list(MODELS.keys()),
        "seeded_models": [name for name, spec in MODELS.items() if spec["seeded"]],
        "zero_shot_models": [name for name, spec in MODELS.items() if not spec["seeded"]],
        "all_saved_truths_match_canonical": bool(audit_df["truth_matches_canonical"].all()),
        "all_predictions_have_expected_shape_and_are_finite": bool(
            (audit_df["prediction_shape"] == str((N_TEST, FORECAST))).all()
            and audit_df["prediction_finite"].all()
        ),
        "all_saved_metric_csvs_match_recomputed_metrics": bool(
            audit_df["saved_metrics_match_recomputed"].dropna().astype(bool).all()
        ),
        "note": (
            "This audit checks saved prediction integrity and recomputes point metrics. "
            "It does not perform statistical significance tests, resolve overlapping-window dependence, "
            "or establish that all architectures are exact reproductions of their reference libraries."
        ),
        "outputs": [
            "recomputed_per_run_metrics.csv",
            "prediction_integrity_audit.csv",
            "recomputed_model_summary_long.csv",
            "recomputed_model_summary_wide.csv",
        ],
    }
    (OUTPUT / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n====================================================")
    print("UCI PREDICTION AUDIT COMPLETED")
    print("====================================================")
    print("All saved truths match the canonical clean targets:", report["all_saved_truths_match_canonical"])
    print("All prediction arrays have the expected shape and finite values:", report["all_predictions_have_expected_shape_and_are_finite"])
    print("All saved metric CSVs match recomputed metrics:", report["all_saved_metric_csvs_match_recomputed_metrics"])
    print("Audit outputs:", OUTPUT.resolve())
    print("\nThis audit is not a significance test. Statistical inference is a separate next step.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nAUDIT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
