#!/usr/bin/env python3
"""Hierarchical paired moving-block bootstrap for corrected UCI forecasts.

Primary question: does each alternative have lower/higher test loss than the
Vanilla Transformer reference on the SAME 192 clean test forecast windows?

Design:
- Per-forecast-window MSE and MAE are the resampling units.
- Candidate origins are daily (stride=1,440 minutes). The test manifest
  preserves all candidate slots, including the 23 rejected windows.
- A non-circular moving-block bootstrap resamples contiguous calendar blocks
  of candidate-origin slots; rejected slots remain absent in each resample.
- For five-seed models, model seeds are independently resampled with
  replacement in each bootstrap replicate to reflect training randomness.
- TimesFM is a single fixed zero-shot forecast; it is not assigned fake seeds.
- Five comparisons against Vanilla Transformer are Holm-adjusted separately
  for RMSE and MAE, at each block length.
- Primary block length is 7 daily candidate origins; 3 and 14 days are
  sensitivity analyses.

The bootstrap p-value is a two-sided centered-bootstrap approximation.
This is not a Diebold-Mariano test, MCS procedure, or cross-household test.
It conditions on the one UCI household and chronological test period.

Run from the DAT_Reproducible_V2 project root:
    python statistical_analysis_uci.py
Optional:
    python statistical_analysis_uci.py --bootstrap-reps 10000

Manifest compatibility:
- Accepts either `candidate_window` (current project manifest) or
  `candidate_window_index` (older generator schema).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
UCI_PROCESSED = ROOT / "Data" / "UCI" / "processed"
RESULTS_ROOT = ROOT / "Results" / "UCI"
OUTPUT = RESULTS_ROOT / "Corrected_V2_Stats"

N_TEST = 192
FORECAST = 1440
SEEDS = (0, 1, 2, 3, 4)
REFERENCE = "Vanilla Transformer"
BLOCK_LENGTHS = (3, 7, 14)
DEFAULT_REPS = 5000
RANDOM_SEED = 20261101

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "Vanilla Transformer": {
        "directory": "Corrected_V2_1_VanillaTransformer",
        "prediction_pattern": "VanillaTransformer_seed{seed}_predictions.npy",
        "seeded": True,
    },
    "DAT": {
        "directory": "Corrected_V2_DAT",
        "prediction_pattern": "DAT_seed{seed}_predictions.npy",
        "seeded": True,
    },
    "LSTM": {
        "directory": "Corrected_V2_LSTM",
        "prediction_pattern": "LSTM_seed{seed}_predictions.npy",
        "seeded": True,
    },
    "DLinear": {
        "directory": "Corrected_V2_DLinear",
        "prediction_pattern": "DLinear_seed{seed}_predictions.npy",
        "seeded": True,
    },
    "PatchTST-style": {
        "directory": "Corrected_V2_PatchTST",
        "prediction_pattern": "PatchTST_seed{seed}_predictions.npy",
        "seeded": True,
    },
    "TimesFM 2.5 zero-shot": {
        "directory": "Corrected_V2_TimesFM25",
        "prediction_pattern": "timesfm25_predictions.npy",
        "seeded": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bootstrap-reps", type=int, default=DEFAULT_REPS,
        help=f"Bootstrap replicates per block length (default: {DEFAULT_REPS}).",
    )
    parser.add_argument(
        "--random-seed", type=int, default=RANDOM_SEED,
        help=f"Bootstrap RNG seed (default: {RANDOM_SEED}).",
    )
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
    print(f"[PASS] {message}")


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: truth {yt.shape}, prediction {yp.shape}")
    if not np.isfinite(yt).all() or not np.isfinite(yp).all():
        raise ValueError("Non-finite values found in truth/prediction")
    err = yp - yt
    nonzero = np.abs(yt) > 1e-8
    denom = np.sum((yt - yt.mean()) ** 2)
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE": float(np.mean(np.abs(err[nonzero] / yt[nonzero])) * 100.0),
        "R2": float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float("nan"),
        "StdR": float(np.std(yp) / np.std(yt)) if np.std(yt) > 0 else float("nan"),
    }


def load_manifest_slots() -> tuple[np.ndarray, int]:
    manifest_path = UCI_PROCESSED / "window_manifest.csv"
    require(manifest_path.is_file(), f"UCI window manifest exists: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    test = manifest.loc[manifest["split"].astype(str).str.lower() == "test"].copy()
    require(len(test) > N_TEST, f"Test manifest retains accepted and rejected candidate slots ({len(test)} rows)")

    # The prepared UCI manifest in this project names this field
    # `candidate_window`; an earlier generator version used
    # `candidate_window_index`. Accept either schema, but never infer
    # candidate order from accepted rows alone because rejected slots matter.
    candidate_col = (
        "candidate_window_index"
        if "candidate_window_index" in test.columns
        else "candidate_window"
        if "candidate_window" in test.columns
        else None
    )
    if candidate_col is None:
        raise RuntimeError(
            "UCI test manifest must contain either 'candidate_window' "
            "or 'candidate_window_index'. Found columns: "
            + ", ".join(map(str, test.columns))
        )
    required_cols = {"accepted", "accepted_window_index"}
    missing_cols = required_cols.difference(test.columns)
    if missing_cols:
        raise RuntimeError(
            "UCI test manifest is missing required columns: "
            + ", ".join(sorted(missing_cols))
        )

    accepted_raw = test["accepted"].astype(str).str.strip().str.lower()
    accepted = accepted_raw.isin({"true", "1", "yes"})
    test["accepted_bool"] = accepted
    test["candidate_window_index"] = pd.to_numeric(
        test[candidate_col], errors="raise"
    ).astype(int)
    test["accepted_window_index_num"] = pd.to_numeric(
        test["accepted_window_index"], errors="coerce"
    )

    test = test.sort_values("candidate_window_index").reset_index(drop=True)
    slots = test["candidate_window_index"].to_numpy(dtype=int)
    require(
        np.array_equal(slots, np.arange(len(test))),
        f"Test candidate slots are consecutive using manifest column '{candidate_col}' and preserve rejected origins",
    )

    accepted_rows = test.loc[test["accepted_bool"]].sort_values("accepted_window_index_num")
    accepted_indices = accepted_rows["accepted_window_index_num"].to_numpy(dtype=int)
    require(len(accepted_indices) == N_TEST, f"Exactly {N_TEST} accepted test windows are in the manifest")
    require(np.array_equal(accepted_indices, np.arange(N_TEST)), "Accepted-window manifest order matches prediction row order")

    # Slot -> accepted prediction row. -1 means the candidate slot was rejected.
    slot_to_row = np.full(len(test), -1, dtype=int)
    for _, row in accepted_rows.iterrows():
        slot = int(row["candidate_window_index"])
        accepted_row = int(row["accepted_window_index_num"])
        slot_to_row[slot] = accepted_row

    n_rejected = int((slot_to_row < 0).sum())
    print(f"[INFO] Candidate daily origins: {len(test)}; accepted: {N_TEST}; rejected: {n_rejected}")
    return slot_to_row, len(test)


def load_predictions(y_true: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, float]]]]:
    predictions: dict[str, np.ndarray] = {}
    descriptive: dict[str, list[dict[str, float]]] = {}

    audit_path = RESULTS_ROOT / "Corrected_V2_Audit" / "audit_report.json"
    require(audit_path.is_file(), "Completed cross-model prediction-integrity audit report exists")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(audit.get("all_saved_truths_match_canonical") is True, "Previous audit confirms all saved truths match canonical targets")
    require(audit.get("all_predictions_have_expected_shape_and_are_finite") is True, "Previous audit confirms prediction shapes and finite values")
    require(audit.get("all_saved_metric_csvs_match_recomputed_metrics") is True, "Previous audit confirms saved metrics match recomputation")

    for model_name, spec in MODEL_SPECS.items():
        folder = RESULTS_ROOT / spec["directory"]
        require(folder.is_dir(), f"Results folder exists for {model_name}")
        truth_file = folder / "y_true_raw.npy"
        require(truth_file.is_file(), f"{model_name}: saved ground truth exists")
        saved_truth = np.load(truth_file, allow_pickle=False)
        require(saved_truth.shape == y_true.shape and np.array_equal(saved_truth, y_true), f"{model_name}: ground truth equals canonical clean test targets")

        if spec["seeded"]:
            arrays = []
            rows = []
            for seed in SEEDS:
                path = folder / spec["prediction_pattern"].format(seed=seed)
                require(path.is_file(), f"{model_name} seed {seed}: prediction file exists")
                pred = np.load(path, allow_pickle=False)
                require(pred.shape == y_true.shape, f"{model_name} seed {seed}: prediction shape is {pred.shape}")
                require(np.isfinite(pred).all(), f"{model_name} seed {seed}: predictions are finite")
                arrays.append(np.asarray(pred, dtype=np.float64))
                rows.append(metric_values(y_true, pred))
            predictions[model_name] = np.stack(arrays, axis=0)
            descriptive[model_name] = rows
        else:
            path = folder / spec["prediction_pattern"]
            require(path.is_file(), f"{model_name}: zero-shot prediction file exists")
            pred = np.load(path, allow_pickle=False)
            require(pred.shape == y_true.shape, f"{model_name}: prediction shape is {pred.shape}")
            require(np.isfinite(pred).all(), f"{model_name}: predictions are finite")
            predictions[model_name] = np.asarray(pred, dtype=np.float64)[None, ...]
            descriptive[model_name] = [metric_values(y_true, pred)]

    return predictions, descriptive


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in original order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p[idx])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted.tolist()


def moving_block_sample_rows(
    rng: np.random.Generator,
    slot_to_row: np.ndarray,
    block_length: int,
) -> np.ndarray:
    n_slots = len(slot_to_row)
    n_blocks = int(math.ceil(n_slots / block_length))
    max_start = n_slots - block_length
    starts = rng.integers(0, max_start + 1, size=n_blocks)
    sampled_slots = np.concatenate([
        np.arange(start, start + block_length, dtype=int)
        for start in starts
    ])[:n_slots]
    accepted_rows = slot_to_row[sampled_slots]
    rows = accepted_rows[accepted_rows >= 0]
    if len(rows) == 0:
        raise RuntimeError("A bootstrap replicate contained no accepted test windows")
    return rows


def resampled_model_losses(
    rng: np.random.Generator,
    mse: np.ndarray,
    mae: np.ndarray,
    sampled_rows: np.ndarray,
) -> tuple[float, float]:
    n_seeds = mse.shape[0]
    if n_seeds == 1:
        seed_indices = np.array([0], dtype=int)
    else:
        seed_indices = rng.integers(0, n_seeds, size=n_seeds)
    # Resample training runs, then retain their paired losses over sampled calendar blocks.
    mean_mse = float(np.mean(mse[seed_indices][:, sampled_rows]))
    mean_mae = float(np.mean(mae[seed_indices][:, sampled_rows]))
    return float(np.sqrt(mean_mse)), mean_mae


def centered_bootstrap_p(observed: float, boot: np.ndarray) -> float:
    centered = boot - np.mean(boot)
    exceed = int(np.count_nonzero(np.abs(centered) >= abs(observed)))
    return float((exceed + 1) / (len(boot) + 1))


def run_block_bootstrap(
    predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    slot_to_row: np.ndarray,
    n_candidate_slots: int,
    reps: int,
    random_seed: int,
    block_length: int,
) -> list[dict[str, Any]]:
    # Per-run, per-window loss arrays: (seed, forecast_window).
    window_losses: dict[str, dict[str, np.ndarray]] = {}
    for model, pred in predictions.items():
        err = pred - y_true[None, :, :]
        window_losses[model] = {
            "MSE": np.mean(err ** 2, axis=2),
            "MAE": np.mean(np.abs(err), axis=2),
        }

    competitors = [name for name in MODEL_SPECS if name != REFERENCE]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(random_seed + block_length * 1009)
    print(f"\n[INFO] Bootstrap block length={block_length} daily candidate origins; replicates={reps:,}")

    anchor_mse = window_losses[REFERENCE]["MSE"]
    anchor_mae = window_losses[REFERENCE]["MAE"]

    for competitor in competitors:
        comp_mse = window_losses[competitor]["MSE"]
        comp_mae = window_losses[competitor]["MAE"]

        observed_anchor_rmse = float(np.sqrt(np.mean(anchor_mse)))
        observed_comp_rmse = float(np.sqrt(np.mean(comp_mse)))
        observed_anchor_mae = float(np.mean(anchor_mae))
        observed_comp_mae = float(np.mean(comp_mae))
        observed_rmse_diff = observed_comp_rmse - observed_anchor_rmse
        observed_mae_diff = observed_comp_mae - observed_anchor_mae

        boot_rmse = np.empty(reps, dtype=np.float64)
        boot_mae = np.empty(reps, dtype=np.float64)

        for b in range(reps):
            sampled_rows = moving_block_sample_rows(rng, slot_to_row, block_length)
            anchor_rmse, anchor_mae_value = resampled_model_losses(
                rng, anchor_mse, anchor_mae, sampled_rows
            )
            comp_rmse_value, comp_mae_value = resampled_model_losses(
                rng, comp_mse, comp_mae, sampled_rows
            )
            boot_rmse[b] = comp_rmse_value - anchor_rmse
            boot_mae[b] = comp_mae_value - anchor_mae_value

        rmse_p = centered_bootstrap_p(observed_rmse_diff, boot_rmse)
        mae_p = centered_bootstrap_p(observed_mae_diff, boot_mae)

        for metric, observed_diff, boot, p_raw, anchor_value, comp_value in (
            ("RMSE", observed_rmse_diff, boot_rmse, rmse_p, observed_anchor_rmse, observed_comp_rmse),
            ("MAE", observed_mae_diff, boot_mae, mae_p, observed_anchor_mae, observed_comp_mae),
        ):
            ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
            rows.append({
                "block_length_candidate_days": int(block_length),
                "Comparison": f"{competitor} - {REFERENCE}",
                "Model": competitor,
                "Reference": REFERENCE,
                "Metric": metric,
                "Reference_estimate": anchor_value,
                "Model_estimate": comp_value,
                "Difference_model_minus_reference": observed_diff,
                "Bootstrap_CI_95_low": float(ci_low),
                "Bootstrap_CI_95_high": float(ci_high),
                "p_value_centered_bootstrap": p_raw,
                "bootstrap_replicates": int(reps),
                "test_candidate_slots": int(n_candidate_slots),
                "accepted_test_windows": int(N_TEST),
                "seed_resampling": "resample 5 training seeds independently; TimesFM fixed single zero-shot run",
            })

        print(
            f"  {competitor}: ΔRMSE={observed_rmse_diff:+.5f} "
            f"(95% CI {np.quantile(boot_rmse, .025):+.5f}, {np.quantile(boot_rmse, .975):+.5f}); "
            f"ΔMAE={observed_mae_diff:+.5f}"
        )

    # Holm correction across five model-vs-reference comparisons separately per metric.
    for metric in ("RMSE", "MAE"):
        indices = [i for i, row in enumerate(rows) if row["Metric"] == metric]
        adjusted = holm_adjust([rows[i]["p_value_centered_bootstrap"] for i in indices])
        for i, value in zip(indices, adjusted):
            rows[i]["p_value_holm_adjusted"] = value
            rows[i]["holm_family_size"] = len(indices)
            rows[i]["significant_holm_0_05"] = bool(value < 0.05)
    return rows


def main() -> int:
    args = parse_args()
    if args.bootstrap_reps < 1000:
        raise ValueError("Use at least 1,000 bootstrap replicates; default is 5,000.")

    print("\n====================================================")
    print("CORRECTED UCI HIERARCHICAL BLOCK-BOOTSTRAP ANALYSIS")
    print("====================================================")
    print(f"Project root: {ROOT}")
    print(f"Bootstrap replicates per block length: {args.bootstrap_reps:,}")

    canonical_path = UCI_PROCESSED / "test_windows.npz"
    require(canonical_path.is_file(), "Canonical corrected UCI test windows exist")
    with np.load(canonical_path, allow_pickle=False) as data:
        y_true = np.asarray(data["y_raw"], dtype=np.float64)
        starts = np.asarray(data["starts_global"], dtype=np.int64)
    require(y_true.shape == (N_TEST, FORECAST), f"Canonical targets shape is {y_true.shape}")
    require(starts.shape == (N_TEST,), "Canonical test start indices have expected length")

    slot_to_row, n_candidate_slots = load_manifest_slots()
    predictions, descriptive = load_predictions(y_true)

    # Save recomputed seed-level descriptive metrics. These are not p-values.
    desc_rows: list[dict[str, Any]] = []
    for model, model_rows in descriptive.items():
        for seed_idx, metrics in enumerate(model_rows):
            seed_label: int | str = seed_idx if MODEL_SPECS[model]["seeded"] else "zero-shot"
            desc_rows.append({"Model": model, "Seed": seed_label, **metrics})
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(OUTPUT / "seed_level_recomputed_metrics.csv", index=False) if OUTPUT.exists() else None

    # Keep an archive of all per-window losses for follow-up figures and tests.
    loss_archive: dict[str, np.ndarray] = {}
    for model, pred in predictions.items():
        error = pred - y_true[None, :, :]
        loss_archive[f"{model}_window_mse"] = np.mean(error ** 2, axis=2)
        loss_archive[f"{model}_window_mae"] = np.mean(np.abs(error), axis=2)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    desc_df.to_csv(OUTPUT / "seed_level_recomputed_metrics.csv", index=False)
    np.savez_compressed(OUTPUT / "per_window_losses_by_seed.npz", **loss_archive)

    all_rows: list[dict[str, Any]] = []
    for block_length in BLOCK_LENGTHS:
        rows = run_block_bootstrap(
            predictions=predictions,
            y_true=y_true,
            slot_to_row=slot_to_row,
            n_candidate_slots=n_candidate_slots,
            reps=args.bootstrap_reps,
            random_seed=args.random_seed,
            block_length=block_length,
        )
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUTPUT / "paired_block_bootstrap_vs_vanilla.csv", index=False)
    results_df.loc[results_df["block_length_candidate_days"] == 7].to_csv(
        OUTPUT / "primary_7day_bootstrap_holm_results.csv", index=False
    )
    results_df.loc[results_df["block_length_candidate_days"].isin((3, 14))].to_csv(
        OUTPUT / "block_length_sensitivity_3_and_14_days.csv", index=False
    )

    config = {
        "analysis": "paired_hierarchical_moving_block_bootstrap",
        "reference_model": REFERENCE,
        "comparisons": [f"{name} - {REFERENCE}" for name in MODEL_SPECS if name != REFERENCE],
        "primary_outcomes": ["RMSE", "MAE"],
        "block_lengths_candidate_days": list(BLOCK_LENGTHS),
        "primary_block_length_candidate_days": 7,
        "bootstrap_replicates_per_block_length": int(args.bootstrap_reps),
        "bootstrap_rng_seed": int(args.random_seed),
        "test_windows_accepted": N_TEST,
        "test_candidate_daily_origin_slots": int(n_candidate_slots),
        "rejected_candidate_slots_preserved_as_missing": int((slot_to_row < 0).sum()),
        "forecast_horizon_minutes": FORECAST,
        "forecast_origin_stride_minutes": 1440,
        "seed_handling": (
            "For each 5-seed model, independently resample its five seed runs with replacement; "
            "for TimesFM, retain the single fixed zero-shot prediction."
        ),
        "temporal_handling": (
            "Non-circular moving blocks of daily candidate-origin slots; the complete test manifest "
            "is the time index, so rejected candidate windows remain absent rather than being compressed away."
        ),
        "test_statistic": (
            "Difference in pooled per-window RMSE or MAE, model minus Vanilla Transformer; "
            "RMSE is computed as square root of the mean per-window MSE across sampled seeds/windows."
        ),
        "p_value_method": "two-sided centered hierarchical bootstrap approximation",
        "multiplicity": "Holm correction across five model-vs-reference comparisons separately for RMSE and MAE, within each block length",
        "limitations": [
            "This is not a Diebold-Mariano test or a Model Confidence Set procedure.",
            "Inference is conditional on one UCI household and one chronological test period.",
            "Only comparisons against Vanilla Transformer are included; this is not an all-pairs model ranking test.",
            "TimesFM has one fixed zero-shot run and no training-seed uncertainty estimate.",
            "The PatchTST and iTransformer files are custom style adaptations, not exact reproductions of the reference library implementations; iTransformer is excluded from the main set because the UCI input is univariate.",
            "MAPE is descriptive only and is not used for significance testing because values near zero can make it unstable.",
        ],
    }
    (OUTPUT / "statistical_analysis_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Human-readable report.
    primary = results_df.loc[results_df["block_length_candidate_days"] == 7].copy()
    lines = [
        "CORRECTED UCI STATISTICAL ANALYSIS",
        "==================================",
        f"Reference: {REFERENCE}",
        "Primary block length: 7 daily candidate origins",
        f"Bootstrap replicates per block length: {args.bootstrap_reps:,}",
        f"Accepted test windows: {N_TEST}; candidate daily slots: {n_candidate_slots}; rejected slots: {(slot_to_row < 0).sum()}",
        "Negative difference means the comparator has lower error than the Vanilla Transformer.",
        "",
        "PRIMARY RESULTS (7-day blocks; Holm correction within metric family)",
        "---------------------------------------------------------------------",
    ]
    for _, row in primary.iterrows():
        lines.append(
            f"{row['Metric']:4s} | {row['Model']:<22s} | "
            f"difference={row['Difference_model_minus_reference']:+.6f} | "
            f"95% CI [{row['Bootstrap_CI_95_low']:+.6f}, {row['Bootstrap_CI_95_high']:+.6f}] | "
            f"p={row['p_value_centered_bootstrap']:.5f} | "
            f"Holm p={row['p_value_holm_adjusted']:.5f} | "
            f"significant={row['significant_holm_0_05']}"
        )
    lines.extend([
        "",
        "INTERPRETATION GUIDE",
        "--------------------",
        "- A negative comparator-minus-reference difference favors the comparator.",
        "- A 95% interval excluding zero is evidence of a difference under this bootstrap design.",
        "- Use Holm-adjusted p-values for the five comparisons within each metric family.",
        "- Compare 3-day and 14-day sensitivity results before making strong claims.",
        "- These tests do not establish cross-household or cross-dataset generalization.",
        "- This analysis is not a DM test or MCS; do not label it as either in the paper.",
        "",
        "Files written:",
        "- seed_level_recomputed_metrics.csv",
        "- per_window_losses_by_seed.npz",
        "- paired_block_bootstrap_vs_vanilla.csv",
        "- primary_7day_bootstrap_holm_results.csv",
        "- block_length_sensitivity_3_and_14_days.csv",
        "- statistical_analysis_config.json",
        "- statistical_analysis_report.txt",
    ])
    (OUTPUT / "statistical_analysis_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n====================================================")
    print("UCI BLOCK-BOOTSTRAP ANALYSIS COMPLETED")
    print("====================================================")
    print(primary[["Metric", "Model", "Difference_model_minus_reference", "Bootstrap_CI_95_low", "Bootstrap_CI_95_high", "p_value_centered_bootstrap", "p_value_holm_adjusted", "significant_holm_0_05"]].to_string(index=False))
    print(f"\nOutputs saved under: {OUTPUT.resolve()}")
    print("Review 3-day and 14-day sensitivity results before writing inferential claims.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nSTATISTICAL ANALYSIS FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
