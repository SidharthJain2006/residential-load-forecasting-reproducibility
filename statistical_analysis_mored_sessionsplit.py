#!/usr/bin/env python3
"""Recording-session-clustered comparison for corrected MORED V3 results.

Run from the project root:
    python statistical_analysis_mored_sessionsplit.py

The script does NOT treat minute-level errors or overlapping forecast windows as
independent observations. It first computes metrics within each of the four original raw recording sessions (IDs 6-9). Five-seed model metrics
are averaged within each raw recording session; random seeds do not inflate the inferential n.
Pairwise comparisons against VanillaTransformer use an exact two-sided sign test
across independent original raw recording sessions, Holm correction across RMSE/MAE comparisons,
and a recording-session-cluster bootstrap percentile interval for the mean recording-session-level
difference. With only four original raw recording sessions, inference is intentionally labelled
low-powered and exploratory.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import binomtest, friedmanchisquare

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data" / "MORED" / "Corrected_V3_SessionSplit"
RES = ROOT / "Results" / "MORED"
OUT = RES / "Corrected_V3_SessionSplit_Statistics"
SEEDS = (0, 1, 2, 3, 4)
N_BOOT = 20000
BOOT_SEED = 20261101
MODELS = {
    "VanillaTransformer": ("Corrected_V3_SessionSplit_Vanilla_DAT", "VanillaTransformer", "mored_corrected_results.csv", "VanillaTransformer"),
    "DAT": ("Corrected_V3_SessionSplit_Vanilla_DAT", "DAT", "mored_corrected_results.csv", "DAT"),
    "LSTM": ("Corrected_V3_SessionSplit_LSTM", "LSTM", "lstm_corrected_results.csv", "LSTM"),
    "DLinear": ("Corrected_V3_SessionSplit_DLinear", "DLinear", "dlinear_corrected_results.csv", "DLinear"),
    "PatchTST-style": ("Corrected_V3_SessionSplit_PatchTST", "PatchTST", "patchtst_corrected_results.csv", "PatchTST-style"),
}
BASELINE = "VanillaTransformer"
COMPARE = ["DAT", "LSTM", "DLinear", "PatchTST-style", "TimesFM 2.5 zero-shot"]
METRICS = ("RMSE", "MAE", "MAPE", "R2", "StdR")


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise RuntimeError(msg)


def load_array(path: Path) -> np.ndarray:
    require(path.is_file(), f"Missing file: {path}")
    return np.load(path, allow_pickle=False)


def calculate_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y, dtype=np.float64).reshape(-1)
    yp = np.asarray(p, dtype=np.float64).reshape(-1)
    require(yt.shape == yp.shape and np.isfinite(yt).all() and np.isfinite(yp).all(),
            "Invalid truth/prediction values while calculating metrics")
    nonzero = np.abs(yt) > 1e-8
    return {
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE": float(mean_absolute_error(yt, yp)),
        "MAPE": float(np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])) * 100.0),
        "R2": float(r2_score(yt, yp)),
        "StdR": float(np.std(yp) / (np.std(yt) + 1e-12)),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise adjustment, returned in original order."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * float(p_values[idx]))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted.tolist()


def cluster_bootstrap_ci(diffs: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    n = len(diffs)
    draws = rng.integers(0, n, size=(N_BOOT, n))
    boot_means = diffs[draws].mean(axis=1)
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    return float(lo), float(hi)


def sign_test(diffs: np.ndarray) -> tuple[int, int, int, float]:
    tol = 1e-12
    pos = int(np.sum(diffs > tol))
    neg = int(np.sum(diffs < -tol))
    ties = int(len(diffs) - pos - neg)
    n = pos + neg
    p = float(binomtest(min(pos, neg), n=n, p=0.5, alternative="two-sided").pvalue) if n else 1.0
    return pos, neg, ties, p


def main() -> int:
    meta_path = DATA / "preprocessing_metadata.json"
    manifest_path = DATA / "window_manifest.csv"
    test_npz = DATA / "test_windows.npz"
    for p in (meta_path, manifest_path, test_npz):
        require(p.is_file(), f"Missing corrected MORED file: {p}")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    require(metadata.get("experiment") == "mored_corrected_v3_session_split_no_interpolation_raw_count_filter",
            "This is not the expected corrected no-interpolation MORED dataset")
    policy = str(metadata.get("minute_aggregation", {}).get("missing_sample_policy", "")).lower()
    require("no interpolation" in policy, "Metadata does not confirm no interpolation")

    with np.load(test_npz, allow_pickle=False) as z:
        test = {k: z[k].copy() for k in z.files}
    y_true = np.asarray(test["y_raw"], dtype=np.float64)
    if y_true.ndim == 3 and y_true.shape[-1] == 1:
        y_true = y_true[..., 0]
    require(y_true.shape == (359, 1440), f"Unexpected y_raw shape: {y_true.shape}")
    starts = np.asarray(test["starts_global"], dtype=np.int64)
    require(starts.shape == (len(y_true),), "starts_global shape mismatch")

    manifest = pd.read_csv(manifest_path)
    mtest = manifest.loc[manifest["split"].astype(str).eq("test")].copy()
    mtest = mtest.sort_values("global_start_minute_row").reset_index(drop=True)
    require(len(mtest) == len(y_true), f"Manifest test count {len(mtest)} != {len(y_true)}")
    require(np.array_equal(mtest["global_start_minute_row"].to_numpy(dtype=np.int64), starts),
            "Manifest window ordering/starts do not match test_windows.npz")
    require("raw_segment_id" in mtest.columns, "Manifest has no raw_segment_id field")
    segment_ids = mtest["raw_segment_id"].to_numpy()
    unique_segments = list(pd.unique(segment_ids))
    n_segments = len(unique_segments)
    require(unique_segments == [6, 7, 8, 9], f"Expected raw test recording sessions [6, 7, 8, 9], found {unique_segments}")
    print(f"Corrected MORED test windows: {len(y_true)} x {y_true.shape[1]} minutes")
    print(f"Original raw recording-session clusters: {n_segments}")
    print("Within-session overlapping windows are aggregated; they are not treated as independent.")

    # Per-seed, per-segment metric records. For each segment, all target minutes
    # from all forecast origins in that segment are pooled to yield one score.
    segment_rows: list[dict[str, Any]] = []
    by_model_segment: dict[str, dict[Any, dict[str, float]]] = {}
    seed_variability_rows: list[dict[str, Any]] = []

    for model, (folder, prefix, csv_name, csv_model_name) in MODELS.items():
        model_dir = RES / folder
        saved = pd.read_csv(model_dir / csv_name)
        subset = saved.loc[saved["Model"].astype(str).eq(csv_model_name)]
        require(set(subset["Seed"].astype(int)) == set(SEEDS),
                f"{model}: expected seeds {SEEDS}; found {subset['Seed'].tolist()}")
        truth_saved = load_array(model_dir / "y_true_raw.npy")
        if truth_saved.ndim == 3 and truth_saved.shape[-1] == 1:
            truth_saved = truth_saved[..., 0]
        require(np.array_equal(truth_saved, y_true), f"{model}: y_true differs from canonical data")
        pred_by_seed: dict[int, np.ndarray] = {}
        for seed in SEEDS:
            pred = load_array(model_dir / f"{prefix}_seed{seed}_predictions.npy")
            require(pred.shape == y_true.shape and np.isfinite(pred).all(),
                    f"{model} seed {seed}: invalid predictions shape/content")
            pred_by_seed[seed] = pred.astype(np.float64)
            saved_row = subset.loc[subset["Seed"].astype(int).eq(seed)].iloc[0]
            full = calculate_metrics(y_true, pred)
            for metric in METRICS:
                require(abs(float(saved_row[metric]) - full[metric]) <= max(1e-6, abs(full[metric]) * 1e-7),
                        f"{model} seed {seed}: saved {metric} fails recomputation")
            for seg in unique_segments:
                ix = np.flatnonzero(segment_ids == seg)
                sm = calculate_metrics(y_true[ix], pred[ix])
                segment_rows.append({"Model": model, "Seed": seed, "Raw_recording_session_ID": seg,
                                     "N_windows": len(ix), **sm})

        model_seg: dict[Any, dict[str, float]] = {}
        for seg in unique_segments:
            ix = np.flatnonzero(segment_ids == seg)
            seed_metric_dicts = [calculate_metrics(y_true[ix], pred_by_seed[s][ix]) for s in SEEDS]
            model_seg[seg] = {metric: float(np.mean([d[metric] for d in seed_metric_dicts])) for metric in METRICS}
            for metric in METRICS:
                seed_vals = [d[metric] for d in seed_metric_dicts]
                seed_variability_rows.append({"Model": model, "Raw_recording_session_ID": seg, "Metric": metric,
                                              "Seed_mean": float(np.mean(seed_vals)),
                                              "Seed_sd": float(np.std(seed_vals, ddof=1))})
        by_model_segment[model] = model_seg

    # TimesFM: one fixed zero-shot inference run, no random-seed replication.
    tf_dir = RES / "Corrected_V3_SessionSplit_TimesFM25"
    tf_pred = load_array(tf_dir / "TimesFM25_predictions_raw.npy")
    tf_truth = load_array(tf_dir / "TimesFM25_y_true_raw.npy")
    if tf_truth.ndim == 3 and tf_truth.shape[-1] == 1:
        tf_truth = tf_truth[..., 0]
    require(tf_pred.shape == y_true.shape and np.isfinite(tf_pred).all(), "TimesFM predictions malformed")
    require(np.array_equal(tf_truth, y_true), "TimesFM ground truth differs from canonical data")
    tf_saved = pd.read_csv(tf_dir / "TimesFM25_metrics.csv")
    tf_full = calculate_metrics(y_true, tf_pred)
    require(len(tf_saved) == 1, "TimesFM metrics file should have exactly one row")
    for metric in METRICS:
        require(abs(float(tf_saved.iloc[0][metric]) - tf_full[metric]) <= max(1e-6, abs(tf_full[metric]) * 1e-7),
                f"TimesFM saved {metric} fails recomputation")
    tf_seg: dict[Any, dict[str, float]] = {}
    for seg in unique_segments:
        ix = np.flatnonzero(segment_ids == seg)
        sm = calculate_metrics(y_true[ix], tf_pred[ix])
        tf_seg[seg] = sm
        segment_rows.append({"Model": "TimesFM 2.5 zero-shot", "Seed": "zero-shot", "Raw_recording_session_ID": seg,
                             "N_windows": len(ix), **sm})
    by_model_segment["TimesFM 2.5 zero-shot"] = tf_seg

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(segment_rows).to_csv(OUT / "mored_recording_session_level_metrics.csv", index=False)
    pd.DataFrame(seed_variability_rows).to_csv(OUT / "mored_seed_variability_by_recording_session.csv", index=False)

    # Macro-average across independent segments, plus descriptive SD between segments.
    summary_rows = []
    model_order = list(MODELS) + ["TimesFM 2.5 zero-shot"]
    for model in model_order:
        row: dict[str, Any] = {"Model": model, "N_independent_recording_sessions": n_segments}
        for metric in METRICS:
            vals = np.array([by_model_segment[model][seg][metric] for seg in unique_segments], dtype=float)
            row[f"{metric}_recording_session_macro_mean"] = float(vals.mean())
            row[f"{metric}_between_recording_session_sd"] = float(vals.std(ddof=1))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUT / "mored_recording_session_macro_summary.csv", index=False)

    # Pairwise comparisons, each difference is alternative minus baseline, so
    # positive RMSE/MAE means alternative is worse. Paired sign test across segments.
    pair_rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for metric in ("RMSE", "MAE"):
        for model in COMPARE:
            diffs = np.array([by_model_segment[model][seg][metric] - by_model_segment[BASELINE][seg][metric]
                              for seg in unique_segments], dtype=float)
            pos, neg, ties, p = sign_test(diffs)
            lo, hi = cluster_bootstrap_ci(diffs, np.random.default_rng(BOOT_SEED + len(pair_rows)))
            p_values.append(p)
            pair_rows.append({
                "Metric": metric, "Comparison": f"{model} - {BASELINE}",
                "Mean_segment_difference": float(diffs.mean()),
                "Median_segment_difference": float(np.median(diffs)),
                "Cluster_bootstrap_95CI_low": lo, "Cluster_bootstrap_95CI_high": hi,
                "Sessions_alternative_worse": pos, "Sessions_alternative_better": neg,
                "Tied_sessions": ties, "N_recording_sessions": n_segments,
                "Exact_two_sided_sign_test_p": p,
            })
    adjusted = holm_adjust(p_values)
    for row, p_adj in zip(pair_rows, adjusted):
        row["Holm_adjusted_p_10_tests"] = p_adj
        row["Significant_Holm_0.05"] = bool(p_adj < 0.05)
    pd.DataFrame(pair_rows).to_csv(OUT / "mored_pairwise_vs_transformer_recording_session_sign_tests.csv", index=False)

    # Friedman repeated-block omnibus test across the six model macro-segment scores.
    omnibus_rows = []
    for metric in ("RMSE", "MAE"):
        arrays = [np.array([by_model_segment[m][seg][metric] for seg in unique_segments], dtype=float)
                  for m in model_order]
        try:
            stat, p = friedmanchisquare(*arrays)
            kendall_w = float(stat / (n_segments * (len(model_order) - 1))) if n_segments and len(model_order) > 1 else np.nan
            omnibus_rows.append({"Metric": metric, "Test": "Friedman repeated-block (exploratory)",
                                 "Statistic": float(stat), "p_value_approx": float(p),
                                 "Kendall_W": kendall_w, "N_recording_sessions": n_segments,
                                 "N_models": len(model_order)})
        except Exception as exc:
            omnibus_rows.append({"Metric": metric, "Test": "Friedman repeated-block (exploratory)",
                                 "Statistic": np.nan, "p_value_approx": np.nan,
                                 "Kendall_W": np.nan, "N_recording_sessions": n_segments,
                                 "N_models": len(model_order), "Note": str(exc)})
    pd.DataFrame(omnibus_rows).to_csv(OUT / "mored_friedman_omnibus_exploratory.csv", index=False)

    report = {
        "status": "PASS",
        "experiment": "MORED corrected V3 whole-session split no-interpolation",
        "n_test_windows": int(len(y_true)),
        "forecast_horizon_minutes": int(y_true.shape[1]),
        "stride_minutes": int(metadata.get("window_config", {}).get("stride_minutes", 60)),
        "n_independent_test_recording_sessions": int(n_segments),
        "models": model_order,
        "trained_seeds_per_neural_model": 5,
        "timesfm_runs": 1,
        "primary_pairwise_method": "exact two-sided paired sign test across four raw recording-session clusters (raw_segment_id 6-9)",
        "multiplicity": "Holm adjustment across 10 RMSE/MAE pairwise comparisons vs VanillaTransformer",
        "interval": f"{N_BOOT:,}-replicate percentile bootstrap resampling whole raw recording sessions; descriptive with four clusters",
        "limitations": [
            "Only four independent original raw recording sessions; exact sign tests have low power.",
            "Overlapping 24-hour forecast windows are kept within their recording-session cluster, not treated as independent units.",
            "The five random seeds quantify training variability but are averaged within recording sessions and do not increase the data sample size.",
            "TimesFM is a single zero-shot run and has no across-seed uncertainty estimate.",
            "Friedman results use an asymptotic approximation and are exploratory with four recording-session blocks."
        ],
        "outputs": [
            "mored_recording_session_level_metrics.csv",
            "mored_seed_variability_by_recording_session.csv",
            "mored_recording_session_macro_summary.csv",
            "mored_pairwise_vs_transformer_recording_session_sign_tests.csv",
            "mored_friedman_omnibus_exploratory.csv"
        ]
    }
    (OUT / "statistical_analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nMORED CORRECTED-V3 SESSION-SPLIT SEGMENT-CLUSTERED STATISTICAL ANALYSIS: COMPLETED")
    print(f"Test windows: {len(y_true)}; horizon: {y_true.shape[1]} minutes; independent segments: {n_segments}")
    print("Inference unit: original raw recording session (raw_segment_id; not individual minutes or overlapping windows).")
    print("Five seeds are averaged within each model/segment; TimesFM remains a single zero-shot run.")
    print("\nPairwise tests vs VanillaTransformer (positive difference = alternative worse):")
    for row in pair_rows:
        print(f"{row['Metric']:4s} | {row['Comparison']:30s} | mean diff={row['Mean_segment_difference']:+.4f} | "
              f"95% cluster bootstrap CI [{row['Cluster_bootstrap_95CI_low']:+.4f}, {row['Cluster_bootstrap_95CI_high']:+.4f}] | "
              f"sign-test p={row['Exact_two_sided_sign_test_p']:.4f} | Holm p={row['Holm_adjusted_p_10_tests']:.4f} | "
              f"direction {row['Sessions_alternative_worse']}/{n_segments} worse")
    print(f"\nSaved five analysis files under: {OUT}")
    print("CAUTION: Small cluster count; report effects and limitations, not only p-values.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nSTATISTICAL ANALYSIS FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
