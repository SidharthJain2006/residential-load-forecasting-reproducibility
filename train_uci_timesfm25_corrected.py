#!/usr/bin/env python3
"""Zero-shot TimesFM 2.5 evaluation on leakage-aware UCI windows.

This is an inference/evaluation script, not a training script. It uses the
original verified UCI NPZ windows created by prepare_uci.py and the shared
uci_window_loader.py. No old interpolated split CSVs are read.

Default mode: load TimesFM 2.5 and run a one-window, 1,440-step smoke test.
Full mode: pass --run-eval to forecast all 192 clean UCI test origins.

The official TimesFM 2.5 PyTorch implementation selects CUDA when available
and otherwise CPU; it does not automatically select Apple MPS. On Apple
Silicon without CUDA, CPU inference is therefore expected and is recorded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from uci_window_loader import load_uci_windows


MODEL_ID = "google/timesfm-2.5-200m-pytorch"
LOOKBACK = 1440
FORECAST = 1440
EXPECTED_TEST_WINDOWS = 192
OUTPUT_DIR_NAME = "Corrected_V2_TimesFM25"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Zero-shot TimesFM 2.5 evaluation on corrected UCI windows."
    )
    parser.add_argument(
        "--run-eval",
        action="store_true",
        help="Run TimesFM on all 192 corrected UCI test windows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing final TimesFM prediction/metric files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Optional local TimesFM 2.5 Hugging Face snapshot directory.",
    )
    return parser.parse_args()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if yt.shape != yp.shape:
        raise ValueError(f"Metric arrays differ: {yt.shape} vs {yp.shape}")
    if not np.isfinite(yt).all() or not np.isfinite(yp).all():
        raise ValueError("Metric arrays contain NaN or infinite values.")

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    nonzero = np.abs(yt) > 1e-8
    mape = (
        float(np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])) * 100.0)
        if nonzero.any()
        else float("nan")
    )
    r2 = float(r2_score(yt, yp))
    true_std = float(np.std(yt))
    stdr = float(np.std(yp) / true_std) if true_std > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "StdR": stdr}


def load_timesfm_model(checkpoint_dir: Path | None):
    try:
        import timesfm
    except ImportError as exc:
        raise RuntimeError(
            "TimesFM is not installed in this virtual environment.\n"
            "Activate .venv and install the pinned package with:\n"
            "python -m pip install 'timesfm[torch]==3.0.2'\n"
            "Then run this script again."
        ) from exc

    package_version = importlib.metadata.version("timesfm")
    print(f"TimesFM package version: {package_version}")
    print(f"Checkpoint: {MODEL_ID}")

    if checkpoint_dir is not None:
        checkpoint_dir = checkpoint_dir.expanduser().resolve()
        if not checkpoint_dir.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")
        model_source = str(checkpoint_dir)
        revision = checkpoint_dir.name
        print(f"Using local Hugging Face snapshot: {checkpoint_dir}")
    else:
        # The official model card recommends this checkpoint and API.
        # The model is downloaded once and cached by Hugging Face Hub (~0.8 GB).
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is missing. Install TimesFM with the [torch] extra."
            ) from exc

        print("Resolving/downloading TimesFM 2.5 checkpoint from Hugging Face...")
        print("First download is approximately 0.8 GB; subsequent runs use the cache.")
        model_source = snapshot_download(
            repo_id=MODEL_ID,
            allow_patterns=["config.json", "model.safetensors"],
        )
        model_source = str(Path(model_source).resolve())
        revision = Path(model_source).name

    print("Loading pretrained TimesFM 2.5 model (no fine-tuning)...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        model_source,
        torch_compile=False,
    )

    # This mirrors the prior UCI TimesFM 2.5 configuration, updated to use
    # only clean raw UCI windows. The fixed quantile head is required because
    # the internally rounded decode horizon is 1,536 (> the 1,024 continuous
    # quantile-head limit).
    model.compile(
        timesfm.ForecastConfig(
            max_context=LOOKBACK,
            max_horizon=FORECAST,
            normalize_inputs=True,
            per_core_batch_size=1,
            use_continuous_quantile_head=False,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=False,
        )
    )

    actual_device = getattr(model.model, "device", torch.device("cpu"))
    print(f"TimesFM PyTorch inference device: {actual_device}")
    if torch.cuda.is_available():
        expected_device = "cuda"
    else:
        expected_device = "cpu"
        if torch.backends.mps.is_available():
            print(
                "Note: Apple MPS is available for the custom PyTorch baselines, "
                "but this TimesFM 2.5 PyTorch wrapper selects CPU when CUDA is absent."
            )

    if str(actual_device).split(":")[0] != expected_device:
        print(
            f"Notice: TimesFM wrapper reported {actual_device}; "
            f"CUDA availability implies expected backend {expected_device}."
        )

    return model, timesfm, package_version, revision, str(actual_device)


def predict_one(model: Any, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(inputs, dtype=np.float32).reshape(-1)
    if x.shape != (LOOKBACK,):
        raise ValueError(f"Expected one {LOOKBACK}-point input window, got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("TimesFM input contains non-finite values.")

    point, quantiles = model.forecast(
        horizon=FORECAST,
        inputs=[x],
    )

    point = np.asarray(point, dtype=np.float32)
    quantiles = np.asarray(quantiles, dtype=np.float32)

    if point.shape != (1, FORECAST):
        raise RuntimeError(f"Unexpected TimesFM point forecast shape: {point.shape}")
    if quantiles.shape != (1, FORECAST, 10):
        raise RuntimeError(f"Unexpected TimesFM quantile shape: {quantiles.shape}")
    if not np.isfinite(point).all() or not np.isfinite(quantiles).all():
        raise RuntimeError("TimesFM produced non-finite predictions.")

    # Public TimesFM API returns the point (median) forecast separately.
    return point[0], quantiles[0]


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, array)
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    out_dir = root / "Results" / "UCI" / OUTPUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Device note: TimesFM 2.5 is a pretrained zero-shot baseline, not a five-seed trained model.")
    data = load_uci_windows(root)
    test = data["test"]
    X_test_raw = np.asarray(test["X_raw"], dtype=np.float32)
    y_test_raw = np.asarray(test["y_raw"], dtype=np.float32)
    starts = np.asarray(test["starts_global"], dtype=np.int64)

    if X_test_raw.shape != (EXPECTED_TEST_WINDOWS, LOOKBACK):
        raise ValueError(f"Unexpected corrected UCI X_test_raw shape: {X_test_raw.shape}")
    if y_test_raw.shape != (EXPECTED_TEST_WINDOWS, FORECAST):
        raise ValueError(f"Unexpected corrected UCI y_test_raw shape: {y_test_raw.shape}")
    if not np.isfinite(X_test_raw).all() or not np.isfinite(y_test_raw).all():
        raise ValueError("Corrected test inputs/targets contain non-finite values.")

    # These are raw kW readings. TimesFM performs its own per-window
    # normalization, so do not feed the MinMax-scaled X array here.
    print("\nCorrected test data:")
    print(f"  test input windows: {X_test_raw.shape}")
    print(f"  test target windows: {y_test_raw.shape}")
    print("  model inputs: original observed kW values (not MinMax-scaled)")
    print("  target values: original observed kW values")
    print("  imputation in exported windows: none")
    print(f"Results directory: {out_dir}")

    model, timesfm_module, package_version, revision, device_name = load_timesfm_model(
        args.checkpoint_dir
    )

    config = {
        "experiment": "uci_timesfm_2p5_zero_shot_corrected_no_interpolated_windows",
        "model": "TimesFM 2.5 200M PyTorch",
        "model_id": MODEL_ID,
        "checkpoint_snapshot_revision": revision,
        "timesfm_package_version": package_version,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": device_name,
        "training_or_finetuning": False,
        "evaluation_protocol": "zero-shot; one pretrained forecast per clean test origin; no random seeds",
        "lookback_minutes": LOOKBACK,
        "forecast_minutes": FORECAST,
        "max_context_configured": LOOKBACK,
        "max_horizon_configured": FORECAST,
        "max_horizon_internal_rounded_to_multiple_of_output_patch": 1536,
        "input_data": "original observed Global_active_power values in kW; TimesFM internal normalization enabled",
        "target_data": "original observed Global_active_power values in kW",
        "input_windows": int(X_test_raw.shape[0]),
        "test_windows": int(y_test_raw.shape[0]),
        "source_window_starts_sha256": hashlib.sha256(starts.tobytes()).hexdigest(),
        "test_targets_sha256": sha256_array(y_test_raw),
        "forecast_config": {
            "normalize_inputs": True,
            "per_core_batch_size": 1,
            "use_continuous_quantile_head": False,
            "force_flip_invariance": True,
            "infer_is_positive": True,
            "fix_quantile_crossing": False,
        },
        "data_source": "Data/UCI/processed/test_windows.npz through uci_window_loader.py",
        "old_interpolated_csvs_used": False,
        "old_timesfm_predictions_reused": False,
    }

    if not args.run_eval:
        print("\nRunning one-window TimesFM 2.5 smoke test...")
        t0 = time.perf_counter()
        point, quantiles = predict_one(model, X_test_raw[0])
        elapsed = time.perf_counter() - t0
        print(f"Smoke-test point forecast shape: {point.shape}")
        print(f"Smoke-test quantile forecast shape: {quantiles.shape}")
        print(f"First-window inference time: {elapsed:.2f} seconds")
        print("TimesFM 2.5 zero-shot smoke test PASSED.")
        print("No full test evaluation was run and no model was trained.")
        print("When ready, run this script again with --run-eval.")
        return 0

    final_pred_path = out_dir / "timesfm25_predictions.npy"
    final_quantile_path = out_dir / "timesfm25_quantiles.npy"
    final_metrics_path = out_dir / "timesfm25_metrics.csv"
    final_metrics_json_path = out_dir / "timesfm25_metrics.json"
    final_config_path = out_dir / "run_config.json"
    final_timestamps_path = out_dir / "test_target_timestamps.npy"
    final_truth_path = out_dir / "y_true_raw.npy"

    if final_pred_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Existing TimesFM results found at {final_pred_path}. "
            "Use --overwrite only if you deliberately want to replace them."
        )

    print("\nStarting full zero-shot evaluation on 192 clean test windows...")
    predictions = np.empty((EXPECTED_TEST_WINDOWS, FORECAST), dtype=np.float32)
    all_quantiles = np.empty((EXPECTED_TEST_WINDOWS, FORECAST, 10), dtype=np.float32)
    runtimes: list[float] = []

    # Generate exact target timestamps from original full-series timestamps.
    all_timestamps = np.asarray(data["all_timestamps"]).astype("datetime64[ns]")
    target_timestamps = np.stack([
        all_timestamps[s + LOOKBACK:s + LOOKBACK + FORECAST]
        for s in starts
    ]).astype("datetime64[ns]")

    if target_timestamps.shape != (EXPECTED_TEST_WINDOWS, FORECAST):
        raise RuntimeError(f"Unexpected target timestamp shape: {target_timestamps.shape}")

    start_all = time.perf_counter()
    for i in range(EXPECTED_TEST_WINDOWS):
        t0 = time.perf_counter()
        point, quantiles = predict_one(model, X_test_raw[i])
        predictions[i] = point
        all_quantiles[i] = quantiles
        elapsed = time.perf_counter() - t0
        runtimes.append(elapsed)

        # Save resumable partial outputs every 5 windows and at the end.
        if (i + 1) % 5 == 0 or (i + 1) == EXPECTED_TEST_WINDOWS:
            atomic_save_npy(out_dir / "timesfm25_partial_predictions.npy", predictions[:i + 1])
            atomic_save_npy(out_dir / "timesfm25_partial_quantiles.npy", all_quantiles[:i + 1])
            print(
                f"TimesFM 2.5: {i + 1}/{EXPECTED_TEST_WINDOWS} "
                f"windows | latest={elapsed:.2f}s | elapsed={time.perf_counter() - start_all:.1f}s"
            )

    total_runtime = time.perf_counter() - start_all
    metrics = compute_metrics(y_test_raw, predictions)

    atomic_save_npy(final_pred_path, predictions)
    atomic_save_npy(final_quantile_path, all_quantiles)
    atomic_save_npy(final_truth_path, y_test_raw)
    atomic_save_npy(final_timestamps_path, target_timestamps)

    pd.DataFrame([{
        "Model": "TimesFM 2.5 zero-shot",
        "Seed": "zero-shot",
        **metrics,
    }]).to_csv(final_metrics_path, index=False)

    metrics_record = {
        "model": "TimesFM 2.5 zero-shot",
        "metrics": metrics,
        "test_windows": EXPECTED_TEST_WINDOWS,
        "forecast_points_per_window": FORECAST,
        "total_forecast_points": int(EXPECTED_TEST_WINDOWS * FORECAST),
        "mean_inference_seconds_per_window": float(np.mean(runtimes)),
        "median_inference_seconds_per_window": float(np.median(runtimes)),
        "total_inference_seconds": float(total_runtime),
    }
    with final_metrics_json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_record, f, indent=2)

    config.update({
        "full_evaluation_completed": True,
        "mean_inference_seconds_per_window": float(np.mean(runtimes)),
        "median_inference_seconds_per_window": float(np.median(runtimes)),
        "total_inference_seconds": float(total_runtime),
        "prediction_sha256": sha256_array(predictions),
        "quantile_sha256": sha256_array(all_quantiles),
        "y_true_sha256": sha256_array(y_test_raw),
    })
    with final_config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Partial files are no longer needed after final outputs have been saved.
    for partial_name in (
        "timesfm25_partial_predictions.npy",
        "timesfm25_partial_quantiles.npy",
    ):
        partial = out_dir / partial_name
        if partial.exists():
            partial.unlink()

    print("\nTimesFM 2.5 zero-shot metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6f}")
    print(f"\nMean inference time/window: {np.mean(runtimes):.2f} seconds")
    print(f"Total inference time: {total_runtime:.1f} seconds")
    print(f"Saved all outputs under: {out_dir}")
    print("TIMESFM 2.5 CORRECTED UCI EVALUATION COMPLETED.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user. Partial checkpoints, if any, remain in the results folder.")
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
