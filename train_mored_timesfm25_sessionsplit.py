#!/usr/bin/env python3
"""Zero-shot TimesFM 2.5 evaluation on corrected, no-interpolation MORED windows.

Uses the exact held-out test windows from Data/MORED/Corrected_V3_SessionSplit. No training,
window reconstruction, interpolation, or test-time scaler fitting is performed.

Run from the project folder:
    python train_mored_timesfm25_corrected.py
"""
from __future__ import annotations

import hashlib
import json
import pickle
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

LOOKBACK = 1440
FORECAST = 1440
BATCH_SIZE = 4
EXPECTED_TEST_WINDOWS = 359
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "Data" / "MORED" / "Corrected_V3_SessionSplit"
RESULT_DIR = PROJECT_ROOT / "Results" / "MORED" / "Corrected_V3_SessionSplit_TimesFM25"
CHECKPOINT = "google/timesfm-2.5-200m-pytorch"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def as_2d_univariate(name: str, values: np.ndarray) -> np.ndarray:
    """Accept (windows, time) or (windows, time, 1); reject other layouts."""
    arr = np.asarray(values)
    if arr.ndim == 3:
        if arr.shape[-1] != 1:
            raise RuntimeError(
                f"{name} has {arr.shape[-1]} channels; this experiment expects univariate data"
            )
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise RuntimeError(
            f"{name} has shape {arr.shape}; expected (windows, time) "
            "or (windows, time, 1)"
        )
    return arr


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape or not np.isfinite(yt).all() or not np.isfinite(yp).all():
        raise ValueError("Metric arrays are non-finite or have different shapes")
    nonzero = np.abs(yt) > 1e-8
    return {
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE": float(mean_absolute_error(yt, yp)),
        "MAPE": float(np.mean(np.abs((yt[nonzero] - yp[nonzero]) / yt[nonzero])) * 100.0)
        if nonzero.any() else float("nan"),
        "R2": float(r2_score(yt, yp)),
        "StdR": float(np.std(yp) / np.std(yt)) if np.std(yt) > 0 else float("nan"),
    }


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = DATA_DIR / "preprocessing_metadata.json"
    scaler_path = DATA_DIR / "scaler.pkl"
    test_path = DATA_DIR / "test_windows.npz"
    train_csv = DATA_DIR / "train.csv"
    for path in (metadata_path, scaler_path, test_path, train_csv):
        if not path.is_file():
            raise FileNotFoundError(f"Required corrected MORED file is missing: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("experiment") != "mored_corrected_v3_session_split_no_interpolation_raw_count_filter":
        raise RuntimeError("Unexpected preprocessing metadata; refusing to evaluate")
    policy = metadata.get("minute_aggregation", {}).get("missing_sample_policy", "")
    if "no interpolation" not in policy.lower():
        raise RuntimeError("Metadata does not confirm the no-interpolation policy")

    with np.load(test_path, allow_pickle=False) as z:
        test = {k: z[k].copy() for k in z.files}
    required = {"X", "y", "X_raw", "y_raw", "input_start", "input_end", "target_start", "target_end"}
    missing = required.difference(test)
    if missing:
        raise RuntimeError(f"Corrected test NPZ is missing fields: {sorted(missing)}")
    # Corrected MORED windows are stored with a trailing singleton channel:
    # (359, 1440, 1). TimesFM expects one univariate 1-D context per example,
    # so remove only that channel axis; do not alter window or time dimensions.
    for key in ("X", "y", "X_raw", "y_raw"):
        test[key] = as_2d_univariate(key, test[key])

    for key, shape in {
        "X": (EXPECTED_TEST_WINDOWS, LOOKBACK),
        "y": (EXPECTED_TEST_WINDOWS, FORECAST),
        "X_raw": (EXPECTED_TEST_WINDOWS, LOOKBACK),
        "y_raw": (EXPECTED_TEST_WINDOWS, FORECAST),
    }.items():
        if test[key].shape != shape:
            raise RuntimeError(f"{key} has shape {test[key].shape}; expected {shape}")
        if not np.isfinite(test[key]).all():
            raise RuntimeError(f"{key} contains non-finite data")

    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)
    train_df = pd.read_csv(train_csv)
    genuine_train = train_df["consumption"].to_numpy(dtype=np.float64)
    if int(scaler.n_samples_seen_) != len(genuine_train):
        raise RuntimeError("Scaler was not fit on the corrected training-minute rows")
    if not np.isclose(float(scaler.data_min_[0]), float(genuine_train.min())):
        raise RuntimeError("Scaler minimum does not match corrected training data")
    if not np.isclose(float(scaler.data_max_[0]), float(genuine_train.max())):
        raise RuntimeError("Scaler maximum does not match corrected training data")

    # Verify the stored scaled arrays against the training-only scaler.
    x_scaled_check = scaler.transform(test["X_raw"].reshape(-1, 1)).reshape(test["X"].shape)
    y_scaled_check = scaler.transform(test["y_raw"].reshape(-1, 1)).reshape(test["y"].shape)
    if not np.allclose(test["X"], x_scaled_check, atol=2e-6):
        raise RuntimeError("Stored test inputs do not match the training-only scaler")
    if not np.allclose(test["y"], y_scaled_check, atol=2e-6):
        raise RuntimeError("Stored test targets do not match the training-only scaler")

    X_test = np.asarray(test["X"], dtype=np.float32)
    y_true_raw = np.asarray(test["y_raw"], dtype=np.float32)
    target_starts = pd.to_datetime(test["target_start"]).to_numpy(dtype="datetime64[ns]")
    target_timestamps = target_starts[:, None] + np.arange(FORECAST).astype("timedelta64[m]")[None, :]
    if not np.all(np.diff(target_timestamps, axis=1) == np.timedelta64(1, "m")):
        raise RuntimeError("At least one target window is not minute-contiguous")

    print("Corrected MORED TimesFM 2.5 zero-shot evaluation")
    print(f"Test windows: {X_test.shape}; targets: {y_true_raw.shape}")
    print("Scaler: verified against corrected training-minute split")
    print("Preprocessing: no interpolation; 359 whole-session split-safe windows")
    print(f"Checkpoint: {CHECKPOINT}")

    try:
        import timesfm
    except ImportError as exc:
        raise RuntimeError("TimesFM is not installed in this environment. Install timesfm[torch]==3.0.2 first.") from exc

    start_load = time.perf_counter()
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(CHECKPOINT)
    load_seconds = time.perf_counter() - start_load
    model.compile(
        timesfm.ForecastConfig(
            max_context=LOOKBACK,
            max_horizon=FORECAST,
            normalize_inputs=False,
            per_core_batch_size=BATCH_SIZE,
            use_continuous_quantile_head=False,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=False,
        )
    )
    print(f"Checkpoint loaded in {load_seconds:.1f}s. Starting forecasts...")

    batches: list[np.ndarray] = []
    inference_start = time.perf_counter()
    for start in range(0, len(X_test), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(X_test))
        inputs = [row.astype(np.float32, copy=False) for row in X_test[start:end]]
        point_forecast, _ = model.forecast(horizon=FORECAST, inputs=inputs)
        pred = np.asarray(point_forecast, dtype=np.float32)
        if pred.ndim != 2 or pred.shape != (end - start, FORECAST):
            raise RuntimeError(f"Unexpected TimesFM prediction shape: {pred.shape}")
        if not np.isfinite(pred).all():
            raise RuntimeError("TimesFM produced non-finite forecasts")
        batches.append(pred)
        print(f"Forecasted {end}/{len(X_test)} windows")

    inference_seconds = time.perf_counter() - inference_start
    prediction_scaled = np.concatenate(batches, axis=0)
    prediction_raw = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).reshape(prediction_scaled.shape)
    result_metrics = metrics(y_true_raw, prediction_raw)

    np.save(RESULT_DIR / "TimesFM25_predictions_raw.npy", prediction_raw)
    np.save(RESULT_DIR / "TimesFM25_y_true_raw.npy", y_true_raw)
    np.save(RESULT_DIR / "TimesFM25_target_timestamps.npy", target_timestamps)
    np.save(RESULT_DIR / "TimesFM25_predictions_scaled.npy", prediction_scaled)
    pd.DataFrame([{"Model": "TimesFM 2.5 zero-shot", **result_metrics}]).to_csv(
        RESULT_DIR / "TimesFM25_metrics.csv", index=False
    )
    pd.DataFrame({
        "target_start": pd.to_datetime(test["target_start"]),
        "target_end": pd.to_datetime(test["target_end"]),
        "input_start": pd.to_datetime(test["input_start"]),
        "input_end": pd.to_datetime(test["input_end"]),
    }).to_csv(RESULT_DIR / "TimesFM25_window_manifest.csv", index=False)

    run_config = {
        "model": "TimesFM 2.5 200M PyTorch",
        "checkpoint": CHECKPOINT,
        "mode": "zero-shot; no fine-tuning",
        "lookback_minutes": LOOKBACK,
        "forecast_minutes": FORECAST,
        "batch_size": BATCH_SIZE,
        "number_of_test_windows": len(X_test),
        "input_array_shape_on_disk": "(359, 1440, 1)",
        "input_array_shape_used_for_timesfm": "(359, 1440); singleton channel axis removed",
        "preprocessing": "Corrected_V3_SessionSplit; no interpolation; segment-safe minute windows",
        "scaler": "training-split-only MinMaxScaler; inverse transform used for reported metrics",
        "test_windows_sha256": sha256_file(test_path),
        "metadata_sha256": sha256_file(metadata_path),
        "scaler_sha256": sha256_file(scaler_path),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "timesfm_version": getattr(timesfm, "__version__", "not exposed"),
        "checkpoint_load_seconds": load_seconds,
        "forecast_inference_seconds": inference_seconds,
        "mean_seconds_per_window": inference_seconds / len(X_test),
        "metrics": result_metrics,
    }
    (RESULT_DIR / "TimesFM25_run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 62)
    print("CORRECTED MORED — TIMESFM 2.5 ZERO-SHOT RESULTS")
    print("=" * 62)
    for key, value in result_metrics.items():
        print(f"{key:5s}: {value:.6f}")
    print(f"Inference time: {inference_seconds:.1f}s total; {inference_seconds / len(X_test):.2f}s/window")
    print(f"Saved outputs: {RESULT_DIR}")


if __name__ == "__main__":
    main()
