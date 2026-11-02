#!/usr/bin/env python3
"""Corrected five-seed MORED DLinear baseline on audited, segment-safe windows.

Architecture follows the supplied MORED NeuralForecast DLinear configuration:
``moving_avg_window=25``. This direct PyTorch implementation applies an
edge-replicated moving-average decomposition and separate linear projections
for the seasonal and trend components (1440 input steps -> 1440 forecast steps).

Only Data/MORED/Corrected_V3_SessionSplit/{train,val,test}_windows.npz is consumed.
Legacy interpolated split files are not used to build windows or targets.

Default: validate the corrected data and run forward/backward smoke test only.
Training: python train_mored_dlinear_corrected.py --train
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

LOOKBACK = 1440
FORECAST = 1440
MOVING_AVG_WINDOW = 25
EXPECTED_WINDOWS = {"train": 245, "val": 119, "test": 359}
SEEDS_DEFAULT = [0, 1, 2, 3, 4]
MAX_STEPS_DEFAULT = 300
LEARNING_RATE = 1e-3
BATCH_SIZE_DEFAULT = 4
ACCUM_STEPS_DEFAULT = 8

ROOT = Path(__file__).resolve().parent
DATA_DIR_DEFAULT = ROOT / "Data" / "MORED" / "Corrected_V3_SessionSplit"
RESULT_DIR_DEFAULT = ROOT / "Results" / "MORED" / "Corrected_V3_SessionSplit_DLinear"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", action="store_true", help="Train five seeds; otherwise smoke-test only.")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS_DEFAULT)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    p.add_argument("--accum-steps", type=int, default=ACCUM_STEPS_DEFAULT)
    p.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    p.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    p.add_argument("--result-dir", type=Path, default=RESULT_DIR_DEFAULT)
    return p.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(data_dir: Path, split: str) -> dict[str, np.ndarray]:
    path = data_dir / f"{split}_windows.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Corrected MORED window file not found: {path}")
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    for key in ("X", "y"):
        if d[key].ndim == 3 and d[key].shape[-1] == 1:
            d[key] = d[key][..., 0]
    n = EXPECTED_WINDOWS[split]
    checks = {
        "X": (n, LOOKBACK),
        "y": (n, FORECAST),
        "X_raw": (n, LOOKBACK),
        "y_raw": (n, FORECAST),
    }
    for key, shape in checks.items():
        if key not in d or d[key].shape != shape:
            raise RuntimeError(f"{split} {key} has shape {d.get(key, np.empty(0)).shape}; expected {shape}")
        if not np.isfinite(d[key]).all():
            raise RuntimeError(f"{split} {key} contains non-finite values")
    for key in ("input_start", "input_end", "target_start", "target_end"):
        if key not in d or len(d[key]) != n:
            raise RuntimeError(f"{split} missing/invalid timestamp field {key}")
    d["target_start_dt"] = pd.to_datetime(d["target_start"]).to_numpy(dtype="datetime64[ns]")
    d["target_timestamps"] = d["target_start_dt"][:, None] + np.arange(FORECAST).astype("timedelta64[m]")[None, :]
    if not np.all(np.diff(d["target_timestamps"], axis=1) == np.timedelta64(1, "m")):
        raise RuntimeError(f"{split} contains a non-contiguous target window")
    return d


def load_data(data_dir: Path):
    metadata_path = data_dir / "preprocessing_metadata.json"
    scaler_path = data_dir / "scaler.pkl"
    train_csv = data_dir / "train.csv"
    if not metadata_path.is_file() or not scaler_path.is_file() or not train_csv.is_file():
        raise FileNotFoundError(f"Corrected MORED metadata/scaler/train.csv missing in {data_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_experiment = "mored_corrected_v3_session_split_no_interpolation_raw_count_filter"
    if metadata.get("experiment") != expected_experiment:
        raise RuntimeError(
            f"Unexpected MORED preprocessing version: {metadata.get('experiment')!r}; "
            f"expected {expected_experiment!r}"
        )
    missing_policy = metadata.get("minute_aggregation", {}).get("missing_sample_policy", "")
    if "no interpolation" not in missing_policy.lower():
        raise RuntimeError("Metadata does not confirm the no-interpolation missing-sample policy")

    arrays = {s: load_split(data_dir, s) for s in ("train", "val", "test")}
    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)

    # Independently check the scaler against every retained training-minute row.
    train_frame = pd.read_csv(train_csv)
    genuine_train = train_frame["consumption"].to_numpy(dtype=np.float64)
    if not np.isfinite(genuine_train).all():
        raise RuntimeError("Corrected training-minute CSV contains non-finite values")
    if int(scaler.n_samples_seen_) != len(genuine_train):
        raise RuntimeError("Scaler sample count does not match retained training-minute rows")
    if not np.isclose(float(scaler.data_min_[0]), float(genuine_train.min())):
        raise RuntimeError("Scaler minimum does not match corrected training readings")
    if not np.isclose(float(scaler.data_max_[0]), float(genuine_train.max())):
        raise RuntimeError("Scaler maximum does not match corrected training readings")

    for split, d in arrays.items():
        expected_x = scaler.transform(d["X_raw"].reshape(-1, 1)).reshape(d["X"].shape)
        expected_y = scaler.transform(d["y_raw"].reshape(-1, 1)).reshape(d["y"].shape)
        if not np.allclose(d["X"], expected_x, atol=2e-6):
            raise RuntimeError(f"{split} scaled inputs do not match the saved training-only scaler")
        if not np.allclose(d["y"], expected_y, atol=2e-6):
            raise RuntimeError(f"{split} scaled targets do not match the saved training-only scaler")

    print("Loaded corrected MORED windows:")
    for split, d in arrays.items():
        print(f"  {split:5s} X={d['X'].shape}, y={d['y'].shape}, y_raw={d['y_raw'].shape}")
    print("Scaler verified against the corrected training-minute split.")
    print("No interpolation is used in these windows.")
    return arrays, scaler, metadata


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape or not np.isfinite(yt).all() or not np.isfinite(yp).all():
        raise ValueError("Metric arrays are non-finite or have different shapes")
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    mask = np.abs(yt) > 1e-8
    mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100) if mask.any() else float("nan")
    true_sd = float(np.std(yt))
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": float(r2_score(yt, yp)),
        "StdR": float(np.std(yp) / true_sd) if true_sd > 0 else float("nan"),
    }


class MovingAverageDecomposition(nn.Module):
    """Edge-replicated moving-average decomposition used by DLinear."""
    def __init__(self, kernel_size: int = MOVING_AVG_WINDOW) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("moving_avg_window must be odd")
        self.kernel_size = int(kernel_size)
        self.padding = (self.kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, time]
        x_ch = x.unsqueeze(1)
        if self.padding:
            front = x_ch[:, :, :1].expand(-1, -1, self.padding)
            end = x_ch[:, :, -1:].expand(-1, -1, self.padding)
            padded = torch.cat((front, x_ch, end), dim=2)
        else:
            padded = x_ch
        trend = F.avg_pool1d(padded, kernel_size=self.kernel_size, stride=1).squeeze(1)
        seasonal = x - trend
        return seasonal, trend


class NeuralForecastStyleDLinear(nn.Module):
    """Separate direct linear maps for seasonal and trend components."""
    def __init__(self) -> None:
        super().__init__()
        self.decomposition = MovingAverageDecomposition(MOVING_AVG_WINDOW)
        self.linear_trend = nn.Linear(LOOKBACK, FORECAST, bias=True)
        self.linear_seasonal = nn.Linear(LOOKBACK, FORECAST, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomposition(x)
        return self.linear_trend(trend) + self.linear_seasonal(seasonal)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0)


def sample_weighted_val(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.functional.mse_loss(model(xb), yb, reduction="mean")
            n = len(xb)
            total += float(loss.item()) * n
            count += n
    if not count:
        raise RuntimeError("Validation loader is empty")
    return total / count


def smoke_test(device: torch.device) -> tuple[torch.device, int]:
    candidates = [device, torch.device("cpu")] if device.type == "mps" else [device]
    last_error: Exception | None = None
    for candidate in candidates:
        model = NeuralForecastStyleDLinear().to(candidate)
        x = torch.zeros((2, LOOKBACK), dtype=torch.float32, device=candidate)
        y = torch.zeros((2, FORECAST), dtype=torch.float32, device=candidate)
        try:
            model.train()
            pred = model(x)
            if pred.shape != (2, FORECAST):
                raise RuntimeError(f"Unexpected DLinear output shape: {tuple(pred.shape)}")
            nn.functional.mse_loss(pred, y).backward()
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if candidate.type == "mps":
                torch.mps.synchronize()
            print(f"DLinear forward/backward smoke test PASSED on {candidate}: input={tuple(x.shape)}, output={tuple(pred.shape)}")
            del model, x, y, pred
            gc.collect()
            if candidate.type == "mps":
                torch.mps.empty_cache()
            return candidate, count
        except Exception as exc:
            last_error = exc
            del model, x, y
            gc.collect()
            if candidate.type == "mps":
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
    raise RuntimeError(f"DLinear smoke test failed on supported devices: {last_error}")


def inverse_array(a: np.ndarray, scaler: Any) -> np.ndarray:
    return scaler.inverse_transform(np.asarray(a).reshape(-1, 1)).reshape(a.shape)


def run_seed(seed: int, arrays: dict[str, Any], scaler: Any, result_dir: Path,
             device: torch.device, max_steps: int, batch_size: int, accum_steps: int) -> dict[str, Any]:
    set_seed(seed)
    train_loader = make_loader(arrays["train"]["X"], arrays["train"]["y"], batch_size, True)
    val_loader = make_loader(arrays["val"]["X"], arrays["val"]["y"], batch_size, False)
    test_loader = make_loader(arrays["test"]["X"], arrays["test"]["y"], batch_size, False)

    model = NeuralForecastStyleDLinear().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0

    print(f"\n{'=' * 65}\nMORED DLinear — SEED {seed}\n{'=' * 65}")
    while micro_step // accum_steps < max_steps:
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            xb, yb = next(iterator)
        xb, yb = xb.to(device), yb.to(device)
        model.train()
        (criterion(model(xb), yb) / accum_steps).backward()
        micro_step += 1
        if micro_step % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step = micro_step // accum_steps
            if step == 1 or step % 25 == 0 or step == max_steps:
                val_loss = sample_weighted_val(model, val_loader, device)
                print(f"Seed {seed} | optimizer step {step}/{max_steps} | val MSE={val_loss:.7f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"No best checkpoint recorded for seed {seed}")
    model.load_state_dict(best_state)
    model.eval()
    predicted_scaled: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in test_loader:
            predicted_scaled.append(model(xb.to(device)).detach().cpu().numpy())
    prediction_raw = inverse_array(np.concatenate(predicted_scaled, axis=0), scaler).astype(np.float32)
    y_true_raw = arrays["test"]["y_raw"].astype(np.float32)
    if prediction_raw.shape != y_true_raw.shape:
        raise RuntimeError(f"Prediction shape {prediction_raw.shape} != targets {y_true_raw.shape}")

    metrics = compute_metrics(y_true_raw, prediction_raw)
    np.save(result_dir / f"DLinear_seed{seed}_predictions.npy", prediction_raw)
    torch.save(best_state, result_dir / f"DLinear_seed{seed}_best.pt")
    row = {"Model": "DLinear", "Seed": seed, "Best_Val_MSE_Scaled": best_loss, **metrics}
    progress_file = result_dir / "dlinear_corrected_progress.csv"
    old = pd.read_csv(progress_file) if progress_file.is_file() else pd.DataFrame()
    if not old.empty and "Seed" in old.columns:
        old = old[old["Seed"] != seed]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).sort_values("Seed").to_csv(progress_file, index=False)

    print("\nSeed results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    del model, optimizer, train_loader, val_loader, test_loader, iterator
    gc.collect()
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main() -> int:
    args = parse_args()
    if min(args.max_steps, args.batch_size, args.accum_steps) <= 0:
        raise ValueError("max-steps, batch-size and accum-steps must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be non-empty and unique")

    data_dir = args.data_dir.expanduser().resolve()
    result_dir = args.result_dir.expanduser().resolve()
    arrays, scaler, prep_metadata = load_data(data_dir)
    initial_device = choose_device(args.device)
    print("Initial device:", initial_device)
    device, parameter_count = smoke_test(initial_device)

    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / "y_true_raw.npy", arrays["test"]["y_raw"].astype(np.float32))
    np.save(result_dir / "test_target_timestamps.npy", arrays["test"]["target_timestamps"].astype("datetime64[ns]").astype(str))

    config = {
        "experiment": "mored_dlinear_corrected_v3_session_split_no_interpolation_segment_safe_windows",
        "model": "DLinear (NeuralForecast-style direct PyTorch implementation)",
        "architecture_reference": "Original supplied MORED baseline: DLinear(moving_avg_window=25)",
        "architecture": "Edge-replicated moving-average decomposition; separate linear trend and seasonal projections",
        "moving_avg_window": MOVING_AVG_WINDOW,
        "source_data": "Data/MORED/Corrected_V3_SessionSplit/{train,val,test}_windows.npz only",
        "legacy_interpolated_csv_splits_used": False,
        "preprocessing_experiment": prep_metadata.get("experiment"),
        "preprocessing_interpolation": False,
        "seeds": args.seeds,
        "max_optimizer_steps_per_seed": args.max_steps,
        "physical_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accum_steps,
        "effective_batch_size_approx": args.batch_size * args.accum_steps,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "training_loss": "MSE",
        "validation_loss": "MSE",
        "validation_interval_optimizer_steps": 25,
        "validation_loss_aggregation": "sample-weighted MSE across windows; partial final batch weighted by sample count",
        "checkpoint_selection": "best sample-weighted validation MSE; test evaluated only after selection",
        "lookback_minutes": LOOKBACK,
        "forecast_minutes": FORECAST,
        "stride_minutes": 60,
        "train_windows": EXPECTED_WINDOWS["train"],
        "validation_windows": EXPECTED_WINDOWS["val"],
        "test_windows": EXPECTED_WINDOWS["test"],
        "trainable_parameters": parameter_count,
        "device": str(device),
        "metric_aggregation": "flatten all 359 accepted test forecast windows and minute targets; forecast origins overlap at 60-minute stride",
        "source_npz_sha256": {s: sha256_file(data_dir / f"{s}_windows.npz") for s in ("train", "val", "test")},
        "note": "The architecture matches the supplied NeuralForecast DLinear moving_avg_window=25 setting, reimplemented directly in PyTorch to consume the audited segment-safe NPZ windows. The training protocol uses the corrected project's 4-sample physical batches with 8-step gradient accumulation (effective batch approx. 32) and best sample-weighted validation checkpoint selection; it is a reproducibility reconstruction, not an exact rerun of the original NeuralForecast training code.",
    }
    (result_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Device:", device)
    print(f"Moving-average decomposition window: {MOVING_AVG_WINDOW}")
    print(f"Trainable parameters: {parameter_count:,}")
    print("Results directory:", result_dir)

    if not args.train:
        print("\nSmoke test only. No model weights were trained.")
        print("When ready, run this script again with --train.")
        return 0

    rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        row = run_seed(seed, arrays, scaler, result_dir, device, args.max_steps, args.batch_size, args.accum_steps)
        rows.append(row)
        pd.DataFrame(rows).sort_values("Seed").to_csv(result_dir / "dlinear_corrected_results.csv", index=False)

    results = pd.DataFrame(rows).sort_values("Seed")
    results.to_csv(result_dir / "dlinear_corrected_results.csv", index=False)
    metrics = ["RMSE", "MAE", "MAPE", "R2", "StdR"]
    summary = pd.DataFrame({"mean": results[metrics].mean(), "sd": results[metrics].std(ddof=1)})
    summary.to_csv(result_dir / "dlinear_corrected_summary.csv")

    print("\nFive-seed corrected MORED DLinear results:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nMean ± sample standard deviation:")
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))
    print("\nSaved all outputs under:", result_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
