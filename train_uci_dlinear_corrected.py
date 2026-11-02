#!/usr/bin/env python3
"""Leakage-aware five-seed UCI DLinear baseline.

This is a direct PyTorch implementation of Nixtla NeuralForecast's DLinear
architecture: moving-average decomposition (kernel=25) followed by separate
linear trend and seasonal projections from 1440 input steps to 1440 forecast
steps. The original supplied UCI baseline configured DLinear with
moving_avg_window=25 and explicitly passed MSE() for train/validation loss.

Only the independently validated Data/UCI/processed/*.npz windows are used.
No old split CSVs, interpolation, or additional window generation is used.

Usage:
  python train_uci_dlinear_corrected.py          # smoke test only
  python train_uci_dlinear_corrected.py --train  # five-seed training
"""
from __future__ import annotations

import argparse
import gc
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from uci_window_loader import load_uci_windows, inverse_array

LOOKBACK = 1440
FORECAST = 1440
MOVING_AVG_WINDOW = 25
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_MAX_OPT_STEPS = 300
LEARNING_RATE = 1e-3
BATCH_SIZE = 4
ACCUM_STEPS = 8

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "Results" / "UCI" / "Corrected_V2_DLinear"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-aware UCI DLinear baseline on verified NPZ windows."
    )
    parser.add_argument("--train", action="store_true", help="Train five seeds; default is smoke test only.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_OPT_STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--accum-steps", type=int, default=ACCUM_STEPS)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    return parser.parse_args()


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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(f"Metric arrays differ in shape: {yt.shape} vs {yp.shape}")
    if not np.isfinite(yt).all() or not np.isfinite(yp).all():
        raise ValueError("Metric inputs contain non-finite values.")
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    valid = np.abs(yt) > 1e-8
    mape = (
        float(np.mean(np.abs((yt[valid] - yp[valid]) / yt[valid])) * 100.0)
        if valid.any() else float("nan")
    )
    r2 = float(r2_score(yt, yp))
    true_std = float(np.std(yt))
    stdr = float(np.std(yp) / true_std) if true_std > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "StdR": stdr}


class MovingAverageDecomposition(nn.Module):
    """Nixtla-style edge-replicated moving-average decomposition."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("moving_avg_window must be odd.")
        self.kernel_size = int(kernel_size)
        self.padding = (self.kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, time]. Replicate endpoints, then valid average pooling.
        x_channel = x.unsqueeze(1)
        if self.padding:
            front = x_channel[:, :, :1].expand(-1, -1, self.padding)
            end = x_channel[:, :, -1:].expand(-1, -1, self.padding)
            padded = torch.cat((front, x_channel, end), dim=2)
        else:
            padded = x_channel
        trend = F.avg_pool1d(padded, kernel_size=self.kernel_size, stride=1).squeeze(1)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """DLinear: separate direct linear maps for seasonal and trend parts."""

    def __init__(self) -> None:
        super().__init__()
        self.decomposition = MovingAverageDecomposition(MOVING_AVG_WINDOW)
        self.linear_trend = nn.Linear(LOOKBACK, FORECAST, bias=True)
        self.linear_seasonal = nn.Linear(LOOKBACK, FORECAST, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, lookback]; output: [batch, forecast]
        seasonal, trend = self.decomposition(x)
        return self.linear_trend(trend) + self.linear_seasonal(seasonal)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)),
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=False, num_workers=0, pin_memory=False,
    )


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """Sample-weighted validation MSE, including a partial final batch."""
    model.eval()
    weighted_loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            prediction = model(xb)
            n = int(yb.shape[0])
            weighted_loss_sum += float(criterion(prediction, yb).item()) * n
            sample_count += n
    if sample_count == 0:
        raise RuntimeError("Validation loader is empty.")
    return weighted_loss_sum / sample_count


def smoke_test(device: torch.device) -> int:
    model = DLinear().to(device)
    x = torch.zeros((2, LOOKBACK), dtype=torch.float32, device=device)
    y = torch.zeros((2, FORECAST), dtype=torch.float32, device=device)
    model.train()
    prediction = model(x)
    if tuple(prediction.shape) != (2, FORECAST):
        raise RuntimeError(f"Unexpected output shape: {tuple(prediction.shape)}")
    loss = nn.MSELoss()(prediction, y)
    loss.backward()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if device.type == "mps":
        torch.mps.synchronize()
    print(
        f"DLinear forward/backward smoke test PASSED on {device}: "
        f"input={tuple(x.shape)}, output={tuple(prediction.shape)}"
    )
    del model, x, y, prediction, loss
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return param_count


def run_one_seed(
    seed: int,
    arrays: dict[str, Any],
    result_dir: Path,
    device: torch.device,
    max_opt_steps: int,
    batch_size: int,
    accum_steps: int,
) -> dict[str, float]:
    set_seed(seed)
    train_loader = make_loader(arrays["train"]["X"], arrays["train"]["y"], batch_size, True)
    val_loader = make_loader(arrays["val"]["X"], arrays["val"]["y"], batch_size, False)
    test_loader = make_loader(arrays["test"]["X"], arrays["test"]["y"], batch_size, False)

    model = DLinear().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0

    print(f"\n{'=' * 65}\nDLINEAR — SEED {seed}\n{'=' * 65}")
    while micro_step // accum_steps < max_opt_steps:
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            xb, yb = next(iterator)

        xb = xb.to(device)
        yb = yb.to(device)
        model.train()
        prediction = model(xb)
        loss = criterion(prediction, yb) / accum_steps
        loss.backward()
        micro_step += 1

        if micro_step % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step = micro_step // accum_steps
            if opt_step == 1 or opt_step % 25 == 0 or opt_step == max_opt_steps:
                val_loss = evaluate(model, val_loader, criterion, device)
                print(
                    f"Seed {seed} | optimizer step {opt_step}/{max_opt_steps} "
                    f"| val MSE={val_loss:.7f}"
                )
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }

    if best_state is None:
        raise RuntimeError(f"No best checkpoint captured for seed {seed}.")

    model.load_state_dict(best_state)
    model.eval()
    prediction_batches: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in test_loader:
            prediction_batches.append(model(xb.to(device)).detach().cpu().numpy())

    prediction_scaled = np.concatenate(prediction_batches, axis=0)
    prediction_raw = inverse_array(prediction_scaled, arrays["scaler"]).astype(np.float32)
    y_true_raw = arrays["test"]["y_raw"].astype(np.float32)
    if prediction_raw.shape != y_true_raw.shape:
        raise RuntimeError(
            f"Prediction shape {prediction_raw.shape} != target shape {y_true_raw.shape}"
        )

    metrics = compute_metrics(y_true_raw, prediction_raw)
    np.save(result_dir / f"DLinear_seed{seed}_predictions.npy", prediction_raw)
    torch.save(best_state, result_dir / f"DLinear_seed{seed}_best.pt")

    row = {"Model": "DLinear", "Seed": seed, "Best_Val_MSE_Scaled": best_val, **metrics}
    progress_file = result_dir / "dlinear_corrected_progress.csv"
    old = pd.read_csv(progress_file) if progress_file.exists() else pd.DataFrame()
    if not old.empty and "Seed" in old.columns:
        old = old[old["Seed"] != seed]
    progress = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    progress.sort_values("Seed").to_csv(progress_file, index=False)

    print("\nSeed results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    del model, optimizer, train_loader, val_loader, test_loader
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main() -> int:
    args = parse_args()
    if min(args.max_steps, args.batch_size, args.accum_steps) <= 0:
        raise ValueError("max-steps, batch-size and accum-steps must be positive.")
    if not args.seeds:
        raise ValueError("At least one seed is required.")

    device = choose_device(args.device)
    print(f"Device: {device}")
    print("Experiment: corrected UCI DLinear (Nixtla-style direct implementation)")
    print(f"Moving-average decomposition window: {MOVING_AVG_WINDOW}")

    arrays = load_uci_windows(project_root=ROOT)
    expected_counts = {"train": 903, "val": 189, "test": 192}
    for split, expected in expected_counts.items():
        X = arrays[split]["X"]
        y = arrays[split]["y"]
        if X.shape != (expected, LOOKBACK) or y.shape != (expected, FORECAST):
            raise RuntimeError(f"Unexpected {split} shapes: X={X.shape}, y={y.shape}")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise RuntimeError(f"Non-finite values found in {split} windows")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULT_DIR / "y_true_raw.npy", arrays["test"]["y_raw"].astype(np.float32))
    np.save(
        RESULT_DIR / "test_target_timestamps.npy",
        arrays["test"]["target_start"].astype("datetime64[ns]").astype(str),
    )

    parameter_count = smoke_test(device)
    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Results directory: {RESULT_DIR.resolve()}")

    run_config = {
        "experiment": "uci_dlinear_corrected_v2_no_interpolated_splits",
        "model": "DLinear (Nixtla NeuralForecast architecture, direct PyTorch window training)",
        "architecture_source_url": "https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/dlinear.py",
        "architecture": "Edge-replicated moving-average decomposition; separate linear trend and seasonal projections",
        "moving_avg_window": MOVING_AVG_WINDOW,
        "source_data": "Data/UCI/processed/*.npz only",
        "old_interpolated_csv_files_used": False,
        "seeds": args.seeds,
        "max_optimizer_steps_per_seed": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accum_steps,
        "effective_batch_size_approx": args.batch_size * args.accum_steps,
        "validation_interval_optimizer_steps": 25,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "training_loss": "MSE",
        "validation_loss": "MSE",
        "validation_loss_aggregation": "sample-weighted mean across windows; partial final batch weighted by sample count",
        "lookback_minutes": LOOKBACK,
        "forecast_minutes": FORECAST,
        "train_windows": 903,
        "validation_windows": 189,
        "test_windows": 192,
        "trainable_parameters": parameter_count,
        "device": str(device),
        "checkpoint_selection": "Best checkpoint selected only by validation MSE; test set evaluated after selection.",
        "inverse_transform": "Training-only MinMaxScaler from corrected preprocessing.",
        "notes": [
            "The architecture follows Nixtla NeuralForecast DLinear's MovingAvg/SeriesDecomp and separate linear_trend/linear_season projections.",
            "The supplied original UCI baseline explicitly passed MSE() for both loss and valid_loss; this corrected implementation uses MSE for training and validation.",
            "Only accepted, fully observed NPZ windows are consumed. No CSV split or additional window extraction is used.",
            "The DLinear architecture has a substantially larger parameter count than the LSTM and Vanilla Transformer; report the count and do not claim parameter-matched comparison."
        ],
    }
    with (RESULT_DIR / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    if not args.train:
        print("\nSmoke test only. No model weights were trained.")
        print("When ready to train, run this script again with --train.")
        return 0

    all_rows: list[dict[str, Any]] = []
    for seed in args.seeds:
        row = run_one_seed(
            seed=seed,
            arrays=arrays,
            result_dir=RESULT_DIR,
            device=device,
            max_opt_steps=args.max_steps,
            batch_size=args.batch_size,
            accum_steps=args.accum_steps,
        )
        all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(
            RESULT_DIR / "dlinear_corrected_results.csv", index=False
        )

    results = pd.DataFrame(all_rows).sort_values("Seed")
    results.to_csv(RESULT_DIR / "dlinear_corrected_results.csv", index=False)
    metric_columns = ["RMSE", "MAE", "MAPE", "R2", "StdR"]
    summary = results[metric_columns].agg(["mean", "std"]).T
    summary.columns = ["mean", "sd"]
    summary.to_csv(RESULT_DIR / "dlinear_corrected_summary.csv")

    print("\nFive-seed corrected DLinear results:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nMean ± sample standard deviation:")
    print(summary.to_string(float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved all outputs under: {RESULT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
