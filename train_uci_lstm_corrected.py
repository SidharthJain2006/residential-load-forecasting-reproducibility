#!/usr/bin/env python3
"""Leakage-aware five-seed UCI LSTM baseline.

This is a window-based PyTorch implementation of the LSTM baseline used in
this project. It follows the NeuralForecast LSTM architecture used by the
original supplied script: a multi-layer LSTM encoder returns a hidden state
for each input timestamp, and a two-layer MLP decoder maps each hidden state
to the corresponding direct-horizon output. With lookback == forecast ==
1440, the last 1440 encoder states are mapped to the 1440 forecast values.

Crucially, this script consumes ONLY the previously validated NPZ windows in
Data/UCI/processed. It does not regenerate windows from old interpolated CSVs.

Usage:
  python train_uci_lstm_corrected.py
      Load the clean windows and run a forward/backward smoke test only.

  python train_uci_lstm_corrected.py --train
      Train the five seeds and save checkpoints, predictions and metrics.

The training protocol mirrors the corrected Transformer/DAT runs:
  seeds 0..4, 300 optimizer updates per seed, Adam, LR 0.001,
  physical batch 4, accumulation 8 (approx. effective batch 32),
  sample-weighted validation MSE every 25 optimizer updates.
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from uci_window_loader import load_uci_windows, inverse_array

LOOKBACK = 1440
FORECAST = 1440
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_MAX_OPT_STEPS = 300
LEARNING_RATE = 1e-3
BATCH_SIZE = 4
ACCUM_STEPS = 8
ENCODER_HIDDEN_SIZE = 64
ENCODER_LAYERS = 2
ENCODER_DROPOUT = 0.1
DECODER_HIDDEN_SIZE = 64
DECODER_LAYERS = 2

ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "Results" / "UCI" / "Corrected_V2_LSTM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-aware UCI LSTM baseline on verified NPZ windows."
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Train the five seeds. Without this flag, run a smoke test only.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help="Random seeds (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_OPT_STEPS,
        help="Optimizer updates per seed (default: 300).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help="Physical batch size (default: 4).",
    )
    parser.add_argument(
        "--accum-steps", type=int, default=ACCUM_STEPS,
        help="Gradient accumulation steps (default: 8).",
    )
    parser.add_argument(
        "--device", choices=("auto", "mps", "cpu"), default="auto",
        help="Compute device. Auto tries MPS, then CUDA, then CPU.",
    )
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
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(f"Metric arrays differ: {yt.shape} vs {yp.shape}")
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


class NeuralForecastStyleLSTM(nn.Module):
    """LSTM encoder + per-timestep MLP decoder, direct 1440-step output."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=ENCODER_HIDDEN_SIZE,
            num_layers=ENCODER_LAYERS,
            bias=True,
            dropout=ENCODER_DROPOUT,
            batch_first=True,
        )
        decoder_layers: list[nn.Module] = []
        in_features = ENCODER_HIDDEN_SIZE
        for _ in range(DECODER_LAYERS):
            decoder_layers.append(nn.Linear(in_features, DECODER_HIDDEN_SIZE))
            decoder_layers.append(nn.ReLU())
            in_features = DECODER_HIDDEN_SIZE
        decoder_layers.append(nn.Linear(in_features, 1))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, lookback]; output: [batch, forecast]
        encoded, _ = self.encoder(x.unsqueeze(-1))
        horizon_states = encoded[:, -FORECAST:, :]
        prediction = self.decoder(horizon_states)
        return prediction.squeeze(-1)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Sample-weighted validation MSE, including a partial final batch."""
    model.eval()
    weighted_loss = 0.0
    sample_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            n = int(yb.shape[0])
            weighted_loss += float(criterion(pred, yb).item()) * n
            sample_count += n
    if sample_count == 0:
        raise RuntimeError("Validation loader is empty.")
    return weighted_loss / sample_count


def model_smoke_test(device: torch.device) -> tuple[torch.device, int]:
    """Check forward and backward; transparently fall back from unsupported MPS."""
    for candidate in ([device, torch.device("cpu")] if device.type == "mps" else [device]):
        model = NeuralForecastStyleLSTM().to(candidate)
        sample = torch.zeros((1, LOOKBACK), dtype=torch.float32, device=candidate)
        target = torch.zeros((1, FORECAST), dtype=torch.float32, device=candidate)
        try:
            model.train()
            prediction = model(sample)
            if tuple(prediction.shape) != (1, FORECAST):
                raise RuntimeError(f"Unexpected model output shape: {tuple(prediction.shape)}")
            loss = nn.MSELoss()(prediction, target)
            loss.backward()
            parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if candidate.type == "mps":
                torch.mps.synchronize()
            print(
                f"LSTM forward/backward smoke test PASSED on {candidate}: "
                f"input={tuple(sample.shape)}, output={tuple(prediction.shape)}"
            )
            del model, sample, target, prediction, loss
            gc.collect()
            if candidate.type == "mps":
                torch.mps.empty_cache()
            return candidate, parameter_count
        except Exception as exc:
            del model, sample, target
            gc.collect()
            if candidate.type == "mps":
                torch.mps.empty_cache()
                print(f"WARNING: LSTM MPS smoke test failed: {exc}")
                print("Falling back to CPU for this LSTM experiment.")
                continue
            raise
    raise RuntimeError("Could not run the LSTM smoke test on MPS or CPU.")


def run_one_seed(
    seed: int,
    arrays: dict[str, Any],
    scaler: Any,
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

    model = NeuralForecastStyleLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    print(f"\n{'=' * 65}\nLSTM — SEED {seed}\n{'=' * 65}")

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
            prediction = model(xb.to(device))
            prediction_batches.append(prediction.detach().cpu().numpy())

    prediction_scaled = np.concatenate(prediction_batches, axis=0)
    prediction_raw = inverse_array(prediction_scaled, scaler).astype(np.float32)
    y_true_raw = arrays["test"]["y_raw"].astype(np.float32)
    if prediction_raw.shape != y_true_raw.shape:
        raise RuntimeError(
            f"Prediction shape {prediction_raw.shape} != target shape {y_true_raw.shape}"
        )

    metrics = compute_metrics(y_true_raw, prediction_raw)
    np.save(result_dir / f"LSTM_seed{seed}_predictions.npy", prediction_raw)
    torch.save(best_state, result_dir / f"LSTM_seed{seed}_best.pt")

    row = {
        "Model": "LSTM",
        "Seed": seed,
        "Best_Val_MSE_Scaled": best_val,
        **metrics,
    }
    progress_file = result_dir / "lstm_corrected_progress.csv"
    if progress_file.exists():
        old = pd.read_csv(progress_file)
        old = old[old["Seed"] != seed]
        progress = pd.concat([old, pd.DataFrame([row])], ignore_index=True)
    else:
        progress = pd.DataFrame([row])
    progress.sort_values("Seed").to_csv(progress_file, index=False)

    print("\nSeed results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    del model, optimizer, train_loader, val_loader, test_loader
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return row


def main() -> int:
    args = parse_args()
    if min(args.max_steps, args.batch_size, args.accum_steps) <= 0:
        raise ValueError("max-steps, batch-size and accum-steps must be positive.")

    print("Device requested:", args.device)
    arrays = load_uci_windows(project_root=ROOT)
    for split, expected in (("train", 903), ("val", 189), ("test", 192)):
        X = arrays[split]["X"]
        y = arrays[split]["y"]
        if X.shape != (expected, LOOKBACK) or y.shape != (expected, FORECAST):
            raise RuntimeError(f"Unexpected {split} shape: X={X.shape}, y={y.shape}")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise RuntimeError(f"Non-finite values found in {split} windows")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULT_DIR / "y_true_raw.npy", arrays["test"]["y_raw"].astype(np.float32))
    np.save(
        RESULT_DIR / "test_target_timestamps.npy",
        np.stack([
            arrays["all_timestamps"][int(start) + LOOKBACK:
                                     int(start) + LOOKBACK + FORECAST]
            for start in arrays["test"]["starts_global"]
        ]).astype("datetime64[ns]").astype(str),
    )

    initial_device = choose_device(args.device)
    print("Initial device:", initial_device)
    device, parameter_count = model_smoke_test(initial_device)
    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Results directory: {RESULT_DIR.resolve()}")

    run_config = {
        "experiment": "uci_lstm_corrected_v2_no_interpolated_splits",
        "model": "LSTM (NeuralForecast-style direct LSTM encoder + MLP decoder)",
        "architecture_reference": "Nixtla NeuralForecast LSTM: LSTM encoder sequence followed by an MLP decoder; recurrent=False",
        "architecture_source_url": "https://github.com/Nixtla/neuralforecast/blob/main/neuralforecast/models/lstm.py",
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
        "loss": "MSE",
        "lookback_minutes": LOOKBACK,
        "forecast_minutes": FORECAST,
        "train_windows": 903,
        "validation_windows": 189,
        "test_windows": 192,
        "encoder_hidden_size": ENCODER_HIDDEN_SIZE,
        "encoder_layers": ENCODER_LAYERS,
        "encoder_dropout": ENCODER_DROPOUT,
        "decoder_hidden_size": DECODER_HIDDEN_SIZE,
        "decoder_layers": DECODER_LAYERS,
        "decoder_activation": "ReLU",
        "decoder_dropout": 0.0,
        "trainable_parameters": parameter_count,
        "device": str(device),
        "validation_loss_aggregation": "sample-weighted mean across windows; partial final batch weighted by sample count",
        "checkpoint_selection": "Best checkpoint selected only by validation MSE; test set evaluated after checkpoint selection.",
        "inverse_transform": "training-only MinMaxScaler from corrected preprocessing",
        "notes": [
            "The original NeuralForecast-style LSTM encoder+MLP decoder is implemented directly in PyTorch to train on fixed, audited input-target windows.",
            "The LSTM consumes only the 1440-minute input window. No future target value is used as an encoder input.",
            "With lookback equal to forecast, the MLP maps the 1440 encoder hidden states directly to 1440 forecast values.",
            "If MPS does not support the LSTM forward/backward path in this environment, the script falls back to CPU and records the actual device."
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
            scaler=arrays["scaler"],
            result_dir=RESULT_DIR,
            device=device,
            max_opt_steps=args.max_steps,
            batch_size=args.batch_size,
            accum_steps=args.accum_steps,
        )
        all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(RESULT_DIR / "lstm_corrected_results.csv", index=False)

    results = pd.DataFrame(all_rows).sort_values("Seed")
    results.to_csv(RESULT_DIR / "lstm_corrected_results.csv", index=False)

    metric_cols = ["RMSE", "MAE", "MAPE", "R2", "StdR"]
    summary_rows = []
    for metric in metric_cols:
        values = results[metric].astype(float)
        summary_rows.append({"Metric": metric, "mean": values.mean(), "sd": values.std(ddof=1)})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULT_DIR / "lstm_corrected_summary.csv", index=False)

    print("\nFive-seed corrected LSTM results:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nMean ± sample standard deviation:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved all outputs under: {RESULT_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)
