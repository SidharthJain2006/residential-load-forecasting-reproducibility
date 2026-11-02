#!/usr/bin/env python3
"""Train the five-seed Vanilla Transformer on leakage-aware UCI windows.

Version 2.1 fixes validation aggregation: batch losses are weighted by
batch size, so a short final batch is not overweighted. On MPS, attention-
probability dropout is explicitly disabled for backend compatibility; other
Transformer dropout layers remain at 0.1.

Default behavior is a smoke test only. Pass --train to start optimization.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path
from typing import Iterable

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
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-aware UCI Vanilla Transformer experiment."
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Actually train the model. Without this flag, run a smoke test only.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help="Random seeds to run (default: 0 1 2 3 4).",
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
        help="Compute device. CUDA is used automatically only if available under auto.",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable in this PyTorch environment.")
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
    mape = float(np.mean(np.abs((yt[valid] - yp[valid]) / yt[valid])) * 100.0) if valid.any() else float("nan")
    r2 = float(r2_score(yt, yp))
    true_std = float(np.std(yt))
    stdr = float(np.std(yp) / true_std) if true_std > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "StdR": stdr}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class VanillaTransformer(nn.Module):
    def __init__(self, attention_dropout: float) -> None:
        super().__init__()
        self.input_projection = nn.Linear(1, D_MODEL)
        self.position = PositionalEncoding(D_MODEL, LOOKBACK)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        # Keep residual/feed-forward dropout at DROPOUT, but configure
        # attention-probability dropout separately for device compatibility.
        encoder_layer.self_attn.dropout = float(attention_dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.output_head = nn.Sequential(
            nn.Linear(D_MODEL, 128),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, FORECAST),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, lookback], yhat: [batch, forecast]
        x = x.unsqueeze(-1)
        x = self.input_projection(x)
        x = self.position(x)
        x = self.encoder(x)
        representation = x[:, -1, :]
        return self.output_head(representation)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    """Return sample-weighted validation MSE, including a short final batch."""
    model.eval()
    weighted_loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            batch_n = int(yb.shape[0])
            weighted_loss_sum += float(criterion(pred, yb).item()) * batch_n
            sample_count += batch_n
    if sample_count == 0:
        raise RuntimeError("Validation loader is empty.")
    return weighted_loss_sum / sample_count


def run_one_seed(
    seed: int,
    arrays: dict,
    scaler,
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

    attention_dropout = 0.0 if device.type == "mps" else DROPOUT
    model = VanillaTransformer(attention_dropout=attention_dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0

    print(f"\n{'=' * 65}\nVANILLA TRANSFORMER — SEED {seed}\n{'=' * 65}")
    model.train()

    while micro_step // accum_steps < max_opt_steps:
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            xb, yb = next(iterator)

        xb = xb.to(device)
        yb = yb.to(device)
        model.train()
        pred = model(xb)
        loss = criterion(pred, yb) / accum_steps
        loss.backward()
        micro_step += 1

        if micro_step % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step = micro_step // accum_steps

            if opt_step == 1 or opt_step % 25 == 0 or opt_step == max_opt_steps:
                val_loss = evaluate(model, val_loader, criterion, device)
                print(f"Seed {seed} | optimizer step {opt_step}/{max_opt_steps} | val MSE={val_loss:.7f}")
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }

    if best_state is None:
        raise RuntimeError(f"No best model state captured for seed {seed}.")

    model.load_state_dict(best_state)
    model.eval()
    scaled_predictions: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in test_loader:
            pred = model(xb.to(device))
            scaled_predictions.append(pred.detach().cpu().numpy())

    pred_scaled = np.concatenate(scaled_predictions, axis=0)
    pred_raw = inverse_array(pred_scaled, scaler).astype(np.float32)
    y_true_raw = arrays["test"]["y_raw"].astype(np.float32)

    if pred_raw.shape != y_true_raw.shape:
        raise RuntimeError(f"Prediction shape {pred_raw.shape} != target shape {y_true_raw.shape}")

    metrics = compute_metrics(y_true_raw, pred_raw)
    prediction_path = result_dir / f"VanillaTransformer_seed{seed}_predictions.npy"
    np.save(prediction_path, pred_raw)
    torch.save(best_state, result_dir / f"VanillaTransformer_seed{seed}_best.pt")

    row = {"Model": "Vanilla Transformer", "Seed": seed, "Best_Val_MSE_Scaled": best_val, **metrics}
    progress_path = result_dir / "vanilla_transformer_corrected_v2_1_progress.csv"
    previous = pd.read_csv(progress_path) if progress_path.exists() else pd.DataFrame()
    previous = previous[~((previous.get("Model", pd.Series(dtype=str)) == "Vanilla Transformer") & (previous.get("Seed", pd.Series(dtype=float)) == seed))] if not previous.empty else previous
    pd.concat([previous, pd.DataFrame([row])], ignore_index=True).to_csv(progress_path, index=False)

    del model, optimizer, train_loader, val_loader, test_loader
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main() -> int:
    args = parse_args()
    if args.max_steps <= 0 or args.batch_size <= 0 or args.accum_steps <= 0:
        raise ValueError("max-steps, batch-size and accum-steps must be positive.")
    if not args.seeds:
        raise ValueError("At least one seed is required.")

    device = choose_device(args.device)
    print(f"Device: {device}")
    print("Experiment version: corrected v2.1 (sample-weighted validation)")
    print(f"Attention-probability dropout: {0.0 if device.type == 'mps' else DROPOUT}")

    arrays = load_uci_windows()
    scaler = arrays["scaler"]

    for split, expected in (("train", 903), ("val", 189), ("test", 192)):
        X = arrays[split]["X"]
        y = arrays[split]["y"]
        y_raw = arrays[split]["y_raw"]
        if X.shape != (expected, LOOKBACK) or y.shape != (expected, FORECAST):
            raise RuntimeError(f"Unexpected {split} shape: X={X.shape}, y={y.shape}")
        if not np.isfinite(X).all() or not np.isfinite(y).all() or not np.isfinite(y_raw).all():
            raise RuntimeError(f"Non-finite values found in {split} windows")

    result_dir = Path("Results/UCI/Corrected_V2_1_VanillaTransformer")
    result_dir.mkdir(parents=True, exist_ok=True)

    np.save(result_dir / "y_true_raw.npy", arrays["test"]["y_raw"].astype(np.float32))
    np.save(result_dir / "test_target_timestamps.npy", arrays["test"]["target_start"].astype("datetime64[ns]").astype(str))
    run_config = {
        "model": "Vanilla Transformer",
        "dataset": "UCI Household Electric Power Consumption",
        "data_source": "Data/UCI/processed/*.npz",
        "seeds": args.seeds,
        "max_optimizer_steps_per_seed": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accum_steps,
        "validation_interval_optimizer_steps": 25,
        "effective_batch_size_approx": args.batch_size * args.accum_steps,
        "lookback_minutes": LOOKBACK,
        "forecast_minutes": FORECAST,
        "train_windows": 903,
        "validation_windows": 189,
        "test_windows": 192,
        "device": str(device),
        "attention_probability_dropout": 0.0 if device.type == "mps" else DROPOUT,
        "encoder_residual_and_feedforward_dropout": DROPOUT,
        "validation_loss_aggregation": "sample-weighted mean across windows; partial final batch weighted by sample count",
        "script_version": "corrected_v2.1_weighted_validation",
        "selection": "Best checkpoint selected only by validation MSE; test set used only after selection.",
        "old_interpolated_csv_files_used": False,
    }
    with (result_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    print("\nCorrected windows loaded and checked.")
    print(f"  train: X={arrays['train']['X'].shape}, y={arrays['train']['y'].shape}")
    print(f"  val:   X={arrays['val']['X'].shape}, y={arrays['val']['y'].shape}")
    print(f"  test:  X={arrays['test']['X'].shape}, y={arrays['test']['y'].shape}")
    print(f"Results directory: {result_dir.resolve()}")

    # Smoke test: one forward pass on one clean input window.
    attention_dropout = 0.0 if device.type == "mps" else DROPOUT
    model = VanillaTransformer(attention_dropout=attention_dropout).to(device)
    with torch.no_grad():
        sample = torch.from_numpy(arrays["train"]["X"][:1]).to(device)
        output = model(sample)
    if tuple(output.shape) != (1, FORECAST):
        raise RuntimeError(f"Unexpected smoke-test output shape: {tuple(output.shape)}")
    print(f"\nModel forward-pass smoke test PASSED: input={tuple(sample.shape)}, output={tuple(output.shape)}")
    del model, output, sample
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    if not args.train:
        print("\nSmoke test only. No model weights were trained.")
        print("When ready to train, run this script again with --train.")
        return 0

    all_rows = []
    for seed in args.seeds:
        row = run_one_seed(
            seed=seed,
            arrays=arrays,
            scaler=scaler,
            result_dir=result_dir,
            device=device,
            max_opt_steps=args.max_steps,
            batch_size=args.batch_size,
            accum_steps=args.accum_steps,
        )
        all_rows.append(row)
        pd.DataFrame(all_rows).to_csv(result_dir / "vanilla_transformer_corrected_v2_1_results.csv", index=False)

    results_df = pd.DataFrame(all_rows)
    summary = results_df[["RMSE", "MAE", "MAPE", "R2", "StdR"]].agg(["mean", "std"]).T
    summary.columns = ["mean", "sd"]
    summary.to_csv(result_dir / "vanilla_transformer_corrected_v2_1_summary.csv")
    print("\nFive-seed corrected Vanilla Transformer results:")
    print(results_df.to_string(index=False))
    print("\nMean ± sample standard deviation:")
    print(summary.to_string())
    print(f"\nSaved all outputs under: {result_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
