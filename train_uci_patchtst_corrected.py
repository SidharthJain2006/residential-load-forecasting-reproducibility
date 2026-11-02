#!/usr/bin/env python3
"""Train a five-seed PatchTST-style baseline on leakage-aware UCI windows.

The implementation uses overlapping univariate patches (patch length 16,
stride 8), a Transformer encoder, and a direct flattening forecast head.
It consumes only the corrected NPZ windows exported by prepare_uci.py.

Default behavior is a smoke test. Pass --train to train five seeds.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
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
SEEDS = (0, 1, 2, 3, 4)
MAX_OPT_STEPS = 300
LEARNING_RATE = 1e-3
BATCH_SIZE = 4
ACCUM_STEPS = 8
PATCH_LEN = 16
PATCH_STRIDE = 8
D_MODEL = 64
NHEAD = 4
ENCODER_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Corrected UCI PatchTST-style experiment")
    parser.add_argument("--train", action="store_true", help="Run the five-seed experiment; otherwise smoke-test only.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--max-steps", type=int, default=MAX_OPT_STEPS)
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1], :]


class PatchTSTStyle(nn.Module):
    """Univariate PatchTST-style direct forecaster."""
    def __init__(self, attention_dropout: float) -> None:
        super().__init__()
        self.patch_count = (LOOKBACK - PATCH_LEN) // PATCH_STRIDE + 1
        self.patch_embedding = nn.Linear(PATCH_LEN, D_MODEL)
        self.position = PositionalEncoding(D_MODEL, self.patch_count)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        # MPS SDPA requires zero attention-probability dropout in this path.
        layer.self_attn.dropout = float(attention_dropout)
        self.encoder = nn.TransformerEncoder(layer, num_layers=ENCODER_LAYERS)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.patch_count * D_MODEL, 128),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, FORECAST),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, lookback]. Patches overlap by PATCH_LEN - PATCH_STRIDE.
        patches = x.unfold(dimension=1, size=PATCH_LEN, step=PATCH_STRIDE)
        z = self.patch_embedding(patches)
        z = self.position(z)
        z = self.encoder(z)
        return self.head(z)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    loss_sum = 0.0
    n_sum = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss_sum += float(criterion(model(xb), yb).item()) * int(yb.shape[0])
            n_sum += int(yb.shape[0])
    if n_sum == 0:
        raise RuntimeError("Validation loader is empty.")
    return loss_sum / n_sum


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape or not np.isfinite(yt).all() or not np.isfinite(yp).all():
        raise ValueError("Invalid or mismatched metric arrays.")
    mape_mask = np.abs(yt) > 1e-8
    return {
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE": float(mean_absolute_error(yt, yp)),
        "MAPE": float(np.mean(np.abs((yt[mape_mask] - yp[mape_mask]) / yt[mape_mask])) * 100.0) if mape_mask.any() else float("nan"),
        "R2": float(r2_score(yt, yp)),
        "StdR": float(np.std(yp) / np.std(yt)) if np.std(yt) > 0 else float("nan"),
    }


def save_run_config(result_dir: Path, device: torch.device, args: argparse.Namespace) -> None:
    config: dict[str, Any] = {
        "model": "PatchTST-style univariate direct forecaster",
        "dataset": "UCI Household Electric Power Consumption",
        "data_source": "verified corrected Data/UCI/processed/*.npz windows only",
        "seeds": args.seeds,
        "max_optimizer_steps_per_seed": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accum_steps,
        "effective_batch_size_approx": args.batch_size * args.accum_steps,
        "learning_rate": LEARNING_RATE,
        "lookback_minutes": LOOKBACK,
        "forecast_minutes": FORECAST,
        "train_windows": 903,
        "validation_windows": 189,
        "test_windows": 192,
        "patch_length": PATCH_LEN,
        "patch_stride": PATCH_STRIDE,
        "patch_count": (LOOKBACK - PATCH_LEN) // PATCH_STRIDE + 1,
        "d_model": D_MODEL,
        "n_heads": NHEAD,
        "encoder_layers": ENCODER_LAYERS,
        "feedforward_dimension": DIM_FEEDFORWARD,
        "dropout": DROPOUT,
        "attention_probability_dropout": 0.0 if device.type == "mps" else DROPOUT,
        "validation_loss_aggregation": "sample-weighted mean across windows",
        "checkpoint_selection": "best validation MSE; test evaluated only after selection",
        "implementation_note": "Custom PyTorch PatchTST-style univariate implementation with overlapping patches and direct flattening head; not the Nixtla NeuralForecast class implementation.",
        "old_interpolated_csv_files_used": False,
    }
    (result_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def run_one_seed(seed: int, arrays: dict[str, Any], scaler: Any, result_dir: Path,
                 device: torch.device, max_steps: int, batch_size: int, accum_steps: int) -> dict[str, float]:
    set_seed(seed)
    train_loader = make_loader(arrays["train"]["X"], arrays["train"]["y"], batch_size, True)
    val_loader = make_loader(arrays["val"]["X"], arrays["val"]["y"], batch_size, False)
    test_loader = make_loader(arrays["test"]["X"], arrays["test"]["y"], batch_size, False)
    attention_dropout = 0.0 if device.type == "mps" else DROPOUT
    model = PatchTSTStyle(attention_dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    micro_steps = 0
    opt_step = 0

    print(f"\n{'=' * 65}\nPATCHTST-STYLE — SEED {seed}\n{'=' * 65}")
    while opt_step < max_steps:
        model.train()
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            xb, yb = next(iterator)
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(model(xb), yb) / accum_steps
        loss.backward()
        micro_steps += 1

        if micro_steps % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
            if opt_step == 1 or opt_step % 25 == 0 or opt_step == max_steps:
                val_loss = evaluate(model, val_loader, criterion, device)
                print(f"Seed {seed} | optimizer step {opt_step}/{max_steps} | val MSE={val_loss:.7f}")
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"No best checkpoint captured for seed {seed}.")

    model.load_state_dict(best_state)
    model.eval()
    pred_scaled_batches: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in test_loader:
            pred_scaled_batches.append(model(xb.to(device)).detach().cpu().numpy())
    pred_scaled = np.concatenate(pred_scaled_batches, axis=0)
    pred_raw = inverse_array(pred_scaled, scaler).astype(np.float32)
    y_true_raw = arrays["test"]["y_raw"].astype(np.float32)
    if pred_raw.shape != y_true_raw.shape:
        raise RuntimeError(f"Prediction shape {pred_raw.shape} != target shape {y_true_raw.shape}")
    metrics = compute_metrics(y_true_raw, pred_raw)

    np.save(result_dir / f"PatchTST_seed{seed}_predictions.npy", pred_raw)
    torch.save(best_state, result_dir / f"PatchTST_seed{seed}_best.pt")
    row = {"Model": "PatchTST-style", "Seed": seed, "Best_Val_MSE_Scaled": best_val, **metrics}
    progress_path = result_dir / "patchtst_corrected_progress.csv"
    old = pd.read_csv(progress_path) if progress_path.exists() else pd.DataFrame()
    if not old.empty and {"Model", "Seed"}.issubset(old.columns):
        old = old[~((old["Model"] == row["Model"]) & (old["Seed"] == seed))]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(progress_path, index=False)
    del model, optimizer, train_loader, val_loader, test_loader
    gc.collect()
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    return row


def main() -> int:
    args = parse_args()
    if args.max_steps < 1 or args.batch_size < 1 or args.accum_steps < 1:
        raise ValueError("max-steps, batch-size, and accum-steps must be positive.")
    if args.seeds != list(args.seeds) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique integers.")

    device = choose_device(args.device)
    print("Device:", device)
    print("Experiment: corrected UCI PatchTST-style baseline")
    print(f"Patch length={PATCH_LEN}, stride={PATCH_STRIDE}, tokens={(LOOKBACK - PATCH_LEN) // PATCH_STRIDE + 1}")
    print("Attention-probability dropout:", 0.0 if device.type == "mps" else DROPOUT)

    data = load_uci_windows()
    arrays = {name: data[name] for name in ("train", "val", "test")}
    result_dir = Path(data["root"]) / "Results" / "UCI" / "Corrected_V2_PatchTST"
    result_dir.mkdir(parents=True, exist_ok=True)
    save_run_config(result_dir, device, args)
    print("Results directory:", result_dir)

    # Smoke-test forward and backward on the actual corrected data.
    model = PatchTSTStyle(attention_dropout=0.0 if device.type == "mps" else DROPOUT).to(device)
    xb = torch.as_tensor(arrays["train"]["X"][:2], dtype=torch.float32, device=device).requires_grad_(True)
    yhat = model(xb)
    if yhat.shape != (2, FORECAST):
        raise RuntimeError(f"Unexpected PatchTST output shape: {tuple(yhat.shape)}")
    yhat.square().mean().backward()
    if xb.grad is None or not torch.isfinite(xb.grad).all():
        raise RuntimeError("PatchTST smoke-test gradients are missing or non-finite.")
    print(f"PatchTST-style forward/backward smoke test PASSED: input={tuple(xb.shape)}, output={tuple(yhat.shape)}")
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    del model, xb, yhat
    gc.collect()

    if not args.train:
        print("\nSmoke test only. No model weights were trained.")
        print("When ready to train, run this script again with --train.")
        return 0

    rows = []
    for seed in args.seeds:
        row = run_one_seed(seed, arrays, data["scaler"], result_dir, device,
                           args.max_steps, args.batch_size, args.accum_steps)
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("Seed")
    results.to_csv(result_dir / "patchtst_corrected_results.csv", index=False)
    metric_cols = ["RMSE", "MAE", "MAPE", "R2", "StdR"]
    summary = pd.DataFrame({"mean": results[metric_cols].mean(), "sd": results[metric_cols].std(ddof=1)})
    summary.to_csv(result_dir / "patchtst_corrected_summary.csv")

    # Save full target timestamps aligned with every test horizon.
    starts = arrays["test"]["starts_global"]
    all_ts = data["all_timestamps"]
    target_ts = np.stack([all_ts[s + LOOKBACK:s + LOOKBACK + FORECAST] for s in starts])
    np.save(result_dir / "test_target_timestamps.npy", target_ts)
    np.save(result_dir / "y_true_raw.npy", arrays["test"]["y_raw"].astype(np.float32))

    print("\nFive-seed corrected PatchTST-style results:")
    print(results.to_string(index=False))
    print("\nMean ± sample standard deviation:")
    print(summary.to_string())
    print("\nSaved all outputs under:", result_dir)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR:", exc)
        raise SystemExit(1)
