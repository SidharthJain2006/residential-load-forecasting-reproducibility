#!/usr/bin/env python3
"""Corrected five-seed MORED PatchTST-style baseline on audited windows.

The model is a custom PyTorch univariate patch Transformer inspired by the
supplied MORED NeuralForecast PatchTST configuration. It consumes only the
segment-safe, no-interpolation NPZ windows in Data/MORED/Corrected_V3_SessionSplit.

Original supplied MORED PatchTST settings retained where applicable:
  encoder_layers=2, n_heads=4, hidden_size=64, linear_hidden_size=128,
  dropout=0.1, fc_dropout=0.1, head_dropout=0.0, attn_dropout=0.0,
  patch_len=16, stride=8.

This is not an exact reproduction of the NeuralForecast PatchTST class.
The corrected experiment uses the same memory-conscious protocol used by the
other corrected custom MORED models (batch 4, accumulation 8, Adam 1e-3,
300 optimizer updates, best sample-weighted validation-MSE checkpoint).

Default: validate data and run forward/backward smoke test only.
Training: python train_mored_patchtst_corrected.py --train
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

LOOKBACK = 1440
FORECAST = 1440
EXPECTED_WINDOWS = {"train": 245, "val": 119, "test": 359}
SEEDS_DEFAULT = [0, 1, 2, 3, 4]
MAX_STEPS_DEFAULT = 300
LEARNING_RATE = 1e-3
BATCH_SIZE_DEFAULT = 4
ACCUM_STEPS_DEFAULT = 8
VALIDATION_INTERVAL = 25

# Supplied MORED PatchTST hyperparameters.
PATCH_LEN = 16
PATCH_STRIDE = 8
D_MODEL = 64
N_HEADS = 4
ENCODER_LAYERS = 2
LINEAR_HIDDEN_SIZE = 128
DROPOUT = 0.1
FC_DROPOUT = 0.1
HEAD_DROPOUT = 0.0
ATTENTION_DROPOUT = 0.0

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR_DEFAULT = PROJECT_ROOT / "Data" / "MORED" / "Corrected_V3_SessionSplit"
RESULT_DIR_DEFAULT = PROJECT_ROOT / "Results" / "MORED" / "Corrected_V3_SessionSplit_PatchTST"


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
    required_shapes = {
        "X": (n, LOOKBACK), "y": (n, FORECAST),
        "X_raw": (n, LOOKBACK), "y_raw": (n, FORECAST),
    }
    for key, shape in required_shapes.items():
        if key not in d or d[key].shape != shape:
            raise RuntimeError(f"{split} {key}: got {d.get(key, np.empty(0)).shape}; expected {shape}")
        if not np.isfinite(d[key]).all():
            raise RuntimeError(f"{split} {key} contains non-finite values")
    for key in ("input_start", "input_end", "target_start", "target_end"):
        if key not in d or len(d[key]) != n:
            raise RuntimeError(f"{split} missing or invalid timestamps: {key}")
    start = pd.to_datetime(d["target_start"]).to_numpy(dtype="datetime64[ns]")
    d["target_timestamps"] = start[:, None] + np.arange(FORECAST).astype("timedelta64[m]")[None, :]
    if not np.all(np.diff(d["target_timestamps"], axis=1) == np.timedelta64(1, "m")):
        raise RuntimeError(f"{split} has a non-contiguous target window")
    return d


def load_data(data_dir: Path):
    metadata_path = data_dir / "preprocessing_metadata.json"
    scaler_path = data_dir / "scaler.pkl"
    train_csv = data_dir / "train.csv"
    if not metadata_path.is_file() or not scaler_path.is_file() or not train_csv.is_file():
        raise FileNotFoundError(f"Corrected MORED metadata/scaler/train.csv missing in {data_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("experiment") != "mored_corrected_v3_session_split_no_interpolation_raw_count_filter":
        raise RuntimeError("Unexpected corrected MORED preprocessing version")
    missing_policy = metadata.get("minute_aggregation", {}).get("missing_sample_policy", "")
    if "no interpolation" not in missing_policy.lower():
        raise RuntimeError("Metadata does not confirm the no-interpolation policy")
    arrays = {s: load_split(data_dir, s) for s in ("train", "val", "test")}
    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)
    train_frame = pd.read_csv(train_csv)
    genuine_train = train_frame["consumption"].to_numpy(dtype=np.float64)
    if not np.isfinite(genuine_train).all():
        raise RuntimeError("Corrected training CSV contains non-finite readings")
    if int(scaler.n_samples_seen_) != len(genuine_train):
        raise RuntimeError("Scaler sample count differs from retained training-minute rows")
    if not np.isclose(float(scaler.data_min_[0]), float(genuine_train.min())):
        raise RuntimeError("Scaler minimum differs from corrected training readings")
    if not np.isclose(float(scaler.data_max_[0]), float(genuine_train.max())):
        raise RuntimeError("Scaler maximum differs from corrected training readings")
    for name, d in arrays.items():
        ex = scaler.transform(d["X_raw"].reshape(-1, 1)).reshape(d["X"].shape)
        ey = scaler.transform(d["y_raw"].reshape(-1, 1)).reshape(d["y"].shape)
        if not np.allclose(d["X"], ex, atol=2e-6):
            raise RuntimeError(f"{name} scaled inputs disagree with the training-only scaler")
        if not np.allclose(d["y"], ey, atol=2e-6):
            raise RuntimeError(f"{name} scaled targets disagree with the training-only scaler")
    print("Loaded corrected MORED windows:")
    for name, d in arrays.items():
        print(f"  {name:5s} X={d['X'].shape}, y={d['y'].shape}, y_raw={d['y_raw'].shape}")
    print("Scaler verified against corrected training-minute split.")
    print("No interpolation is used in these windows.")
    return arrays, scaler, metadata


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1], :]


class PatchTSTStyle(nn.Module):
    """Custom univariate patch Transformer with a direct 1,440-step head."""
    def __init__(self, attention_dropout: float = ATTENTION_DROPOUT) -> None:
        super().__init__()
        self.patch_count = (LOOKBACK - PATCH_LEN) // PATCH_STRIDE + 1
        self.patch_embedding = nn.Linear(PATCH_LEN, D_MODEL)
        self.position = PositionalEncoding(D_MODEL, self.patch_count)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=LINEAR_HIDDEN_SIZE,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        # Explicitly keep attention-probability dropout at the supplied value.
        layer.self_attn.dropout = float(attention_dropout)
        self.encoder = nn.TransformerEncoder(layer, num_layers=ENCODER_LAYERS)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.patch_count * D_MODEL, LINEAR_HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(FC_DROPOUT),
            nn.Linear(LINEAR_HIDDEN_SIZE, FORECAST),
        )
        self.head_dropout = nn.Dropout(HEAD_DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=PATCH_LEN, step=PATCH_STRIDE)
        z = self.patch_embedding(patches)
        z = self.position(z)
        z = self.encoder(z)
        z = self.head_dropout(z)
        return self.head(z)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0)


def sample_weighted_val(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.functional.mse_loss(model(xb), yb, reduction="mean")
            total += float(loss.item()) * len(xb)
            count += len(xb)
    if count == 0:
        raise RuntimeError("Validation loader is empty")
    return total / count


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape or not np.isfinite(yt).all() or not np.isfinite(yp).all():
        raise ValueError("Metric arrays are non-finite or have different shapes")
    mask = np.abs(yt) > 1e-8
    return {
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAE": float(mean_absolute_error(yt, yp)),
        "MAPE": float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0) if mask.any() else float("nan"),
        "R2": float(r2_score(yt, yp)),
        "StdR": float(np.std(yp) / np.std(yt)) if np.std(yt) > 0 else float("nan"),
    }


def inverse_array(a: np.ndarray, scaler: Any) -> np.ndarray:
    return scaler.inverse_transform(np.asarray(a).reshape(-1, 1)).reshape(a.shape)


def smoke_test(device: torch.device) -> int:
    model = PatchTSTStyle().to(device)
    xb = torch.as_tensor(np.zeros((2, LOOKBACK), dtype=np.float32), device=device).requires_grad_(True)
    yb = torch.zeros((2, FORECAST), dtype=torch.float32, device=device)
    pred = model(xb)
    if pred.shape != (2, FORECAST):
        raise RuntimeError(f"Unexpected PatchTST output shape: {tuple(pred.shape)}")
    nn.functional.mse_loss(pred, yb).backward()
    if xb.grad is None or not torch.isfinite(xb.grad).all():
        raise RuntimeError("PatchTST smoke-test input gradients are missing or non-finite")
    if device.type == "mps":
        torch.mps.synchronize()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PatchTST-style forward/backward smoke test PASSED on {device}: input={tuple(xb.shape)}, output={tuple(pred.shape)}")
    print(f"Trainable parameters: {count:,}")
    del model, xb, yb, pred
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return count


def run_seed(seed: int, arrays: dict[str, Any], scaler: Any, result_dir: Path,
             device: torch.device, max_steps: int, batch_size: int, accum_steps: int) -> dict[str, Any]:
    set_seed(seed)
    train_loader = make_loader(arrays["train"]["X"], arrays["train"]["y"], batch_size, True)
    val_loader = make_loader(arrays["val"]["X"], arrays["val"]["y"], batch_size, False)
    test_loader = make_loader(arrays["test"]["X"], arrays["test"]["y"], batch_size, False)
    model = PatchTSTStyle().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    micro_steps = 0
    optimizer_steps = 0
    print(f"\n{'=' * 65}\nMORED PATCHTST-STYLE — SEED {seed}\n{'=' * 65}")
    while optimizer_steps < max_steps:
        model.train()
        try:
            xb, yb = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            xb, yb = next(iterator)
        xb, yb = xb.to(device), yb.to(device)
        (criterion(model(xb), yb) / accum_steps).backward()
        micro_steps += 1
        if micro_steps % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            if optimizer_steps == 1 or optimizer_steps % VALIDATION_INTERVAL == 0 or optimizer_steps == max_steps:
                val_loss = sample_weighted_val(model, val_loader, device)
                print(f"Seed {seed} | optimizer step {optimizer_steps}/{max_steps} | val MSE={val_loss:.7f}")
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError(f"No best checkpoint captured for seed {seed}")
    model.load_state_dict(best_state)
    model.eval()
    pred_batches: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in test_loader:
            pred_batches.append(model(xb.to(device)).detach().cpu().numpy())
    pred_raw = inverse_array(np.concatenate(pred_batches, axis=0), scaler).astype(np.float32)
    y_true_raw = arrays["test"]["y_raw"].astype(np.float32)
    if pred_raw.shape != y_true_raw.shape:
        raise RuntimeError(f"Prediction shape {pred_raw.shape} != target shape {y_true_raw.shape}")
    metrics = compute_metrics(y_true_raw, pred_raw)
    np.save(result_dir / f"PatchTST_seed{seed}_predictions.npy", pred_raw)
    torch.save(best_state, result_dir / f"PatchTST_seed{seed}_best.pt")
    row = {"Model": "PatchTST-style", "Seed": seed, "Best_Val_MSE_Scaled": best_val, **metrics}
    progress_path = result_dir / "patchtst_corrected_progress.csv"
    prior = pd.read_csv(progress_path) if progress_path.is_file() else pd.DataFrame()
    if not prior.empty and "Seed" in prior.columns:
        prior = prior[prior["Seed"] != seed]
    pd.concat([prior, pd.DataFrame([row])], ignore_index=True).sort_values("Seed").to_csv(progress_path, index=False)
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
    if min(args.max_steps, args.batch_size, args.accum_steps) < 1:
        raise ValueError("max-steps, batch-size and accum-steps must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be non-empty and unique")
    data_dir = args.data_dir.expanduser().resolve()
    result_dir = args.result_dir.expanduser().resolve()
    arrays, scaler, prep_metadata = load_data(data_dir)
    device = choose_device(args.device)
    parameter_count = smoke_test(device)
    result_dir.mkdir(parents=True, exist_ok=True)
    np.save(result_dir / "y_true_raw.npy", arrays["test"]["y_raw"].astype(np.float32))
    np.save(result_dir / "test_target_timestamps.npy", arrays["test"]["target_timestamps"].astype("datetime64[ns]").astype(str))
    config = {
        "experiment": "mored_patchtst_corrected_v3_session_split_no_interpolation_segment_safe_windows",
        "model": "Custom PyTorch PatchTST-style univariate direct forecaster",
        "dataset": "MORED Premises 6, APT6",
        "architecture_reference": "Supplied MORED NeuralForecast PatchTST configuration",
        "implementation_note": "Custom PyTorch patch Transformer with direct flattened horizon head; not an exact reproduction of the NeuralForecast PatchTST implementation.",
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
        "validation_interval_optimizer_steps": VALIDATION_INTERVAL,
        "validation_loss_aggregation": "sample-weighted mean across windows, including partial final batch",
        "checkpoint_selection": "best sample-weighted validation MSE; test evaluated only after checkpoint selection",
        "lookback_minutes": LOOKBACK,
        "forecast_minutes": FORECAST,
        "forecast_origin_stride_minutes": 60,
        "train_windows": EXPECTED_WINDOWS["train"],
        "validation_windows": EXPECTED_WINDOWS["val"],
        "test_windows": EXPECTED_WINDOWS["test"],
        "patch_length": PATCH_LEN,
        "patch_stride": PATCH_STRIDE,
        "patch_tokens": (LOOKBACK - PATCH_LEN) // PATCH_STRIDE + 1,
        "encoder_layers": ENCODER_LAYERS,
        "n_heads": N_HEADS,
        "hidden_size": D_MODEL,
        "linear_hidden_size": LINEAR_HIDDEN_SIZE,
        "dropout": DROPOUT,
        "fc_dropout": FC_DROPOUT,
        "head_dropout": HEAD_DROPOUT,
        "attention_dropout": ATTENTION_DROPOUT,
        "trainable_parameters": parameter_count,
        "device": str(device),
        "metric_aggregation": "Flatten all accepted test-window and forecast-minute values; test forecast origins are separated by 60 minutes and overlap in their 24-hour horizons.",
        "source_npz_sha256": {s: sha256_file(data_dir / f"{s}_windows.npz") for s in ("train", "val", "test")},
        "note": "The architecture hyperparameters follow the supplied MORED PatchTST kwargs, reimplemented as a custom PyTorch univariate patch Transformer to consume audited segment-safe NPZ windows. Training protocol is aligned with other corrected custom MORED experiments (batch 4, accumulation 8, best sample-weighted validation checkpoint), not an exact recreation of NeuralForecast training.",
    }
    (result_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Device:", device)
    print(f"Patch length={PATCH_LEN}, stride={PATCH_STRIDE}, tokens={(LOOKBACK - PATCH_LEN) // PATCH_STRIDE + 1}")
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
        pd.DataFrame(rows).sort_values("Seed").to_csv(result_dir / "patchtst_corrected_results.csv", index=False)
    results = pd.DataFrame(rows).sort_values("Seed")
    results.to_csv(result_dir / "patchtst_corrected_results.csv", index=False)
    cols = ["RMSE", "MAE", "MAPE", "R2", "StdR"]
    summary = pd.DataFrame({"mean": results[cols].mean(), "sd": results[cols].std(ddof=1)})
    summary.to_csv(result_dir / "patchtst_corrected_summary.csv")
    print("\nFive-seed corrected MORED PatchTST-style results:")
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
