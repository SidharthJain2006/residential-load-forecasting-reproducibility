#!/usr/bin/env python3
"""Corrected five-seed MORED LSTM baseline on audited, segment-safe NPZ windows.

This is a direct PyTorch implementation of a NeuralForecast-style LSTM
encoder + MLP decoder. It consumes only Data/MORED/Corrected_V3_SessionSplit/*.npz.
It does not read legacy interpolated CSV splits or create new windows.

Default: load/validate data and run a forward/backward smoke test.
Training: python train_mored_lstm_corrected.py --train
"""
from __future__ import annotations

import argparse
import gc
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
ENCODER_HIDDEN_SIZE = 64
ENCODER_LAYERS = 2
ENCODER_DROPOUT = 0.1
DECODER_HIDDEN_SIZE = 64
DECODER_LAYERS = 2

ROOT = Path(__file__).resolve().parent
DATA_DIR_DEFAULT = ROOT / "Data" / "MORED" / "Corrected_V3_SessionSplit"
RESULT_DIR_DEFAULT = ROOT / "Results" / "MORED" / "Corrected_V3_SessionSplit_LSTM"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", action="store_true", help="Train the five seeds; otherwise smoke-test only.")
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
        "X": (n, LOOKBACK), "y": (n, FORECAST),
        "X_raw": (n, LOOKBACK), "y_raw": (n, FORECAST),
    }
    for key, shape in checks.items():
        if d[key].shape != shape:
            raise RuntimeError(f"{split} {key} has shape {d[key].shape}; expected {shape}")
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
    if metadata.get("experiment") != "mored_corrected_v3_session_split_no_interpolation_raw_count_filter":
        raise RuntimeError("Unexpected MORED preprocessing experiment/version in metadata")
    if metadata.get("minute_aggregation", {}).get("missing_sample_policy", "").lower().find("no interpolation") < 0:
        raise RuntimeError("Metadata does not confirm no interpolation")
    arrays = {s: load_split(data_dir, s) for s in ("train", "val", "test")}
    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)
    # Confirm scaler was fitted on the full retained training-minute split, not just windows.
    train_frame = pd.read_csv(train_csv)
    genuine_train = train_frame["consumption"].to_numpy(dtype=np.float64)
    if not np.isfinite(genuine_train).all():
        raise RuntimeError("Corrected training minute CSV contains non-finite readings")
    if int(scaler.n_samples_seen_) != len(genuine_train):
        raise RuntimeError("MORED scaler sample count does not match retained training-minute rows")
    if not np.isclose(float(scaler.data_min_[0]), float(genuine_train.min())):
        raise RuntimeError("MORED scaler minimum does not match training readings")
    if not np.isclose(float(scaler.data_max_[0]), float(genuine_train.max())):
        raise RuntimeError("MORED scaler maximum does not match training readings")
    for split, d in arrays.items():
        ex = scaler.transform(d["X_raw"].reshape(-1, 1)).reshape(d["X"].shape)
        ey = scaler.transform(d["y_raw"].reshape(-1, 1)).reshape(d["y"].shape)
        if not np.allclose(d["X"], ex, atol=2e-6) or not np.allclose(d["y"], ey, atol=2e-6):
            raise RuntimeError(f"{split} scaled windows do not match saved train-fitted scaler")
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
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": float(r2_score(yt, yp)),
            "StdR": float(np.std(yp) / true_sd) if true_sd > 0 else float("nan")}


class NeuralForecastStyleLSTM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.LSTM(1, ENCODER_HIDDEN_SIZE, num_layers=ENCODER_LAYERS,
                               dropout=ENCODER_DROPOUT, batch_first=True)
        layers: list[nn.Module] = []
        in_features = ENCODER_HIDDEN_SIZE
        for _ in range(DECODER_LAYERS):
            layers.extend([nn.Linear(in_features, DECODER_HIDDEN_SIZE), nn.ReLU()])
            in_features = DECODER_HIDDEN_SIZE
        layers.append(nn.Linear(in_features, 1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x.unsqueeze(-1))
        return self.decoder(encoded[:, -FORECAST:, :]).squeeze(-1)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0)


def sample_weighted_val(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total, n_total = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.functional.mse_loss(model(xb), yb, reduction="mean")
            n = len(xb)
            total += float(loss.item()) * n
            n_total += n
    if not n_total:
        raise RuntimeError("Validation loader is empty")
    return total / n_total


def smoke_test(device: torch.device) -> tuple[torch.device, int]:
    candidates = [device, torch.device("cpu")] if device.type == "mps" else [device]
    last_error: Exception | None = None
    for candidate in candidates:
        model = NeuralForecastStyleLSTM().to(candidate)
        x = torch.zeros((2, LOOKBACK), dtype=torch.float32, device=candidate)
        y = torch.zeros((2, FORECAST), dtype=torch.float32, device=candidate)
        try:
            model.train()
            pred = model(x)
            if pred.shape != (2, FORECAST):
                raise RuntimeError(f"Unexpected output shape: {tuple(pred.shape)}")
            nn.functional.mse_loss(pred, y).backward()
            count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if candidate.type == "mps":
                torch.mps.synchronize()
            print(f"LSTM forward/backward smoke test PASSED on {candidate}: input={tuple(x.shape)}, output={tuple(pred.shape)}")
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
    raise RuntimeError(f"LSTM smoke test failed: {last_error}")


def inverse_array(a: np.ndarray, scaler: Any) -> np.ndarray:
    return scaler.inverse_transform(np.asarray(a).reshape(-1, 1)).reshape(a.shape)


def run_seed(seed: int, arrays: dict[str, Any], scaler: Any, result_dir: Path,
             device: torch.device, max_steps: int, batch_size: int, accum_steps: int) -> dict[str, Any]:
    set_seed(seed)
    train_loader = make_loader(arrays["train"]["X"], arrays["train"]["y"], batch_size, True)
    val_loader = make_loader(arrays["val"]["X"], arrays["val"]["y"], batch_size, False)
    test_loader = make_loader(arrays["test"]["X"], arrays["test"]["y"], batch_size, False)
    model = NeuralForecastStyleLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    print(f"\n{'=' * 60}\nMORED LSTM — SEED {seed}\n{'=' * 60}")
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
    pred_scaled: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in test_loader:
            pred_scaled.append(model(xb.to(device)).detach().cpu().numpy())
    pred_raw = inverse_array(np.concatenate(pred_scaled, axis=0), scaler).astype(np.float32)
    y_true = arrays["test"]["y_raw"].astype(np.float32)
    if pred_raw.shape != y_true.shape:
        raise RuntimeError(f"Prediction shape {pred_raw.shape} != truth shape {y_true.shape}")
    metrics = compute_metrics(y_true, pred_raw)
    np.save(result_dir / f"LSTM_seed{seed}_predictions.npy", pred_raw)
    torch.save(best_state, result_dir / f"LSTM_seed{seed}_best.pt")
    row = {"Model": "LSTM", "Seed": seed, "Best_Val_MSE_Scaled": best_loss, **metrics}
    progress_file = result_dir / "lstm_corrected_progress.csv"
    old = pd.read_csv(progress_file) if progress_file.exists() else pd.DataFrame()
    if not old.empty and "Seed" in old:
        old = old[old["Seed"] != seed]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).sort_values("Seed").to_csv(progress_file, index=False)
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
        "experiment": "mored_lstm_corrected_v3_session_split_no_interpolation_segment_safe_windows",
        "model": "LSTM (NeuralForecast-style direct PyTorch encoder + MLP decoder)",
        "architecture_reference": "Original supplied MORED baseline: encoder_hidden_size=64, encoder_n_layers=2, decoder_hidden_size=64, decoder_layers=2, encoder_dropout=0.1",
        "source_data": "Data/MORED/Corrected_V3_SessionSplit/{train,val,test}_windows.npz only",
        "legacy_interpolated_csv_splits_used": False,
        "seeds": args.seeds,
        "max_optimizer_steps_per_seed": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accum_steps,
        "effective_batch_size_approx": args.batch_size * args.accum_steps,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss": "MSE",
        "validation_interval_optimizer_steps": 25,
        "validation_loss_aggregation": "sample-weighted mean across windows",
        "checkpoint_selection": "best validation MSE; test only evaluated after checkpoint selection",
        "lookback_minutes": LOOKBACK,
        "forecast_minutes": FORECAST,
        "train_windows": EXPECTED_WINDOWS["train"],
        "validation_windows": EXPECTED_WINDOWS["val"],
        "test_windows": EXPECTED_WINDOWS["test"],
        "encoder_hidden_size": ENCODER_HIDDEN_SIZE,
        "encoder_layers": ENCODER_LAYERS,
        "encoder_dropout": ENCODER_DROPOUT,
        "decoder_hidden_size": DECODER_HIDDEN_SIZE,
        "decoder_layers": DECODER_LAYERS,
        "trainable_parameters": parameter_count,
        "device": str(device),
        "metric_aggregation": "flatten all accepted test forecast windows and minute targets; windows overlap at 60-minute stride",
        "preprocessing_experiment": prep_metadata.get("experiment"),
        "preprocessing_interpolation": False,
        "note": "Custom PyTorch direct-window implementation following the architecture settings of the originally supplied NeuralForecast LSTM baseline; not a call to the NeuralForecast class itself.",
    }
    (result_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Trainable parameters: {parameter_count:,}")
    print("Results directory:", result_dir)
    if not args.train:
        print("\nSmoke test only. No model weights were trained.")
        print("When ready, run this script again with --train.")
        return 0
    rows = []
    for seed in args.seeds:
        row = run_seed(seed, arrays, scaler, result_dir, device, args.max_steps, args.batch_size, args.accum_steps)
        rows.append(row)
        pd.DataFrame(rows).sort_values("Seed").to_csv(result_dir / "lstm_corrected_results.csv", index=False)
    results = pd.DataFrame(rows).sort_values("Seed")
    results.to_csv(result_dir / "lstm_corrected_results.csv", index=False)
    metrics = ["RMSE", "MAE", "MAPE", "R2", "StdR"]
    summary = pd.DataFrame({"mean": results[metrics].mean(), "sd": results[metrics].std(ddof=1)})
    summary.to_csv(result_dir / "lstm_corrected_summary.csv")
    print("\nFive-seed corrected MORED LSTM results:")
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
