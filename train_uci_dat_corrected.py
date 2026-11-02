#!/usr/bin/env python3
"""Leakage-aware five-seed UCI DAT experiment.

This script uses ONLY the verified windows from Data/UCI/processed/*.npz.
It regenerates DAT decomposition features from each accepted 1,440-minute
INPUT window, so no old interpolated-split feature cache is reused.

Commands:
  python train_uci_dat_corrected.py
      Run a one-window feature + model forward-pass smoke test.

  python train_uci_dat_corrected.py --prepare-features
      Build versioned DAT caches for train/validation/test.

  python train_uci_dat_corrected.py --train
      Train the five seeds using the prepared corrected caches.

Requires: vmdpy (install with: python -m pip install vmdpy)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
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

try:
    from vmdpy import VMD
except ImportError as exc:  # pragma: no cover - user environment dependent
    raise SystemExit(
        "Missing dependency 'vmdpy'. In your active .venv, run:\n"
        "python -m pip install vmdpy\n"
        "Then run this script again."
    ) from exc

from uci_window_loader import load_uci_windows, inverse_array


# ============================================================
# EXPERIMENT CONFIGURATION (retained from the supplied UCI DAT)
# ============================================================
LOOKBACK = 1440
FORECAST = 1440
SEEDS = (0, 1, 2, 3, 4)
MAX_OPT_STEPS = 300
LEARNING_RATE = 1e-3
BATCH_SIZE = 4
ACCUM_STEPS = 8

D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1

VMD_K = 5
VMD_ALPHA = 2000
BES_PADDING = 144
FE_M = 2
FE_R = 0.2
FE_N = 2
FE_MAX_POINTS = 180
REFINED_K = 2
REFINED_ALPHA = 2000
DAT_CHANNELS = VMD_K * REFINED_K

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "Cache" / "UCI_DAT_Corrected_V2"
RESULT_DIR = ROOT / "Results" / "UCI" / "Corrected_V2_DAT"
CACHE_VERSION = "uci_dat_corrected_v2_no_interpolated_splits"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Corrected UCI DAT five-seed experiment")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--prepare-features", action="store_true", help="Generate corrected DAT decomposition caches")
    mode.add_argument("--train", action="store_true", help="Train five DAT seeds; requires prepared caches")
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--max-steps", type=int, default=MAX_OPT_STEPS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--accum-steps", type=int, default=ACCUM_STEPS)
    p.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
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


def mirror_pad(signal: np.ndarray, padding: int = BES_PADDING) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1 or len(signal) <= padding:
        raise ValueError(f"Expected 1-D signal longer than padding={padding}; got {signal.shape}")
    left = signal[1:padding + 1][::-1]
    right = signal[-padding - 1:-1][::-1]
    return np.concatenate([left, signal, right])


def causal_vmd(signal: np.ndarray) -> np.ndarray:
    padded = mirror_pad(signal)
    modes, _, _ = VMD(padded, VMD_ALPHA, 0, VMD_K, 0, 1, 1e-7)
    modes = modes[:, BES_PADDING:BES_PADDING + len(signal)]
    if modes.shape != (VMD_K, len(signal)):
        raise RuntimeError(f"VMD returned unexpected mode shape {modes.shape}")
    return modes.astype(np.float32)


def fuzzy_entropy(signal: np.ndarray, m: int = FE_M, r: float = FE_R, n: int = FE_N) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) > FE_MAX_POINTS:
        indices = np.linspace(0, len(signal) - 1, FE_MAX_POINTS).astype(int)
        signal = signal[indices]
    signal = signal - np.mean(signal)
    std = np.std(signal)
    if std < 1e-12:
        return 0.0
    tolerance = r * std

    def phi(dimension: int) -> float:
        count = len(signal) - dimension + 1
        if count <= 1:
            return 1e-12
        vectors = np.array([signal[i:i + dimension] for i in range(count)])
        vectors = vectors - vectors.mean(axis=1, keepdims=True)
        differences = np.abs(vectors[:, None, :] - vectors[None, :, :])
        distances = np.max(differences, axis=2)
        similarity = np.exp(-(distances ** n) / (tolerance + 1e-12))
        mask = ~np.eye(count, dtype=bool)
        return float(np.mean(similarity[mask]))

    phi_m = phi(m)
    phi_m1 = phi(m + 1)
    return float(np.log((phi_m + 1e-12) / (phi_m1 + 1e-12)))


def refined_vmd(signal: np.ndarray) -> np.ndarray:
    modes, _, _ = VMD(np.asarray(signal, dtype=np.float64), REFINED_ALPHA, 0, REFINED_K, 0, 1, 1e-7)
    if modes.shape[0] != REFINED_K:
        raise RuntimeError(f"Secondary VMD returned unexpected shape {modes.shape}")
    return modes.astype(np.float32)


def estimate_frequency(signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - np.mean(signal)
    spectrum = np.abs(np.fft.rfft(signal))
    frequencies = np.fft.rfftfreq(len(signal))
    if len(spectrum) <= 1:
        return 0.0
    index = int(np.argmax(spectrum[1:]) + 1)
    return float(frequencies[index])


def process_window(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the supplied DAT feature logic to one historical input window."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape != (LOOKBACK,) or not np.isfinite(signal).all():
        raise ValueError(f"Expected a finite input window of shape ({LOOKBACK},); got {signal.shape}")

    modes = causal_vmd(signal)
    entropy_values = np.asarray([fuzzy_entropy(mode) for mode in modes], dtype=np.float32)
    threshold = float(np.median(entropy_values))

    channels: list[np.ndarray] = []
    frequencies: list[float] = []
    for mode_index in range(VMD_K):
        mode = modes[mode_index]
        if float(entropy_values[mode_index]) > threshold:
            refined = refined_vmd(mode)
            for component in refined:
                channels.append(component)
                frequencies.append(estimate_frequency(component))
        else:
            channels.append(mode)
            frequencies.append(estimate_frequency(mode))
            channels.append(np.zeros_like(mode, dtype=np.float32))
            frequencies.append(0.0)

    feature = np.stack(channels, axis=1).astype(np.float32)
    freq = np.asarray(frequencies, dtype=np.float32)
    if feature.shape != (LOOKBACK, DAT_CHANNELS) or freq.shape != (DAT_CHANNELS,):
        raise RuntimeError(f"DAT feature dimensions are wrong: {feature.shape}, {freq.shape}")
    return feature, freq, entropy_values


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def cache_paths(split: str) -> tuple[Path, Path]:
    return CACHE_DIR / f"{split}_features.npz", CACHE_DIR / f"{split}_features.json"


def prepare_split_features(split: str, X: np.ndarray, force: bool = False) -> dict[str, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file, meta_file = cache_paths(split)
    source_hash = sha256_array(X)

    if cache_file.is_file() and meta_file.is_file() and not force:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if meta.get("cache_version") == CACHE_VERSION and meta.get("source_X_sha256") == source_hash:
            with np.load(cache_file, allow_pickle=False) as a:
                features = {k: np.asarray(a[k]) for k in ("X", "freq", "entropy")}
            expected = {
                "X": (len(X), LOOKBACK, DAT_CHANNELS),
                "freq": (len(X), DAT_CHANNELS),
                "entropy": (len(X), VMD_K),
            }
            if all(features[k].shape == shape for k, shape in expected.items()):
                print(f"[{split}] Reusing verified corrected cache: {cache_file}")
                return features
        print(f"[{split}] Existing cache is stale or incompatible; rebuilding.")

    n = len(X)
    features = np.empty((n, LOOKBACK, DAT_CHANNELS), dtype=np.float32)
    frequencies = np.empty((n, DAT_CHANNELS), dtype=np.float32)
    entropies = np.empty((n, VMD_K), dtype=np.float32)

    print(f"\n[{split}] Generating corrected DAT features for {n} accepted windows...")
    for i, signal in enumerate(X):
        f, fr, en = process_window(signal)
        features[i] = f
        frequencies[i] = fr
        entropies[i] = en
        if i == 0 or (i + 1) % 25 == 0 or i + 1 == n:
            print(f"[{split}] DAT preprocessing: {i + 1}/{n}")

    tmp_file = cache_file.with_suffix(".tmp.npz")
    np.savez_compressed(tmp_file, X=features, freq=frequencies, entropy=entropies)
    tmp_file.replace(cache_file)

    meta = {
        "cache_version": CACHE_VERSION,
        "split": split,
        "windows": n,
        "source_X_sha256": source_hash,
        "feature_shape": list(features.shape),
        "frequency_shape": list(frequencies.shape),
        "entropy_shape": list(entropies.shape),
        "lookback_minutes": LOOKBACK,
        "VMD_K": VMD_K,
        "VMD_ALPHA": VMD_ALPHA,
        "BES_mirror_padding": BES_PADDING,
        "fuzzy_entropy": {"m": FE_M, "r": FE_R, "n": FE_N, "max_points": FE_MAX_POINTS},
        "secondary_VMD": {"K": REFINED_K, "alpha": REFINED_ALPHA},
        "source": "Data/UCI/processed/*.npz corrected fully-observed input windows",
        "target_values_used_for_feature_generation": False,
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"X": features, "freq": frequencies, "entropy": entropies}


def calendar_features(timestamp_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray(timestamp_windows).astype("datetime64[m]")
    # pandas handles calendar fields consistently with the original supplied implementation.
    hours = np.empty(timestamps.shape, dtype=np.int64)
    weekdays = np.empty(timestamps.shape, dtype=np.int64)
    for i, window in enumerate(timestamps):
        dt = pd.DatetimeIndex(window)
        hours[i] = dt.hour.to_numpy(dtype=np.int64)
        weekdays[i] = dt.dayofweek.to_numpy(dtype=np.int64)
    return hours, weekdays


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class DATModel(nn.Module):
    def __init__(self, attention_dropout: float) -> None:
        super().__init__()
        self.mode_convs = nn.ModuleList([
            nn.Conv1d(1, D_MODEL, kernel_size=3, padding=1)
            for _ in range(DAT_CHANNELS)
        ])
        self.frequency_projection = nn.Linear(1, D_MODEL)
        self.hour_embedding = nn.Embedding(24, D_MODEL)
        self.weekday_embedding = nn.Embedding(7, D_MODEL)
        self.position = PositionalEncoding(D_MODEL, LOOKBACK)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        # Match the MPS-safe configuration used for the corrected Transformer baseline.
        encoder_layer.self_attn.dropout = float(attention_dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.output_head = nn.Sequential(
            nn.Linear(D_MODEL, 128),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, FORECAST),
        )

    def forward(self, x: torch.Tensor, frequencies: torch.Tensor, hours: torch.Tensor, weekdays: torch.Tensor) -> torch.Tensor:
        mode_tokens = []
        for channel_index in range(DAT_CHANNELS):
            channel = x[:, :, channel_index].unsqueeze(1)
            projected = self.mode_convs[channel_index](channel).transpose(1, 2)
            frequency = frequencies[:, channel_index].unsqueeze(1)
            projected = projected + self.frequency_projection(frequency).unsqueeze(1)
            mode_tokens.append(projected)
        x = torch.stack(mode_tokens, dim=0).mean(dim=0)
        x = x + self.hour_embedding(hours) + self.weekday_embedding(weekdays)
        x = self.position(x)
        x = self.encoder(x)
        return self.output_head(x[:, -1, :])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if yt.shape != yp.shape or not np.isfinite(yt).all() or not np.isfinite(yp).all():
        raise ValueError("Metric arrays have mismatched shapes or non-finite values.")
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    valid = np.abs(yt) > 1e-8
    mape = float(np.mean(np.abs((yt[valid] - yp[valid]) / yt[valid])) * 100.0) if valid.any() else float("nan")
    r2 = float(r2_score(yt, yp))
    std_true = float(np.std(yt))
    stdr = float(np.std(yp) / std_true) if std_true > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "StdR": stdr}


def make_loader(*arrays: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = []
    for array in arrays:
        dtype = torch.long if np.issubdtype(array.dtype, np.integer) else torch.float32
        tensors.append(torch.from_numpy(np.asarray(array, dtype=np.int64 if dtype == torch.long else np.float32)))
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    weighted_sum = 0.0
    count = 0
    with torch.no_grad():
        for xb, fb, hb, wb, yb in loader:
            xb, fb = xb.to(device), fb.to(device)
            hb, wb, yb = hb.to(device), wb.to(device), yb.to(device)
            pred = model(xb, fb, hb, wb)
            n = int(yb.shape[0])
            weighted_sum += float(criterion(pred, yb).item()) * n
            count += n
    if count == 0:
        raise RuntimeError("Validation loader is empty.")
    return weighted_sum / count


def load_all_features(arrays: dict[str, Any], force: bool = False) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for split in ("train", "val", "test"):
        result[split] = prepare_split_features(split, arrays[split]["X"], force=force)
    return result


def smoke_test(arrays: dict[str, Any], device: torch.device) -> None:
    print("\nRunning one-window DAT feature smoke test...")
    feature, frequency, entropy = process_window(arrays["train"]["X"][0])
    print(f"Feature shapes: X={feature.shape}, freq={frequency.shape}, entropy={entropy.shape}")
    hours, weekdays = calendar_features(arrays["train"]["timestamps"][:1])
    model = DATModel(attention_dropout=0.0 if device.type == "mps" else DROPOUT).to(device)
    with torch.no_grad():
        output = model(
            torch.from_numpy(feature[None]).to(device),
            torch.from_numpy(frequency[None]).to(device),
            torch.from_numpy(hours).to(device),
            torch.from_numpy(weekdays).to(device),
        )
    if tuple(output.shape) != (1, FORECAST):
        raise RuntimeError(f"Unexpected DAT smoke-test output shape: {tuple(output.shape)}")
    print(f"DAT forward-pass smoke test PASSED: output={tuple(output.shape)}")
    del model, output
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


def run_one_seed(seed: int, arrays: dict[str, Any], features: dict[str, dict[str, np.ndarray]], device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    set_seed(seed)
    hours = {}
    weekdays = {}
    for split in ("train", "val", "test"):
        hours[split], weekdays[split] = calendar_features(arrays[split]["timestamps"])

    train_loader = make_loader(
        features["train"]["X"], features["train"]["freq"], hours["train"], weekdays["train"], arrays["train"]["y"],
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = make_loader(
        features["val"]["X"], features["val"]["freq"], hours["val"], weekdays["val"], arrays["val"]["y"],
        batch_size=args.batch_size, shuffle=False,
    )
    test_loader = make_loader(
        features["test"]["X"], features["test"]["freq"], hours["test"], weekdays["test"], arrays["test"]["y"],
        batch_size=args.batch_size, shuffle=False,
    )

    attention_dropout = 0.0 if device.type == "mps" else DROPOUT
    model = DATModel(attention_dropout=attention_dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0

    print(f"\n{'=' * 68}\nCORRECTED UCI DAT — SEED {seed}\n{'=' * 68}")
    while micro_step // args.accum_steps < args.max_steps:
        try:
            xb, fb, hb, wb, yb = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            xb, fb, hb, wb, yb = next(iterator)
        xb, fb = xb.to(device), fb.to(device)
        hb, wb, yb = hb.to(device), wb.to(device), yb.to(device)
        model.train()
        pred = model(xb, fb, hb, wb)
        (criterion(pred, yb) / args.accum_steps).backward()
        micro_step += 1
        if micro_step % args.accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step = micro_step // args.accum_steps
            if opt_step == 1 or opt_step % 25 == 0 or opt_step == args.max_steps:
                val_loss = evaluate(model, val_loader, criterion, device)
                print(f"DAT seed {seed} | optimizer step {opt_step}/{args.max_steps} | weighted val MSE={val_loss:.7f}")
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError(f"No best checkpoint was selected for seed {seed}.")
    model.load_state_dict(best_state)
    model.eval()
    predictions = []
    with torch.no_grad():
        for xb, fb, hb, wb, _ in test_loader:
            pred = model(xb.to(device), fb.to(device), hb.to(device), wb.to(device))
            predictions.append(pred.detach().cpu().numpy())
    pred_scaled = np.concatenate(predictions, axis=0)
    pred_raw = inverse_array(pred_scaled, arrays["scaler"]).astype(np.float32)
    y_true_raw = arrays["test"]["y_raw"].astype(np.float32)
    if pred_raw.shape != y_true_raw.shape:
        raise RuntimeError(f"Prediction shape {pred_raw.shape} != target shape {y_true_raw.shape}")
    metrics = compute_metrics(y_true_raw, pred_raw)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(RESULT_DIR / f"DAT_seed{seed}_predictions.npy", pred_raw)
    torch.save(best_state, RESULT_DIR / f"DAT_seed{seed}_best.pt")
    row = {"Model": "DAT", "Seed": seed, "Best_Val_MSE_Scaled": best_val, **metrics}
    progress = RESULT_DIR / "dat_corrected_progress.csv"
    if progress.is_file():
        prev = pd.read_csv(progress)
        prev = prev[~((prev["Model"] == "DAT") & (prev["Seed"] == seed))]
    else:
        prev = pd.DataFrame()
    pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(progress, index=False)

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
    print(f"Experiment: {CACHE_VERSION}")
    print(f"Attention-probability dropout: {0.0 if device.type == 'mps' else DROPOUT}")

    arrays = load_uci_windows(ROOT)
    for split, expected in (("train", 903), ("val", 189), ("test", 192)):
        if arrays[split]["X"].shape != (expected, LOOKBACK):
            raise RuntimeError(f"Unexpected {split} input shape: {arrays[split]['X'].shape}")
        if arrays[split]["y"].shape != (expected, FORECAST):
            raise RuntimeError(f"Unexpected {split} target shape: {arrays[split]['y'].shape}")

    print("\nCorrected clean windows loaded:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {arrays[split]['X'].shape} input windows")

    if args.prepare_features:
        print("\nGenerating/reusing versioned DAT features from the corrected input windows.")
        features = load_all_features(arrays)
        print("\nCorrected DAT feature preparation completed.")
        for split in ("train", "val", "test"):
            print(f"  {split}: features={features[split]['X'].shape}, freq={features[split]['freq'].shape}, entropy={features[split]['entropy'].shape}")
        print(f"Feature cache directory: {CACHE_DIR}")
        return 0

    if args.train:
        features = {}
        for split in ("train", "val", "test"):
            cache_file, meta_file = cache_paths(split)
            if not cache_file.is_file() or not meta_file.is_file():
                raise FileNotFoundError(
                    f"Missing corrected DAT feature cache for {split}. "
                    "Run: python train_uci_dat_corrected.py --prepare-features"
                )
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            if meta.get("cache_version") != CACHE_VERSION or meta.get("source_X_sha256") != sha256_array(arrays[split]["X"]):
                raise RuntimeError(f"Corrected DAT cache for {split} does not match the current clean windows. Re-run --prepare-features.")
            with np.load(cache_file, allow_pickle=False) as data:
                features[split] = {k: np.asarray(data[k]) for k in ("X", "freq", "entropy")}
            if features[split]["X"].shape != (len(arrays[split]["X"]), LOOKBACK, DAT_CHANNELS):
                raise RuntimeError(f"Bad DAT cache dimensions for {split}: {features[split]['X'].shape}")
        print("\nAll corrected DAT caches verified.")
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        config = {
            "experiment": CACHE_VERSION,
            "model": "DAT",
            "source": "verified corrected UCI NPZ windows only",
            "seeds": args.seeds,
            "max_optimizer_steps_per_seed": args.max_steps,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.accum_steps,
            "learning_rate": LEARNING_RATE,
            "lookback_minutes": LOOKBACK,
            "forecast_minutes": FORECAST,
            "split_windows": {"train": 903, "val": 189, "test": 192},
            "VMD": {"K": VMD_K, "alpha": VMD_ALPHA, "mirror_padding": BES_PADDING},
            "fuzzy_entropy": {"m": FE_M, "r": FE_R, "n": FE_N, "max_points": FE_MAX_POINTS},
            "secondary_VMD": {"K": REFINED_K, "alpha": REFINED_ALPHA},
            "DAT_channels": DAT_CHANNELS,
            "device": str(device),
            "attention_probability_dropout": 0.0 if device.type == "mps" else DROPOUT,
            "encoder_residual_and_feedforward_dropout": DROPOUT,
            "validation_loss_aggregation": "sample-weighted mean across windows",
            "features_use_input_window_only": True,
            "old_interpolated_CSVs_or_old_DAT_cache_used": False,
        }
        (RESULT_DIR / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

        rows = []
        for seed in args.seeds:
            row = run_one_seed(seed, arrays, features, device, args)
            rows.append(row)
            pd.DataFrame(rows).to_csv(RESULT_DIR / "dat_corrected_results.csv", index=False)
        result_df = pd.DataFrame(rows)
        summary = result_df[["RMSE", "MAE", "MAPE", "R2", "StdR"]].agg(["mean", "std"]).T
        summary.columns = ["mean", "sd"]
        summary.to_csv(RESULT_DIR / "dat_corrected_summary.csv")
        np.save(RESULT_DIR / "y_true_raw.npy", arrays["test"]["y_raw"].astype(np.float32))
        np.save(RESULT_DIR / "test_target_start_timestamps.npy", arrays["test"]["target_start"].astype("datetime64[ns]").astype(str))
        print("\nFive-seed corrected DAT results:")
        print(result_df.to_string(index=False))
        print("\nMean ± sample standard deviation:")
        print(summary.to_string())
        print(f"\nSaved results under: {RESULT_DIR}")
        return 0

    # Default smoke test: calculate one window's features and run one DAT forward pass.
    smoke_test(arrays, device)
    print("\nSmoke test only. No full feature cache or model weights were trained.")
    print("Next: run with --prepare-features, then --train.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
