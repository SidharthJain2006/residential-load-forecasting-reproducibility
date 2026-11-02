#!/usr/bin/env python3
"""Corrected MORED Vanilla Transformer + DAT experiments.

Uses the verified, segment-safe NPZ windows produced by
prepare_mored_corrected.py. The legacy MORED CSV splits/results are never read
or overwritten. Training selects the best checkpoint using sample-weighted
validation MSE, and reports metrics in the original MORED consumption unit (W).

Stages:
  python train_mored_vanilla_dat_corrected.py                 # smoke test
  python train_mored_vanilla_dat_corrected.py --prepare-features
  python train_mored_vanilla_dat_corrected.py --train
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
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from vmdpy import VMD

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------ Fixed protocol ------------------------------
LOOKBACK = 1440
FORECAST = 1440
STRIDE = 60
SEEDS = [0, 1, 2, 3, 4]
MAX_OPT_STEPS = 300
LEARNING_RATE = 0.001
BATCH_SIZE = 4
ACCUM_STEPS = 8
VALIDATION_INTERVAL = 25

D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1
ATTENTION_PROBABILITY_DROPOUT = 0.0  # MPS-compatible, recorded explicitly

VMD_K = 5
VMD_ALPHA = 2000
MIRROR_PADDING = 144
FE_M = 2
FE_R = 0.2
FE_N = 2
FE_MAX_POINTS = 180
REFINED_K = 2
REFINED_ALPHA = 2000
DAT_CHANNELS = VMD_K * REFINED_K

EXPECTED_WINDOWS = {"train": 245, "val": 119, "test": 359}

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data" / "MORED" / "Corrected_V3_SessionSplit"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "Cache" / "MORED_DAT_Corrected_V3_SessionSplit"
DEFAULT_RESULT_DIR = PROJECT_ROOT / "Results" / "MORED" / "Corrected_V3_SessionSplit_Vanilla_DAT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prepare-features", action="store_true")
    group.add_argument("--train", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
    print(f"[PASS] {message}")


def load_split(data_dir: Path, split: str) -> dict[str, np.ndarray]:
    path = data_dir / f"{split}_windows.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing corrected MORED windows: {path}")
    with np.load(path, allow_pickle=False) as z:
        result = {k: z[k].copy() for k in z.files}

    # This pipeline stores model arrays as [N, L, 1]. Flatten the univariate channel.
    for key in ("X", "y"):
        arr = result[key]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            result[key] = arr[..., 0]

    expected_n = EXPECTED_WINDOWS[split]
    check(result["X"].shape == (expected_n, LOOKBACK),
          f"{split}: input windows have shape ({expected_n}, {LOOKBACK})")
    check(result["y"].shape == (expected_n, FORECAST),
          f"{split}: target windows have shape ({expected_n}, {FORECAST})")
    check(result["X_raw"].shape == (expected_n, LOOKBACK),
          f"{split}: raw input windows have correct shape")
    check(result["y_raw"].shape == (expected_n, FORECAST),
          f"{split}: raw target windows have correct shape")
    for key in ("X", "y", "X_raw", "y_raw"):
        check(np.isfinite(result[key]).all(), f"{split}: {key} contains only finite values")

    # The NPZ records input_start and target_start; reconstruct full minute stamps.
    input_starts = pd.to_datetime(result["input_start"]).to_numpy(dtype="datetime64[ns]")
    target_starts = pd.to_datetime(result["target_start"]).to_numpy(dtype="datetime64[ns]")
    result["input_timestamps"] = input_starts[:, None] + np.arange(LOOKBACK).astype("timedelta64[m]")[None, :]
    result["target_timestamps"] = target_starts[:, None] + np.arange(FORECAST).astype("timedelta64[m]")[None, :]

    check(np.all(np.diff(result["input_timestamps"], axis=1) == np.timedelta64(1, "m")),
          f"{split}: all input windows are continuous at one-minute frequency")
    check(np.all(np.diff(result["target_timestamps"], axis=1) == np.timedelta64(1, "m")),
          f"{split}: all target windows are continuous at one-minute frequency")
    check(np.all(result["target_timestamps"][:, 0] - result["input_timestamps"][:, -1] == np.timedelta64(1, "m")),
          f"{split}: target windows immediately follow their input windows")
    return result


def load_all_data(data_dir: Path):
    print("\nLoading corrected MORED NPZ windows...")
    if not (data_dir / "preprocessing_metadata.json").is_file():
        raise FileNotFoundError(f"Corrected MORED metadata not found in {data_dir}")
    train = load_split(data_dir, "train")
    val = load_split(data_dir, "val")
    test = load_split(data_dir, "test")

    scaler_path = data_dir / "scaler.pkl"
    with scaler_path.open("rb") as f:
        scaler = pickle.load(f)

    # Validate scaler against observed training targets in the corrected minute table.
    train_csv = pd.read_csv(data_dir / "train.csv")
    train_observed = train_csv["consumption"].to_numpy(dtype=np.float64)
    check(int(scaler.n_samples_seen_) == len(train_observed),
          "MORED scaler was fitted on all retained training-minute observations only")
    check(np.isclose(float(scaler.data_min_[0]), float(train_observed.min())),
          "MORED scaler minimum matches the retained training data")
    check(np.isclose(float(scaler.data_max_[0]), float(train_observed.max())),
          "MORED scaler maximum matches the retained training data")

    # Confirm saved NPZ scaled values are consistent with the scaler and raw values.
    for split_name, d in (("train", train), ("val", val), ("test", test)):
        expected_x = scaler.transform(d["X_raw"].reshape(-1, 1)).reshape(d["X"].shape)
        expected_y = scaler.transform(d["y_raw"].reshape(-1, 1)).reshape(d["y"].shape)
        check(np.allclose(d["X"], expected_x, atol=2e-6), f"{split_name}: scaled inputs match training-fitted scaler")
        check(np.allclose(d["y"], expected_y, atol=2e-6), f"{split_name}: scaled targets match training-fitted scaler")

    print("\nCorrected MORED windows:")
    for split_name, d in (("train", train), ("val", val), ("test", test)):
        print(f"  {split_name:5s}: X={d['X'].shape}, y={d['y'].shape}")
    return train, val, test, scaler


def mirror_pad(signal: np.ndarray) -> np.ndarray:
    left = signal[1:MIRROR_PADDING + 1][::-1]
    right = signal[-MIRROR_PADDING - 1:-1][::-1]
    return np.concatenate([left, signal, right])


def causal_vmd(signal: np.ndarray):
    padded = mirror_pad(np.asarray(signal, dtype=np.float64))
    modes, _, omega = VMD(padded, VMD_ALPHA, 0, VMD_K, 0, 1, 1e-7)
    modes = modes[:, MIRROR_PADDING:MIRROR_PADDING + len(signal)]
    return modes.astype(np.float32), omega


def fuzzy_entropy(signal, m=FE_M, r=FE_R, n=FE_N) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    if len(signal) > FE_MAX_POINTS:
        idx = np.linspace(0, len(signal) - 1, FE_MAX_POINTS).astype(int)
        signal = signal[idx]
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
        distances = np.max(np.abs(vectors[:, None, :] - vectors[None, :, :]), axis=2)
        mask = ~np.eye(count, dtype=bool)
        similarity = np.exp(-(distances ** n) / (tolerance + 1e-12))
        return float(np.mean(similarity[mask]))

    return float(np.log((phi(m) + 1e-12) / (phi(m + 1) + 1e-12)))


def refined_vmd(signal: np.ndarray) -> np.ndarray:
    modes, _, _ = VMD(np.asarray(signal, dtype=np.float64), REFINED_ALPHA, 0, REFINED_K, 0, 1, 1e-7)
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


def process_dat_window(signal: np.ndarray):
    modes, _ = causal_vmd(signal)
    entropies = np.asarray([fuzzy_entropy(mode) for mode in modes], dtype=np.float32)
    threshold = np.median(entropies)
    channels, frequencies = [], []
    for mode_index in range(VMD_K):
        mode = modes[mode_index]
        if entropies[mode_index] > threshold:
            refined = refined_vmd(mode)
            for component in refined:
                channels.append(component)
                frequencies.append(estimate_frequency(component))
        else:
            channels.append(mode)
            frequencies.append(estimate_frequency(mode))
            channels.append(np.zeros_like(mode, dtype=np.float32))
            frequencies.append(0.0)
    features = np.stack(channels, axis=1).astype(np.float32)
    frequencies = np.asarray(frequencies, dtype=np.float32)
    if features.shape != (LOOKBACK, DAT_CHANNELS):
        raise RuntimeError(f"Unexpected DAT feature shape: {features.shape}")
    if frequencies.shape != (DAT_CHANNELS,):
        raise RuntimeError(f"Unexpected frequency feature shape: {frequencies.shape}")
    return features, frequencies, entropies


def make_cache(X: np.ndarray, split: str, cache_dir: Path, source_hash: str,
               force: bool = False):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split}_features.npz"
    meta_path = cache_dir / f"{split}_features.json"
    if cache_path.exists() and meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("source_npz_sha256") == source_hash and meta.get("feature_version") == "mored_corrected_v3_session_split":
            print(f"[CACHE] Reusing verified {split} feature cache.")
            with np.load(cache_path, allow_pickle=False) as z:
                return z["X"].copy(), z["freq"].copy(), z["entropy"].copy()
        print(f"[CACHE] Existing {split} cache is stale; rebuilding.")

    features, freqs, entropies = [], [], []
    print(f"\n[{split}] Extracting DAT features from {len(X)} corrected windows...")
    for i, signal in enumerate(X):
        f, fr, ent = process_dat_window(signal)
        features.append(f)
        freqs.append(fr)
        entropies.append(ent)
        if i == 0 or (i + 1) % 25 == 0 or (i + 1) == len(X):
            print(f"  {split}: {i + 1}/{len(X)}")
    Xf = np.asarray(features, dtype=np.float32)
    frf = np.asarray(freqs, dtype=np.float32)
    enf = np.asarray(entropies, dtype=np.float32)
    np.savez_compressed(cache_path, X=Xf, freq=frf, entropy=enf)
    meta = {
        "feature_version": "mored_corrected_v3_session_split",
        "split": split,
        "source_npz_sha256": source_hash,
        "windows": int(len(X)),
        "lookback": LOOKBACK,
        "channels": DAT_CHANNELS,
        "vmd_k": VMD_K,
        "vmd_alpha": VMD_ALPHA,
        "mirror_padding": MIRROR_PADDING,
        "fuzzy_entropy_m": FE_M,
        "fuzzy_entropy_r": FE_R,
        "fuzzy_entropy_n": FE_N,
        "fuzzy_entropy_max_points": FE_MAX_POINTS,
        "secondary_vmd_k": REFINED_K,
        "secondary_vmd_alpha": REFINED_ALPHA,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return Xf, frf, enf


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


def make_encoder():
    layer = nn.TransformerEncoderLayer(
        d_model=D_MODEL,
        nhead=NHEAD,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        batch_first=True,
        activation="gelu",
    )
    # MPS compatibility: set attention-probability dropout to zero explicitly.
    layer.self_attn.dropout = ATTENTION_PROBABILITY_DROPOUT
    return nn.TransformerEncoder(layer, num_layers=NUM_LAYERS)


class VanillaTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(1, D_MODEL)
        self.position = PositionalEncoding(D_MODEL, LOOKBACK)
        self.encoder = make_encoder()
        self.output_head = nn.Sequential(
            nn.Linear(D_MODEL, 128), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(128, FORECAST)
        )

    def forward(self, x):
        x = self.input_projection(x.unsqueeze(-1))
        x = self.position(x)
        x = self.encoder(x)
        return self.output_head(x[:, -1, :])


class DATModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode_convs = nn.ModuleList([
            nn.Conv1d(1, D_MODEL, kernel_size=3, padding=1)
            for _ in range(DAT_CHANNELS)
        ])
        self.frequency_projection = nn.Linear(1, D_MODEL)
        self.hour_embedding = nn.Embedding(24, D_MODEL)
        self.weekday_embedding = nn.Embedding(7, D_MODEL)
        self.position = PositionalEncoding(D_MODEL, LOOKBACK)
        self.encoder = make_encoder()
        self.output_head = nn.Sequential(
            nn.Linear(D_MODEL, 128), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(128, FORECAST)
        )

    def forward(self, x, frequencies, hours, weekdays):
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


def calendar_features(timestamp_windows: np.ndarray):
    dt = pd.DatetimeIndex(timestamp_windows.reshape(-1))
    hours = dt.hour.to_numpy(dtype=np.int64).reshape(timestamp_windows.shape)
    weekdays = dt.dayofweek.to_numpy(dtype=np.int64).reshape(timestamp_windows.shape)
    return hours, weekdays


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inverse_array(array: np.ndarray, scaler) -> np.ndarray:
    shape = array.shape
    inverse = scaler.inverse_transform(array.reshape(-1, 1))
    return inverse.reshape(shape).astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    mask = np.abs(yt) > 1e-8
    mape = float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100.0)
    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": float(r2_score(yt, yp)),
        "StdR": float(np.std(yp) / (np.std(yt) + 1e-12)),
    }


def weighted_validation_mse(model, loader, device, dat_mode: bool = False) -> float:
    model.eval()
    squared_error_sum = 0.0
    element_count = 0
    with torch.no_grad():
        for batch in loader:
            if dat_mode:
                xb, fb, hb, wb, yb = batch
                xb, fb = xb.to(device), fb.to(device)
                hb, wb, yb = hb.to(device), wb.to(device), yb.to(device)
                pred = model(xb, fb, hb, wb)
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
            squared_error_sum += torch.sum((pred - yb) ** 2).item()
            element_count += yb.numel()
    if element_count == 0:
        raise RuntimeError("Validation loader produced no samples.")
    return squared_error_sum / element_count


def train_one_model(
    model_name: str,
    seed: int,
    model: nn.Module,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    test_inputs: Any,
    y_test_raw: np.ndarray,
    scaler,
    result_dir: Path,
    device: torch.device,
):
    prediction_path = result_dir / f"{model_name}_seed{seed}_predictions.npy"
    checkpoint_path = result_dir / f"{model_name}_seed{seed}_best.pt"
    if prediction_path.exists() and checkpoint_path.exists():
        pred = np.load(prediction_path)
        if pred.shape != y_test_raw.shape or not np.isfinite(pred).all():
            raise RuntimeError(f"Existing {model_name} seed {seed} prediction file is invalid; refusing to overwrite.")
        print(f"[RESUME] {model_name} seed {seed}: using existing verified predictions.")
        return compute_metrics(y_test_raw, pred), float("nan")

    set_seed(seed)
    model = model.to(device)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    opt_step = 0
    best_val = float("inf")
    best_step = 0
    last_val = float("nan")
    dat_mode = model_name == "DAT"

    print(f"\n{'=' * 65}\n{model_name} — SEED {seed}\n{'=' * 65}")
    while opt_step < MAX_OPT_STEPS:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)

        if dat_mode:
            xb, fb, hb, wb, yb = batch
            xb, fb = xb.to(device), fb.to(device)
            hb, wb, yb = hb.to(device), wb.to(device), yb.to(device)
            pred = model(xb, fb, hb, wb)
        else:
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)

        model.train()
        loss = criterion(pred, yb) / ACCUM_STEPS
        loss.backward()
        micro_step += 1
        if micro_step % ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
            if opt_step == 1 or opt_step % VALIDATION_INTERVAL == 0 or opt_step == MAX_OPT_STEPS:
                last_val = weighted_validation_mse(model, val_loader, device, dat_mode)
                print(f"{model_name} seed {seed} | step {opt_step}/{MAX_OPT_STEPS} | sample-weighted val MSE={last_val:.7f}")
                if last_val < best_val:
                    best_val = last_val
                    best_step = opt_step
                    torch.save(model.state_dict(), checkpoint_path)

    if not checkpoint_path.exists():
        raise RuntimeError(f"Best checkpoint was not saved: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    outputs = []
    with torch.no_grad():
        if dat_mode:
            test_loader = DataLoader(test_inputs, batch_size=BATCH_SIZE, shuffle=False)
            for xb, fb, hb, wb in test_loader:
                pred = model(xb.to(device), fb.to(device), hb.to(device), wb.to(device))
                outputs.append(pred.cpu().numpy())
        else:
            test_loader = DataLoader(test_inputs, batch_size=BATCH_SIZE, shuffle=False)
            for xb in test_loader:
                outputs.append(model(xb.to(device)).cpu().numpy())

    pred_scaled = np.concatenate(outputs, axis=0)
    predictions_raw = inverse_array(pred_scaled, scaler)
    check(predictions_raw.shape == y_test_raw.shape, f"{model_name} seed {seed}: prediction shape matches test targets")
    check(np.isfinite(predictions_raw).all(), f"{model_name} seed {seed}: predictions are finite")
    np.save(prediction_path, predictions_raw)
    metrics = compute_metrics(y_test_raw, predictions_raw)
    del model, train_loader, val_loader, test_loader, iterator
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics, best_val


def create_run_config(data_dir: Path, result_dir: Path, cache_dir: Path, device: torch.device):
    split_hashes = {s: sha256(data_dir / f"{s}_windows.npz") for s in ("train", "val", "test")}
    return {
        "experiment": "MORED corrected V3 session split vanilla transformer and DAT",
        "source": "Data/MORED/Corrected_V3_SessionSplit/*.npz only; no legacy interpolated splits or caches",
        "seeds": SEEDS,
        "max_optimizer_steps_per_seed": MAX_OPT_STEPS,
        "physical_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": ACCUM_STEPS,
        "effective_batch_size_approx": BATCH_SIZE * ACCUM_STEPS,
        "learning_rate": LEARNING_RATE,
        "lookback_minutes": LOOKBACK,
        "forecast_minutes": FORECAST,
        "stride_minutes": STRIDE,
        "split_windows": EXPECTED_WINDOWS,
        "validation_interval_optimizer_steps": VALIDATION_INTERVAL,
        "validation_loss_aggregation": "sample-weighted MSE across all target points/windows",
        "checkpoint_selection": "lowest sample-weighted validation MSE; test used only after selection",
        "device": str(device),
        "attention_probability_dropout": ATTENTION_PROBABILITY_DROPOUT,
        "encoder_residual_and_feedforward_dropout": DROPOUT,
        "vmd": {"K": VMD_K, "alpha": VMD_ALPHA, "mirror_padding": MIRROR_PADDING},
        "fuzzy_entropy": {"m": FE_M, "r": FE_R, "n": FE_N, "max_points": FE_MAX_POINTS},
        "secondary_vmd": {"K": REFINED_K, "alpha": REFINED_ALPHA},
        "dat_channels": DAT_CHANNELS,
        "input_windows_only_for_decomposition": True,
        "source_npz_sha256": split_hashes,
    }


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    result_dir = args.result_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    if not torch.backends.mps.is_available() and not torch.cuda.is_available():
        device = torch.device("cpu")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda")

    print("Device:", device)
    print("Experiment: MORED corrected V3 whole-session split — Vanilla Transformer + DAT")
    print("Validation: sample-weighted MSE; best checkpoint selected")
    train, val, test, scaler = load_all_data(data_dir)

    X_train, y_train = train["X"], train["y"]
    X_val, y_val = val["X"], val["y"]
    X_test = test["X"]
    y_test_raw = test["y_raw"].astype(np.float32)
    input_times = train["input_timestamps"]
    val_times = val["input_timestamps"]
    test_target_times = test["target_timestamps"]

    # Persist canonical test target and per-minute target timestamps once.
    np.save(result_dir / "y_true_raw.npy", y_test_raw)
    np.save(result_dir / "test_target_timestamps.npy", test_target_times.astype("datetime64[ns]"))

    # Prepare calendar embeddings from the input windows only.
    hour_train, weekday_train = calendar_features(input_times)
    hour_val, weekday_val = calendar_features(val_times)
    hour_test, weekday_test = calendar_features(test["input_timestamps"])

    print("\nCorrected split sizes:")
    print(f"  train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
    print(f"  raw MORED metric unit: W (watts)")
    print(f"  feature cache: {cache_dir}")
    print(f"  result folder: {result_dir}")

    # One-window smoke test is the default path.
    vanilla = VanillaTransformer().to(device)
    dat = DATModel().to(device)
    # DAT smoke test only; full VMD cache is created in --prepare-features stage.
    sample_feature, sample_freq, sample_entropy = process_dat_window(X_train[0])
    sample_hour = torch.from_numpy(hour_train[:1]).to(device)
    sample_weekday = torch.from_numpy(weekday_train[:1]).to(device)
    v_input = torch.from_numpy(X_train[:1]).to(device)
    d_input = torch.from_numpy(sample_feature[None]).to(device)
    d_freq = torch.from_numpy(sample_freq[None]).to(device)
    # Run forward and backward smoke tests to catch device/autograd issues.
    vanilla.train()
    v_out = vanilla(v_input)
    v_out.mean().backward()
    dat.train()
    d_out = dat(d_input, d_freq, sample_hour, sample_weekday)
    d_out.mean().backward()
    vanilla.zero_grad(set_to_none=True)
    dat.zero_grad(set_to_none=True)
    print(f"Vanilla Transformer smoke test: input={tuple(X_train[:1].shape)}, output={tuple(v_out.shape)}")
    print(f"DAT smoke test: features={sample_feature.shape}, frequencies={sample_freq.shape}, entropy={sample_entropy.shape}, output={tuple(d_out.shape)}")
    check(tuple(v_out.shape) == (1, FORECAST), "Vanilla Transformer forward pass works")
    check(tuple(d_out.shape) == (1, FORECAST), "DAT forward pass works")
    del vanilla, dat
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    if not args.prepare_features and not args.train:
        print("\nSmoke test only. No feature cache or model weights were trained.")
        print("Next: run --prepare-features, then --train.")
        return

    # Feature cache tied to each split NPZ by SHA-256.
    feature_sets = {}
    for split_name, split_data, X in (
        ("train", train, X_train),
        ("val", val, X_val),
        ("test", test, X_test),
    ):
        source_hash = sha256(data_dir / f"{split_name}_windows.npz")
        feature_sets[split_name] = make_cache(X, split_name, cache_dir, source_hash)

    X_train_dat, freq_train, entropy_train = feature_sets["train"]
    X_val_dat, freq_val, entropy_val = feature_sets["val"]
    X_test_dat, freq_test, entropy_test = feature_sets["test"]
    print("\nDAT feature shapes:")
    for name, (features, freqs, entropies) in feature_sets.items():
        print(f"  {name}: X={features.shape}, freq={freqs.shape}, entropy={entropies.shape}")
        check(features.shape[1:] == (LOOKBACK, DAT_CHANNELS), f"{name}: DAT feature dimensions valid")

    if args.prepare_features:
        print("\nMORED corrected DAT feature preparation completed.")
        return

    # Build train/validation datasets for each model family.
    vanilla_train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    vanilla_val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    vanilla_test_ds = torch.from_numpy(X_test)

    dat_train_ds = TensorDataset(
        torch.from_numpy(X_train_dat), torch.from_numpy(freq_train),
        torch.from_numpy(hour_train), torch.from_numpy(weekday_train), torch.from_numpy(y_train)
    )
    dat_val_ds = TensorDataset(
        torch.from_numpy(X_val_dat), torch.from_numpy(freq_val),
        torch.from_numpy(hour_val), torch.from_numpy(weekday_val), torch.from_numpy(y_val)
    )
    dat_test_ds = TensorDataset(
        torch.from_numpy(X_test_dat), torch.from_numpy(freq_test),
        torch.from_numpy(hour_test), torch.from_numpy(weekday_test)
    )

    config = create_run_config(data_dir, result_dir, cache_dir, device)
    (result_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    all_results = []
    for model_name in ("VanillaTransformer", "DAT"):
        for seed in SEEDS:
            # Seed before model construction so initial parameters are reproducible.
            set_seed(seed)
            if model_name == "VanillaTransformer":
                model = VanillaTransformer()
                train_ds, val_ds, test_ds = vanilla_train_ds, vanilla_val_ds, vanilla_test_ds
            else:
                model = DATModel()
                train_ds, val_ds, test_ds = dat_train_ds, dat_val_ds, dat_test_ds

            metrics, best_val = train_one_model(
                model_name=model_name,
                seed=seed,
                model=model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                test_inputs=test_ds,
                y_test_raw=y_test_raw,
                scaler=scaler,
                result_dir=result_dir,
                device=device,
            )
            row = {"Model": model_name, "Seed": seed, "Best_Val_MSE_Scaled": best_val, **metrics}
            all_results.append(row)
            pd.DataFrame(all_results).to_csv(result_dir / "mored_corrected_progress.csv", index=False)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(result_dir / "mored_corrected_results.csv", index=False)
    summary_rows = []
    for model_name in ("VanillaTransformer", "DAT"):
        subset = results_df[results_df["Model"] == model_name]
        row = {"Model": model_name}
        for metric in ("RMSE", "MAE", "MAPE", "R2", "StdR"):
            row[f"{metric}_mean"] = float(subset[metric].mean())
            row[f"{metric}_sd"] = float(subset[metric].std(ddof=1))
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(result_dir / "mored_corrected_summary.csv", index=False)
    print("\nMORED corrected Vanilla Transformer + DAT results:")
    print(results_df.to_string(index=False))
    print("\nSummary:")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("\nOutputs saved under:", result_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
