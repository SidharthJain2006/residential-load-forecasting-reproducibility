#!/usr/bin/env python3
"""Corrected-architecture London DAT experiment.

This is a NEW full-package BES ablation script. It does not overwrite any
previous script or result folder. It reuses recovered London-specific VMD/BES
helpers and the Conv1D + frequency + calendar DAT architecture. Unlike the
previous corrected run, this script toggles all three BES components together:
edge extension and overlap fusion. This keeps this result set distinct from the
prior corrected run and the historical manuscript benchmark.

Usage (from the project root):
  python run_london_dat_corrected_fullbes.py --smoke-test
  python run_london_dat_corrected_fullbes.py

This full-package ablation treats BES as all three boundary components together:
left reflection + historical right extension + weighted overlap fusion. BES OFF
uses unpadded window decomposition and disables overlap fusion. This is separate
from earlier runs. Outputs are in Results_DAT_corrected_fullbes/ and
Cache/London_DAT_corrected_fullbes/.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import the recovered script as a library so we reuse its London-specific
# configuration, VMD implementation, and historical right-boundary extension.
# Its training entry point must be guarded by if __name__ == '__main__'.
import run_bes_ablation as london_base

LOOKBACK = int(london_base.LOOKBACK)
FORECAST = int(london_base.FORECAST)
STRIDE = int(london_base.STRIDE)
VMD_K = int(london_base.VMD_K)
D_MODEL = int(london_base.D_MODEL)
NHEAD = int(london_base.NHEAD)
NUM_LAYERS = int(london_base.NUM_LAYERS)
DIM_FEEDFORWARD = int(london_base.DIM_FEEDFORWARD)
DROPOUT = float(london_base.DROPOUT)
BATCH_SIZE = int(london_base.BATCH_SIZE)
MAX_EPOCHS = int(london_base.MAX_EPOCHS)
LEARNING_RATE = float(london_base.LEARNING_RATE)
LR_PATIENCE = int(london_base.LR_PATIENCE)
MIN_DELTA = float(london_base.MIN_DELTA)
PATIENCE = int(london_base.EARLY_STOP_PATIENCE)
BES_PADDING = int(london_base.BES_PADDING)
SEEDS = list(london_base.SEEDS)

TRAIN_FILE = Path(london_base.TRAIN_FILE)
VAL_FILE = Path(london_base.VAL_FILE)
TEST_FILE = Path(london_base.TEST_FILE)
SCALER_FILE = Path(london_base.SCALER_FILE)
RESULT_DIR = Path("Results_DAT_corrected_fullbes")
CACHE_DIR = Path("Cache/London_DAT_corrected_fullbes")
CACHE_VERSION = "london-dat-conv-calendar-fullbes-v1"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_split_windows(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Make chronological windows and calendar features within this split only."""
    required = {"datetime", "consumption", "scaled_consumption"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Split missing required columns: {sorted(missing)}")
    frame = df.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    if frame["datetime"].isna().any():
        raise ValueError("Found missing timestamps")
    if not frame["datetime"].is_monotonic_increasing:
        raise ValueError("Split timestamps are not chronological")
    dts = frame["datetime"].to_numpy(dtype="datetime64[ns]")
    scaled = frame["scaled_consumption"].to_numpy(dtype=np.float32)
    starts = list(range(0, len(frame) - LOOKBACK - FORECAST + 1, STRIDE))
    if not starts:
        raise ValueError(f"Not enough rows ({len(frame)}) for lookback + horizon")
    X, y, time_windows, target_times = [], [], [], []
    for start in starts:
        X.append(scaled[start : start + LOOKBACK])
        y.append(scaled[start + LOOKBACK : start + LOOKBACK + FORECAST])
        time_windows.append(dts[start : start + LOOKBACK])
        target_times.append(dts[start + LOOKBACK : start + LOOKBACK + FORECAST])
    tw = np.asarray(time_windows, dtype="datetime64[ns]")
    return {
        "raw_windows": np.asarray(X, dtype=np.float32),
        "targets": np.asarray(y, dtype=np.float32),
        "time_windows": tw,
        "target_times": np.asarray(target_times, dtype="datetime64[ns]"),
        "starts": np.asarray(starts, dtype=np.int64),
        "series_scaled": scaled,
        "timestamps": dts,
    }


def dominant_frequency(signal: np.ndarray) -> float:
    """Match the frequency-estimation method in the recovered UCI DAT script."""
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    spectrum = np.abs(np.fft.rfft(x))
    frequencies = np.fft.rfftfreq(len(x))
    if len(spectrum) <= 1:
        return 0.0
    idx = int(np.argmax(spectrum[1:]) + 1)  # ignore DC
    return float(frequencies[idx])


def ten_channel_refinement(modes: np.ndarray, primary_frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 2 channels per primary VMD mode, retaining VMD centre frequencies."""
    entropy_values = np.asarray(
        [london_base.fuzzy_entropy(mode, m=2, r_ratio=0.20, n=2) for mode in modes],
        dtype=np.float32,
    )
    primary_frequencies = np.asarray(primary_frequencies, dtype=np.float32).reshape(-1)
    if primary_frequencies.shape != (VMD_K,):
        raise ValueError(f"Expected {VMD_K} primary centre frequencies, got {primary_frequencies.shape}")
    threshold = float(np.median(entropy_values))
    channels: list[np.ndarray] = []
    frequencies: list[float] = []
    for mode_index in range(VMD_K):
        mode = np.asarray(modes[mode_index], dtype=np.float32)
        if float(entropy_values[mode_index]) > threshold:
            # Use the same secondary VMD setup (K=2, alpha=2000) as run_uci_dat.py.
            refined, refined_frequencies = london_base.run_vmd(mode, k=2)
            if refined.shape != (2, LOOKBACK):
                raise ValueError(f"Unexpected secondary VMD shape: {refined.shape}")
            for component in refined:
                component = np.asarray(component, dtype=np.float32)
                channels.append(component)
            frequencies.extend(np.asarray(refined_frequencies, dtype=np.float32).reshape(-1).tolist())
        else:
            channels.append(mode)
            frequencies.append(float(primary_frequencies[mode_index]))
            channels.append(np.zeros_like(mode, dtype=np.float32))
            frequencies.append(0.0)
    channel_tensor = np.stack(channels, axis=1).astype(np.float32)
    frequency_vector = np.asarray(frequencies, dtype=np.float32)
    if channel_tensor.shape != (LOOKBACK, VMD_K * 2):
        raise ValueError(f"Expected {(LOOKBACK, VMD_K * 2)}, got {channel_tensor.shape}")
    if not np.isfinite(channel_tensor).all() or not np.isfinite(frequency_vector).all():
        raise ValueError("Non-finite value in refined channels/frequencies")
    return channel_tensor, frequency_vector, entropy_values


def cache_key(split_name: str, use_bes: bool, split: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    h.update(CACHE_VERSION.encode())
    h.update(split_name.encode())
    h.update(str(use_bes).encode())
    h.update(json.dumps({
        "lookback": LOOKBACK, "forecast": FORECAST, "stride": STRIDE,
        "vmd_k": VMD_K, "alpha": int(london_base.VMD_ALPHA),
        "bes_padding": BES_PADDING,
    }, sort_keys=True).encode())
    h.update(np.ascontiguousarray(split["series_scaled"]).view(np.uint8))
    h.update(np.ascontiguousarray(split["timestamps"]).view(np.uint8))
    return h.hexdigest()[:16]


def make_features(split_name: str, split: dict[str, np.ndarray], use_bes: bool) -> tuple[np.ndarray, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = CACHE_DIR / f"{split_name}_{'bes_on' if use_bes else 'bes_off'}_{cache_key(split_name, use_bes, split)}.npz"
    if filename.exists():
        cached = np.load(filename)
        features, frequencies = cached["features"], cached["frequencies"]
        expected_x = (len(split["raw_windows"]), LOOKBACK, VMD_K * 2)
        expected_f = (len(split["raw_windows"]), VMD_K * 2)
        if features.shape == expected_x and frequencies.shape == expected_f:
            print(f"Loading cache: {filename}")
            return features.astype(np.float32), frequencies.astype(np.float32)
        print("Ignoring cache with unexpected dimensions:", filename)

    raw_windows = split["raw_windows"]
    starts = split["starts"]
    whole_series = split["series_scaled"]
    per_window_features: list[np.ndarray] = []
    per_window_freq: list[np.ndarray] = []
    for i, (window, start) in enumerate(zip(raw_windows, starts), start=1):
        if use_bes:
            # London-specific BES: mirror the left boundary and extend the right
            # from a historically similar segment using only earlier observations.
            left = np.asarray(window[:BES_PADDING][::-1], dtype=np.float32)
            right = np.asarray(
                london_base.historical_right_extension(whole_series, int(start), window),
                dtype=np.float32,
            )
            if right.shape != (BES_PADDING,):
                raise ValueError(f"Historical extension should have {BES_PADDING} values, got {right.shape}")
            padded = np.concatenate([left, window, right]).astype(np.float32)
            padded_modes, primary_frequencies = london_base.run_vmd(padded, k=VMD_K)
            modes = padded_modes[:, BES_PADDING : BES_PADDING + LOOKBACK]
        else:
            modes, primary_frequencies = london_base.run_vmd(window, k=VMD_K)
        if modes.shape != (VMD_K, LOOKBACK):
            raise ValueError(f"Unexpected primary VMD shape {modes.shape}")
        channel_tensor, frequency_vector, _ = ten_channel_refinement(modes, primary_frequencies)
        per_window_features.append(channel_tensor)
        per_window_freq.append(frequency_vector)
        if i % 50 == 0 or i == len(raw_windows):
            print(f"  {split_name} BES={'ON' if use_bes else 'OFF'}: {i}/{len(raw_windows)} windows")

    features = np.asarray(per_window_features, dtype=np.float32)
    frequencies = np.asarray(per_window_freq, dtype=np.float32)

    # Weighted overlap fusion is a component of the BES package. Apply it
    # only when BES is ON. The prior window ends before the current forecast
    # origin, so this does not use a later window to revise an earlier one.
    # Frequency embeddings remain those computed for the current window; the
    # manuscript describes overlap fusion of decomposed modes, not frequencies.
    if use_bes and STRIDE < LOOKBACK and len(features) > 1:
        overlap = LOOKBACK - STRIDE
        weights = np.linspace(1.0, 0.0, overlap, dtype=np.float32)[:, None]
        for i in range(1, len(features)):
            prev_tail = features[i - 1, STRIDE : STRIDE + overlap, :].copy()
            cur_head = features[i, :overlap, :].copy()
            features[i, :overlap, :] = weights * prev_tail + (1.0 - weights) * cur_head

    np.savez_compressed(filename, features=features, frequencies=frequencies)
    print(f"Saved cache: {filename}")
    return features, frequencies


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"Sequence length {x.size(1)} > positional encoding length {self.pe.size(1)}")
        return self.dropout(x + self.pe[:, : x.size(1), :])


class MPSCompatibleTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Transformer encoder layer that avoids MPS SDPA's training-dropout limitation.

    PyTorch's default TransformerEncoderLayer uses scaled_dot_product_attention
    when need_weights=False. Some MPS/PyTorch combinations do not support
    nonzero attention dropout through that path. Requesting attention weights
    uses the explicit attention-matrix path instead, preserving the configured
    attention dropout rather than disabling dropout or moving training to CPU.
    """
    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal=False):
        attn_output = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            is_causal=is_causal,
        )[0]
        return self.dropout1(attn_output)


class LondonDAT(nn.Module):
    """Mode-specific Conv1D DAT with per-channel frequency and calendar embeddings."""
    def __init__(self):
        super().__init__()
        channels = VMD_K * 2
        self.mode_convs = nn.ModuleList([
            nn.Conv1d(1, D_MODEL, kernel_size=3, padding=1)
            for _ in range(channels)
        ])
        self.frequency_projection = nn.Linear(1, D_MODEL)
        self.hour_embedding = nn.Embedding(24, D_MODEL)
        self.weekday_embedding = nn.Embedding(7, D_MODEL)
        self.position = PositionalEncoding(D_MODEL, LOOKBACK)
        encoder_layer = MPSCompatibleTransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.output_head = nn.Sequential(
            nn.Linear(D_MODEL, 128),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, FORECAST),
        )

    def forward(self, x: torch.Tensor, frequencies: torch.Tensor,
                hours: torch.Tensor, weekdays: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, 10], frequencies: [batch, 10], calendar: [batch, time]
        mode_tokens = []
        for channel_index, conv in enumerate(self.mode_convs):
            channel = x[:, :, channel_index].unsqueeze(1)  # [B, 1, T]
            projected = conv(channel).transpose(1, 2)       # [B, T, D]
            freq = frequencies[:, channel_index].unsqueeze(1)  # [B, 1]
            freq_embedding = self.frequency_projection(freq).unsqueeze(1)  # [B, 1, D]
            mode_tokens.append(projected + freq_embedding)
        token = torch.stack(mode_tokens, dim=0).mean(dim=0)
        token = token + self.hour_embedding(hours) + self.weekday_embedding(weekdays)
        token = self.position(token)
        encoded = self.transformer(token)
        return self.output_head(encoded[:, -1, :])


def add_calendar(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # times are [n_windows, LOOKBACK] local-naive timestamps from the supplied dataset.
    flat = pd.to_datetime(times.reshape(-1))
    hours = flat.hour.to_numpy(dtype=np.int64).reshape(times.shape)
    weekdays = flat.day_of_week.to_numpy(dtype=np.int64).reshape(times.shape)
    return hours, weekdays


def create_loader(features: np.ndarray, frequencies: np.ndarray, hours: np.ndarray,
                  weekdays: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(frequencies.astype(np.float32)),
        torch.from_numpy(hours.astype(np.int64)),
        torch.from_numpy(weekdays.astype(np.int64)),
        torch.from_numpy(y.astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(0)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, generator=generator if shuffle else None)


def inverse_scale(scaler: Any, arr: np.ndarray) -> np.ndarray:
    shape = arr.shape
    return scaler.inverse_transform(arr.reshape(-1, 1)).reshape(shape)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    a = actual.reshape(-1)
    p = predicted.reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(a, p)))
    mae = float(mean_absolute_error(a, p))
    r2 = float(r2_score(a, p)) if len(a) > 1 else float("nan")
    actual_sd = float(np.std(a, ddof=0))
    pred_sd = float(np.std(p, ddof=0))
    stdr = pred_sd / actual_sd if actual_sd > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "StdR": stdr}


def train_one_seed(seed: int, train: dict[str, Any], val: dict[str, Any], test: dict[str, Any],
                   scaler: Any, bes_label: str) -> tuple[dict[str, Any], pd.DataFrame]:
    set_seed(seed)
    model = LondonDAT().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=LR_PATIENCE)
    loss_fn = nn.MSELoss()
    train_loader = create_loader(train["features"], train["frequencies"], train["hours"], train["weekdays"], train["targets"], True)
    val_loader = create_loader(val["features"], val["frequencies"], val["hours"], val["weekdays"], val["targets"], False)
    best_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_sum = 0.0
        for xb, fb, hb, wb, yb in train_loader:
            xb, fb, hb, wb, yb = (v.to(DEVICE) for v in (xb, fb, hb, wb, yb))
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb, fb, hb, wb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_sum += float(loss.item()) * xb.size(0)
        train_loss = train_sum / len(train_loader.dataset)

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for xb, fb, hb, wb, yb in val_loader:
                xb, fb, hb, wb, yb = (v.to(DEVICE) for v in (xb, fb, hb, wb, yb))
                pred = model(xb, fb, hb, wb)
                val_sum += float(loss_fn(pred, yb).item()) * xb.size(0)
        val_loss = val_sum / len(val_loader.dataset)
        scheduler.step(val_loss)
        print(f"BES={bes_label} seed={seed} epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_loss - MIN_DELTA:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    test_loader = create_loader(test["features"], test["frequencies"], test["hours"], test["weekdays"], test["targets"], False)
    predictions = []
    truths = []
    with torch.no_grad():
        for xb, fb, hb, wb, yb in test_loader:
            xb, fb, hb, wb = (v.to(DEVICE) for v in (xb, fb, hb, wb))
            pred = model(xb, fb, hb, wb).detach().cpu().numpy()
            predictions.append(pred)
            truths.append(yb.numpy())
    pred_scaled = np.concatenate(predictions, axis=0)
    true_scaled = np.concatenate(truths, axis=0)
    pred_original = inverse_scale(scaler, pred_scaled)
    true_original = inverse_scale(scaler, true_scaled)
    result = metrics(true_original, pred_original)
    result.update({"seed": seed, "BES": bes_label, "best_val_mse_scaled": best_loss, "best_epoch_stop_or_end": epoch})

    rows = []
    target_times = pd.to_datetime(test["target_times"].reshape(-1))
    origin_times = pd.to_datetime(test["time_windows"][:, -1])
    n_windows = pred_original.shape[0]
    for i in range(n_windows):
        for h in range(FORECAST):
            rows.append({
                "BES": bes_label,
                "seed": seed,
                "origin_id": i,
                "input_end_datetime": origin_times[i],
                "forecast_step": h + 1,
                "forecast_datetime": pd.Timestamp(test["target_times"][i, h]),
                "actual": float(true_original[i, h]),
                "predicted": float(pred_original[i, h]),
            })
    return result, pd.DataFrame(rows)


def prepare_split(name: str, path: Path, use_bes: bool) -> dict[str, Any]:
    print(f"\nLoading {name}: {path}")
    df = pd.read_csv(path, parse_dates=["datetime"])
    split = make_split_windows(df)
    features, frequencies = make_features(name, split, use_bes)
    hours, weekdays = add_calendar(split["time_windows"])
    out = dict(split)
    out.update({"features": features, "frequencies": frequencies, "hours": hours, "weekdays": weekdays})
    print(f"{name}: features={features.shape}, freq={frequencies.shape}, y={out['targets'].shape}, calendar={hours.shape}")
    return out


def smoke_test() -> None:
    print("Device:", DEVICE)
    df = pd.read_csv(TRAIN_FILE, parse_dates=["datetime"])
    split = make_split_windows(df)
    print("Train windows:", split["raw_windows"].shape, split["targets"].shape)
    for use_bes in (False, True):
        # Only run a single window through preprocessing for smoke-test speed.
        one = {k: (v[:1] if isinstance(v, np.ndarray) and len(v.shape) and v.shape[0] == len(split["raw_windows"]) else v)
               for k, v in split.items()}
        x, freq = make_features_smoke(one, use_bes)
        hours, weekdays = add_calendar(one["time_windows"])
        model = LondonDAT().to(DEVICE)
        with torch.no_grad():
            yhat = model(
                torch.from_numpy(x).to(DEVICE),
                torch.from_numpy(freq).to(DEVICE),
                torch.from_numpy(hours).to(DEVICE),
                torch.from_numpy(weekdays).to(DEVICE),
            )
        print(f"BES={'ON' if use_bes else 'OFF'}: x={x.shape}, freq={freq.shape}, calendar={hours.shape}, prediction={tuple(yhat.shape)}")
        assert x.shape == (1, LOOKBACK, VMD_K * 2)
        assert freq.shape == (1, VMD_K * 2)
        assert hours.shape == weekdays.shape == (1, LOOKBACK)
        assert tuple(yhat.shape) == (1, FORECAST)
        assert torch.isfinite(yhat).all().item()
    print("SMOKE TEST PASSED. No model training was performed.")


def make_features_smoke(split: dict[str, np.ndarray], use_bes: bool) -> tuple[np.ndarray, np.ndarray]:
    window = split["raw_windows"][0]
    start = int(split["starts"][0])
    whole_series = split["series_scaled"]
    if use_bes:
        left = window[:BES_PADDING][::-1]
        right = np.asarray(london_base.historical_right_extension(whole_series, start, window), dtype=np.float32)
        padded = np.concatenate([left, window, right]).astype(np.float32)
        modes, primary_frequencies = london_base.run_vmd(padded, k=VMD_K)
        modes = modes[:, BES_PADDING:BES_PADDING + LOOKBACK]
    else:
        modes, primary_frequencies = london_base.run_vmd(window, k=VMD_K)
    channels, freq, _ = ten_channel_refinement(modes, primary_frequencies)
    return channels[None, ...], freq[None, ...]


def run_experiment() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if any(RESULT_DIR.iterdir()):
        raise SystemExit(
            f"Refusing to overwrite existing output in {RESULT_DIR}. "
            "Rename/archive that folder or remove it intentionally before rerunning."
        )
    import pickle
    with SCALER_FILE.open("rb") as f:
        scaler = pickle.load(f)

    all_results: list[dict[str, Any]] = []
    # Prepare caches independently for BES OFF / ON, and use a fresh result folder.
    for use_bes, label in ((False, "OFF"), (True, "ON")):
        print("\n" + "=" * 72)
        print(f"CORRECTED LONDON DAT — BES {label}")
        print("=" * 72)
        train = prepare_split("train", TRAIN_FILE, use_bes)
        val = prepare_split("val", VAL_FILE, use_bes)
        test = prepare_split("test", TEST_FILE, use_bes)
        for seed in SEEDS:
            result, predictions = train_one_seed(seed, train, val, test, scaler, label)
            print(f"RESULT BES={label} seed={seed}: RMSE={result['RMSE']:.6f} MAE={result['MAE']:.6f} StdR={result['StdR']:.6f}")
            all_results.append(result)
            pred_path = RESULT_DIR / f"predictions_bes_{label.lower()}_seed{seed}.csv"
            predictions.to_csv(pred_path, index=False)
            print("Saved predictions:", pred_path)
            pd.DataFrame(all_results).to_csv(RESULT_DIR / "per_seed_results.csv", index=False)

    per_seed = pd.DataFrame(all_results)
    per_seed.to_csv(RESULT_DIR / "per_seed_results.csv", index=False)
    summary_rows = []
    for label in ("OFF", "ON"):
        part = per_seed[per_seed["BES"] == label]
        row: dict[str, Any] = {"Configuration": f"DAT (corrected architecture, BES {label.lower()})", "BES": label}
        for metric_name in ("RMSE", "MAE", "R2", "StdR"):
            row[f"{metric_name}_mean"] = float(part[metric_name].mean())
            row[f"{metric_name}_sd"] = float(part[metric_name].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULT_DIR / "summary_corrected_dat.csv", index=False)
    metadata = {
        "cache_version": CACHE_VERSION,
        "device": str(DEVICE),
        "seeds": SEEDS,
        "lookback": LOOKBACK,
        "forecast": FORECAST,
        "stride": STRIDE,
        "primary_vmd_k": VMD_K,
        "primary_vmd_alpha": int(london_base.VMD_ALPHA),
        "secondary_vmd_k": 2,
        "bes_padding": BES_PADDING,
        "architecture": "mode-specific Conv1d; VMD centre-frequency projection per channel; hour and weekday embeddings; positional encoding; Transformer encoder; direct forecast head",
        "channels": VMD_K * 2,
        "training": {"batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "learning_rate": LEARNING_RATE,
                     "lr_patience": LR_PATIENCE, "early_stopping_patience": PATIENCE, "min_delta": MIN_DELTA},
        "bes_definition": "BES ON applies left-edge reflection, a historical-similarity right extension, and weighted overlap fusion. BES OFF applies none of these three boundary components.",
        "causality_note": "When BES is ON, overlap fusion uses the preceding window and current window only; no later window revises an earlier forecast origin.",
        "interpretation_note": "This is a new corrected-architecture experiment. Do not conflate these outputs with the previously recovered five-mode linear-projection BES run or the manuscript's historical multi-household London benchmark.",
    }
    (RESULT_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("\n===== CORRECTED DAT SUMMARY =====")
    print(summary.to_string(index=False))
    print("\nAll corrected-architecture outputs saved under:", RESULT_DIR.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true", help="Validate preprocessing/model shapes on one window, without training")
    args = parser.parse_args()
    print(f"Using device: {DEVICE}")
    if args.smoke_test:
        smoke_test()
    else:
        run_experiment()


if __name__ == "__main__":
    main()
