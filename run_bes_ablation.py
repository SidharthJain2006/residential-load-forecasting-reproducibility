#!/usr/bin/env python3

import os
import math
import random
import pickle

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error
from vmdpy import VMD


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = "Data/London/splits/train.csv"
VAL_FILE = "Data/London/splits/val.csv"
TEST_FILE = "Data/London/splits/test.csv"
SCALER_FILE = "Data/London/splits/scaler.pkl"

RESULT_DIR = "Results"
CACHE_DIR = "Cache"

SEEDS = [0, 1, 2, 3, 4]

# London = 30 minute sampling
LOOKBACK = 336             # 7 days
FORECAST = 48              # 24 hours
STRIDE = 48                # one forecast every 24 hours

# VMD
VMD_K = 5
VMD_ALPHA = 2000
VMD_TAU = 0
VMD_DC = 0
VMD_INIT = 1
VMD_TOL = 1e-7

# FE
FE_M = 2
FE_R_RATIO = 0.20
FE_N = 2

# BES
BES_PADDING = 48

# Transformer
D_MODEL = 64
NHEAD = 4
NUM_LAYERS = 2
DIM_FEEDFORWARD = 128
DROPOUT = 0.1

# Training
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 100

LR_PATIENCE = 5
EARLY_STOP_PATIENCE = 10
MIN_DELTA = 0.0001

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("Using device:", DEVICE)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# LOAD DATA
# ============================================================

def load_split(path):
    df = pd.read_csv(path)

    return df["scaled_consumption"].astype(float).values


train_values = load_split(TRAIN_FILE)
val_values = load_split(VAL_FILE)
test_values = load_split(TEST_FILE)

with open(SCALER_FILE, "rb") as f:
    scaler = pickle.load(f)


# ============================================================
# WINDOW CREATION
# ============================================================

def make_windows(values):
    X = []
    y = []

    starts = range(
        0,
        len(values) - LOOKBACK - FORECAST + 1,
        STRIDE
    )

    for start in starts:

        X.append(
            values[start:start + LOOKBACK]
        )

        y.append(
            values[
                start + LOOKBACK:
                start + LOOKBACK + FORECAST
            ]
        )

    return np.asarray(X), np.asarray(y)


X_train_raw, y_train = make_windows(train_values)
X_val_raw, y_val = make_windows(val_values)
X_test_raw, y_test = make_windows(test_values)

print("\nWindow counts")
print("Train:", X_train_raw.shape, y_train.shape)
print("Val  :", X_val_raw.shape, y_val.shape)
print("Test :", X_test_raw.shape, y_test.shape)


# ============================================================
# FUZZY ENTROPY
# ============================================================

def fuzzy_entropy(x, m=2, r_ratio=0.2, n=2):

    x = np.asarray(x, dtype=np.float64)

    std = np.std(x)

    if std < 1e-12:
        return 0.0

    r = r_ratio * std

    def phi(dim):

        N = len(x) - dim + 1

        patterns = np.array(
            [x[i:i + dim] for i in range(N)]
        )

        # Remove local mean
        patterns = (
            patterns
            - patterns.mean(axis=1, keepdims=True)
        )

        total = 0.0
        count = 0

        for i in range(N):

            dist = np.max(
                np.abs(patterns - patterns[i]),
                axis=1
            )

            mask = np.arange(N) != i

            d = dist[mask]

            similarity = np.exp(
                -(d ** n) / (r + 1e-12)
            )

            total += similarity.sum()
            count += len(similarity)

        return total / max(count, 1)

    p1 = phi(m)
    p2 = phi(m + 1)

    if p1 <= 0 or p2 <= 0:
        return 0.0

    return float(np.log(p1 / p2))


# ============================================================
# VMD
# ============================================================

def run_vmd(signal, k=VMD_K):

    u, u_hat, omega = VMD(
        signal,
        VMD_ALPHA,
        VMD_TAU,
        k,
        VMD_DC,
        VMD_INIT,
        VMD_TOL
    )

    final_omega = omega[-1]

    return u, final_omega


# ============================================================
# FUZZY-ENTROPY REFINEMENT
# ============================================================

def fe_refine(modes):

    entropies = np.array(
        [fuzzy_entropy(mode) for mode in modes]
    )

    # Dynamic threshold for this window
    threshold = np.median(entropies)

    refined = modes.copy()

    for i, entropy in enumerate(entropies):

        if entropy > threshold:

            sub_modes, _ = run_vmd(
                modes[i],
                k=2
            )

            # Explicit rerun choice:
            # retain lower-frequency refined component
            refined[i] = sub_modes[-1]

    return refined


# ============================================================
# HISTORICAL SIMILARITY EXTENSION
# ============================================================

def historical_right_extension(
    whole_series,
    window_start,
    window
):

    M = BES_PADDING

    if window_start <= LOOKBACK + M:
        return window[-M:][::-1]

    terminal = window[-M:]

    best_distance = np.inf
    best_extension = None

    # Search previous daily-aligned candidates.
    # Another explicit rerun choice.
    candidate_step = 48

    latest_candidate = (
        window_start - M - LOOKBACK
    )

    for start in range(
        0,
        max(0, latest_candidate),
        candidate_step
    ):

        candidate = whole_series[
            start + LOOKBACK - M:
            start + LOOKBACK
        ]

        extension = whole_series[
            start + LOOKBACK:
            start + LOOKBACK + M
        ]

        if (
            len(candidate) != M
            or len(extension) != M
        ):
            continue

        distance = np.linalg.norm(
            terminal - candidate
        )

        if distance < best_distance:
            best_distance = distance
            best_extension = extension

    if best_extension is None:
        return window[-M:][::-1]

    return best_extension


# ============================================================
# DECOMPOSITION FOR ONE SPLIT
# ============================================================

def decompose_windows(
    values,
    use_bes,
    split_name
):

    cache_file = os.path.join(
        CACHE_DIR,
        f"{split_name}_bes_{'on' if use_bes else 'off'}.npz"
    )

    if os.path.exists(cache_file):

        print("Loading cache:", cache_file)

        z = np.load(cache_file)

        return z["X"], z["omega"]

    windows, _ = make_windows(values)

    processed_modes = []
    processed_omega = []

    previous_modes = None

    starts = list(
        range(
            0,
            len(values) - LOOKBACK - FORECAST + 1,
            STRIDE
        )
    )

    print(
        f"\nPreprocessing {split_name} "
        f"BES={'ON' if use_bes else 'OFF'}"
    )

    for idx, (start, window) in enumerate(
        zip(starts, windows)
    ):

        if use_bes:

            M = BES_PADDING

            left = window[:M][::-1]

            right = historical_right_extension(
                values,
                start,
                window
            )

            padded = np.concatenate([
                left,
                window,
                right
            ])

            modes, omega = run_vmd(padded)

            modes = modes[:, M:M + LOOKBACK]

        else:

            modes, omega = run_vmd(window)

        modes = fe_refine(modes)

        # Weighted overlap fusion
        if (
            use_bes
            and previous_modes is not None
            and STRIDE < LOOKBACK
        ):

            overlap = LOOKBACK - STRIDE

            prev_overlap = previous_modes[
                :,
                STRIDE:
            ]

            curr_overlap = modes[
                :,
                :overlap
            ]

            weights = np.linspace(
                1.0,
                0.0,
                overlap
            )

            modes[:, :overlap] = (
                prev_overlap * weights
                + curr_overlap * (1.0 - weights)
            )

        previous_modes = modes.copy()

        processed_modes.append(modes.T)
        processed_omega.append(omega)

        if (
            (idx + 1) % 50 == 0
            or idx == len(windows) - 1
        ):
            print(
                f"  {idx + 1}/{len(windows)}"
            )

    X = np.asarray(
        processed_modes,
        dtype=np.float32
    )

    omega = np.asarray(
        processed_omega,
        dtype=np.float32
    )

    np.savez_compressed(
        cache_file,
        X=X,
        omega=omega
    )

    print("Saved cache:", cache_file)

    return X, omega


# ============================================================
# POSITIONAL ENCODING
# ============================================================

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len):

        super().__init__()

        pe = torch.zeros(
            max_len,
            d_model
        )

        position = torch.arange(
            0,
            max_len
        ).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(
                0,
                d_model,
                2
            ).float()
            * (
                -math.log(10000.0)
                / d_model
            )
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):

        return (
            x
            + self.pe[:, :x.size(1)]
        )


# ============================================================
# DAT TRANSFORMER
# ============================================================

class DATTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.mode_projection = nn.Linear(
            VMD_K,
            D_MODEL
        )

        self.frequency_projection = nn.Linear(
            VMD_K,
            D_MODEL
        )

        self.position = PositionalEncoding(
            D_MODEL,
            LOOKBACK
        )

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=NUM_LAYERS
        )

        self.head = nn.Sequential(
            nn.Linear(D_MODEL, 128),
            nn.GELU(),
            nn.Linear(128, FORECAST)
        )

    def forward(self, x, omega):

        # x:
        # batch x LOOKBACK x K

        z = self.mode_projection(x)

        frequency_embedding = (
            self.frequency_projection(omega)
            .unsqueeze(1)
        )

        z = z + frequency_embedding

        z = self.position(z)

        z = self.encoder(z)

        # Last temporal representation
        z = z[:, -1, :]

        return self.head(z)


# ============================================================
# DATA LOADERS
# ============================================================

def create_loader(
    X,
    omega,
    y,
    shuffle
):

    dataset = TensorDataset(
        torch.tensor(
            X,
            dtype=torch.float32
        ),
        torch.tensor(
            omega,
            dtype=torch.float32
        ),
        torch.tensor(
            y,
            dtype=torch.float32
        )
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle
    )


# ============================================================
# TRAIN ONE SEED
# ============================================================

def train_one_seed(
    seed,
    X_train,
    omega_train,
    X_val,
    omega_val,
    X_test,
    omega_test
):

    set_seed(seed)

    model = DATTransformer().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=LR_PATIENCE
    )

    loss_fn = nn.MSELoss()

    train_loader = create_loader(
        X_train,
        omega_train,
        y_train,
        True
    )

    val_loader = create_loader(
        X_val,
        omega_val,
        y_val,
        False
    )

    # y_train/y_val above were already made using STRIDE.
    # Fix indexing:
    train_targets = y_train
    val_targets = y_val

    train_loader = create_loader(
        X_train,
        omega_train,
        train_targets,
        True
    )

    val_loader = create_loader(
        X_val,
        omega_val,
        val_targets,
        False
    )

    best_loss = np.inf
    best_state = None
    bad_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()

        train_loss = 0.0

        for xb, ob, yb in train_loader:

            xb = xb.to(DEVICE)
            ob = ob.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            pred = model(xb, ob)

            loss = loss_fn(
                pred,
                yb
            )

            loss.backward()

            optimizer.step()

            train_loss += (
                loss.item()
                * xb.size(0)
            )

        train_loss /= len(
            train_loader.dataset
        )

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for xb, ob, yb in val_loader:

                xb = xb.to(DEVICE)
                ob = ob.to(DEVICE)
                yb = yb.to(DEVICE)

                pred = model(xb, ob)

                loss = loss_fn(
                    pred,
                    yb
                )

                val_loss += (
                    loss.item()
                    * xb.size(0)
                )

        val_loss /= len(
            val_loader.dataset
        )

        scheduler.step(val_loss)

        print(
            f"Seed {seed} "
            f"Epoch {epoch:03d} "
            f"train={train_loss:.6f} "
            f"val={val_loss:.6f}"
        )

        if (
            best_loss - val_loss
            > MIN_DELTA
        ):

            best_loss = val_loss

            best_state = {
                k: v.detach().cpu().clone()
                for k, v
                in model.state_dict().items()
            }

            bad_epochs = 0

        else:

            bad_epochs += 1

        if (
            bad_epochs
            >= EARLY_STOP_PATIENCE
        ):

            print(
                "Early stopping."
            )

            break

    model.load_state_dict(
        best_state
    )

    test_loader = create_loader(
        X_test,
        omega_test,
        y_test,
        False
    )

    predictions = []
    truths = []

    model.eval()

    with torch.no_grad():

        for xb, ob, yb in test_loader:

            xb = xb.to(DEVICE)
            ob = ob.to(DEVICE)

            pred = model(
                xb,
                ob
            )

            predictions.append(
                pred.cpu().numpy()
            )

            truths.append(
                yb.numpy()
            )

    pred_scaled = np.concatenate(
        predictions,
        axis=0
    )

    true_scaled = np.concatenate(
        truths,
        axis=0
    )

    # Convert back to kWh
    pred = scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).reshape(pred_scaled.shape)

    truth = scaler.inverse_transform(
        true_scaled.reshape(-1, 1)
    ).reshape(true_scaled.shape)

    return truth, pred


# ============================================================
# METRICS
# ============================================================

def metrics(y_true, y_pred):

    yt = y_true.ravel()
    yp = y_pred.ravel()

    rmse = np.sqrt(
        mean_squared_error(
            yt,
            yp
        )
    )

    mae = mean_absolute_error(
        yt,
        yp
    )

    stdr = (
        np.std(yp)
        / np.std(yt)
    )

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "StdR": float(stdr)
    }


# ============================================================
# RUN BES CONFIGURATION
# ============================================================

def run_configuration(use_bes):

    X_train, omega_train = decompose_windows(
        train_values,
        use_bes,
        "train"
    )

    X_val, omega_val = decompose_windows(
        val_values,
        use_bes,
        "val"
    )

    X_test, omega_test = decompose_windows(
        test_values,
        use_bes,
        "test"
    )

    rows = []

    for seed in SEEDS:

        print("\n" + "=" * 60)
        print(
            f"BES={'ON' if use_bes else 'OFF'} "
            f"SEED={seed}"
        )
        print("=" * 60)

        truth, pred = train_one_seed(
            seed,
            X_train,
            omega_train,
            X_val,
            omega_val,
            X_test,
            omega_test
        )

        result = metrics(
            truth,
            pred
        )

        result["seed"] = seed
        result["BES"] = (
            "ON"
            if use_bes
            else "OFF"
        )

        rows.append(result)

        print(
            f"\nRESULT "
            f"RMSE={result['RMSE']:.5f} "
            f"MAE={result['MAE']:.5f} "
            f"StdR={result['StdR']:.5f}"
        )

        prediction_file = os.path.join(
            RESULT_DIR,
            f"predictions_bes_"
            f"{'on' if use_bes else 'off'}"
            f"_seed{seed}.csv"
        )

        pd.DataFrame({
            "truth": truth.ravel(),
            "prediction": pred.ravel()
        }).to_csv(
            prediction_file,
            index=False
        )

    return rows


# ============================================================
# SUMMARY
# ============================================================

def summarise(rows):

    output = {}

    for metric in [
        "RMSE",
        "MAE",
        "StdR"
    ]:

        values = np.array([
            r[metric]
            for r in rows
        ])

        output[metric] = (
            values.mean(),
            values.std(ddof=1)
        )

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("DAT BES ABLATION")
    print("London Smart Meter")
    print("=" * 70)

    print("LOOKBACK:", LOOKBACK)
    print("FORECAST:", FORECAST)
    print("STRIDE:", STRIDE)
    print("VMD K:", VMD_K)
    print("VMD alpha:", VMD_ALPHA)
    print("BES padding:", BES_PADDING)
    print("Seeds:", SEEDS)

    off_rows = run_configuration(
        False
    )

    on_rows = run_configuration(
        True
    )

    all_rows = (
        off_rows
        + on_rows
    )

    pd.DataFrame(
        all_rows
    ).to_csv(
        os.path.join(
            RESULT_DIR,
            "per_seed_results.csv"
        ),
        index=False
    )

    off = summarise(
        off_rows
    )

    on = summarise(
        on_rows
    )

    def cell(x):

        return (
            f"{x[0]:.3f} "
            f"± "
            f"{x[1]:.3f}"
        )

    print("\n")
    print("=" * 70)
    print("TABLE 6")
    print("=" * 70)

    print(
        f"{'Configuration':<18}"
        f"{'BES':<8}"
        f"{'RMSE':<20}"
        f"{'MAE':<20}"
        f"{'StdR':<20}"
    )

    print("-" * 86)

    print(
        f"{'DAT (BES off)':<18}"
        f"{'Off':<8}"
        f"{cell(off['RMSE']):<20}"
        f"{cell(off['MAE']):<20}"
        f"{cell(off['StdR']):<20}"
    )

    print(
        f"{'DAT (BES on)':<18}"
        f"{'On':<8}"
        f"{cell(on['RMSE']):<20}"
        f"{cell(on['MAE']):<20}"
        f"{cell(on['StdR']):<20}"
    )

    table = pd.DataFrame([
        {
            "Configuration":
                "DAT (BES off)",
            "BES": "Off",
            "RMSE_mean":
                off["RMSE"][0],
            "RMSE_sd":
                off["RMSE"][1],
            "MAE_mean":
                off["MAE"][0],
            "MAE_sd":
                off["MAE"][1],
            "StdR_mean":
                off["StdR"][0],
            "StdR_sd":
                off["StdR"][1]
        },
        {
            "Configuration":
                "DAT (BES on)",
            "BES": "On",
            "RMSE_mean":
                on["RMSE"][0],
            "RMSE_sd":
                on["RMSE"][1],
            "MAE_mean":
                on["MAE"][0],
            "MAE_sd":
                on["MAE"][1],
            "StdR_mean":
                on["StdR"][0],
            "StdR_sd":
                on["StdR"][1]
        }
    ])

    table.to_csv(
        os.path.join(
            RESULT_DIR,
            "table6.csv"
        ),
        index=False
    )

    print("\nEffect of enabling BES")

    print(
        "ΔRMSE:",
        on["RMSE"][0]
        - off["RMSE"][0]
    )

    print(
        "ΔMAE:",
        on["MAE"][0]
        - off["MAE"][0]
    )

    print(
        "ΔStdR:",
        on["StdR"][0]
        - off["StdR"][0]
    )

    print("\nSaved:")
    print(
        "Results/per_seed_results.csv"
    )
    print(
        "Results/table6.csv"
    )


if __name__ == "__main__":
    main()