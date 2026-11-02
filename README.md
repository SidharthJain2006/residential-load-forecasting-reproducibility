# Residential Load Forecasting Reproducibility Study

**Does Causal Decomposition Improve Residential Load Forecasting? A Leakage-Aware, Multi-Dataset Reproducibility Study of a Decomposition-Augmented Transformer**

This is a **pre-release research repository** containing the corrected experiment scripts, selected processed inputs, final manuscript, result summaries, and the authoritative London BES reproducibility package.

> **Release status:** Complete private pre-release staging package. The final UCI corrected V2 and MORED corrected V3 prediction archives, run configurations, integrity audits and statistical outputs are now included in `supplementary/FINAL_UCI_MORED_V3_PREDICTIONS_AND_AUDITS.zip`. Public release is still blocked pending dataset-rights confirmation, exact environment capture, licence selection and co-author approval.

## Study scope

Three distinct experimental components are reported. They are **not** a uniform three-dataset benchmark:

1. **UCI Household Power Consumption:** corrected chronological 70/15/15 split, one-minute resolution, no imputation, training-only scaling; 903 train, 189 validation, and 192 test windows.
2. **MORED Premises 6:** corrected V3 whole-original-session split; raw five-second readings aggregated to minute means only when at least 10 genuine valid readings are present; no interpolation; training-only scaling; 245 train, 119 validation and 359 test windows. The 359 windows are distributed over four original recording sessions (IDs 6–9), with seven within-session continuity runs that contribute forecast windows.
3. **London Smart Meter:** a separate corrected DAT boundary-end-effect mitigation (BES) ablation on household MAC000147 only; 336 half-hourly inputs and 48 half-hourly outputs (24-hour forecast); 117 test origins. This is a single-household case study, not a London-wide benchmark.

All experiments use load history only; no weather or other exogenous covariates are included. UCI and MORED results are reconstructions under corrected protocols, not exact reproductions of unrecoverable older configurations.

## Corrected results

Trainable models are mean ± sample standard deviation across five seeds. TimesFM 2.5 is one zero-shot evaluation and has no across-seed standard deviation.

### UCI (RMSE and MAE in kW)

| Model | RMSE | MAE | MAPE (%) | R² | StdR |
|---|---:|---:|---:|---:|---:|
| Vanilla Transformer | 0.782735 ± 0.009840 | 0.567287 ± 0.022962 | 107.809627 ± 14.785082 | 0.173908 ± 0.020779 | 0.541370 ± 0.036280 |
| LSTM | 0.820548 ± 0.004493 | 0.632089 ± 0.006222 | 135.306918 ± 3.781718 | 0.092260 ± 0.009930 | 0.425470 ± 0.052066 |
| DLinear | 0.849813 ± 0.011597 | 0.610658 ± 0.012286 | 105.389372 ± 4.979169 | 0.026233 ± 0.026542 | 0.681856 ± 0.025599 |
| PatchTST-style | 0.859433 ± 0.004283 | 0.615488 ± 0.001901 | 105.965146 ± 1.050743 | 0.004192 ± 0.009906 | 0.471566 ± 0.006315 |
| TimesFM 2.5 (zero-shot) | 0.890434 | 0.587980 | 79.179894 | −0.068923 | 0.514655 |
| DAT | 0.994470 ± 0.033898 | 0.761134 ± 0.028644 | 158.953 ± 8.718 | −0.334534 ± 0.091419 | 0.819308 ± 0.063380 |

The one-token univariate iTransformer experiment was excluded from the principal ranking because cross-variable attention cannot be meaningfully exercised with a single input variate.

### MORED V3 (RMSE and MAE in W)

| Model | RMSE | MAE | MAPE (%) | R² | StdR |
|---|---:|---:|---:|---:|---:|
| Vanilla Transformer | 126.478 ± 0.185 | 86.924 ± 0.075 | 139.805 ± 1.764 | −0.0192 ± 0.0030 | 0.0978 ± 0.0076 |
| LSTM | 126.118 ± 0.207 | 85.970 ± 0.107 | 137.440 ± 1.205 | −0.0134 ± 0.0033 | 0.0178 ± 0.0180 |
| DAT | 128.751 ± 0.390 | 89.323 ± 0.409 | 134.209 ± 0.700 | −0.0561 ± 0.0064 | 0.1778 ± 0.0180 |
| TimesFM 2.5 (zero-shot) | 130.868 | 90.388 | 145.327 | −0.0911 | 0.3873 |
| DLinear | 146.011 ± 0.328 | 103.382 ± 0.289 | 149.267 ± 1.853 | −0.3583 ± 0.0061 | 0.6202 ± 0.0052 |
| PatchTST-style | 146.976 ± 1.417 | 105.164 ± 1.261 | 113.900 ± 0.706 | −0.3764 ± 0.0264 | 0.3531 ± 0.0136 |

LSTM has the lowest mean RMSE and MAE, narrowly ahead of the Vanilla Transformer; its very low StdR indicates severe under-dispersion. All six models have negative R². DAT does not improve RMSE or MAE over the Vanilla Transformer on this MORED V3 split.

### London BES ablation (kWh per half-hour reading)

| Configuration | RMSE | MAE | R² | StdR |
|---|---:|---:|---:|---:|
| DAT, BES off | 0.164207 ± 0.002561 | 0.108367 ± 0.002577 | −0.539934 ± 0.047940 | 0.559785 ± 0.027437 |
| DAT, BES on | 0.164217 ± 0.002546 | 0.108376 ± 0.002560 | −0.540119 ± 0.047667 | 0.559913 ± 0.027340 |

The difference is negligible and does not demonstrate a practical accuracy benefit from BES in this one-household case study.

## Statistical analysis

- **UCI:** primary seven-day hierarchical moving-block bootstrap with 5,000 replicates; 215 candidate test origins, 192 accepted complete windows, 23 rejected. Seed variability is resampled for trained models; TimesFM is a single fixed zero-shot run. All alternatives had significantly higher RMSE than the Vanilla Transformer after Holm correction (adjusted p = 0.001). For MAE, TimesFM was not significant (Holm p = 0.298). Three-day and fourteen-day sensitivity analyses were also run. These are the implemented moving-block-bootstrap results; do not relabel them as DM/MCS tests.
- **MORED V3:** errors are aggregated by the four original raw recording-session clusters (IDs 6–9); overlapping 24-hour forecast windows are not treated as independent. Directional differences are consistent across four clusters, but the exact two-sided sign-test p-value is 0.125 for every pairwise comparison, and Holm-adjusted p = 1.000. Bootstrap intervals are exploratory effect summaries only. Four clusters are insufficient for strong inferential claims.
- **London:** no inferential paired test is claimed for the BES ablation. Five seeds are reported descriptively.

## Repository layout

The exact project-relative paths used by the scripts are retained. Run commands from the repository root.

- `prepare_uci.py`, `validate_uci_release.py`, `uci_window_loader.py`, `train_uci_*.py`, `audit_uci_predictions.py`, `statistical_analysis_uci_v2.py`
- `prepare_mored_sessionsplit.py`, `validate_mored_sessionsplit.py`, `train_mored_*_sessionsplit.py`, `audit_mored_predictions_sessionsplit.py`, `statistical_analysis_mored_sessionsplit.py`
- `run_bes_ablation.py`, `run_london_dat_corrected_fullbes.py`
- `Data/`: selected processed UCI and MORED V3 split data; London split files
- `Cache/`: final London BES feature cache
- `Results_DAT_corrected_fullbes/`: London predictions, per-seed metrics and metadata
- `Results/`: current metric summaries, detailed UCI/MORED V3 statistical tables and per-seed metrics
- `supplementary/FINAL_UCI_MORED_V3_PREDICTIONS_AND_AUDITS.zip`: full final UCI and MORED V3 predictions, timestamps, targets, configs, audit and statistical outputs
- `figures/`, `manuscript/`, `docs/`

## Reproduction workflow

Create a Python environment compatible with your hardware and install dependencies from `requirements.in` (then record the exact environment with `python -m pip freeze`). The experiments were run on Apple Silicon using PyTorch MPS; CPU/GPU availability and package compatibility can change runtime and numerical details.

### UCI

The processed corrected windows are included under `Data/UCI/processed/`. Full final predictions, metrics, run configs and audit outputs are in the supplementary results ZIP. Validate the packaged split:

```bash
python validate_uci_release.py
```

Smoke-test and trainable model runners generally use a no-flag smoke test and `--train` for five-seed training. See each script's `--help` output. TimesFM is zero-shot and uses its script-specific run flag. The final corrected UCI prediction files and audit outputs are included in the supplementary results ZIP. Re-run the audit after extracting the relevant `Results/UCI/` outputs into the project root, or regenerate them by running the documented model commands.

To regenerate the UCI processed split from raw data, download the official source file from the UCI Machine Learning Repository and place it at `Data/UCI/raw/household_power_consumption.txt`, then run:

```bash
python prepare_uci.py
```

### MORED V3

Obtain the official Premises 6 source (`APT6.csv`) from the MORED dataset site and place it at `Data/MORED/raw/APT6.csv`. Then run:

```bash
python prepare_mored_sessionsplit.py
python validate_mored_sessionsplit.py
python train_mored_vanilla_dat_sessionsplit.py --prepare-features
python train_mored_vanilla_dat_sessionsplit.py --train
python train_mored_lstm_sessionsplit.py --train
python train_mored_dlinear_sessionsplit.py --train
python train_mored_patchtst_sessionsplit.py --train
python train_mored_timesfm25_sessionsplit.py
python audit_mored_predictions_sessionsplit.py
python statistical_analysis_mored_sessionsplit.py
```

### London

The final corrected FullBES experiment package is included. It expects `Data/London/splits/{train,val,test}.csv`, `Data/London/splits/scaler.pkl`, and `run_bes_ablation.py` in the project root. Smoke-test and reproduce from the repository root:

```bash
python run_london_dat_corrected_fullbes.py --smoke-test
python run_london_dat_corrected_fullbes.py
```

The included final predictions and feature cache are preserved for audit; a full training run re-trains the five seeds for BES OFF and BES ON.

## Dataset sources and citations

- UCI Individual Household Electric Power Consumption: Hebrail, G. & Berard, A. (2006), UCI Machine Learning Repository, DOI `10.24432/C58K54`. UCI lists the dataset under CC BY 4.0; attribution is required.
- MORED: Ahajjam, M. A., Bonilla Licea, D., Essayeh, C., Ghogho, M., & Kobbane, A. (2020). “MORED: A Moroccan Buildings’ Electricity Consumption Dataset.” *Energies*, 13(24), 6737. DOI `10.3390/en13246737`. The official dataset site provides download links; this repository does not include the original raw APT6.csv file. Verify the applicable dataset terms before publicly redistributing derived MORED data.
- London Smart Meter: Low Carbon London data published by UK Power Networks through the London Datastore. The London Datastore record lists a Creative Commons Attribution licence. Confirm that the split files included here originate from this open UKPN/GLA release; do not redistribute UK Data Service Study 7857 data under its separate End User Licence.

## Important limitations

- This is a load-only comparison; no climate-informed baseline is included.
- UCI and MORED are corrected within-dataset reconstructions; the original earlier run configurations were not fully recoverable.
- MORED V3 uses only four independent original test sessions. Its exact tests are underpowered; no statistical superiority claim is made.
- London evaluates one household only and has no matched London model benchmark.
- All datasets/models have important under-dispersion or negative R² results; report these, not just RMSE.
- MAPE is sensitive to low/near-zero actual demand and is not the primary ranking metric.
- Custom PatchTST-style models are custom PyTorch implementations, not claimed to be exact upstream NeuralForecast PatchTST runs.
- TimesFM 2.5 is zero-shot, not fine-tuned.

## Citation and release

A complete paper citation and DOI will be added after the authors approve the final manuscript and archive. See `CITATION.cff`, `LICENSE_SELECTION_REQUIRED.md`, `DATA_RIGHTS_REVIEW.md`, and `PUBLIC_RELEASE_CHECKLIST.md` before making the repository public.
