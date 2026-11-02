# Residential Load Forecasting Reproducibility Study

**Does Causal Decomposition Improve Residential Load Forecasting? A Leakage-Aware, Multi-Dataset Reproducibility Study of a Decomposition-Augmented Transformer**

This repository contains the corrected experiment scripts, selected processed inputs, final manuscript, result summaries, audit outputs, and the final London Boundary End-effect mitigation (BES) reproducibility package. It is a **private pre-release research archive**, not yet a public software/data release.

> **Release status:** The corrected UCI V2 and MORED V3 prediction/audit archive is included at `supplementary/FINAL_UCI_MORED_V3_PREDICTIONS_AND_AUDITS.zip`. The exact experiment environment is also captured in `ENVIRONMENT_CAPTURE.md` and `requirements-macos-mps-lock.txt`. Public release remains on hold for confirmation of MORED-derived-data redistribution terms, code-licence selection, and co-author approval. The MORED processed data and actual-target arrays in this private repository must not be made public until those terms are confirmed.

## Study scope

The repository reports three distinct experimental components. They do **not** form a uniform three-dataset model-ranking benchmark.

1. **UCI Individual Household Electric Power Consumption.** One French household, one-minute readings, December 2006–November 2010. The corrected chronological split is 70/15/15, with no imputation and scaling fitted on observed training data only. Only complete 48-hour windows are retained: 903 training, 189 validation, and 192 test windows.
2. **MORED Premises 6 (whole-premises data).** The source is the MORED `APT6.csv` file for Premises 6. The raw five-second readings are aggregated to minute means only when at least 10 genuine valid readings are available. Negative readings are invalidated; no interpolation, forward fill, or backward fill is used. The corrected V3 split assigns whole original recording sessions to partitions: sessions 0–4 for training, session 5 for validation, and sessions 6–9 for testing. Scaling is fitted on training data only. The split contains 245 training, 119 validation, and 359 test windows. The 359 test windows belong to four original recording-session clusters and seven within-session continuity runs.
3. **London SmartMeter Energy Consumption Data in London Households.** A separate corrected DAT BES ablation is run on household `MAC000147` only. Each example uses 336 half-hourly readings (seven days) to predict 48 half-hourly readings (24 hours); there are 117 test forecast origins. This is a one-household case study, not a London-wide benchmark.

No measured external covariates such as weather are used. However, the DAT implementations also use deterministic hour-of-day and weekday embeddings derived from the input-window timestamps (UCI, MORED, and London). The baseline models do not all receive these calendar embeddings. Therefore, the UCI and MORED comparisons evaluate the complete implemented architectures; they do not isolate the effect of decomposition alone. UCI and MORED are corrected reconstructions under the documented protocols, not exact reproductions of earlier configurations that could not be fully recovered.

## Corrected results

Trainable-model values are means ± sample standard deviations across five random seeds. These standard deviations describe seed-to-seed variation; they are not confidence intervals. TimesFM 2.5 is one zero-shot evaluation and therefore has no across-seed standard deviation. `StdR` is the standard deviation of predictions divided by the standard deviation of the observed targets. MAPE excludes exact-zero actual values but remains sensitive to very small actual loads.

### UCI (RMSE and MAE in kW)

| Model | RMSE | MAE | MAPE (%) | R² | StdR |
|---|---:|---:|---:|---:|---:|
| Vanilla Transformer | 0.782735 ± 0.009840 | 0.567287 ± 0.022962 | 107.809627 ± 14.785082 | 0.173908 ± 0.020779 | 0.541370 ± 0.036280 |
| LSTM | 0.820548 ± 0.004493 | 0.632089 ± 0.006222 | 135.306918 ± 3.781718 | 0.092260 ± 0.009930 | 0.425470 ± 0.052066 |
| DLinear | 0.849813 ± 0.011597 | 0.610658 ± 0.012286 | 105.389372 ± 4.979169 | 0.026233 ± 0.026542 | 0.681856 ± 0.025599 |
| PatchTST-style | 0.859433 ± 0.004283 | 0.615488 ± 0.001901 | 105.965146 ± 1.050743 | 0.004192 ± 0.009906 | 0.471566 ± 0.006315 |
| TimesFM 2.5 (zero-shot) | 0.890434 | 0.587980 | 79.179894 | −0.068923 | 0.514655 |
| DAT | 0.994470 ± 0.033898 | 0.761134 ± 0.028644 | 158.953 ± 8.718 | −0.334534 ± 0.091419 | 0.819308 ± 0.063380 |

The Vanilla Transformer has the lowest mean RMSE and MAE on this corrected UCI test set. DAT does not improve either metric over the Vanilla Transformer. The one-token univariate iTransformer run is excluded from the principal ranking because cross-variable attention cannot be meaningfully exercised with a single input variate.

### MORED V3 (RMSE and MAE in W)

| Model | RMSE | MAE | MAPE (%) | R² | StdR |
|---|---:|---:|---:|---:|---:|
| Vanilla Transformer | 126.478 ± 0.185 | 86.924 ± 0.075 | 139.805 ± 1.764 | −0.0192 ± 0.0030 | 0.0978 ± 0.0076 |
| LSTM | 126.118 ± 0.207 | 85.970 ± 0.107 | 137.440 ± 1.205 | −0.0134 ± 0.0033 | 0.0178 ± 0.0180 |
| DAT | 128.751 ± 0.390 | 89.323 ± 0.409 | 134.209 ± 0.700 | −0.0561 ± 0.0064 | 0.1778 ± 0.0180 |
| TimesFM 2.5 (zero-shot) | 130.868 | 90.388 | 145.327 | −0.0911 | 0.3873 |
| DLinear | 146.011 ± 0.328 | 103.382 ± 0.289 | 149.267 ± 1.853 | −0.3583 ± 0.0061 | 0.6202 ± 0.0052 |
| PatchTST-style | 146.976 ± 1.417 | 105.164 ± 1.261 | 113.900 ± 0.706 | −0.3764 ± 0.0264 | 0.3531 ± 0.0136 |

LSTM has the lowest mean RMSE and MAE numerically, narrowly ahead of the Vanilla Transformer. Its very low `StdR` indicates severe prediction-variance collapse. All six models have negative R². DAT does not improve RMSE or MAE over the Vanilla Transformer on this MORED V3 split.

### London BES ablation (RMSE and MAE in kWh per half-hour reading)

| Configuration | RMSE | MAE | R² | StdR |
|---|---:|---:|---:|---:|
| DAT, BES off | 0.164207 ± 0.002561 | 0.108367 ± 0.002577 | −0.539934 ± 0.047940 | 0.559785 ± 0.027437 |
| DAT, BES on | 0.164217 ± 0.002546 | 0.108376 ± 0.002560 | −0.540119 ± 0.047667 | 0.559913 ± 0.027340 |

BES ON combines left-edge reflection, a historical-similarity right extension, and weighted overlap fusion. BES OFF uses none of these boundary components. The difference is negligible: BES ON has a very slightly higher RMSE (about 0.006%). This one-household ablation does not demonstrate a practical accuracy benefit from BES.

## Statistical analysis

- **UCI:** the primary analysis is a seven-day hierarchical moving-block bootstrap with 5,000 replicates per block length. The test manifest has 215 candidate origins: 192 complete windows were accepted and 23 were rejected. Seed variability is resampled for trained models; TimesFM is treated as one fixed zero-shot run. All five alternatives have higher RMSE than the Vanilla Transformer after Holm correction (adjusted p = 0.001). For MAE, the trained alternatives are significant after Holm correction; TimesFM is not (Holm p = 0.298). Three-day and fourteen-day block-length sensitivity analyses were also run. These are moving-block-bootstrap results, not Diebold–Mariano or Model Confidence Set tests.
- **MORED V3:** errors are aggregated by the four original test recording-session clusters (IDs 6–9); overlapping 24-hour windows are not treated as independent observations. The exact two-sided sign-test p-value is 0.125 for each pairwise comparison, with Holm-adjusted p = 1.000. Bootstrap intervals are exploratory effect summaries. Four independent test sessions are insufficient for strong inferential claims.
- **London:** the five seeds are reported descriptively. No inferential paired test is claimed for the BES ablation.

## Repository layout

Run scripts from the repository root. The principal files and folders are:

- UCI: `prepare_uci.py`, `validate_uci_release.py`, `uci_window_loader.py`, `train_uci_*.py`, `audit_uci_predictions.py`, and `statistical_analysis_uci_v2.py`.
- MORED V3: `prepare_mored_sessionsplit.py`, `validate_mored_sessionsplit.py`, `train_mored_*_sessionsplit.py`, `audit_mored_predictions_sessionsplit.py`, and `statistical_analysis_mored_sessionsplit.py`.
- London BES: `run_london_dat_corrected_fullbes.py` and its helper module `run_bes_ablation.py`.
- `Data/UCI/processed/`: corrected no-imputation UCI windows, manifest, scaler, and preprocessing metadata.
- `Data/MORED/Corrected_V3_SessionSplit/`: corrected V3 processed data, windows, manifest, scaler, and metadata. These MORED-derived measurements are present for private reproducibility only; do not publish them until redistribution terms are confirmed.
- `Data/London/splits/`: the selected household's train, validation, and test CSVs plus scaler.
- `Cache/London_DAT_corrected_fullbes/`: final London BES feature caches.
- `Results/`: current UCI and MORED summaries, per-seed metrics, and statistical tables.
- `Results_DAT_corrected_fullbes/`: London predictions, per-seed metrics, summaries, and metadata.
- `supplementary/FINAL_UCI_MORED_V3_PREDICTIONS_AND_AUDITS.zip`: selected UCI V2 and MORED V3 predictions, timestamps, actual targets, run configurations, audits, and statistical outputs.
- `figures/`, `manuscript/`, and `docs/`: figure, revised manuscript, and protocol/validation/statistical documentation.
- `requirements.in`: direct dependencies; `requirements-macos-mps-lock.txt`: captured package versions for the experiment machine; `ENVIRONMENT_CAPTURE.md`: hardware and software environment details.
- `PROVENANCE.md`, `DATA_RIGHTS_REVIEW.md`, `LICENSE_SELECTION_REQUIRED.md`, and `PUBLIC_RELEASE_CHECKLIST.md`: archive history and release checks.

**Checkpoint note:** trained model checkpoint files are not included. The repository preserves prediction arrays, actual targets, configurations, metrics, audits, and logs needed to inspect the reported outputs. To recreate model weights, rerun the training scripts.

## Environment setup

The reported experiments were run with Python 3.12.4 on Apple Silicon/macOS 26.5, PyTorch 2.15.0, and the Apple MPS backend. The exact package freeze is included in `requirements-macos-mps-lock.txt`.

For the closest match to the recorded environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-macos-mps-lock.txt
```

See `ENVIRONMENT_CAPTURE.md` for verification and platform notes. The lock file is specific to the recorded macOS/Apple Silicon environment; a different operating system, accelerator, or package build may change runtime and numerical details. `requirements.in` is an unpinned direct-dependency file, not a replacement for the captured lock.

## Reproduction workflow

The processed UCI and MORED V3 inputs, London splits, and saved predictions are included in this private package. The original raw UCI and MORED source files and the full original London archive are **not** included. Download raw files from their official sources only when regenerating the corresponding processed data.

### UCI

The corrected processed windows are already included under `Data/UCI/processed/`. Validate them with:

```bash
python validate_uci_release.py
```

To regenerate the UCI processed split, download `household_power_consumption.txt` from the [UCI dataset record](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), place it at `Data/UCI/raw/household_power_consumption.txt`, and run:

```bash
python prepare_uci.py
python validate_uci_release.py
```

To rerun the five-seed models from the repository root:

```bash
python train_uci_vanilla_transformer_corrected_v2_1.py --train
python train_uci_dat_corrected.py --prepare-features
python train_uci_dat_corrected.py --train
python train_uci_lstm_corrected.py --train
python train_uci_dlinear_corrected.py --train
python train_uci_patchtst_corrected.py --train
python train_uci_timesfm25_corrected.py --run-eval
python audit_uci_predictions.py
python statistical_analysis_uci_v2.py --bootstrap-reps 5000
```

The final prediction files are also in the supplementary archive. To extract those saved outputs into the expected `Results/` folders before auditing them, run from the repository root:

```bash
unzip -o supplementary/FINAL_UCI_MORED_V3_PREDICTIONS_AND_AUDITS.zip -d .
python audit_uci_predictions.py
python statistical_analysis_uci_v2.py --bootstrap-reps 5000
```

### MORED V3

The original `APT6.csv` file is not bundled. Obtain the **WP data → Semi-Detached House in an affluent neighbourhood → Premises 6** source from the [official MORED dataset page](https://moredataset.github.io/MORED/), or use the [Premises 6 WP-data download folder](https://drive.google.com/drive/folders/1KSk8HO0wgxMvQxkfyY1Pf7NlRbnUSyvb?usp=sharing) linked by the [official repository README](https://github.com/MOREDataset/MORED). Place the downloaded `APT6.csv` at `Data/MORED/raw/APT6.csv`. The source file used in this study contains 1,346,393 five-second readings from 5 May to 26 July 2020.

Then run:

```bash
python prepare_mored_sessionsplit.py
python validate_mored_sessionsplit.py
python train_mored_vanilla_dat_sessionsplit.py
python train_mored_vanilla_dat_sessionsplit.py --prepare-features
python train_mored_vanilla_dat_sessionsplit.py --train
python train_mored_lstm_sessionsplit.py --train
python train_mored_dlinear_sessionsplit.py --train
python train_mored_patchtst_sessionsplit.py --train
python train_mored_timesfm25_sessionsplit.py
python audit_mored_predictions_sessionsplit.py
python statistical_analysis_mored_sessionsplit.py
```

The first no-flag Vanilla Transformer/DAT command is a smoke test. The five-seed training commands are separate. The MORED source page describes the dataset as open, but the official repository contains no standalone data-licence file that clearly grants redistribution of raw or derived measurement values. This private package therefore retains the processed V3 measurements and actual targets for audit, but **do not publish the MORED processed files or MORED-derived actual-target/prediction archive until the applicable terms are confirmed**. Publishing code and aggregate metrics separately does not require bundling the measurement-level data.

### London BES ablation

The corrected London split files, scaler, feature caches, final predictions, metadata, and helper code are included. The original full London archive (about 167 million readings) is not bundled. The case study uses the selected household `MAC000147` from the UK Power Networks Low Carbon London dataset listed in the [London Datastore catalogue](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d). The catalogue identifies UK Power Networks as the author, half-hourly energy in kWh per half-hour, the November 2011–February 2014 coverage, and a Creative Commons Attribution licence. The earlier study manuscript also identifies this London Datastore source. The raw source archive is not retained here; the household-level split files used in the final BES run are included.

From the repository root, first run a smoke test and then the full five-seed BES OFF/ON experiment:

```bash
python run_london_dat_corrected_fullbes.py --smoke-test
python run_london_dat_corrected_fullbes.py
```

The final predictions and feature cache are preserved for audit. The run imports `run_bes_ablation.py` as a helper and expects the supplied `Data/London/splits/{train,val,test}.csv` files and `scaler.pkl`.

## Dataset sources and citations

- **UCI Individual Household Electric Power Consumption:** Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption* [Dataset]. UCI Machine Learning Repository. [Dataset record](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), DOI [10.24432/C58K54](https://doi.org/10.24432/C58K54). UCI identifies the dataset licence as CC BY 4.0; provide attribution.
- **MORED:** Ahajjam, M. A., Bonilla Licea, D., Essayeh, C., Ghogho, M., & Kobbane, A. (2020). “MORED: A Moroccan Buildings’ Electricity Consumption Dataset.” *Energies*, 13(24), 6737. [Official dataset page](https://moredataset.github.io/MORED/), [official repository](https://github.com/MOREDataset/MORED), DOI [10.3390/en13246737](https://doi.org/10.3390/en13246737). The study uses the Premises 6 whole-premises (`APT6.csv`) source. The official site describes MORED as an open dataset and provides download links; however, this repository does not assert a specific data licence for redistribution of raw or derived measurement values.
- **London SmartMeter Energy Consumption Data in London Households:** UK Power Networks, Low Carbon London project, published via the Greater London Authority's London Datastore. [Dataset record](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d). The record lists Creative Commons Attribution, half-hourly kWh readings, and November 2011–February 2014 coverage. Attribute UK Power Networks and the London Datastore/Greater London Authority as required by the source terms.

## Important limitations

- ## Important limitations

- The UCI and MORED experiments are corrected reconstructions because the complete configurations of earlier experimental runs were not recoverable.
- MORED statistical analysis is limited by four independent test recording-session clusters. The five model seeds measure optimisation variability, not additional independent data sessions.
- The London experiment evaluates only household MAC000147 and does not support conclusions about the wider household population.
- DAT uses deterministic hour-of-day and weekday embeddings that are not included in every baseline. Therefore, the comparisons evaluate complete model configurations and do not isolate the contribution of decomposition alone.
- No weather-informed or other exogenous-variable forecasting baseline is evaluated.
  
## Citation and release

The associated article has not yet been assigned a publication DOI. Until the article and archive are approved, cite the dataset sources listed above and the repository's pre-release version in `CITATION.cff`. A code licence has intentionally not been selected; all authors and code contributors must agree before public release. See `DATA_RIGHTS_REVIEW.md`, `LICENSE_SELECTION_REQUIRED.md`, and `PUBLIC_RELEASE_CHECKLIST.md` before changing repository visibility.
