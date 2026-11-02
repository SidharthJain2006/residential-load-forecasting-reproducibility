# Does Causal Decomposition Improve Residential Load Forecasting?

### A Leakage-Aware, Multi-Dataset Reproducibility Study of a Decomposition-Augmented Transformer

This repository contains the research code, experimental configurations, preprocessing and evaluation scripts, audit outputs, and experimental results supporting the study.

**Repository status:** Public research repository.

The study investigates whether a Decomposition-Augmented Transformer (DAT) improves residential electricity load forecasting under leakage-aware evaluation protocols. It evaluates three distinct experimental components across UCI, MORED Premises 6, and one London SmartMeter household.

The experiments do not constitute a uniform three-dataset model-ranking benchmark.

---

## 1. Study Overview

The study evaluates the Decomposition-Augmented Transformer (DAT) against selected forecasting baselines, with emphasis on:

- Chronological data partitioning.
- Leakage-aware preprocessing and scaling.
- Window-based multi-horizon forecasting.
- Comparison with established forecasting architectures.
- Statistical evaluation that accounts for temporal dependence.
- Reproducibility and auditability of experimental outputs.

No measured external covariates, such as weather variables, are used.

The DAT implementations use deterministic hour-of-day and weekday embeddings derived from input-window timestamps. These calendar embeddings are not supplied to every baseline. Consequently, the comparisons evaluate the complete implemented configurations and do not isolate the effect of decomposition alone.

UCI and MORED are corrected reconstructions under the documented protocols, rather than exact reproductions of earlier configurations that could not be fully recovered.

---

## 2. Experimental Datasets

The manuscript reports three separate experimental components.

### 2.1 UCI Individual Household Electric Power Consumption

Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

- One French household.
- One-minute electricity consumption readings.
- Coverage: December 2006 to November 2010.
- Chronological split: 70% training, 15% validation, and 15% testing.
- No imputation is performed.
- Scaling is fitted using training data only.
- Only complete 48-hour windows are retained.

| Partition | Complete windows |
|---|---:|
| Training | 903 |
| Validation | 189 |
| Testing | 192 |

### 2.2 MORED Premises 6

Dataset: [MORED official website](https://moredataset.github.io/MORED/)

The experiment uses the whole-premises `APT6.csv` source for Premises 6.

- Original readings: five-second sampling.
- Valid readings are aggregated into minute means only when at least 10 genuine valid readings are available.
- Negative readings are invalidated.
- No interpolation, forward filling, or backward filling is performed.
- Scaling is fitted on training data only.

The corrected V3 split assigns complete original recording sessions to partitions:

| Partition | Recording sessions | Windows |
|---|---|---:|
| Training | 0–4 | 245 |
| Validation | 5 | 119 |
| Testing | 6–9 | 359 |

The test windows belong to four original recording-session clusters and seven within-session continuity runs.

Additional split documentation: [MORED V3 Session Split](README_MORED_V3_SessionSplit.md).

### 2.3 London SmartMeter

Dataset: [London Datastore – SmartMeter Energy Consumption Data in London Households](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d)

This is a separate corrected DAT Building Energy Signature (BES) ablation using household `MAC000147` only.

- Input: 336 half-hourly readings (seven days).
- Forecast horizon: 48 half-hourly readings (24 hours).
- Test forecast origins: 117.
- Five random seeds.
- BES ON/OFF configurations are compared descriptively.

This is a single-household case study, not a London-wide benchmark or a cross-household generalisation experiment.

---

## 3. Models Evaluated

The reported comparisons include:

- Decomposition-Augmented Transformer (DAT).
- Vanilla Transformer.
- Long Short-Term Memory (LSTM).
- DLinear.
- PatchTST-style forecasting baseline.
- TimesFM 2.5 zero-shot forecasting.

The one-token univariate iTransformer run is excluded from the principal ranking because cross-variable attention cannot be meaningfully exercised with a single input variate.

The model comparisons are dataset-specific. Not every model or experimental configuration is evaluated identically across all three experimental components.

---

## 4. Corrected Experimental Results

Trainable-model values are reported as mean ± sample standard deviation across five random seeds.

These standard deviations represent seed-to-seed variation, not confidence intervals. TimesFM 2.5 is a single zero-shot evaluation and therefore has no across-seed standard deviation.

`StdR` is the standard deviation of predictions divided by the standard deviation of observed targets.

MAPE excludes exact-zero actual values but remains sensitive to very small actual loads.

### 4.1 UCI Results

RMSE and MAE are reported in kW.

| Model | RMSE | MAE |
|---|---:|---:|
| Vanilla Transformer | 0.782735 ± 0.009840 | 0.567287 ± 0.022962 |
| LSTM | 0.820548 ± 0.004493 | 0.632089 ± 0.006222 |
| DLinear | 0.849813 ± 0.011597 | 0.610658 ± 0.012286 |
| PatchTST-style | 0.859433 ± 0.004283 | 0.615488 ± 0.001901 |
| TimesFM 2.5 (zero-shot) | 0.890434 | 0.587980 |
| DAT | 0.994470 ± 0.033898 | 0.761134 ± 0.028644 |

The Vanilla Transformer achieves the lowest mean RMSE and MAE on the corrected UCI test set. DAT does not improve either metric over the Vanilla Transformer.

### 4.2 MORED V3 Results

RMSE and MAE are reported in W.

| Model | RMSE | MAE |
|---|---:|---:|
| Vanilla Transformer | 126.478 ± 0.185 | 86.924 ± 0.075 |
| LSTM | 126.118 ± 0.207 | 85.970 ± 0.107 |
| DAT | 128.751 ± 0.390 | 89.323 ± 0.409 |
| TimesFM 2.5 (zero-shot) | 130.868 | 90.388 |
| DLinear | 146.011 ± 0.328 | 103.382 ± 0.289 |
| PatchTST-style | 146.976 ± 1.417 | 105.164 ± 1.261 |

LSTM has the lowest mean RMSE and MAE numerically, narrowly ahead of the Vanilla Transformer. Its very low `StdR` indicates severe prediction-variance collapse.

All six models have negative R². DAT does not improve RMSE or MAE over the Vanilla Transformer on this MORED V3 split.

### 4.3 London BES Ablation

RMSE and MAE are reported in kWh per half-hour reading.

| Configuration | RMSE | MAE |
|---|---:|---:|
| DAT, BES off | 0.164207 ± 0.002561 | 0.108367 ± 0.002577 |
| DAT, BES on | 0.164217 ± 0.002546 | 0.108376 ± 0.002560 |

BES ON combines left-edge reflection, a historical-similarity right extension, and weighted overlap fusion. BES OFF uses none of these boundary components.

The difference is negligible. BES ON has a very slightly higher RMSE (approximately 0.006%). This one-household ablation does not demonstrate a practical accuracy benefit from BES.

---

## 5. Statistical Analysis

### UCI

- Seven-day hierarchical moving-block bootstrap.
- 5,000 replicates per block length.
- The test manifest contains 215 candidate origins: 192 complete windows accepted and 23 rejected.
- Seed variability is resampled for trained models.
- TimesFM is treated as one fixed zero-shot run.
- All five alternatives have higher RMSE than the Vanilla Transformer after Holm correction (adjusted p = 0.001).
- For MAE, trained alternatives are significant after Holm correction; TimesFM is not (Holm p = 0.298).
- Three-day and fourteen-day block-length sensitivity analyses were also performed.

These are moving-block-bootstrap results, not Diebold–Mariano or Model Confidence Set tests.

### MORED V3

- Errors are aggregated by the four original test recording-session clusters (IDs 6–9).
- Overlapping 24-hour windows are not treated as independent observations.
- Exact two-sided sign-test p-value: 0.125 for each pairwise comparison.
- Holm-adjusted p-value: 1.000.
- Bootstrap intervals are exploratory effect summaries.

Four independent test sessions are insufficient for strong inferential claims.

### London

The five seeds are reported descriptively. No inferential paired test is claimed for the BES ablation.

---

## 6. Repository Structure

The principal scripts and directories are:

### UCI

- `prepare_uci.py`
- `validate_uci_release.py`
- `uci_window_loader.py`
- `train_uci_*.py`
- `audit_uci_predictions.py`
- `statistical_analysis_uci_v2.py`

### MORED V3

- `prepare_mored_sessionsplit.py`
- `validate_mored_sessionsplit.py`
- `train_mored_*_sessionsplit.py`
- `audit_mored_predictions_sessionsplit.py`
- `statistical_analysis_mored_sessionsplit.py`

### London BES

- `run_london_dat_corrected_fullbes.py`
- `run_bes_ablation.py`

### Data, results, and documentation

| Path | Description |
|---|---|
| `Cache/London_DAT_corrected_fullbes/` | Six cached London BES feature arrays (train/validation/test; BES ON/OFF) |
| `Results/UCI/` | Aggregate UCI summaries, audited metrics, and bootstrap results |
| `Results/MORED/` | Aggregate MORED V3 summaries, per-seed metrics, and session-level analyses |
| `Results_DAT_corrected_fullbes/` | London per-seed metrics, aggregate summaries, and run metadata |
| `figures/` | Research figures currently included in the repository |
| `manuscript/` | Manuscript DOCX included with the repository |
| `docs/` | Experiment protocol, validation, artifact-completeness, and statistical documentation |

Additional repository documentation:

- `ENVIRONMENT_CAPTURE.md`
- `PROVENANCE.md`
- `DATA_RIGHTS_REVIEW.md`
- `LICENSE_SELECTION_REQUIRED.md`
- `PUBLIC_RELEASE_CHECKLIST.md`
- `README_MORED_V3_SessionSplit.md`

The `Data/` and `supplementary/` directories are **not included in the current public repository tree**. Users must obtain the original datasets and create the required local data files by running the documented preprocessing workflows.

The London experiment additionally requires household split CSVs and a scaler described in `PROVENANCE.md`. These files are not currently committed. The London feature caches and aggregate result files listed above are committed.

Trained model checkpoint files and the complete per-timestamp prediction/actual-target archive are not included in the current public repository. The committed materials include source scripts, aggregate result tables, per-seed metrics, statistical summaries, audit scripts, London feature caches, run metadata, and checksum/provenance documentation. Recreating processed inputs and model weights requires obtaining the original datasets and running the documented preparation and training workflows.

---

## 7. Environment Setup

The reported experiments were run using:

- Python 3.12.4
- Apple Silicon / macOS 26.5
- PyTorch 2.15.0
- Apple MPS backend

The captured package versions are documented in `requirements-macos-mps-lock.txt`.

For the closest match to the recorded environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-macos-mps-lock.txt
```

See `ENVIRONMENT_CAPTURE.md` for environment verification and platform notes.

The lock file is specific to the recorded macOS/Apple Silicon environment. A different operating system, accelerator, or package build may change runtime and numerical details.

`requirements.in` contains direct dependencies and is not a replacement for the captured lock file.

---

## 8. Reproduction Workflow

Run scripts from the repository root.

The original raw UCI, MORED, and full London source archives are not redistributed as original datasets. Obtain original data from the respective dataset sources when regenerating processed data.

### 8.1 UCI

The `Data/UCI/` directory is created locally; it is not included in the current public repository. Obtain the original UCI dataset and run the preparation script before validation or training.

Validate the locally generated release:

```bash
python validate_uci_release.py
```

To regenerate the processed split:

1. Download `household_power_consumption.txt` from the [UCI dataset record](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption).
2. Place it at `Data/UCI/raw/household_power_consumption.txt`.
3. Run:

```bash
python prepare_uci.py
python validate_uci_release.py
```

To rerun the experiments:

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

The complete supplementary prediction and audit archive is not included in the current public repository. Aggregate metrics and statistical outputs are available under `Results/UCI/`.

### 8.2 MORED V3

The original `APT6.csv` source is available through the [official MORED dataset page](https://moredataset.github.io/MORED/).

The Premises 6 WP-data download is listed under:

**Semi-Detached House in an affluent neighbourhood → Premises 6**

The [Premises 6 WP-data download folder](https://drive.google.com/drive/folders/1KSk8HO0wgxMvQxkfyY1Pf7NlRbnUSyvb?usp=sharing) is linked by the [official MORED repository](https://github.com/MOREDataset/MORED).

Place the downloaded file at:

```text
Data/MORED/raw/APT6.csv
```

The source file used in this study contains 1,346,393 five-second readings from 5 May to 26 July 2020.

Run preprocessing and validation:

```bash
python prepare_mored_sessionsplit.py
python validate_mored_sessionsplit.py
```

Run the Vanilla Transformer/DAT smoke test:

```bash
python train_mored_vanilla_dat_sessionsplit.py
```

Prepare features and train:

```bash
python train_mored_vanilla_dat_sessionsplit.py --prepare-features
python train_mored_vanilla_dat_sessionsplit.py --train
```

Run the other models and evaluation:

```bash
python train_mored_lstm_sessionsplit.py --train
python train_mored_dlinear_sessionsplit.py --train
python train_mored_patchtst_sessionsplit.py --train
python train_mored_timesfm25_sessionsplit.py

python audit_mored_predictions_sessionsplit.py
python statistical_analysis_mored_sessionsplit.py
```

**Data rights note:** Consult `DATA_RIGHTS_REVIEW.md` before redistributing MORED-derived measurement-level data or prediction archives. The official dataset page describes MORED as an open dataset, but the repository does not assert a specific licence granting redistribution of raw or derived measurement values.

### 8.3 London BES Ablation

The experiment uses household `MAC000147` from the UK Power Networks Low Carbon London dataset listed in the [London Datastore catalogue](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d).

The original full London archive is not bundled.

The repository includes London BES feature caches, per-seed metrics, aggregate summaries, run metadata, and helper code. It does not currently include the household split CSVs or scaler required to regenerate those caches. `PROVENANCE.md` records that these files came from an earlier archive; obtain and place the permitted source-derived split files locally before attempting a fresh run.

After obtaining and placing all four required local split files and the scaler listed above, run the smoke test:

```bash
python run_london_dat_corrected_fullbes.py --smoke-test
```

Run the full five-seed BES OFF/ON experiment:

```bash
python run_london_dat_corrected_fullbes.py
```

The script imports `run_bes_ablation.py` and expects the following local files, which are not included in the current public repository:

```text
Data/London/splits/train.csv
Data/London/splits/val.csv
Data/London/splits/test.csv
Data/London/splits/scaler.pkl
```

A smoke test that requires these inputs cannot be considered a clean-clone test until the split files and scaler are supplied locally.

---

## 9. Dataset Sources and Citations

### UCI Individual Household Electric Power Consumption

Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption* [Dataset]. UCI Machine Learning Repository.

- [Dataset record](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- DOI: [10.24432/C58K54](https://doi.org/10.24432/C58K54)

UCI identifies the dataset licence as CC BY 4.0. Provide the required attribution.

### MORED

Ahajjam, M. A., Bonilla Licea, D., Essayeh, C., Ghogho, M., & Kobbane, A. (2020). "MORED: A Moroccan Buildings’ Electricity Consumption Dataset." *Energies*, 13(24), 6737.

- [Official dataset page](https://moredataset.github.io/MORED/)
- [Official repository](https://github.com/MOREDataset/MORED)
- DOI: [10.3390/en13246737](https://doi.org/10.3390/en13246737)

The study uses the Premises 6 whole-premises (`APT6.csv`) source. Consult the dataset's applicable terms before redistributing its raw or derived measurements.

### London SmartMeter Energy Consumption Data

UK Power Networks, Low Carbon London project, published through the Greater London Authority's London Datastore.

- [Dataset record](https://data.london.gov.uk/dataset/smartmeter-energy-consumption-data-in-london-households-vqm0d)

The record lists half-hourly kWh readings and November 2011–February 2014 coverage. Follow the dataset's attribution and licence conditions.

---

## 10. Important Limitations

- UCI and MORED are corrected reconstructions because the complete configurations of earlier experimental runs were not recoverable.
- MORED statistical analysis is limited by four independent test recording-session clusters. Five model seeds measure optimisation variability, not additional independent data sessions.
- The London experiment evaluates household `MAC000147` only and does not support conclusions about the wider household population.
- DAT uses deterministic hour-of-day and weekday embeddings that are not included in every baseline. Comparisons therefore evaluate complete model configurations and do not isolate the contribution of decomposition alone.
- No weather-informed or other exogenous-variable forecasting baseline is evaluated.
- DAT does not demonstrate accuracy superiority over the Vanilla Transformer on UCI or MORED.
- The London BES ablation does not demonstrate a practical accuracy improvement.
- Negative R² and under-dispersed forecasts are retained and reported rather than hidden.

---

## 11. Code Availability

The source code, model implementations, experimental configurations, preprocessing and evaluation scripts, and aggregate experimental results supporting this study are publicly available at:

[https://github.com/SidharthJain2006/residential-load-forecasting-reproducibility](https://github.com/SidharthJain2006/residential-load-forecasting-reproducibility)

The original datasets are not redistributed in the repository and should be obtained from their respective sources, as cited in this manuscript.

---

## 12. Citation

The associated manuscript is:

**Does Causal Decomposition Improve Residential Load Forecasting? A Leakage-Aware, Multi-Dataset Reproducibility Study of a Decomposition-Augmented Transformer.**

A publication DOI has not yet been assigned. Please check the repository's `CITATION.cff` for citation metadata and update it when the manuscript is published.

---

## 13. Licence

No software licence should be inferred from the public visibility of this repository.

See `LICENSE_SELECTION_REQUIRED.md` and obtain the necessary author and contributor agreement before selecting and adding a software licence.

