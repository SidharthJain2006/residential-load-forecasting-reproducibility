# MORED V3 whole-recording-session split

This is a new, separate MORED experiment. It does **not** overwrite or replace the existing `Corrected_V2` data/results. The raw source is the original `APT6.csv`.

## Session assignment

The original raw file contains 10 recording sessions, defined by a raw timestamp gap greater than 60 seconds. The V3 chronological allocation is:

- Training: raw sessions 0, 1, 2, 3, 4
- Validation: raw session 5
- Testing: raw sessions 6, 7, 8, 9

Each raw session is assigned wholly to one partition. A minute is retained only if at least 10 genuine valid five-second readings are present. There is no interpolation, forward fill, or backward fill. Windows do not cross a raw-session boundary or a missing retained minute. The MinMaxScaler is fitted only on retained training-session minute means.

Expected windows (24h input, 24h target, 60-minute stride):

| Split | Windows |
|---|---:|
| Train | 245 |
| Validation | 119 |
| Test | 359 |

The 359 test windows come from four raw recording-session clusters (session IDs 6-9), with 140, 157, 27, and 35 windows respectively. Those four sessions contain seven eligible continuous runs; all remain inside their original raw-session cluster for inference.

## Files

Copy all `.py` files in this kit into the root of your `DAT_Reproducible_V2` project folder. Ensure `APT6.csv` is at `Data/MORED/raw/APT6.csv`.

## Run order in the VS Code terminal

1. Prepare the new split:

   `python prepare_mored_sessionsplit.py`

2. Independently validate it:

   `python validate_mored_sessionsplit.py`

   Do not train unless all checks pass and the printed counts are 245 / 119 / 359.

3. Run the Vanilla Transformer + DAT smoke test:

   `python train_mored_vanilla_dat_sessionsplit.py`

4. Build the new DAT feature cache:

   `python train_mored_vanilla_dat_sessionsplit.py --prepare-features`

5. Train Vanilla Transformer and DAT (five seeds each):

   `python train_mored_vanilla_dat_sessionsplit.py --train`

6. Train the other neural baselines (five seeds each):

   `python train_mored_lstm_sessionsplit.py --train`

   `python train_mored_dlinear_sessionsplit.py --train`

   `python train_mored_patchtst_sessionsplit.py --train`

7. Run TimesFM 2.5 zero-shot evaluation (no training):

   `python train_mored_timesfm25_sessionsplit.py`

8. Audit every model's predictions, target arrays, timestamps and saved metrics:

   `python audit_mored_predictions_sessionsplit.py`

9. Run the four-session-clustered statistical analysis:

   `python statistical_analysis_mored_sessionsplit.py`

## Output locations

Data: `Data/MORED/Corrected_V3_SessionSplit/`

Results:

- `Results/MORED/Corrected_V3_SessionSplit_Vanilla_DAT/`
- `Results/MORED/Corrected_V3_SessionSplit_LSTM/`
- `Results/MORED/Corrected_V3_SessionSplit_DLinear/`
- `Results/MORED/Corrected_V3_SessionSplit_PatchTST/`
- `Results/MORED/Corrected_V3_SessionSplit_TimesFM25/`
- `Results/MORED/Corrected_V3_SessionSplit_Audit/`
- `Results/MORED/Corrected_V3_SessionSplit_Statistics/`

## Interpretation warning

This split increases the number of test recording-session clusters from two to four and expands the test set from 62 to 359 windows. Four clusters are still a small inferential sample: the exact two-sided sign test cannot reach p < 0.05 with only four non-tied pairs, even before Holm adjustment. Report the session-level differences and uncertainty descriptively; do not claim conventional statistical significance or equivalence from these four clusters. Five training seeds measure optimisation variability, not additional independent data sessions.

The V3 results must be reported as a separate split experiment. Do not combine V2 and V3 scores in one model table or describe V3 as a direct extension of the V2 held-out test results.
