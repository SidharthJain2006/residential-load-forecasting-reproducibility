# Experiment protocol summary

## Common
- Direct 24-hour forecast from load history only.
- Train-only scaling.
- Trainable models: seeds 0–4.
- TimesFM 2.5: one zero-shot run; no fine-tuning and no seed SD.
- Metrics: RMSE, MAE, MAPE, R², StdR (prediction standard deviation / actual standard deviation). StdR is a dispersion diagnostic, not accuracy.

## UCI
- Frequency: 1 minute.
- Lookback: 1,440 minutes; horizon: 1,440 minutes; stride: 1,440.
- Chronological 70/15/15 split.
- No interpolation; any complete 48-hour candidate window containing an unobserved/non-finite target or input value is rejected.
- Accepted windows: 903 train, 189 validation, 192 test.

## MORED V3
- Raw source: APT6.csv, 1,346,393 five-second samples.
- 1,036 negative raw real-power readings invalidated.
- One-minute bins retained only with at least 10 genuine valid raw samples. No interpolation/forward fill/back fill.
- Chronological whole-original-session assignment: train raw sessions 0–4, validation 5, test 6–9.
- Lookback 1,440 minutes; horizon 1,440; stride 60.
- Accepted windows: 245 train, 119 validation, 359 test.
- Four original raw test sessions are the inference clusters; seven within-session continuity runs contribute windows.

## London
- Household MAC000147 only.
- 30-minute resolution; lookback 336 steps; horizon 48 steps; stride 48.
- 117 test windows.
- Five seeds each for BES off/on.
- BES on combines left reflection, historical-similarity right extension and weighted overlap fusion; BES off disables those boundary components.
