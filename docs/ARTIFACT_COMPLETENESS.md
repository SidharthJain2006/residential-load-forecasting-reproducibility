# Result artifact completeness and provenance

This private pre-release package now includes:
- UCI corrected V2 final prediction arrays for the five trainable models and TimesFM 2.5 zero-shot output; target arrays/timestamps; per-seed results; run configurations; integrity audit; and the implemented seven-day moving-block bootstrap and sensitivity outputs.
- MORED corrected V3 whole-session final prediction arrays for five seeds of Vanilla Transformer, DAT, LSTM, DLinear and PatchTST-style; TimesFM 2.5 zero-shot output; target arrays/timestamps; run configurations; prediction-integrity audit; and four-original-session statistical analysis.
- London authoritative final FullBES runner, helper dependency, split inputs/scaler, caches, predictions, per-seed results, metadata and checksums.
- Summary tables, final manuscript and figures.

The complete UCI and MORED V3 prediction and audit files are packed in `supplementary/FINAL_UCI_MORED_V3_PREDICTIONS_AND_AUDITS.zip`. This archive contains only the selected final UCI and MORED V3 experiments. It intentionally excludes superseded MORED V2 result folders and the superseded UCI `Corrected_V2_VanillaTransformer` folder; the final UCI Transformer output is `Corrected_V2_1_VanillaTransformer`.

The user's original collector archive had 216 files and its SHA-256 manifest was verified with no mismatches. A filtered archive was then assembled from 140 selected result files, plus its README and checksum manifest. The original upload also contained superseded MORED V2 results and an earlier UCI Transformer folder; those are retained by the user locally but not promoted as current results.

The included `validate_uci_release.py` is a structural integrity validator for the packaged UCI split; it is not represented as the original local validation script.
