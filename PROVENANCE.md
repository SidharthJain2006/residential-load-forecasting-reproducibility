# Provenance and archive reconciliation

Three uploaded archives were reviewed:

1. `DAT_London_BES_Completed_20261028_144903.zip` contains an earlier BES experiment whose own metrics (RMSE about 0.1565/0.1561) are not the manuscript's final corrected-architecture results. Its `Results/` files are NOT carried into this final package. The helper module `run_bes_ablation.py` and split/scaler files are used only because the final fullBES runner imports that module and expects those split paths.
2. `London_DAT_Corrected_20261028_152629.zip` contains an intermediate corrected-architecture run with slightly different summary values. It is preserved by the user as a historical run but is NOT the selected final London result set.
3. `London_DAT_FullBES_Final_20261028_191159(2).zip` is the authoritative final experiment. Its `summary_corrected_dat.csv` matches the London table in the manuscript. Its per-seed file has five seeds for each BES setting.

## Validation performed on the uploaded archives

- All ten final prediction files contain 5,616 rows each (= 117 origins × 48 forecast steps).
- Every prediction timestamp maps to a test-split timestamp.
- Every saved actual value matches the test CSV consumption at the corresponding timestamp to a maximum absolute difference of approximately 2.8×10⁻⁷ (floating-point serialization precision).
- The final per-seed metrics recompute to the final summary means and sample standard deviations.
- The final script imports `run_bes_ablation.py`; that dependency is supplied in this consolidated package.
- The helper expects `Data/London/splits/{train,val,test}.csv` and `scaler.pkl`; those files are supplied from the earlier archive.
- Independently recomputed the final runner's cache keys from these split CSVs and the final script configuration. All six keys (train/validation/test × BES ON/OFF) exactly match the cache filenames in the final FullBES archive, confirming that the supplied splits are the ones used to create the final feature caches.

## Not found in the uploaded archives

- Trained model checkpoint files for each seed/configuration.
- A pinned Python dependency lock file / exact full package-version list.
- The original raw London dataset file.

Therefore, this package preserves final source, split data, cached features, predictions, metrics, metadata and log, and it supports rerunning training in a recreated environment. It does not contain saved trained checkpoints or a complete environment lock.


## UCI and MORED V3 result archive received 2026-11-02

The user supplied `final_corrected_result_artifacts.zip` containing 216 result files plus a checksum manifest. All 216 listed files were SHA-256 verified successfully. The archive included final UCI corrected V2 outputs, final MORED corrected V3 outputs, and older MORED V2 results.

For the current publication package, 140 selected result files were retained: UCI corrected V2 outputs (using `Corrected_V2_1_VanillaTransformer` as the final Transformer run) and MORED corrected V3 whole-session outputs. The 65 older MORED V2 files and 11 superseded UCI `Corrected_V2_VanillaTransformer` files were intentionally omitted from the current supplementary results archive. Original local copies should remain untouched.

The filtered result archive is `supplementary/FINAL_UCI_MORED_V3_PREDICTIONS_AND_AUDITS.zip`; it contains all selected predictions, y_true arrays, timestamps, configurations, per-seed metrics, integrity reports and statistical outputs, with its own SHA-256 manifest.
