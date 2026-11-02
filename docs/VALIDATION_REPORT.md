# Validation report for the staged repository

Validation was run after assembling the pre-release repository bundle.

- `validate_uci_release.py`: all checks passed for 903 train, 189 validation and 192 test windows, including finite arrays, target alignment, scaler transformation consistency, split metadata and manifest counts.
- `validate_mored_sessionsplit.py`: all checks passed for 245 train, 119 validation and 359 test windows; whole-session assignments 0–4 / 5 / 6–9; no interpolation; training-only scaler; four original test recording-session clusters.
- ZIP integrity test: passed.
- Scan for obvious absolute user-local paths, credential-like strings and API tokens in included text/code: no matches.

The container's installed scikit-learn version (1.8.0) emitted an `InconsistentVersionWarning` when loading scalers saved under scikit-learn 1.9.1. The validator still passed, but the experiment's exact lockfile must be captured from the original Mac `.venv` before public release. Do not use this container's environment as the experiment environment.

The London package's original provenance record and checksums are included. Its full end-to-end smoke test should be re-run from the authors' Mac environment after a fresh extraction because MPS and dependency versions differ by host.
