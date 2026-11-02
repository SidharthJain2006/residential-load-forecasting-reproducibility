# Public-release checklist

- [ ] All co-authors approve repository and archive contents.
- [ ] Code licence selected by all relevant authors.
- [ ] MORED redistribution terms confirmed for derived V3 split files and predictions.
- [ ] London source confirmed as open UKPN/GLA DataStore release (not restricted UKDS Study 7857).
- [ ] Exact Python/PyTorch/TimesFM environment lock captured from original experiment environment.
- [ ] Final UCI corrected V2 and MORED corrected V3 prediction files, configurations, audits and statistics present in the supplementary archive.
- [ ] Superseded MORED V2 outputs and superseded UCI Vanilla Transformer folder are not represented as current results.
- [ ] London final FullBES kit provenance and SHA256 checks reverified.
- [ ] UCI validation script used on Mac copied if it is part of the actual released protocol; `validate_uci_release.py` is an independent structural validator only.
- [ ] No `.venv`, `__pycache__`, `.DS_Store`, `.git`, raw dataset dumps, credentials, tokens, or private editorial material.
- [ ] README run commands tested from a fresh clone.
- [ ] Repository history contains no accidentally committed secrets or restricted files.
- [ ] Create a version tag only after review; archive the tag on Zenodo and add DOI to the manuscript.
