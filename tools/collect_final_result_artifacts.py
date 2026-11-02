#!/usr/bin/env python3
"""Collect the final corrected UCI V2 and MORED V3 results from a project folder.

Usage:
  python tools/collect_final_result_artifacts.py --project-root /path/to/DAT_Reproducible_V2

The archive includes UCI corrected V2 outputs (using the V2.1 Vanilla Transformer run),
MORED corrected V3 whole-session outputs, prediction arrays, metrics, audits and statistics.
It deliberately excludes superseded MORED V2 outputs, the superseded UCI V2 Vanilla Transformer
folder, model checkpoints, training logs and raw datasets.
"""
from pathlib import Path
import argparse, zipfile, hashlib

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--project-root", type=Path, default=Path.cwd())
p.add_argument("--output", type=Path, default=None)
args = p.parse_args()
ROOT = args.project_root.expanduser().resolve()
OUTPUT = args.output.expanduser().resolve() if args.output else ROOT / "final_corrected_result_artifacts.zip"
EXCLUDE_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".log"}
EXCLUDE_NAMES = {".DS_Store"}
files=[]

# UCI corrected V2: include the final V2.1 Transformer and the other final corrected models.
uci=ROOT/"Results"/"UCI"
if uci.exists():
    for f in uci.rglob("*"):
        if not f.is_file() or f.name in EXCLUDE_NAMES or f.suffix.lower() in EXCLUDE_SUFFIXES:
            continue
        rel=f.relative_to(uci)
        if not rel.parts or not rel.parts[0].startswith("Corrected_V2_"):
            continue
        if rel.parts[0] == "Corrected_V2_VanillaTransformer":
            continue  # superseded by Corrected_V2_1_VanillaTransformer
        files.append((f,f.relative_to(ROOT).as_posix()))

# MORED corrected V3 only; superseded V2 outputs stay in the local project but are not in the main release.
mored=ROOT/"Results"/"MORED"
if mored.exists():
    for f in mored.rglob("*"):
        if not f.is_file() or f.name in EXCLUDE_NAMES or f.suffix.lower() in EXCLUDE_SUFFIXES:
            continue
        rel=f.relative_to(mored)
        if not rel.parts or not rel.parts[0].startswith("Corrected_V3_SessionSplit_"):
            continue
        files.append((f,f.relative_to(ROOT).as_posix()))

if not files:
    raise SystemExit("No final UCI corrected V2 or MORED corrected V3 result files found.")
OUTPUT.parent.mkdir(parents=True,exist_ok=True)
checks=[]
with zipfile.ZipFile(OUTPUT,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for f,rel in sorted(files,key=lambda x:x[1]):
        z.write(f,rel)
        checks.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {rel}")
    z.writestr("COLLECTED_SHA256SUMS.txt","\n".join(checks)+"\n")
print(f"Collected {len(files)} current result files")
print(f"Created: {OUTPUT}")
print("Included: UCI corrected V2, MORED corrected V3, prediction arrays, run configs, audits and statistics.")
print("Excluded: superseded MORED V2, superseded UCI VanillaTransformer V2 folder, checkpoints, logs and raw data.")
