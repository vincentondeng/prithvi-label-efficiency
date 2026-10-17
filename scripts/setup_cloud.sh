#!/usr/bin/env bash
# Prepare a cloud notebook session. Safe to re-run.
set -euo pipefail
pip install -q -r requirements-cloud.txt
python - << 'PY'
import importlib.metadata as md, sys
print("python", sys.version.split()[0])
for p in ["terratorch", "torch", "torchvision", "torchgeo", "lightning", "numpy"]:
    print(f"{p:12s} {md.version(p)}")
PY
