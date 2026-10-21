#!/usr/bin/env bash
# One-command setup for a fresh Kaggle session. Safe to re-run.
# Never imports torch in the notebook kernel: all checks run in fresh processes.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/kaggle/working/prithvi-label-efficiency}"
DATA_ROOT="${DATA_ROOT:-/kaggle/temp/sen1floods11}"
SPLITS_DIR="${SPLITS_DIR:-/kaggle/temp/splits}"
BUCKET="https://storage.googleapis.com/sen1floods11/v1.1/splits/flood_handlabeled"

cd "$REPO_DIR"

echo "[${SECONDS}s] == 1/6 Installing pinned packages"
if ! pip install -q -r requirements-cloud.txt > /tmp/pip_install.log 2>&1; then
  echo "pip install failed. Last lines of the log:"; tail -n 30 /tmp/pip_install.log; exit 1
fi
echo "   done (warnings about unrelated preinstalled packages are expected)"

echo "[${SECONDS}s] == 2/6 GPUs"
nvidia-smi -L || { echo "No GPU found: enable a GPU accelerator in the notebook settings."; exit 1; }

echo "[${SECONDS}s] == 3/6 Import check (fresh process)"
python - << 'PY'
import importlib.metadata as md, sys
import torch
assert torch.cuda.is_available(), "CUDA not available in torch"
import terratorch
import torchgeo.trainers.utils
print("   python", sys.version.split()[0])
for p in ["terratorch", "torch", "torchvision", "torchgeo", "lightning", "numpy"]:
    print(f"   {p:12s} {md.version(p)}")
print("   GPU:", torch.cuda.get_device_name(0), "| count:", torch.cuda.device_count())
PY

echo "[${SECONDS}s] == 4/6 Split files"
mkdir -p "$SPLITS_DIR"
for f in flood_train_data flood_valid_data flood_test_data flood_bolivia_data; do
  curl -sf -o "$SPLITS_DIR/$f.csv" "$BUCKET/$f.csv" || { echo "failed to fetch $f.csv"; exit 1; }
done
wc -l "$SPLITS_DIR"/*.csv

echo "[${SECONDS}s] == 5/6 Chips and labels"
python scripts/download_sen1floods11.py --splits-dir "$SPLITS_DIR" --root "$DATA_ROOT" 2>&1 | tail -n 3
echo "   S2 chips:     $(ls "$DATA_ROOT/S2Hand" | wc -l)  (expect 446)"
echo "   label chips:  $(ls "$DATA_ROOT/LabelHand" | wc -l)  (expect 446)"
du -sh "$DATA_ROOT"

echo "[${SECONDS}s] == 6/6 Recording environment"
pip freeze > /kaggle/working/env_versions.txt
echo "   wrote /kaggle/working/env_versions.txt"
echo "== Setup complete"
echo "   total setup time: ${SECONDS}s"
