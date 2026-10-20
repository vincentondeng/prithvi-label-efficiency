"""Step 3c: print the normalisation constants the pretrained Prithvi expects."""
import json
from huggingface_hub import hf_hub_download

REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-100M-TL"
path = hf_hub_download(repo_id=REPO, filename="config.json")
cfg = json.load(open(path))
print("top-level keys:", list(cfg.keys()))
pc = cfg.get("pretrained_cfg", cfg)
for k in ["bands", "mean", "std", "img_size", "num_frames", "patch_size", "coords_encoding"]:
    print(f"{k}: {pc.get(k)}")
