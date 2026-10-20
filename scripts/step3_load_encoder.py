"""Step 3a: build the pretrained Prithvi encoder and inspect its inputs/outputs."""
import torch
from terratorch.registry import BACKBONE_REGISTRY

NAME = "terratorch_prithvi_eo_v2_100_tl"

model = BACKBONE_REGISTRY.build(NAME, pretrained=True, num_frames=1)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"{NAME}: {n_params / 1e6:.1f}M parameters")
for attr in ["embed_dim", "out_channels", "patch_size", "num_frames", "img_size"]:
    if hasattr(model, attr):
        print(f"  {attr}: {getattr(model, attr)}")

# Fake input: (batch, channels=6 bands, time=1, height, width)
x = torch.randn(2, 6, 1, 224, 224)
with torch.no_grad():
    out = model(x)

outs = out if isinstance(out, (list, tuple)) else [out]
print(f"output: {len(outs)} tensor(s)")
for i, o in enumerate(outs):
    print(f"  [{i}] shape={tuple(o.shape)}")
