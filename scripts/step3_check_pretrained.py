"""Step 3b: confirm pretrained weights are really loaded (differ from random init)."""
import torch
from terratorch.registry import BACKBONE_REGISTRY

NAME = "terratorch_prithvi_eo_v2_100_tl"

torch.manual_seed(0)
pre = BACKBONE_REGISTRY.build(NAME, pretrained=True, num_frames=1).eval()
torch.manual_seed(0)
rnd = BACKBONE_REGISTRY.build(NAME, pretrained=False, num_frames=1).eval()

differ, same_names = 0, []
for (n1, p1), (n2, p2) in zip(pre.named_parameters(), rnd.named_parameters()):
    assert n1 == n2, f"parameter order mismatch: {n1} vs {n2}"
    if torch.equal(p1, p2):
        same_names.append(n1)
    else:
        differ += 1
print(f"parameter tensors: {differ} differ from random init, {len(same_names)} identical")
if same_names:
    print("identical tensors (first 8):", same_names[:8])

x = torch.randn(2, 6, 1, 224, 224)
with torch.no_grad():
    a, b = pre(x)[-1], rnd(x)[-1]
print(f"last-layer output: pretrained std={a.std():.3f}, random-init std={b.std():.3f}")
print(f"mean abs difference between them: {(a - b).abs().mean():.3f}")
print("any NaN in pretrained output:", bool(torch.isnan(a).any()))
