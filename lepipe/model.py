"""Prithvi encoder + segmentation head (identical across arms within an experiment).

head="decoder" (default): multi-level decoder used in all main experiments.
head="linear": linear probe on the last tapped block: per-channel normalisation without learnable
    parameters (as in MAE linear probing), a 1x1 convolution to 2 classes on the 32x32 token grid,
    and bilinear upsampling to the input size.
freeze_encoder=True: encoder weights are not trained, it runs without gradients and always in eval mode
    (dropout/drop-path off); only the head is optimised.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from terratorch.registry import BACKBONE_REGISTRY

BACKBONE = "terratorch_prithvi_eo_v2_100_tl"


class SegModel(nn.Module):
    def __init__(self, pretrained, img_size=512, layers=(2, 5, 8, 11), dec_dim=256,
                 head="decoder", freeze_encoder=False):
        super().__init__()
        if head not in ("decoder", "linear"):
            raise ValueError(f"unknown head {head!r}")
        self.encoder = BACKBONE_REGISTRY.build(BACKBONE, pretrained=pretrained, num_frames=1, img_size=img_size)
        self.head_type = head
        self.freeze_encoder = freeze_encoder
        self.layers = list(layers) if head == "decoder" else [layers[-1]]
        self.grid = img_size // 16
        d = self.encoder.embed_dim
        if head == "decoder":
            self.fuse = nn.Sequential(nn.Conv2d(d * len(self.layers), dec_dim, 1),
                                      nn.GroupNorm(8, dec_dim), nn.GELU())
            ups, ch = [], dec_dim
            for _ in range(4):  # 16x upsampling: 32 -> 64 -> 128 -> 256 -> 512
                ups += [nn.ConvTranspose2d(ch, ch // 2, 2, stride=2), nn.GroupNorm(8, ch // 2), nn.GELU()]
                ch //= 2
            self.up = nn.Sequential(*ups)
            self.head = nn.Conv2d(ch, 2, 1)
        else:
            self.norm = nn.BatchNorm2d(d, affine=False)
            self.head = nn.Conv2d(d, 2, 1)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()  # keep dropout/drop-path off in a frozen encoder
        return self

    def _token_maps(self, x):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_encoder):
            feats = self.encoder(x.unsqueeze(2))  # encoder expects (B, 6, T=1, H, W)
        maps = []
        for i in self.layers:
            f = feats[i]
            assert f.shape[1] == self.grid * self.grid + 1, f"unexpected token count {f.shape}"
            f = f[:, 1:, :]  # drop class token (assumed first)
            maps.append(f.reshape(f.shape[0], self.grid, self.grid, -1).permute(0, 3, 1, 2).contiguous())
        return maps

    def forward(self, x):  # x: (B, 6, H, W) normalised
        maps = self._token_maps(x)
        if self.head_type == "decoder":
            return self.head(self.up(self.fuse(torch.cat(maps, 1))))
        logits = self.head(self.norm(maps[0]))
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

    def param_groups(self, lr_encoder, lr_decoder):
        dec = [p for n, p in self.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
        groups = [{"params": dec, "lr": lr_decoder}]
        if not self.freeze_encoder:
            groups.insert(0, {"params": list(self.encoder.parameters()), "lr": lr_encoder})
        return groups

    def n_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
