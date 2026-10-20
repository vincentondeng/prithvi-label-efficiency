"""Prithvi encoder + small segmentation decoder (identical across all arms)."""
import torch
import torch.nn as nn
from terratorch.registry import BACKBONE_REGISTRY

BACKBONE = "terratorch_prithvi_eo_v2_100_tl"


class SegModel(nn.Module):
    def __init__(self, pretrained, img_size=512, layers=(2, 5, 8, 11), dec_dim=256):
        super().__init__()
        self.encoder = BACKBONE_REGISTRY.build(BACKBONE, pretrained=pretrained, num_frames=1, img_size=img_size)
        self.layers = list(layers)
        self.grid = img_size // 16
        d = self.encoder.embed_dim
        self.fuse = nn.Sequential(nn.Conv2d(d * len(self.layers), dec_dim, 1),
                                  nn.GroupNorm(8, dec_dim), nn.GELU())
        ups, ch = [], dec_dim
        for _ in range(4):  # 16x upsampling: 32 -> 64 -> 128 -> 256 -> 512
            ups += [nn.ConvTranspose2d(ch, ch // 2, 2, stride=2), nn.GroupNorm(8, ch // 2), nn.GELU()]
            ch //= 2
        self.up = nn.Sequential(*ups)
        self.head = nn.Conv2d(ch, 2, 1)

    def forward(self, x):  # x: (B, 6, H, W) normalised
        feats = self.encoder(x.unsqueeze(2))  # encoder expects (B, 6, T=1, H, W)
        maps = []
        for i in self.layers:
            f = feats[i]
            assert f.shape[1] == self.grid * self.grid + 1, f"unexpected token count {f.shape}"
            f = f[:, 1:, :]  # drop class token (assumed first)
            maps.append(f.reshape(f.shape[0], self.grid, self.grid, -1).permute(0, 3, 1, 2).contiguous())
        return self.head(self.up(self.fuse(torch.cat(maps, 1))))

    def param_groups(self, lr_encoder, lr_decoder):
        dec = [p for n, p in self.named_parameters() if not n.startswith("encoder.")]
        return [{"params": list(self.encoder.parameters()), "lr": lr_encoder},
                {"params": dec, "lr": lr_decoder}]
