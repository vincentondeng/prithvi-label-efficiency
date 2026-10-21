"""Segmentation metrics from a confusion matrix accumulated over a whole split."""
import torch


class ConfusionMeter:
    def __init__(self, ignore_index=255, n_classes=2):
        self.n, self.ignore = n_classes, ignore_index
        self.cm = torch.zeros(n_classes, n_classes, dtype=torch.long)  # rows = target, cols = prediction

    @torch.no_grad()
    def update(self, logits, target):
        pred = logits.argmax(1)
        valid = target != self.ignore
        idx = (target[valid] * self.n + pred[valid]).cpu()
        self.cm += torch.bincount(idx, minlength=self.n * self.n).reshape(self.n, self.n)

    def compute(self):
        cm = self.cm.double()
        tp = cm.diag()
        fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
        iou = tp / (tp + fp + fn).clamp(min=1)
        return {"iou_water": iou[1].item(), "iou_land": iou[0].item(), "miou": iou.mean().item(),
                "f1_water": (2 * tp[1] / (2 * tp[1] + fp[1] + fn[1]).clamp(min=1)).item(),
                "acc": (tp.sum() / cm.sum().clamp(min=1)).item(),
                "cm": self.cm.tolist()}
