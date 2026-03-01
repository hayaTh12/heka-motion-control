import torch
from torch import nn


class WeightedMSELoss(nn.Module):
    def __init__(self, weights, mode="time"):
        super().__init__()
        self.mode = mode

        w = torch.tensor(weights, dtype=torch.float32)

        if mode == "time":
            # weights = [w0, w1, ..., wH-1]
            w = w / w.sum()
            self.register_buffer("w_time", w)
            self.w_out = None

        elif mode == "output":
            # weights = [w_hip, w_knee, ...]
            w = w / w.mean()
            self.register_buffer("w_out", w)
            self.w_time = None

        else:
            raise ValueError("mode must be 'time' or 'output'")

    def forward(self, output, target):
        se = (output - target) ** 2   # (B, H, O)

        if self.w_time is not None:
            se = se * self.w_time.view(1, -1, 1)

        if self.w_out is not None:
            se = se * self.w_out.view(1, 1, -1)

        return se.mean()
