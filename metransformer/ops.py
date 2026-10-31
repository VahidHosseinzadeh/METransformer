"""Translation-equivariant building blocks on the periodic grid."""
import math

import torch
import torch.nn as nn


def torus_shift(x, d):
    """Translate every map of x by its own integer displacement on the torus.

    x: (..., C, H, W); d: (..., 2) holding (dx, dy), same leading shape as x.
    out[..., i, j] = x[..., (i - dy) % H, (j - dx) % W], so content moves by +d, as
    torch.roll(x, (dy, dx), (-2, -1)) does for one map. It is a gather, so it is exact and
    differentiable in x, also on MPS, which has no grid_sample backward.
    """
    *lead, C, H, W = x.shape
    n = math.prod(lead)
    x = x.reshape(n, C, H * W)
    d = d.reshape(n, 2).long()
    iy = (torch.arange(H, device=x.device) - d[:, 1:2]) % H
    ix = (torch.arange(W, device=x.device) - d[:, 0:1]) % W
    idx = (iy[:, :, None] * W + ix[:, None, :]).reshape(n, 1, H * W)
    return x.gather(2, idx.expand(n, C, H * W)).reshape(*lead, C, H, W)


def circular_conv(c_in, c_out, kernel):
    return nn.Conv2d(c_in, c_out, kernel, padding=kernel // 2, padding_mode="circular")


class ChannelNorm(nn.Module):
    """LayerNorm over the channels of each pixel; the affine is per channel, so it commutes
    with translations. Computed in place along dim -3: nn.LayerNorm on channels-last copies
    had a slow backward with only 32 channels per row."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(dim, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=-3, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-3, keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias
