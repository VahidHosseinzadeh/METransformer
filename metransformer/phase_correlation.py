"""Fourier phase correlation, as in MELSTM (moving_mnist/velocity_predictor_model.py)."""
import torch
import torch.nn as nn


class PhaseCorrelation(nn.Module):
    """The n_modes strongest peaks of the whitened cross-power spectrum of two frames.

    Channels are averaged first. For b = torch.roll(a, (dy, dx), (-2, -1)) the strongest peak
    is (dx, dy): the displacement that carries the content of a into b. The estimate is an
    integer (the peak is an argmax over pixels), and the module has no parameters and no
    gradient. Same arithmetic as MELSTM's legacy path (pad_factor 1, periodic, alpha 1).
    """

    def __init__(self, n_modes=1, eps=1e-8):
        super().__init__()
        self.n_modes = n_modes
        self.eps = eps

    def surface(self, a, b):
        """Correlation surface (N, H, W) of a: (N, C, H, W) against b: (N, C, H, W)."""
        H, W = a.shape[-2:]
        R = torch.fft.rfft2(a.mean(dim=1)) * torch.conj(torch.fft.rfft2(b.mean(dim=1)))
        R = R / (R.abs() + self.eps)
        return torch.fft.irfft2(R, s=(H, W))

    @staticmethod
    def index_to_velocity(idx, H, W):
        y = torch.div(idx, W, rounding_mode="floor")
        x = idx % W
        x = torch.where(x > W / 2, x - W, x)
        y = torch.where(y > H / 2, y - H, y)
        return torch.stack((-x, -y), dim=-1)

    @torch.no_grad()
    def forward(self, a, b):
        """-> velocities (N, n_modes, 2) as int64 (dx, dy), scores (N, n_modes)."""
        H, W = a.shape[-2:]
        corr = self.surface(a.float(), b.float())
        scores, idx = corr.reshape(corr.shape[0], -1).topk(self.n_modes, dim=1)
        return self.index_to_velocity(idx, H, W), scores
