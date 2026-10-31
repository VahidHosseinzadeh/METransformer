"""The motion equivariant transformer for next-frame prediction.

Every frame is embedded by a circular convolution and lifted into the K slot frames
(lift.py): slot k sees frame t translated by -P^k_t. A causal transformer, whose projections
are convolutions and whose norm acts per pixel, runs along time in each slot, with weights
shared across slots. A pixel of a slot frame at time t therefore attends to the same object
point at earlier times: in image coordinates, the query of frame t at x reads the keys and
values of frame s at x - (P_t - P_s), i.e. transported by T_{t,s} (MELSTM, Sec. 6). To predict
frame t+1, each slot's output is moved to P^k_{t+1}, the slots are max-pooled and a
convolutional decoder renders the frame. Past the context the paths continue at the last
velocity, and the predictions are fed back as the next tokens.

With every operation translation-equivariant, and the slot tokens of psi X equal to those of X
up to the global offset psi_0 (lift.py), the predictions transform with the motion psi
(continued at its last velocity past the context).
"""
import torch
import torch.utils.checkpoint
import torch.nn as nn

from .ops import ChannelNorm, circular_conv, torus_shift


class TemporalAttention(nn.Module):
    """Causal multi-head self-attention along time within one slot frame.

    mode 'pixel': every pixel attends to its own past, one softmax per pixel. mode 'frame':
    one weight per pair of frames from the inner product of the whole maps (as in the FET).
    The logits get a learned bias per head and time lag, b[min(t - s, max_dist - 1)];
    window, if set, hides frames more than window - 1 steps back.
    """

    def __init__(self, dim, heads, mode="pixel", max_dist=16, window=None):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} is not divisible by heads {heads}")
        if mode not in ("pixel", "frame"):
            raise ValueError(f"unknown attention mode {mode!r}")
        self.heads, self.mode, self.max_dist, self.window = heads, mode, max_dist, window
        self.qkv = circular_conv(dim, 3 * dim, 1)
        self.proj = circular_conv(dim, dim, 1)
        self.time_bias = nn.Parameter(torch.zeros(heads, max_dist))

    def forward(self, x, cache=None, t0=0):
        """x: (N, T, C, H, W), the frames t0..t0+T-1; cache: keys and values of frames < t0.
        -> (N, T, C, H, W) and the cache extended by these frames.

        The cache is a list of (keys, values) chunks, one per call. With gradients on they
        are not concatenated: a concatenation per rollout step would keep a copy of the
        whole past for the backward pass. Without gradients they are merged into one.

        Queries, keys and values are laid out (N, heads, head_dim, time, pixel). Pixel
        attention takes its dot products as broadcast products summed over head_dim, along
        contiguous pixel rows: as a batched matmul it would be one tiny (T x head_dim) GEMM
        per pixel, which ran at ~1% of an A100's throughput."""
        N, T, C, H, W = x.shape
        h, d = self.heads, C // self.heads
        qkv = self.qkv(x.reshape(N * T, C, H, W)).view(N, T, 3, h, d, H * W)
        q, k, v = qkv.permute(2, 0, 3, 4, 1, 5).unbind(0)
        cache = list(cache or []) + [(k, v)]
        if not torch.is_grad_enabled() and len(cache) > 1:
            cache = [tuple(torch.cat(c, dim=3) for c in zip(*cache))]
        sizes = [k.shape[3] for k, _ in cache]
        lag = (torch.arange(t0, t0 + T, device=x.device)[:, None]
               - torch.arange(sum(sizes), device=x.device)[None, :])
        hidden = lag < 0
        if self.window:
            hidden = hidden | (lag >= self.window)
        bias = self.time_bias[:, lag.clamp(0, self.max_dist - 1)].masked_fill(hidden, float("-inf"))
        if self.mode == "pixel":
            logits = torch.cat([(q.unsqueeze(4) * k.unsqueeze(3)).sum(dim=2) for k, _ in cache],
                               dim=3)
            att = (logits * d ** -0.5 + bias[..., None]).softmax(dim=3).split(sizes, dim=3)
            out = sum((a.unsqueeze(2) * v.unsqueeze(3)).sum(dim=4) for a, (_, v) in zip(att, cache))
        else:
            logits = torch.cat([torch.einsum("nhdtp,nhdsp->nhts", q, k) for k, _ in cache], dim=-1)
            att = (logits * (d * H * W) ** -0.5 + bias).softmax(dim=-1).split(sizes, dim=-1)
            out = sum(torch.einsum("nhts,nhdsp->nhdtp", a, v) for a, (_, v) in zip(att, cache))
        out = out.permute(0, 3, 1, 2, 4).reshape(N * T, C, H, W)
        return self.proj(out).view(N, T, C, H, W), cache


class Block(nn.Module):
    """Pre-norm transformer block; the MLP's first convolution mixes space."""

    def __init__(self, dim, heads, mlp_hidden, kernel=3, **attn_kw):
        super().__init__()
        self.norm1 = ChannelNorm(dim)
        self.attn = TemporalAttention(dim, heads, **attn_kw)
        self.norm2 = ChannelNorm(dim)
        self.mlp = nn.Sequential(circular_conv(dim, mlp_hidden, kernel), nn.GELU(),
                                 circular_conv(mlp_hidden, dim, 1))

    def forward(self, x, cache=None, t0=0):
        a, cache = self.attn(self.norm1(x), cache, t0)
        x = x + a
        N, T, C, H, W = x.shape
        x = x + self.mlp(self.norm2(x).reshape(N * T, C, H, W)).view(N, T, C, H, W)
        return x, cache


class METransformer(nn.Module):
    """Next-frame predictor: lift, causal transformer per slot, slot pooling, decoder.

    The lift decides the model: NoLift gives a plain transformer, FlowLift the flow
    equivariant transformer, MotionLift the motion equivariant one. All share these weights.
    """

    def __init__(self, lift, in_channels=1, dim=32, depth=2, heads=4, mlp_hidden=48, kernel=3,
                 attn="pixel", max_dist=16, window=None, slot_reduce="max", decoder_hidden=32,
                 detach_feedback=False, checkpoint=False):
        super().__init__()
        if slot_reduce not in ("max", "mean", "sum"):
            raise ValueError(f"unknown slot_reduce {slot_reduce!r}")
        self.lift = lift
        self.slot_reduce = slot_reduce
        self.detach_feedback = detach_feedback
        self.checkpoint = checkpoint
        self.embed = circular_conv(in_channels, dim, kernel)
        self.blocks = nn.ModuleList(
            Block(dim, heads, mlp_hidden, kernel, mode=attn, max_dist=max_dist, window=window)
            for _ in range(depth))
        self.norm = ChannelNorm(dim)
        self.decoder = nn.Sequential(circular_conv(dim, decoder_hidden, 3), nn.GELU(),
                                     circular_conv(decoder_hidden, in_channels, 3))

    def tokens(self, frames, P):
        """frames (B, T, C, H, W), P (B, K, T, 2) -> slot tokens (B*K, T, D, H, W)."""
        B, T, C, H, W = frames.shape
        K = P.shape[1]
        z = self.embed(frames.reshape(B * T, C, H, W)).view(B, 1, T, -1, H, W)
        return torus_shift(z.expand(B, K, T, -1, H, W), -P).reshape(B * K, T, -1, H, W)

    def transformer(self, tokens, caches=None, t0=0):
        """With checkpoint on (and gradients), each block keeps only its input and outputs
        and recomputes the rest in the backward pass."""
        caches = caches or [None] * len(self.blocks)
        new = []
        for block, cache in zip(self.blocks, caches):
            if self.checkpoint and torch.is_grad_enabled():
                tokens, cache = torch.utils.checkpoint.checkpoint(block, tokens, cache, t0,
                                                                  use_reentrant=False)
            else:
                tokens, cache = block(tokens, cache, t0)
            new.append(cache)
        return tokens, new

    def read(self, y, P):
        """Slot outputs y (B, K, ..., D, H, W) in the slot frames, placed at P (B, K, ..., 2)
        -> frames (B, ..., C, H, W)."""
        y = torus_shift(self.norm(y), P)
        if self.slot_reduce == "max":
            y = y.max(dim=1).values
        else:
            y = y.mean(dim=1) if self.slot_reduce == "mean" else y.sum(dim=1)
        out = self.decoder(y.reshape(-1, *y.shape[-3:]))
        return out.view(*y.shape[:-3], *out.shape[-3:])

    def forward(self, context, pred_len, gt_motion=None, context_predictions=False):
        """context (B, T, C, H, W) -> predictions of frames T..T+pred_len-1 and an aux dict.

        aux: P, U from the lift; with context_predictions, also 'context_pred', the predictions
        of frames 1..T-1 read at the lifted paths (the velocity measured on the target frame).
        """
        B, T = context.shape[:2]
        P, u, U = self.lift(context, gt_motion)
        K = P.shape[1]
        h, caches = self.transformer(self.tokens(context, P))
        h = h.view(B, K, T, *h.shape[2:])
        aux = {"P": P, "U": U}
        if context_predictions:
            aux["context_pred"] = self.read(h[:, :, :-1], P[:, :, 1:])

        y, Pt, preds = h[:, :, -1], P[:, :, -1], []
        for j in range(pred_len):
            Pt = Pt + u
            frame = self.read(y, Pt)
            preds.append(frame)
            if j + 1 == pred_len:
                break
            fed = frame.detach() if self.detach_feedback else frame
            h, caches = self.transformer(self.tokens(fed[:, None], Pt[:, :, None]), caches, T + j)
            y = h.view(B, K, *h.shape[2:])
        return torch.stack(preds, dim=1), aux
