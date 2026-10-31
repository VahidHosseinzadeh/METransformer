"""Figures of a model's state, for wandb and the run directory.

slot_states is the transformer's counterpart of MELSTM's hidden-state plots: for every token
(the context frames, then the fed-back predictions) and every slot, what the slot holds in
its own frame, where it places that content for the next frame, and the channel mean of the
last block's output.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from metransformer.ops import torus_shift  # noqa: E402


@torch.no_grad()
def slot_states(model, seq, motion, input_frames, pred_len, oracle=False):
    """seq (n_frames, C, H, W) on the model's device -> dict of numpy arrays over the
    n = input_frames + pred_len - 1 tokens: 'frames' (n, H, W) the tokens' frames, 'targets'
    (n, H, W) the frames they predict, and per slot (K, n, H, W): 'content' the decoded output
    in the slot frame, 'placed' the same moved to the slot's next position, 'hidden' the channel
    mean of the last block's output in the slot frame; 'paths' (K, n + 1, 2)."""
    model.eval()
    pred, aux = model(seq[None, :input_frames], pred_len,
                      gt_motion=motion[None] if oracle else None, return_slots=True)
    h, paths = aux["slot_h"][0], aux["paths"][0]
    K, n = h.shape[:2]
    y = model.norm(h)

    def decode(z):
        return model.decoder(z.reshape(K * n, *z.shape[2:])).view(K, n, *z.shape[-2:])

    frames = torch.cat([seq[:input_frames], pred[0, :-1]])[:, 0]
    out = dict(frames=frames, targets=seq[1:n + 1, 0], content=decode(y),
               placed=decode(torus_shift(y, paths[:, 1:])), hidden=h.mean(dim=2), paths=paths)
    return {k: v.float().cpu().numpy() for k, v in out.items()}


def _row(images, every, gap=1):
    """(n, H, W) -> one strip of every `every`-th image, separated by `gap` blank columns."""
    images = images[::every]
    H, W = images.shape[1:]
    strip = np.full((H, len(images) * (W + gap) - gap), np.nan)
    for i, im in enumerate(images):
        strip[:, i * (W + gap):i * (W + gap) + W] = im
    return strip


def state_figure(states, input_frames, title="", every=1):
    """Rows: tokens, targets, and per slot its content, placement and hidden mean."""
    K = states["content"].shape[0]
    rows = [("input token", states["frames"], "gray"), ("target frame", states["targets"], "gray")]
    for k in range(K):
        rows += [(f"slot {k}: content", states["content"][k], "gray"),
                 (f"slot {k}: placed", states["placed"][k], "gray"),
                 (f"slot {k}: hidden mean", states["hidden"][k], "RdBu_r")]
    n = states["frames"].shape[0]
    times = np.arange(n)[::every]
    fig, axes = plt.subplots(len(rows), 1, figsize=(0.42 * len(times) + 1.6, 0.48 * len(rows) + 0.5))
    for ax, (label, images, cmap) in zip(axes, rows):
        strip = _row(images, every)
        if cmap == "gray":
            ax.imshow(np.clip(strip, 0, 1), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
        else:
            lim = np.nanpercentile(np.abs(strip), 99) or 1.0
            ax.imshow(strip, cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7)
    W = states["frames"].shape[-1]
    for i, t in enumerate(times):
        axes[0].text(i * (W + 1) + W / 2, -4, str(t), ha="center", va="bottom", fontsize=6,
                     color="red" if t >= input_frames else "black")
    fig.suptitle(title + "  (token time; red: fed-back predictions)", fontsize=8)
    fig.tight_layout(pad=0.3)
    return fig


def rollout_figure(gt, pred, input_frames, every=5, title=""):
    """gt (B, T, H, W) full sequences, pred (B, T - input_frames, H, W) -> GT / prediction rows
    for each sequence, every `every`-th frame."""
    B, T = gt.shape[:2]
    full = np.concatenate([gt[:, :input_frames], pred], axis=1)
    times = np.arange(T)[::every]
    fig, axes = plt.subplots(2 * B, 1, figsize=(0.42 * len(times) + 1.2, 0.5 * 2 * B + 0.5))
    for b in range(B):
        for r, (label, x) in enumerate((("truth", gt[b]), ("model", full[b]))):
            ax = axes[2 * b + r]
            ax.imshow(np.clip(_row(x, every), 0, 1), cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(f"{label} {b}", rotation=0, ha="right", va="center", fontsize=7)
    W = gt.shape[-1]
    for i, t in enumerate(times):
        axes[0].text(i * (W + 1) + W / 2, -4, str(t + 1), ha="center", va="bottom", fontsize=6,
                     color="red" if t >= input_frames else "black")
    fig.suptitle(title + "  (frame t, 1-indexed; red: predicted)", fontsize=8)
    fig.tight_layout(pad=0.3)
    return fig


def lag_bias_figure(model):
    """The learned attention bias of every block and head against the time lag."""
    blocks = [b.attn.time_bias.detach().float().cpu().numpy() for b in model.blocks]
    fig, axes = plt.subplots(1, len(blocks), figsize=(3.2 * len(blocks), 2.4), squeeze=False)
    for l, (ax, bias) in enumerate(zip(axes[0], blocks)):
        for j, b in enumerate(bias):
            ax.plot(np.arange(len(b)), b, marker=".", label=f"head {j}")
        ax.set_title(f"block {l}", fontsize=8)
        ax.set_xlabel("lag t - s (last bin: all larger lags)", fontsize=7)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("attention bias", fontsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    return fig


def velocity_figure(per_step):
    """Velocity accuracy at every context step t = 1..T-1."""
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    t = np.arange(1, len(per_step) + 1)
    ax.plot(t, 100 * np.asarray(per_step), marker="o")
    ax.set_xlabel("context step t", fontsize=7)
    ax.set_ylabel("velocity accuracy (%)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def boundary_age(motion, input_frames):
    """Steps since each digit's last velocity change before the context ends, (B, N): 1 = the
    velocity changed between the last two context displacements (as in MELSTM's analysis)."""
    f = input_frames
    m = motion[:, :f - 1]
    changed = (m[:, 1:] != m[:, :-1]).any(dim=-1)
    idx = torch.arange(1, f - 1, device=motion.device).view(1, -1, 1)
    t_last = torch.where(changed, idx, torch.zeros_like(idx)).amax(dim=1)
    return (f - 1) - t_last
