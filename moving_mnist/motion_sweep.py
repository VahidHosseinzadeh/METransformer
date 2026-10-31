"""Motion generalization: trained checkpoints scored on motions they were not trained on.

The regimes are MELSTM's one-factor sweeps (motion_factors_sweep.py there), on the same
5,000 test sequences (seed 42, 100-frame stream): a centre close to the training motion, and
three axes, the switching rate (A), the jump size at a fixed realised rate (B) and the speed
range (C). Each checkpoint predicts frames 16-25; per-sequence MSE is saved per regime, and
make_figures.py normalises every model by its own centre.

    python -m moving_mnist.motion_sweep --runs runs --arms met tf
"""
import argparse
import glob
import json
import os
import re
import time

import numpy as np
import torch

from moving_mnist.time_dependent_moving_mnist_dataset import TDMovingMNISTDataset
from moving_mnist.train import build_model

D_GRID = (0.0, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00)
S_GRID = (1.0, 0.9, 0.6, 0.4, 0.2, 0.0)
N_GRID = (1, 3, 4, 5)
TRAIN_D, TRAIN_S, TRAIN_N = 0.50, 0.80, 2
UNIT_OFFSETS = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]


def velocity_grid(n):
    return [(a, b) for a in range(-n, n + 1) for b in range(-n, n + 1) if (a, b) != (0, 0)]


def acceptance(n):
    g = set(velocity_grid(n))
    return float(np.mean([sum((v[0] + o[0], v[1] + o[1]) in g for o in UNIT_OFFSETS) / 8 for v in g]))


def compensated_p(target, s, n):
    """p_change that keeps the realised switching rate at target for smoothness s, speed n."""
    return target / (1.0 - s * (1.0 - acceptance(n)))


TARGET_RATE = TRAIN_D ** 2.170 * (1.0 - TRAIN_S * (1.0 - acceptance(TRAIN_N)))


def cells():
    """(name, axis, level, dataset kwargs), as in MELSTM's sweep."""
    base = dict(neighbor_kernel="symmetric", motion_mode="stochastic", transition_mode="smooth")
    out = [("centre", "centre", float("nan"),
            dict(base, p_change=TRAIN_D ** 2.170, smooth_probability=TRAIN_S, max_speed=TRAIN_N))]
    out += [(f"A:d={d:.2f}", "rate", d,
             dict(base, p_change=d ** 2.170, smooth_probability=TRAIN_S, max_speed=TRAIN_N))
            for d in D_GRID if d != TRAIN_D]
    out += [(f"B:s={s:.2f}", "jump", s,
             dict(base, p_change=compensated_p(TARGET_RATE, s, TRAIN_N), smooth_probability=s,
                  max_speed=TRAIN_N)) for s in S_GRID]
    out += [(f"C:N={n}", "speed", float(n),
             dict(base, p_change=compensated_p(TARGET_RATE, TRAIN_S, n),
                  smooth_probability=TRAIN_S, max_speed=n)) for n in N_GRID]
    return out


def slug(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def generate(root, kwargs, n, input_frames=15, seq_len=100, seed=42):
    ds = TDMovingMNISTDataset(root=root, train=False, seq_len=seq_len, num_digits=2,
                              image_size=36, freeze_after=input_frames, min_center_distance=20,
                              reject_overlap=True, require_distinct_velocities=True,
                              return_motion=True, return_positions=False, transform=None,
                              download=False, random=False, seed=seed, max_tries=300, **kwargs)
    ds.reset_rng()
    items = [ds[i] for i in range(n)]
    return torch.stack([s for s, _, _ in items]), torch.stack([m for _, _, m in items])


def motion_stats(motion, input_frames=15):
    """Realised switching rate and mean jump |dv| over the context (as MELSTM reports them)."""
    ctx = motion[:, :input_frames - 1].numpy().astype(np.int64)
    changed = (ctx[:, 1:] != ctx[:, :-1]).any(axis=-1)
    dv = np.abs(ctx[:, 1:] - ctx[:, :-1]).max(axis=-1)
    return dict(rate=float(changed.mean()),
                mean_dv=float(dv[changed].mean()) if changed.any() else 0.0)


@torch.no_grad()
def score(model, seqs, motion, device, batch, input_frames=15, pred_len=10, oracle=False):
    model.eval()
    errs = []
    for s in range(0, len(seqs), batch):
        x = seqs[s:s + batch].to(device)
        m = motion[s:s + batch].to(device) if oracle else None
        pred, _ = model(x[:, :input_frames], pred_len, gt_motion=m)
        target = x[:, input_frames:input_frames + pred_len]
        errs.append((pred - target).pow(2).mean(dim=(2, 3, 4)).cpu())
    return torch.cat(errs).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--arms", nargs="+", default=["met", "tf"])
    p.add_argument("--root", default="data")
    p.add_argument("--n_sequences", type=int, default=5000)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    for arm in args.arms:
        for d in sorted(glob.glob(os.path.join(args.runs, f"{arm}_ms*"))):
            if not os.path.exists(os.path.join(d, "DONE")):
                print(f"skipping {d}: not finished")
                continue
            run_args = argparse.Namespace(**json.load(open(os.path.join(d, "results.json")))["args"])
            model = build_model(run_args)
            model.load_state_dict(torch.load(os.path.join(d, "best_model.pth"), map_location="cpu"))
            models[d] = (model.to(device), run_args.lift == "motion" and run_args.velocity == "oracle")
    print(f"{len(models)} checkpoints, {len(cells())} regimes, {args.n_sequences} sequences each, "
          f"device {device}", flush=True)

    t0 = time.time()
    for name, axis, level, kwargs in cells():
        seqs, motion = generate(args.root, kwargs, args.n_sequences)
        stats = motion_stats(motion)
        for d, (model, oracle) in models.items():
            err = score(model, seqs, motion, device, args.batch, oracle=oracle)
            out = os.path.join(d, "motion_factors")
            os.makedirs(out, exist_ok=True)
            np.savez_compressed(os.path.join(out, f"{slug(name)}.npz"), per_seq_err=err,
                                regime=name, axis=axis, level=level,
                                max_speed=kwargs["max_speed"], **{f"motion_{k}": v for k, v in stats.items()})
            print(f"[{time.time() - t0:6.0f}s] {name:<10} {os.path.basename(d):<10} rate {stats['rate']:.4f} "
                  f"|dv| {stats['mean_dv']:.2f}  median MSE t=16-25 {np.median(err.mean(1)):.3e}", flush=True)


if __name__ == "__main__":
    main()
