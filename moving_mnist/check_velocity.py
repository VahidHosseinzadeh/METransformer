"""Velocity models and parameter-free predictors on the Moving MNIST test set.

For every velocity model: the accuracy of its slot velocities, at all context steps and at
the boundary (the velocity frozen for the rollout), and the test MSE of a parameter-free
predictor that renders each slot's co-moving template (the mean or the pixel-wise minimum
of its context frames) along the frozen path, max-pooled over the slots. With the copy-last
baseline, this says how much of the task the lift alone solves before anything is trained.

    python -m moving_mnist.check_velocity --root data --n 2000
"""
import argparse
import time

import torch

from metransformer.lift import MotionLift, comoving_template
from metransformer.ops import torus_shift
from moving_mnist.data import LEN_GEN_SEED, TEST_SEED, fixed_set, make_dataset
from moving_mnist.velocity_metrics import summarize, velocity_matches

VELOCITY_MODELS = {
    "pair": dict(velocity="pair"),
    **{f"track{s}": dict(velocity="track", template="mean", pair_steps=s) for s in (2, 3, 4, 5)},
    **{f"track{s}_min": dict(velocity="track", template="min", pair_steps=s) for s in (2, 3, 4, 5)},
    "oracle": dict(velocity="oracle"),
}
DEFAULT = "track4_min"


def transport(template, P_last, u, steps):
    """Render the slot templates (B, K, C, H, W) at P_last + j u, j = 1..steps; max over slots."""
    j = torch.arange(1, steps + 1)[None, None, :, None]
    paths = P_last[:, :, None] + j * u[:, :, None]
    frames = torus_shift(template[:, :, None].expand(-1, -1, steps, -1, -1, -1), paths)
    return frames.max(dim=1).values


def evaluate(x, motion, input_frames, batch, horizons):
    ctx_len = input_frames
    target = x[:, ctx_len:]
    steps = target.shape[1]
    rows = {}
    copy_last = ((x[:, ctx_len - 1:ctx_len] - target) ** 2).mean(dim=(2, 3, 4))
    rows["copy last frame"] = {"mse_t": copy_last.mean(0)}
    for name, kw in VELOCITY_MODELS.items():
        lift = MotionLift(2, **kw)
        matches, mse = [], {"mean": [], "min": []}
        for s in range(0, len(x), batch):
            xb, mb = x[s:s + batch, :ctx_len], motion[s:s + batch]
            P, u, U = lift(xb, mb)
            matches.append(velocity_matches(U, mb))
            for kind in mse:
                pred = transport(comoving_template(xb, P, kind), P[:, :, -1], u, steps)
                mse[kind].append(((pred - target[s:s + batch]) ** 2).mean(dim=(2, 3, 4)))
        acc = summarize(torch.cat(matches))
        for kind in mse:
            rows[f"{name} + {kind} template"] = {"mse_t": torch.cat(mse[kind]).mean(0), **acc}
    print(f"{'predictor':28s} {'vel acc':>8s} {'boundary':>9s} {'both@bnd':>9s} "
          + " ".join(f"{'MSE t=' + str(ctx_len + h):>12s}" for h in horizons)
          + f" {'MSE t=' + str(ctx_len + 1) + '..' + str(ctx_len + min(10, steps)):>15s}")
    for name, r in rows.items():
        accs = (f"{100 * r['vel_acc']:7.2f}% {100 * r['vel_acc_boundary']:8.2f}% "
                f"{100 * r['vel_both_boundary']:8.2f}%" if "vel_acc" in r else " " * 28)
        mse = " ".join(f"{r['mse_t'][h - 1]:12.3e}" for h in horizons)
        print(f"{name:28s} {accs} {mse} {r['mse_t'][:10].mean():15.3e}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data")
    p.add_argument("--split", choices=["test", "train"], default="test",
                   help="MNIST digits of the sequences; choose settings on train, confirm on test")
    p.add_argument("--seed", type=int, default=None,
                   help=f"stream seed (default {TEST_SEED} for test, 42 for train)")
    p.add_argument("--n", type=int, default=2000, help="sequences: the first n of the stream")
    p.add_argument("--n_len_gen", type=int, default=500, help="100-frame sequences (seed 42)")
    p.add_argument("--batch", type=int, default=250)
    p.add_argument("--input_frames", type=int, default=15)
    args = p.parse_args()
    seed = args.seed if args.seed is not None else (TEST_SEED if args.split == "test" else 42)

    t0 = time.time()
    x, motion = fixed_set(make_dataset(args.root, args.split == "train", 25, args.input_frames,
                                       random=False, seed=seed), args.n)
    print(f"{args.split} split: first {args.n} sequences of seed {seed} ({time.time() - t0:.0f}s)")
    rows = evaluate(x, motion, args.input_frames, args.batch, horizons=(1, 10))
    print(f"per-step velocity accuracy, {DEFAULT} (t = 1..{args.input_frames - 1}):",
          " ".join(f"{100 * a:.1f}" for a in rows[f"{DEFAULT} + min template"]["vel_acc_per_step"]))

    if args.n_len_gen:
        t0 = time.time()
        x, motion = fixed_set(make_dataset(args.root, False, 100, args.input_frames, random=False,
                                           seed=LEN_GEN_SEED), args.n_len_gen)
        print(f"\nlength generalization: first {args.n_len_gen} sequences of seed {LEN_GEN_SEED}, "
              f"100 frames ({time.time() - t0:.0f}s)")
        evaluate(x, motion, args.input_frames, args.batch, horizons=(1, 10, 85))


if __name__ == "__main__":
    main()
