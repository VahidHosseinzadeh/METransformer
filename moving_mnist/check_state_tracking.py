"""Slot-state tracking on trained MET checkpoints, without retraining.

Each slot tracks its digit with its own transformer state, the MET counterpart of MELSTM's
track_mode="h": at step t, the slot's last-block output at t-1 (its prediction of frame t, in
its co-moving frame) is placed at P^k_{t-1} and phase-correlated against X_t, which gives
u^k_t. The context pass becomes sequential, one token per step with the key/value cache.
Steps t < pair_steps use frame-pair velocities, as the default tracker does.

Templates made from the state h^k_{t-1}:
  mean   its channel mean (MELSTM tracks with the channel mean of h). In the trained MET
         checkpoints it is anti-correlated with the slot's digit (r = -0.89), so the
         phase-correlation peak becomes a trough and this template fails
  nmean  the channel mean after the read-out norm N_f (r = +0.97 with the rendered slot)
  dec    the slot rendered alone: decoder(N_f(h^k_{t-1})), without the max over slots

Compared on the same sequences with the checkpoint's own tracker (track4_min) and with the
dataset velocities (oracle). The prediction protocol is unchanged: frames 16-25 from 15
context frames, the paths continued at the last context velocity.

    python -m moving_mnist.check_state_tracking --runs runs/met_ms1 --n 500
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from metransformer.ops import torus_shift
from moving_mnist.data import TEST_SEED, fixed_set, make_dataset
from moving_mnist.motion_sweep import cells, generate
from moving_mnist.train import build_model
from moving_mnist.velocity_metrics import velocity_matches

LOST = 1e-2


def state_template(model, h, P, kind):
    """h (B, K, D, H, W) slot states at t-1 in the slot frames, P (B, K, 2) = P_{t-1}
    -> templates (B, K, 1, H, W) in the coordinates of frame t-1."""
    B, K, D, H, W = h.shape
    if kind == "mean":
        tm = h.mean(dim=2, keepdim=True)
    elif kind == "nmean":
        tm = model.norm(h).mean(dim=2, keepdim=True)
    elif kind == "dec":
        tm = model.decoder(model.norm(h).reshape(B * K, D, H, W)).view(B, K, -1, H, W)
    else:
        raise ValueError(kind)
    return torus_shift(tm, P)


@torch.no_grad()
def predict(model, x, pred_len, mode, pair_steps=4, U_given=None):
    """Sequential context pass and rollout. mode: 'given' (velocities U_given), or a state
    template kind. -> predictions (B, pred_len, C, H, W), U (B, K, T, 2)."""
    lift = model.lift
    B, T, C, H, W = x.shape
    K = lift.n_slots
    f = x.mean(dim=2, keepdim=True)
    U = torch.zeros(B, K, T, 2, dtype=torch.long, device=x.device)
    P = torch.zeros(B, K, 2, dtype=torch.long, device=x.device)
    caches, h = None, None
    for t in range(T):
        if mode == "given":
            U[:, :, t] = U_given[:, :, t]
        elif t == 1:
            U[:, :, 1] = lift.pc_boot(x[:, 0], x[:, 1])[0]
        elif 2 <= t < pair_steps:
            U[:, :, t] = lift._pair_step(x, U, t)
        elif t >= 2:
            tmpl = state_template(model, h, P, mode).reshape(B * K, 1, H, W)
            frame = f[:, t, None].expand(B, K, 1, H, W).reshape(B * K, 1, H, W)
            U[:, :, t] = lift.pc_track(tmpl, frame)[0].view(B, K, 2)
        P = P + U[:, :, t]
        out, caches = model.transformer(model.tokens(x[:, t:t + 1], P[:, :, None]), caches, t)
        h = out.view(B, K, *out.shape[2:])
    u, preds = U[:, :, -1], []
    for j in range(pred_len):
        P = P + u
        frame = model.read(h, P)
        preds.append(frame)
        if j + 1 == pred_len:
            break
        out, caches = model.transformer(model.tokens(frame[:, None], P[:, :, None]), caches, T + j)
        h = out.view(B, K, *out.shape[2:])
    return torch.stack(preds, dim=1), U


def boundary_set_correct(U, motion):
    """The two slot velocities at the last context step equal the two digit velocities as a
    set (a slot swap is harmless: every slot carries the whole frame)."""
    T = U.shape[2]
    p, m = U[:, :, -1].cpu(), motion[:, T - 2].long().cpu()
    eq = lambda a, b: (a == b).all(dim=-1)
    return ((eq(p[:, 0], m[:, 0]) & eq(p[:, 1], m[:, 1]))
            | (eq(p[:, 0], m[:, 1]) & eq(p[:, 1], m[:, 0])))


@torch.no_grad()
def score(model, x, motion, arms, input_frames, pred_len, batch, device):
    """-> {arm: dict(mse (n,), set_ok (n,), vel_acc (n, T-1, 2))}."""
    res = {a: {"mse": [], "set_ok": [], "vel": []} for a in arms}
    for s in range(0, len(x), batch):
        xb = x[s:s + batch].to(device)
        mb = motion[s:s + batch].to(device)
        ctx, target = xb[:, :input_frames], xb[:, input_frames:input_frames + pred_len]
        for arm in arms:
            if arm == "default":
                pred, aux = model(ctx, pred_len)
                U = aux["U"]
            elif arm == "oracle":
                U_or = torch.zeros(len(xb), 2, input_frames, 2, dtype=torch.long, device=device)
                U_or[:, :, 1:] = mb[:, :input_frames - 1].transpose(1, 2).long()
                pred, U = predict(model, ctx, pred_len, "given", U_given=U_or)
            else:
                kind, steps = arm.split("@")
                pred, U = predict(model, ctx, pred_len, kind, pair_steps=int(steps))
            r = res[arm]
            r["mse"].append((pred - target).pow(2).mean(dim=(1, 2, 3, 4)).cpu())
            r["set_ok"].append(boundary_set_correct(U, mb))
            r["vel"].append(velocity_matches(U, mb))
    return {a: {k: torch.cat(v).numpy() for k, v in r.items()} for a, r in res.items()}


def line(arm, r):
    mse, ok = r["mse"], r["set_ok"]
    lost = mse > LOST
    return (f"{arm:12s} mean {mse.mean():.3e}  median {np.median(mse):.2e}  lost {100 * lost.mean():5.2f}%"
            f"  boundary set wrong {100 * (1 - ok.mean()):5.2f}%  step acc {100 * r['vel'].mean():6.2f}%"
            f"  mean|set ok {mse[ok].mean() if ok.any() else float('nan'):.2e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["runs/met_ms1", "runs/met_ms2", "runs/met_ms3"])
    p.add_argument("--arms", nargs="+",
                   default=["default", "oracle", "mean@4", "dec@4"],
                   help="default, oracle, or <template>@<pair_steps> with template mean|dec")
    p.add_argument("--data", choices=["test", "sweep"], default="test")
    p.add_argument("--regimes", nargs="+", default=["centre", "A:d=1.00", "C:N=1"],
                   help="--data sweep: regimes of motion_sweep.cells()")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--root", default="data")
    p.add_argument("--cache", default=None, help="folder for the generated sequences")
    p.add_argument("--out", default=None, help="json file for the per-arm summaries")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)
    input_frames, pred_len = 15, 10

    def load(name, make):
        path = os.path.join(args.cache, f"{name}_{args.n}.pt") if args.cache else None
        if path and os.path.exists(path):
            return torch.load(path)
        data = make()
        if path:
            os.makedirs(args.cache, exist_ok=True)
            torch.save(data, path)
        return data

    if args.data == "test":
        sets = {"test": load("test", lambda: fixed_set(make_dataset(
            args.root, False, 25, input_frames, random=False, seed=TEST_SEED), args.n))}
    else:
        kw = {name: k for name, _, _, k in cells()}
        sets = {r: load(r.replace(":", "_").replace("=", ""),
                        lambda r=r: tuple(t[:, :25] for t in generate(args.root, kw[r], args.n)))
                for r in args.regimes}

    summary = {}
    for run in args.runs:
        run_args = argparse.Namespace(**json.load(open(os.path.join(run, "results.json")))["args"])
        model = build_model(run_args)
        model.load_state_dict(torch.load(os.path.join(run, "best_model.pth"), map_location="cpu"))
        model.to(device).eval()
        for name, (x, motion) in sets.items():
            t0 = time.time()
            res = score(model, x, motion, args.arms, input_frames, pred_len, args.batch, device)
            print(f"\n{os.path.basename(run)}  {name}  ({len(x)} sequences, {time.time() - t0:.0f}s)")
            for arm, r in res.items():
                print(line(arm, r), flush=True)
                summary[f"{os.path.basename(run)}|{name}|{arm}"] = {
                    "mean": float(r["mse"].mean()), "median": float(np.median(r["mse"])),
                    "lost": float((r["mse"] > LOST).mean()),
                    "set_wrong": float(1 - r["set_ok"].mean()),
                    "step_acc": float(r["vel"].mean()),
                    "per_step_acc": r["vel"].mean(axis=(0, 2)).tolist(),
                    "lost_given_set_ok": float((r["mse"][r["set_ok"]] > LOST).mean()),
                    "lost_given_set_wrong": float((r["mse"][~r["set_ok"]] > LOST).mean())
                    if (~r["set_ok"]).any() else None}
    if args.out:
        json.dump(summary, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
