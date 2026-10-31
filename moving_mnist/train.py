"""Train a transformer, plain, flow or motion equivariant, on Moving MNIST next-frame prediction.

The protocol is MELSTM's (run_comparison.sh): 36x36 torus, two digits, piecewise-constant
velocities |v| <= 2 frozen from the last context step on; 15 context frames, 10 predicted,
fed back autoregressively; loss MSE + L1 on the predicted frames; 54k / 6k train / val
sequences per epoch, generated on the fly; Adam, cosine schedule, grad clip 1. The best-val
checkpoint is scored once, at the end, on the fixed test set (first n_test sequences of seed
123) and on 100-frame rollouts (seed 42), with per-sequence errors saved.

    python -m moving_mnist.train --lift motion --model_seed 1 --save_dir runs/met_ms1
"""
import argparse
import hashlib
import contextlib
import json
import math
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split

from metransformer.lift import make_lift
from metransformer.model import METransformer
from moving_mnist.data import LEN_GEN_SEED, TEST_SEED, fixed_set, make_dataset
from moving_mnist.velocity_metrics import summarize, velocity_matches
from moving_mnist.visualize import (boundary_age, lag_bias_figure, plt, rollout_figure,
                                    slot_states, state_figure, velocity_figure)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a = p.add_argument_group("data")
    a.add_argument("--root", default="data", help="torchvision MNIST root")
    a.add_argument("--image_size", type=int, default=36)
    a.add_argument("--seq_len", type=int, default=25)
    a.add_argument("--input_frames", type=int, default=15)
    a.add_argument("--gen_seq_len", type=int, default=100, help="length of the rollout benchmark")
    a.add_argument("--data_v_range", type=int, default=2)
    a.add_argument("--min_segment", type=int, default=3)
    a.add_argument("--max_segment", type=int, default=6)
    a.add_argument("--smooth_probability", type=float, default=0.8)
    a.add_argument("--data_seed", type=int, default=42, help="train/val split and the val probe")
    a.add_argument("--max_train_samples", type=int, default=None, help="smoke tests")
    a.add_argument("--max_val_samples", type=int, default=None, help="smoke tests")
    a.add_argument("--n_test", type=int, default=10000)
    a.add_argument("--n_len_gen", type=int, default=10000)

    m = p.add_argument_group("model")
    m.add_argument("--lift", choices=["none", "flow", "motion"], default="motion")
    m.add_argument("--n_slots", type=int, default=2, help="--lift motion")
    m.add_argument("--flow_v_range", type=int, default=2, help="--lift flow: |v| <= this")
    m.add_argument("--velocity", choices=["track", "pair", "oracle"], default="track")
    m.add_argument("--template", choices=["min", "mean"], default="min")
    m.add_argument("--decay", type=float, default=1.0, help="--template mean: EMA decay")
    m.add_argument("--pair_steps", type=int, default=4,
                   help="--velocity track: steps t < this use frame-pair velocities")
    m.add_argument("--dim", type=int, default=32)
    m.add_argument("--depth", type=int, default=2)
    m.add_argument("--heads", type=int, default=4)
    m.add_argument("--mlp_hidden", type=int, default=48)
    m.add_argument("--kernel", type=int, default=3)
    m.add_argument("--attn", choices=["pixel", "frame"], default="pixel")
    m.add_argument("--max_dist", type=int, default=16, help="time lags >= this share a bias")
    m.add_argument("--window", type=int, default=None, help="attend to the last N frames only")
    m.add_argument("--slot_reduce", choices=["max", "mean", "sum"], default="max")
    m.add_argument("--decoder_hidden", type=int, default=32)
    m.add_argument("--detach_feedback", action="store_true")
    m.add_argument("--checkpoint", action="store_true",
                   help="recompute block activations in the backward pass (less memory)")

    o = p.add_argument_group("optimization")
    o.add_argument("--epochs", type=int, default=50)
    o.add_argument("--batch_size", type=int, default=32)
    o.add_argument("--eval_batch_size", type=int, default=64)
    o.add_argument("--gen_batch_size", type=int, default=16,
                   help="the 100-frame rollout, whose key/value cache grows to 100 frames")
    o.add_argument("--lr", type=float, default=1e-3)
    o.add_argument("--final_lr", type=float, default=1e-5)
    o.add_argument("--warmup_epochs", type=float, default=1.0)
    o.add_argument("--weight_decay", type=float, default=0.0)
    o.add_argument("--grad_clip", type=float, default=1.0)
    o.add_argument("--context_loss_weight", type=float, default=0.0,
                   help="weight of an extra loss on the predictions of context frames 1..T-1")
    o.add_argument("--model_seed", type=int, required=True, help="init, shuffle, training stream")
    o.add_argument("--init_seed", type=int, default=None, help="overrides the init only")
    o.add_argument("--amp", action="store_true", help="bfloat16 autocast (CUDA)")
    o.add_argument("--num_workers", type=int, default=4)

    g = p.add_argument_group("logging")
    g.add_argument("--save_dir", required=True)
    g.add_argument("--wandb_project", default="METransformer")
    g.add_argument("--wandb_name", default=None)
    g.add_argument("--wandb_group", default=None, help="e.g. the arm, to group its seeds")
    g.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    g.add_argument("--log_every", type=int, default=25, help="batches between train-curve points")
    g.add_argument("--val_probe_size", type=int, default=256)
    g.add_argument("--val_probe_every", type=int, default=50, help="batches; 0 = off")
    g.add_argument("--image_every", type=int, default=5, help="epochs between prediction strips")
    args = p.parse_args(argv)
    if args.input_frames >= args.seq_len:
        p.error("--input_frames must be < --seq_len")
    if args.velocity == "oracle" and args.n_slots != 2:
        p.error("--velocity oracle needs --n_slots 2 (one slot per digit)")
    return args


def code_version():
    """-> (git commit, dirty flag, hash of the package and benchmark sources). The hash
    identifies the code also where it was copied without its git history."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sha = hashlib.sha1()
    for pkg in ("metransformer", "moving_mnist"):
        for name in sorted(os.listdir(os.path.join(root, pkg))):
            if name.endswith(".py"):
                with open(os.path.join(root, pkg, name), "rb") as f:
                    sha.update(name.encode() + f.read())
    try:
        run = lambda *c: subprocess.run(["git", "-C", root, *c], capture_output=True,
                                        text=True, check=True).stdout.strip()
        commit, dirty = run("rev-parse", "--short", "HEAD"), bool(run("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = "none", True
    return commit, dirty, sha.hexdigest()[:12]


def build_model(args):
    lift = make_lift(args.lift, args.n_slots, args.flow_v_range, args.velocity, args.template,
                     args.decay, args.pair_steps)
    return METransformer(lift, in_channels=1, dim=args.dim, depth=args.depth, heads=args.heads,
                         mlp_hidden=args.mlp_hidden, kernel=args.kernel, attn=args.attn,
                         max_dist=args.max_dist, window=args.window,
                         slot_reduce=args.slot_reduce, decoder_hidden=args.decoder_hidden,
                         detach_feedback=args.detach_feedback, checkpoint=args.checkpoint)


def save_atomic(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


class Runner:
    """The model call shared by training and evaluation."""

    def __init__(self, model, args, device):
        self.model, self.args, self.device = model, args, device
        self.oracle = args.lift == "motion" and args.velocity == "oracle"
        self.amp = args.amp and device.type == "cuda"

    def __call__(self, seq, motion, pred_len, context_predictions=False):
        ctx = seq[:, :self.args.input_frames]
        amp = (torch.autocast("cuda", dtype=torch.bfloat16) if self.amp
               else contextlib.nullcontext())
        with amp:
            pred, aux = self.model(ctx, pred_len, gt_motion=motion if self.oracle else None,
                                   context_predictions=context_predictions)
        return pred.float(), aux


def frame_errors(pred, target):
    """-> per-sequence, per-frame MSE and L1, each (B, T)."""
    d = pred - target
    return d.pow(2).mean(dim=(2, 3, 4)), d.abs().mean(dim=(2, 3, 4))


@torch.no_grad()
def evaluate(runner, batches, pred_len, with_velocity, keep=0):
    """batches: iterable of (seq, motion) on the CPU -> dict of per-sequence arrays: MSE and L1
    per frame, the boundary age of each digit, the velocity matches; with keep, also the
    frames, motions and predictions of the first `keep` sequences."""
    runner.model.eval()
    mse, l1, ages, matches, kept = [], [], [], [], None
    for seq, motion in batches:
        seq, motion = seq.to(runner.device), motion.to(runner.device)
        target = seq[:, runner.args.input_frames:runner.args.input_frames + pred_len]
        pred, aux = runner(seq, motion, pred_len)
        e2, e1 = frame_errors(pred, target)
        mse.append(e2.cpu())
        l1.append(e1.cpu())
        ages.append(boundary_age(motion, runner.args.input_frames).cpu())
        if with_velocity:
            matches.append(velocity_matches(aux["U"], motion))
        if keep and kept is None:
            kept = (seq[:keep].cpu(), motion[:keep].cpu(), pred[:keep].cpu())
    out = {"mse": torch.cat(mse).numpy(), "l1": torch.cat(l1).numpy(),
           "boundary_age": torch.cat(ages).numpy()}
    if matches:
        out["velocity_matches"] = torch.cat(matches).numpy()
    if kept is not None:
        out["kept_seq"], out["kept_motion"], out["kept_pred"] = (k.numpy() for k in kept)
    return out


def fixed_batches(dataset, n, batch_size):
    """The first n sequences of a fixed dataset's stream, drawn here batch by batch."""
    dataset.reset_rng()
    for s in range(0, n, batch_size):
        items = [dataset[i] for i in range(min(batch_size, n - s))]
        yield torch.stack([x for x, _, _ in items]), torch.stack([m for _, _, m in items])


def loader_batches(loader):
    for seq, _, motion in loader:
        yield seq, motion


def chunks(data, batch_size):
    seq, motion = data
    for s in range(0, len(seq), batch_size):
        yield seq[s:s + batch_size], motion[s:s + batch_size]


def save_figures(figs, fig_dir, prefix=""):
    """Save matplotlib figures as PNGs (overwriting earlier ones) -> {name: wandb.Image}."""
    import wandb
    os.makedirs(fig_dir, exist_ok=True)
    out = {}
    for name, fig in figs.items():
        path = os.path.join(fig_dir, f"{prefix}{name}.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        out[name] = wandb.Image(path)
    return out


def diagnostics(runner, args, probe, probe_long, pred_len, epoch, val_matches):
    """Figures of the model's state on fixed probe sequences, for an image epoch."""
    import wandb
    model, device, T = runner.model, runner.device, args.input_frames
    model.eval()
    with torch.no_grad():
        pred, _ = runner(probe[0][:4].to(device), probe[1][:4].to(device), pred_len)
        long_seq, long_motion = probe_long
        long_pred, _ = runner(long_seq.to(device), long_motion.to(device), long_seq.shape[1] - T)
    long_pred = long_pred.cpu()
    states = slot_states(model, probe[0][0].to(device), probe[1][0].to(device), T, pred_len,
                         oracle=runner.oracle)
    figs = {
        "probe_strip": rollout_figure(probe[0][:4, :, 0].numpy(), pred.cpu()[:, :, 0].numpy(), T,
                                      every=1, title=f"val probe, epoch {epoch}"),
        "slot_states": state_figure(states, T, title=f"val probe 0, epoch {epoch}"),
        "extrapolation": rollout_figure(long_seq[:, :, 0].numpy(), long_pred[:, :, 0].numpy(), T,
                                        every=5, title=f"100-frame rollout, epoch {epoch}"),
        "lag_bias": lag_bias_figure(model),
    }
    if val_matches is not None:
        figs["velocity_per_step"] = velocity_figure(
            summarize(torch.from_numpy(val_matches))["vel_acc_per_step"])
    log = save_figures(figs, os.path.join(args.save_dir, "figures"))
    long_mse = (long_pred - long_seq[:, T:]).pow(2).mean(dim=(0, 2, 3, 4)).tolist()
    log["extrapolation_mse"] = wandb.plot.line_series(
        xs=list(range(T + 1, long_seq.shape[1] + 1)), ys=[long_mse], keys=["4 probe sequences"],
        title="rollout MSE on the 100-frame probe", xname="frame t")
    return log


def final_figures(runner, args, test, gen, pred_len):
    """Figures of the best checkpoint on the test and 100-frame sets, and horizon curves."""
    import wandb
    model, device, T = runner.model, runner.device, args.input_frames
    figs = {}
    for name, d, every in (("test", test, 1), ("len_gen", gen, 5)):
        seq = torch.from_numpy(d["kept_seq"][0]).to(device)
        motion = torch.from_numpy(d["kept_motion"][0]).to(device)
        states = slot_states(model, seq, motion, T, seq.shape[0] - T, oracle=runner.oracle)
        figs[f"final_slot_states_{name}"] = state_figure(states, T, title=f"best model, {name} 0",
                                                         every=every)
    figs["final_extrapolation"] = rollout_figure(gen["kept_seq"][:, :, 0], gen["kept_pred"][:, :, 0],
                                                 T, every=5, title="best model, 100-frame set")
    log = save_figures(figs, os.path.join(args.save_dir, "figures"))
    t = list(range(T + 1, T + gen["mse"].shape[1] + 1))
    log["final_rollout_mse"] = wandb.plot.line_series(
        xs=t, ys=[gen["mse"].mean(0).tolist(), np.median(gen["mse"], 0).tolist()],
        keys=["mean", "median"], title="rollout MSE, 100-frame set", xname="frame t")
    return log


def main(argv=None):
    args = parse_args(argv)
    commit, dirty, code_hash = code_version()
    os.makedirs(args.save_dir, exist_ok=True)
    print(f">>> commit {commit}{' (dirty)' if dirty else ''}  code {code_hash}  lift={args.lift}  "
          f"velocity={args.velocity}  model_seed={args.model_seed}  init_seed={args.init_seed}  "
          f"data_seed={args.data_seed}  save_dir={args.save_dir}", flush=True)
    print(">>> args " + json.dumps(vars(args)), flush=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    if device.type == "mps":  # torch.fft on MPS warns about an internal resize on every call
        warnings.filterwarnings("ignore", message="An output with one or more elements was resized")
    ckpt_path = os.path.join(args.save_dir, "checkpoint.pth")
    best_path = os.path.join(args.save_dir, "best_model.pth")
    resume = torch.load(ckpt_path, map_location="cpu", weights_only=False) if os.path.exists(ckpt_path) else None
    if resume is not None and resume["args"] != vars(args):
        changed = {k: (resume["args"].get(k), v) for k, v in vars(args).items()
                   if resume["args"].get(k) != v}
        sys.exit(f"{ckpt_path} was written with other arguments {changed}; "
                 f"use another --save_dir or delete it")

    import wandb
    run = wandb.init(project=args.wandb_project, name=args.wandb_name, group=args.wandb_group,
                     mode=args.wandb_mode, dir=args.save_dir,
                     id=resume["wandb_id"] if resume else None,
                     resume="allow" if resume else None,
                     config={**vars(args), "git_commit": commit, "git_dirty": dirty,
                             "code_hash": code_hash})

    pred_len = args.seq_len - args.input_frames
    data_kw = dict(image_size=args.image_size, v_range=args.data_v_range,
                   min_segment=args.min_segment, max_segment=args.max_segment,
                   smooth_probability=args.smooth_probability)
    full = make_dataset(args.root, True, args.seq_len, args.input_frames, random=True, **data_kw)
    n_val = len(full) // 10
    train_ds, val_ds = random_split(full, [len(full) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(args.data_seed))
    if args.max_train_samples:
        train_ds = Subset(train_ds, range(min(args.max_train_samples, len(train_ds))))
    if args.max_val_samples:
        val_ds = Subset(val_ds, range(min(args.max_val_samples, len(val_ds))))
    probe = fixed_set(make_dataset(args.root, True, args.seq_len, args.input_frames, random=False,
                                   seed=args.data_seed, **data_kw), args.val_probe_size)
    probe_long = fixed_set(make_dataset(args.root, True, args.gen_seq_len, args.input_frames,
                                        random=False, seed=args.data_seed + 1, **data_kw), 4)

    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)
    if args.init_seed is not None:
        rng = torch.get_rng_state()
        torch.manual_seed(args.init_seed)
    model = build_model(args)
    if args.init_seed is not None:
        torch.set_rng_state(rng)
    model.to(device)
    runner = Runner(model, args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}  ("
          + ", ".join(f"{n} {sum(p.numel() for p in c.parameters()):,}"
                      for n, c in model.named_children() if any(True for _ in c.parameters()))
          + f")  slots: {model.lift.n_slots}  device: {device}", flush=True)
    run.config.update({"num_parameters": n_params}, allow_val_change=True)

    loader_kw = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, **loader_kw)
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup = max(1, round(args.warmup_epochs * steps_per_epoch))

    def lr_at(step):
        if step < warmup:
            return args.lr * (step + 1) / warmup
        prog = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return args.final_lr + 0.5 * (args.lr - args.final_lr) * (1 + math.cos(math.pi * prog))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_mse": [], "epoch_time_sec": [],
               "val_vel_acc": [], "val_vel_acc_boundary": [], "probe_step": [], "probe_loss": [],
               "train_step": [], "train_curve": [], "steps_per_epoch": None}
    start_epoch, best_val, best_epoch = 1, float("inf"), None
    if resume is not None:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        history, best_val, best_epoch = resume["history"], resume["best_val"], resume["best_epoch"]
        start_epoch = resume["epoch"] + 1
        torch.set_rng_state(resume["torch_rng"])
        print(f">>> resumed after epoch {resume['epoch']} (best val {best_val:.4e})", flush=True)

    with_velocity = args.lift == "motion" and args.velocity != "oracle"
    use_ctx = args.context_loss_weight > 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, running, n_seen = time.time(), 0.0, 0
        for i, (seq, _, motion) in enumerate(train_loader):
            step = (epoch - 1) * steps_per_epoch + i
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step)
            seq, motion = seq.to(device, non_blocking=True), motion.to(device, non_blocking=True)
            target = seq[:, args.input_frames:]
            pred, aux = runner(seq, motion, pred_len, context_predictions=use_ctx)
            loss = F.mse_loss(pred, target) + F.l1_loss(pred, target)
            if use_ctx:
                ctx_pred, ctx_tgt = aux["context_pred"].float(), seq[:, 1:args.input_frames]
                loss = loss + args.context_loss_weight * (F.mse_loss(ctx_pred, ctx_tgt)
                                                          + F.l1_loss(ctx_pred, ctx_tgt))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += loss.item() * seq.shape[0]
            n_seen += seq.shape[0]

            if (step + 1) % args.log_every == 0:
                history["train_step"].append(step + 1)
                history["train_curve"].append(loss.item())
                wandb.log({"train_loss_step": loss.item(), "grad_norm": float(grad_norm),
                           "lr": lr_at(step), "global_step": step + 1})
            if args.val_probe_every and (step + 1) % args.val_probe_every == 0:
                r = evaluate(runner, chunks(probe, args.eval_batch_size), pred_len, False)
                probe_loss = float((r["mse"] + r["l1"]).mean())
                history["probe_step"].append(step + 1)
                history["probe_loss"].append(probe_loss)
                wandb.log({"val_probe_loss": probe_loss, "global_step": step + 1})
                model.train()
        epoch_time = time.time() - t0

        t0 = time.time()
        r = evaluate(runner, loader_batches(val_loader), pred_len, with_velocity)
        val_time = time.time() - t0
        val_mse, val_loss = float(r["mse"].mean()), float((r["mse"] + r["l1"]).mean())
        train_loss = running / max(1, n_seen)
        for k, v in (("epoch", epoch), ("train_loss", train_loss), ("val_loss", val_loss),
                     ("val_mse", val_mse), ("epoch_time_sec", epoch_time)):
            history[k].append(v)
        log = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_mse": val_mse,
               "epoch_time_sec": epoch_time, "global_step": epoch * steps_per_epoch}
        history["steps_per_epoch"] = steps_per_epoch
        vel = ""
        if "velocity_matches" in r:
            acc = summarize(torch.from_numpy(r["velocity_matches"]))
            log.update(val_vel_acc=acc["vel_acc"], val_vel_acc_boundary=acc["vel_acc_boundary"])
            history["val_vel_acc"].append(acc["vel_acc"])
            history["val_vel_acc_boundary"].append(acc["vel_acc_boundary"])
            vel = f" | vel {100 * acc['vel_acc']:.2f}% (boundary {100 * acc['vel_acc_boundary']:.2f}%)"
        if device.type == "cuda":
            log["max_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
            vel += f" | peak {log['max_memory_gb']:.1f} GB"
        if args.image_every and (epoch % args.image_every == 0 or epoch == 1):
            log.update(diagnostics(runner, args, probe, probe_long, pred_len, epoch,
                                   r.get("velocity_matches")))
        improved = val_loss < best_val
        if improved:
            best_val, best_epoch = val_loss, epoch
            save_atomic(model.state_dict(), best_path)
        log.update(best_val_loss=best_val, best_epoch=best_epoch)
        wandb.log(log)
        print(f"epoch {epoch}/{args.epochs} | train {train_loss:.4e} | val {val_loss:.4e} "
              f"(MSE {val_mse:.4e}){vel} | train {epoch_time:.0f}s, val {val_time:.0f}s"
              f"{' | best' if improved else ''}", flush=True)
        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "best_val": best_val, "best_epoch": best_epoch,
                     "history": history, "args": vars(args), "wandb_id": run.id,
                     "torch_rng": torch.get_rng_state()}, ckpt_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    data_kw_test = dict(data_kw, random=False)
    t0 = time.time()
    test = evaluate(runner, fixed_batches(
        make_dataset(args.root, False, args.seq_len, args.input_frames, seed=TEST_SEED,
                     **data_kw_test), args.n_test, args.eval_batch_size), pred_len, with_velocity,
        keep=4)
    gen = evaluate(runner, fixed_batches(
        make_dataset(args.root, False, args.gen_seq_len, args.input_frames, seed=LEN_GEN_SEED,
                     **data_kw_test), args.n_len_gen, args.gen_batch_size),
        args.gen_seq_len - args.input_frames, with_velocity, keep=3)
    summary = {"best_epoch": best_epoch, "best_val_loss": best_val,
               "test_mse": float(test["mse"].mean()), "test_l1": float(test["l1"].mean()),
               "test_mse_median": float(np.median(test["mse"].mean(1))),
               "len_gen_mse_t25": float(gen["mse"][:, pred_len - 1].mean()),
               "len_gen_mse_t100": float(gen["mse"][:, -1].mean()),
               "len_gen_mse_mean": float(gen["mse"].mean()),
               "num_parameters": n_params, "git_commit": commit, "git_dirty": dirty,
               "code_hash": code_hash,
               "eval_seconds": time.time() - t0}
    if "velocity_matches" in test:
        acc = summarize(torch.from_numpy(test["velocity_matches"]))
        summary.update(test_vel_acc=acc["vel_acc"], test_vel_acc_boundary=acc["vel_acc_boundary"],
                       test_vel_both_boundary=acc["vel_both_boundary"])
    np.savez_compressed(
        os.path.join(args.save_dir, "test_errors.npz"),
        test_mse=test["mse"], test_l1=test["l1"], test_boundary_age=test["boundary_age"],
        len_gen_mse=gen["mse"], len_gen_boundary_age=gen["boundary_age"],
        strip_seq=gen["kept_seq"], strip_pred=gen["kept_pred"],
        **{f"{split}_velocity_matches": d["velocity_matches"]
           for split, d in (("test", test), ("len_gen", gen)) if "velocity_matches" in d})
    with open(os.path.join(args.save_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "history": history, "args": vars(args)}, f, indent=1)
    run.summary.update(summary)
    wandb.log(final_figures(runner, args, test, gen, pred_len))
    print(">>> " + "  ".join(f"{k} {v:.4e}" if isinstance(v, float) else f"{k} {v}"
                             for k, v in summary.items()), flush=True)
    open(os.path.join(args.save_dir, "DONE"), "w").close()
    wandb.finish()


if __name__ == "__main__":
    main()
