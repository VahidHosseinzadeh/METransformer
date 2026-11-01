"""MELSTM-style figures and table for METransformer runs, with MELSTM's runs as references.

    python -m moving_mnist.make_figures --runs runs --out figures \
        --reference ~/Documents/Repos/FEConvLSTM/cluster_runs/pure_h

--runs holds one directory per run, <arm>_ms<seed>, with results.json and test_errors.npz.
--reference holds MELSTM's paper runs, ms<seed>/results/{history,len_gen}_<model>_<id>.*.
MELSTM's loaders drew the 100-frame set in 4 workers that each restarted the stream, so its
rows are 4 copies of the first 2,512 sequences of our 100-frame set. Comparisons with the
references therefore use exactly those 2,512 sequences; the script checks that the saved
strips agree frame for frame.
"""
import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Reference categorical palette, slots 1-5 in fixed order; MELSTM's models are dashed.
STYLE = {
    "met": dict(label="MET", color="#2a78d6", ls="-"),
    "tf": dict(label="Transformer", color="#eb6834", ls="-"),
    "melstm": dict(label="MELSTM", color="#1baf7a", ls="--"),
    "felstm": dict(label="FELSTM", color="#eda100", ls="--"),
    "lstm": dict(label="LSTM", color="#e87ba4", ls="--"),
}
OURS, REFS = ("met", "tf"), ("melstm", "felstm", "lstm")
N_SHARED = 2512  # unique sequences in MELSTM's 100-frame set
FAIL = 1e-2  # a sequence whose MSE over t=16-25 exceeds this has lost a digit
MELSTM_TABLE = {  # MELSTM paper, Fig. 2 table: test MSE t=16-25 and t=100, s/epoch (A100)
    "lstm": ("3.26e-2 ± 0.05e-2", "2.01e-1 ± 1.34e-1", "77"),
    "felstm": ("1.54e-3 ± 0.42e-3", "8.83e-2 ± 6.58e-2", "918"),
    "melstm": ("2.87e-4 ± 2.20e-4", "1.16e-3 ± 1.05e-3", "157"),
}


def load_runs(root):
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "*_ms*"))):
        m = re.fullmatch(r"(.+)_ms(\d+)", os.path.basename(d))
        if not m or not os.path.exists(os.path.join(d, "test_errors.npz")):
            continue
        res = json.load(open(os.path.join(d, "results.json")))
        err = dict(np.load(os.path.join(d, "test_errors.npz")))
        runs.setdefault(m.group(1), []).append(dict(seed=int(m.group(2)), dir=d, **res, **err))
    return runs


def dedupe_rows(n_rows, batch=32, workers=4):
    """Row indices of MELSTM's 4-worker evaluation that hold stream sequences 0..N-1, in order."""
    rows = []
    for r in range(n_rows):
        b, i = divmod(r, batch)
        if b % workers == 0:
            rows.append((batch * (b // workers) + i, r))
    return np.array([r for _, r in sorted(rows)])


def load_references(root):
    refs = {}
    for d in sorted(glob.glob(os.path.join(root, "ms*"))):
        seed = int(os.path.basename(d)[2:])
        for model in REFS:
            h = glob.glob(os.path.join(d, "results", f"history_{model}_*.json"))
            g = glob.glob(os.path.join(d, "results", f"len_gen_{model}_*.npz"))
            if not (h and g):
                continue
            hist = json.load(open(h[0]))
            lg = dict(np.load(g[0], allow_pickle=True))
            keep = dedupe_rows(len(lg["per_seq_err"]))
            refs.setdefault(model, []).append(dict(
                seed=seed, val_curve=hist.get("val_curve"), epoch_time=hist["history"]["epoch_time_sec"],
                len_gen_mse=lg["per_seq_err"][keep], len_gen_boundary_age=lg["boundary_age"][keep],
                strip_gt=lg["strip_gt"], strip_pred=lg["strip_pred"], strip_input=lg["strip_input"]))
    return refs


def style_axes(ax):
    ax.grid(alpha=0.25, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(out, name + ".{pdf,png}"))


def fig_training(runs, refs, out):
    """Validation criterion on the fixed probe against epochs: seeds thin, median bold."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for model in (*OURS, *REFS):
        curves = []
        for r in runs.get(model, []):
            h = r["history"]
            if h["probe_step"]:
                curves.append((np.array(h["probe_step"]) / h["steps_per_epoch"], np.array(h["probe_loss"])))
        for r in refs.get(model, []):
            vc = r["val_curve"]
            if vc and vc["step"]:
                curves.append((np.array(vc["step"]) / 1688, np.array(vc["loss_mean"])))
        if not curves:
            continue
        s = STYLE[model]
        for x, y in curves:
            ax.plot(x, y, color=s["color"], ls=s["ls"], lw=0.6, alpha=0.35)
        n = min(len(y) for _, y in curves)
        med = np.median(np.stack([y[:n] for _, y in curves]), axis=0)
        ax.plot(curves[0][0][:n], med, color=s["color"], ls=s["ls"], lw=1.6,
                label=f"{s['label']} ({len(curves)} seeds)")
    ax.set_yscale("log")
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel("validation MSE + L1 (probe)", fontsize=9)
    style_axes(ax)
    ax.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, out, "training")


def seed_medians(entries, key, n=None):
    """Per seed, the median over sequences at every frame -> (n_seeds, T)."""
    return np.stack([np.median(e[key][:n], axis=0) for e in entries])


def fig_rollout(runs, refs, out):
    """MELSTM's generalization figure: median rollout MSE; the cost of a velocity change just
    before the context boundary at t=16; the age-1 group over the rollout. All models are
    scored on the first 2,512 sequences of the 100-frame set, MELSTM's references included."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    t = np.arange(16, 101)
    groups = [(m, runs.get(m, [])) for m in OURS] + [(m, refs.get(m, [])) for m in REFS]
    width = 0.8 / max(1, sum(bool(e) for _, e in groups))
    k = 0
    for model, entries in groups:
        if not entries:
            continue
        s = STYLE[model]
        med = seed_medians(entries, "len_gen_mse", N_SHARED)
        for curve in med:
            axes[0].plot(t, curve, color=s["color"], ls=s["ls"], lw=0.6, alpha=0.35)
        axes[0].plot(t, np.median(med, axis=0), color=s["color"], ls=s["ls"], lw=1.6, label=s["label"])

        ratios, age1 = [], []
        for e in entries:
            mse = e["len_gen_mse"][:N_SHARED]
            age = e["len_gen_boundary_age"][:N_SHARED].min(axis=1)
            settled = np.median(mse[age >= 4], axis=0)
            ratios.append([np.median(mse[age == a, 0]) / settled[0] for a in (1, 2, 3)])
            age1.append(np.median(mse[age == 1], axis=0) / settled)
        ratios = np.array(ratios)
        x = np.arange(3) - 0.4 + width * (k + 0.5)
        axes[1].bar(x, np.median(ratios, axis=0), width=width * 0.9, color=s["color"], label=s["label"])
        for j in range(3):
            axes[1].scatter(np.full(len(ratios), x[j]), ratios[:, j], s=8, color="black", zorder=3)
        axes[2].plot(t, np.median(np.array(age1), axis=0), color=s["color"], ls=s["ls"], lw=1.6,
                     label=s["label"])
        k += 1
    axes[0].set_yscale("log")
    axes[0].set_ylabel("median MSE", fontsize=9)
    axes[0].set_title("rollout (thin: seeds)", fontsize=9)
    axes[1].set_xticks(range(3), ["1", "2", "3"])
    axes[1].set_xlabel("boundary velocity age (frames)", fontsize=9)
    axes[1].set_ylabel("MSE at t=16 / settled (age ≥ 4)", fontsize=9)
    axes[1].axhline(1, color="0.4", lw=0.8)
    axes[1].set_title("velocity change before the boundary (dots: seeds)", fontsize=9)
    axes[2].axhline(1, color="0.4", lw=0.8)
    axes[2].set_ylabel("age-1 / settled median MSE", fontsize=9)
    axes[2].set_title("age-1 group over the rollout", fontsize=9)
    for ax in axes[[0, 2]]:
        ax.set_xlabel("frame t", fontsize=9)
    for ax in axes:
        style_axes(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, out, "rollout")


def fig_strips(runs, refs, out, seqs=(0, 1), every=5):
    """Ground truth and every model's rollout of the same 100-frame sequences, to t = 100."""
    first = {m: sorted(e, key=lambda r: r["seed"])[0] for m, e in {**refs, **runs}.items() if e}
    gt = None
    rows = []
    for model in ("tf", "lstm", "felstm", "melstm", "met"):
        e = first.get(model)
        if e is None:
            continue
        if model in REFS:
            full = np.concatenate([e["strip_input"], e["strip_pred"]], axis=1)[:, :, 0]
            truth = np.concatenate([e["strip_input"], e["strip_gt"]], axis=1)[:, :, 0]
        else:
            full = np.concatenate([e["strip_seq"][:, :15], e["strip_pred"]], axis=1)[:, :, 0]
            truth = e["strip_seq"][:, :, 0]
        if gt is None:
            gt = truth
        elif not np.allclose(gt[:len(truth)], truth[:len(gt)], atol=1e-6):
            print(f"WARNING: {model}'s strip sequences differ from the others")
        rows.append((f"{STYLE[model]['label']} (seed {e['seed']})", full))
    if gt is None:
        return
    rows.insert(0, ("ground truth", gt))
    times = np.arange(0, gt.shape[1], every)
    fig, axes = plt.subplots(len(seqs) * len(rows), 1,
                             figsize=(0.42 * len(times) + 1.6, 0.48 * len(seqs) * len(rows) + 0.4))
    for si, s_ in enumerate(seqs):
        for ri, (label, x) in enumerate(rows):
            ax = axes[si * len(rows) + ri]
            strip = np.concatenate([np.pad(x[s_, t], ((0, 0), (0, 1)), constant_values=1) for t in times], axis=1)
            ax.imshow(np.clip(strip, 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7)
    W = gt.shape[-1] + 1
    for i, tt in enumerate(times):
        axes[0].text(i * W + W / 2, -3, str(tt + 1), ha="center", va="bottom", fontsize=6,
                     color="red" if tt >= 25 else ("black" if tt < 15 else "0.4"))
    fig.tight_layout(pad=0.3)
    fig.suptitle("100-frame rollouts, frame t (gray: trained horizon 16-25; red: beyond it)",
                 fontsize=8, y=1.0 + 0.25 / fig.get_figheight())
    save(fig, out, "strips")


AXES = (("rate", "A", "realised switching rate", "motion_rate"),
        ("jump", "B", "mean jump |Δv| (rate fixed)", "motion_mean_dv"),
        ("speed", "C", "maximum speed", "max_speed"))


def load_motion(runs, refs_root):
    """{model: [ {regime: (npz, median MSE t=16-25, share above FAIL)} per seed ]} for ours and
    MELSTM's."""
    def cell(z):
        err = z["per_seq_err"][:, :10].mean(1)
        return z, float(np.median(err)), float((err > FAIL).mean())

    out = {}
    for model, entries in runs.items():
        for r in entries:
            cells = {}
            for f in glob.glob(os.path.join(r["dir"], "motion_factors", "*.npz")):
                d = np.load(f)
                cells[str(d["regime"])] = cell(d)
            if cells:
                out.setdefault(model, []).append(cells)
    for d in sorted(glob.glob(os.path.join(refs_root or "", "ms*", "motion_factors"))):
        for model in REFS:
            cells = {}
            for f in glob.glob(os.path.join(d, f"motion_factors_{model}_*.npz")):
                z = np.load(f, allow_pickle=True)
                cells[str(z["regime"])] = cell(z)
            if cells:
                out.setdefault(model, []).append(cells)
    return out


def fig_motion(motion, out):
    """MELSTM's motion-generalization sensitivity (top): median MSE over t=16-25 on each regime,
    divided by the model's own median at the centre (1 = the new motion costs nothing). A median
    does not see the sequences a model loses, so the bottom row gives their share: MSE above
    FAIL, on a log scale (a seed with none drops out)."""
    if not motion:
        return
    ours = next((c for m in OURS for c in motion.get(m, [])), None)
    ref = next((c for m in REFS for c in motion.get(m, [])), None)
    if ours and ref and "centre" in ours and "centre" in ref:
        a, b = float(ours["centre"][0]["motion_rate"]), float(ref["centre"][0]["motion_rate"])
        print(f"centre realised rate: ours {a:.5f}, MELSTM's {b:.5f}"
              + ("  (same sequences)" if abs(a - b) < 1e-9 else "  WARNING: different sequences"))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), sharey="row")
    for col, (axis, code, xlabel, key) in enumerate(AXES):
        for model in (*OURS, *REFS):
            seeds = motion.get(model, [])
            if not seeds:
                continue
            s = STYLE[model]
            names = sorted({n for c in seeds for n in c if n.startswith(code + ":")} | {"centre"},
                           key=lambda n: float(seeds[0][n][0][key]) if n in seeds[0] else 0.0)
            xs, ratios, shares = [], [], []
            for n in names:
                cells = [c for c in seeds if n in c and "centre" in c]
                if not cells:
                    continue
                x = float(seeds[0][n][0][key])
                ratio = [c[n][1] / c["centre"][1] for c in cells]
                share = [100 * c[n][2] for c in cells]
                xs.append(x)
                ratios.append(np.median(ratio))
                shares.append(np.median(share))
                axes[0, col].scatter([x] * len(ratio), ratio, s=6, color=s["color"], alpha=0.5, zorder=3)
                axes[1, col].scatter([x] * len(share), share, s=6, color=s["color"], alpha=0.5, zorder=3)
            for ax, ys in ((axes[0, col], ratios), (axes[1, col], shares)):
                ax.plot(xs, ys, color=s["color"], ls=s["ls"], lw=1.6, marker="o", ms=3, label=s["label"])
        axes[0, col].axhline(1, color="0.4", lw=0.8)
        axes[0, col].set_title(f"axis {code}", fontsize=9)
        axes[1, col].set_xlabel(xlabel, fontsize=9)
        for ax in axes[:, col]:
            ax.set_yscale("log", nonpositive="mask")
            style_axes(ax)
    axes[0, 0].set_ylabel("median MSE t=16-25 / own centre", fontsize=9)
    axes[1, 0].set_ylabel(f"sequences with MSE > {FAIL:g} (%)", fontsize=9)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, out, "motion_generalization")


def shared_errors(e):
    """Per-sequence MSE over t=16-25 on the first 2,512 sequences of the 100-frame set, the
    sequences every model (MELSTM's references included) was scored on."""
    return e["len_gen_mse"][:N_SHARED, :10].mean(axis=1)


def fig_errors(runs, refs, out):
    """Where the mean comes from: the distribution of per-sequence errors on the shared
    sequences, and the parameter-free tracker's failures per context step (the same for every
    MET seed, since no weights enter the velocities)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), gridspec_kw=dict(width_ratios=(1.4, 1)))
    ax = axes[0]
    for model in (*OURS, *REFS):
        entries = runs.get(model) or refs.get(model) or []
        s = STYLE[model]
        for i, e in enumerate(sorted(entries, key=lambda r: r["seed"])):
            x = np.sort(shared_errors(e))
            ax.plot(x, 1 - np.arange(len(x)) / len(x), color=s["color"], ls=s["ls"], lw=1,
                    alpha=0.8, label=f"{s['label']} ({len(entries)} seeds)" if i == 0 else None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("per-sequence MSE, t=16-25", fontsize=9)
    ax.set_ylabel("fraction of sequences with larger MSE", fontsize=9)
    ax.set_title("error distribution on the shared 2,512 sequences (lines: seeds)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    style_axes(ax)

    ax = axes[1]
    met = sorted(runs.get("met", []), key=lambda r: r["seed"])
    if met and "test_velocity_matches" in met[0]:
        wrong = 100 * (~met[0]["test_velocity_matches"]).mean(axis=(0, 2))
        steps = np.arange(1, len(wrong) + 1)
        ax.plot(steps, wrong, color=STYLE["met"]["color"], lw=1.6, marker="o", ms=3)
        hand_over = met[0]["args"]["pair_steps"]
        ax.axvline(hand_over - 0.5, color="0.4", lw=0.8, ls=":")
        ax.text(hand_over - 0.3, 0.5 * wrong.max(), "co-moving template\ntakes over", fontsize=7)
        ax.set_xticks(steps)
    ax.set_xlabel("context step t (frame t-1 → t)", fontsize=9)
    ax.set_ylabel("digits with a wrong velocity (%)", fontsize=9)
    ax.set_title("MET tracker failures, 10k test sequences", fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    save(fig, out, "errors")


def fmt(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return f"{values[0]:.2e}"
    return f"{values.mean():.2e} ± {values.std(ddof=1):.1e}"


def table(runs, refs, out):
    """Test MSE (t = 16-25), rollout MSE at t = 25 and t = 100, velocity accuracy, cost."""
    lines = ["| model | seeds | test MSE t=16-25 (10k) | median | shared: t=16-25 | shared: t=100 "
             "| shared: MSE > 1e-2 | boundary vel. acc. | s/epoch |",
             "|---|---|---|---|---|---|---|---|---|"]
    for model in OURS:
        e = runs.get(model, [])
        if not e:
            continue
        test = [r["test_mse"].mean() for r in e]
        med = [np.median(r["test_mse"].mean(1)) for r in e]
        shared = [shared_errors(r) for r in e]
        t100 = [r["len_gen_mse"][:N_SHARED, -1].mean() for r in e]
        vel = [r["summary"].get("test_vel_acc_boundary") for r in e]
        vel = fmt_pct(vel) if all(v is not None for v in vel) else "-"
        sec = [np.median(r["history"]["epoch_time_sec"]) for r in e]
        lines.append(f"| {STYLE[model]['label']} | {len(e)} | {fmt(test)} | {fmt(med)} | "
                     f"{fmt([x.mean() for x in shared])} | {fmt(t100)} | "
                     f"{fmt_pct([(x > FAIL).mean() for x in shared])} | {vel} | {np.mean(sec):.0f} |")
    for model in REFS:
        e = refs.get(model, [])
        if not e:
            continue
        shared = [shared_errors(r) for r in e]
        t100 = [r["len_gen_mse"][:, -1].mean() for r in e]
        rep = MELSTM_TABLE[model]
        lines.append(f"| {STYLE[model]['label']} (MELSTM paper) | {len(e)} | {rep[0]} | - | "
                     f"{fmt([x.mean() for x in shared])} | {fmt(t100)} (paper: {rep[1]}) | "
                     f"{fmt_pct([(x > FAIL).mean() for x in shared])} | - | {rep[2]} (A100) |")
    text = "\n".join(lines)
    note = ("\nMeans ± s.d. over seeds. Test: first 10,000 test sequences (seed 123); MELSTM's "
            "column is its paper's, over 4 copies of the first 2,512 of them. Shared: the first "
            "2,512 sequences of the 100-frame set (seed 42), the unique sequences of MELSTM's "
            "evaluation, scored by every model: mean MSE over t=16-25, MSE at t=100 (paper: over "
            "MELSTM's 10,000 rows, 4 copies of those), and the share of sequences whose t=16-25 "
            "MSE exceeds 1e-2. MET's velocities come from a parameter-free tracker, so its "
            "velocity accuracy is the same for every seed. s/epoch: training pass only, V100 for "
            "MET/Transformer, A100 for MELSTM's runs.\n")
    with open(os.path.join(out, "table.md"), "w") as f:
        f.write(text + "\n" + note)
    print(text + "\n" + note)


def fmt_pct(values):
    values = 100 * np.asarray(values, dtype=float)
    return f"{values.mean():.2f}%" if len(values) == 1 else f"{values.mean():.2f} ± {values.std(ddof=1):.2f}%"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--reference", default=None, help="MELSTM's cluster_runs/pure_h")
    p.add_argument("--out", default="figures")
    p.add_argument("--wandb", action="store_true",
                   help="also log the figures and the table to one W&B run, next to the training runs")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load_runs(args.runs)
    refs = load_references(args.reference) if args.reference else {}
    print("runs:", {m: sorted(r["seed"] for r in e) for m, e in runs.items()},
          "references:", {m: sorted(r["seed"] for r in e) for m, e in refs.items()})
    fig_training(runs, refs, args.out)
    fig_rollout(runs, refs, args.out)
    fig_strips(runs, refs, args.out)
    fig_errors(runs, refs, args.out)
    fig_motion(load_motion(runs, args.reference), args.out)
    table(runs, refs, args.out)
    if args.wandb:
        log_wandb(args.out)


def log_wandb(out, project="METransformer"):
    """The figures and the table as the run 'comparison' (group 'summary'); a rerun updates it."""
    import wandb
    run = wandb.init(project=project, id="comparison-mm", name="comparison", group="summary",
                     job_type="analysis", resume="allow")
    run.log({f"figures/{os.path.splitext(os.path.basename(f))[0]}": wandb.Image(f)
             for f in sorted(glob.glob(os.path.join(out, "*.png")))})
    rows = [[c.strip() for c in line.strip().strip("|").split("|")]
            for line in open(os.path.join(out, "table.md")) if line.startswith("| ")]
    run.log({"table": wandb.Table(columns=rows[0], data=rows[1:])})
    run.finish()


if __name__ == "__main__":
    main()
