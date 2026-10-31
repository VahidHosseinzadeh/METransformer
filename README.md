# METransformer

A motion equivariant transformer: the transformer counterpart of MELSTM (the motion
equivariant ConvLSTM) and the generalization of the flow equivariant transformer (FET,
Khaliliya & Keller 2026) from flows to motions whose velocity changes in time.

## Construction

A sequence $X_0,\dots,X_{T-1}$ of frames on the torus is read in $K$ moving frames (slots).
A velocity model returns, for every slot, integer velocities $u^k_t$; their sums
$P^k_t=\sum_{s\le t}u^k_s$ are the slot's path, i.e. the transporter $T^k_{t,s}$ is the
translation by $P^k_t-P^k_s$.

1. **Lift.** Each frame is embedded by a circular convolution, $z_t=E(X_t)$, and enters
   slot $k$ in the slot's co-moving frame, $\tilde z^k_t = S_{-P^k_t}\,z_t$.
2. **Transformer.** A causal pre-norm transformer runs along time in every slot, with
   weights shared across slots: convolutional Q/K/V and MLP, LayerNorm over channels per
   pixel, and a learned bias per head and time lag. Attention is per pixel of the slot
   frame. In image coordinates, the query of frame $t$ at $x$ reads the keys and values of
   frame $s$ at $x-(P_t-P_s)$, i.e. **transported by $T_{t,s}$**, which is the construction
   sketched at the end of the MELSTM paper.
3. **Read-out.** To predict frame $t+1$, every slot's output is moved to $P^k_{t+1}$, the
   slots are max-pooled and a convolutional decoder renders the frame. After the context
   the paths continue at the last velocity, and the predictions are fed back as tokens
   (autoregressive, with a key/value cache).

The lift selects the model; everything else, including the parameter count, is shared:

| lift | slots | paths | model |
|---|---|---|---|
| `none` | 1 | $P=0$ | plain (translation-equivariant) transformer |
| `flow` | $\lvert V\rvert$ | $P^v_t=tv$, every $v$ of a lattice $V$ | FET (enumerates velocities) |
| `motion` | $K$ | $P^k_t=\sum_{s\le t}u^k_s$, $u$ estimated | **METransformer** (estimates them) |

**Equivariance.** If the velocity model obeys MELSTM's gauge law, which for translations reads
$u_t[\psi X]=u_t[X]+\psi_t-\psi_{t-1}$, then $P_t[\psi X]=P_t[X]+\psi_t-\psi_0$. The slot tokens of
$\psi X$ are then those of $X$ translated by $\psi_0$, for any motion $\psi$. Every layer
commutes with translations, so the predictions transform with $\psi$, continued at its
last velocity past the context. Equality is up to the torus: velocities live in
$\mathbb Z_H\times\mathbb Z_W$. `tests/test_equivariance.py` checks this to $10^{-10}$ in
float64 for arbitrary integer motions (up to 6 px/frame), for both attention modes and
both phase-correlation velocity models. The tests also check that the plain transformer
fails the same test, that the cached rollout equals a parallel pass, and that attention is causal.

## Velocity models

All are Fourier phase correlation (`metransformer/phase_correlation.py`, MELSTM's
estimator), parameter-free and integer-valued:

- `track` (default): slot tracking. At $t=1$ the slots take the $K$ strongest peaks of
  $\mathrm{PC}(X_0,X_1)$, MELSTM's bootstrap. From $t=4$ on, each slot correlates its own
  template against $X_t$. The template is the pixel-wise minimum of the frames so far in
  the slot's co-moving frame: there the slot's digit stands still while the other digit
  moves and is erased. Steps $t=2,3$ use frame-pair velocities, because a template of
  only two frames still holds the other digit.
- `pair`: MELSTM's frame-pair velocities, the $K$ peaks of $\mathrm{PC}(X_{t-1},X_t)$
  assigned to the slots by continuity.
- `oracle`: the dataset's true velocities (diagnostic only).

Measured on the first 2000 test sequences (`python -m moving_mnist.check_velocity`).
The "transport" columns predict with no parameters at all: each slot's co-moving
min-template, moved along its frozen path, max-pooled over slots.

| velocity model | velocity acc. | at boundary | both digits at boundary | transport MSE, $t=16..25$ |
|---|---|---|---|---|
| copy last frame | | | | 1.9e-1 |
| `pair` | 90.4% | 83.9% | 81.3% | 4.4e-2 |
| `track`, mean template, $t\ge2$ | 90.4% | 89.7% | 79.4% | 8.2e-3 |
| `track`, min template, $t\ge2$ | 93.0% | 92.2% | 84.4% | 7.9e-3 |
| **`track`, min template, $t\ge4$** | **99.3%** | **99.0%** | **98.1%** | **1.1e-3** |
| `oracle` | 100% | 100% | 100% | 2.5e-4 |

- With true velocities, the parameter-free transport is already at MELSTM's level
  (MELSTM test MSE 2.87e-4). On this benchmark the velocity estimate decides most of the
  error, and a fair account of what the transformer adds must include this baseline.
- Tracking from $t=2$ fails systematically on the first tracked step. A two-frame template
  holds the other digit as two half-strength copies; a light digit (1, 7) next to a heavy
  one (0, 8) loses its slot to the copy whose velocity is $2v_\text{other}-v_\text{own}$. The
  slot then stays on the other digit (collapse).
- The remaining 2% of boundary errors under the default are mostly the same collapse
  (32 of 39), now starting at $t=4$. A learned template, the transformer's own slot
  output as MELSTM uses its hidden state, is the natural next step.

## Moving MNIST protocol

MELSTM's (`run_comparison.sh` there): 36×36 torus, two digits, piecewise-constant
velocities $\lvert v\rvert_\infty\le2$ held 3–6 frames, frozen from the last context step on.
There are 15 context frames and 10 predicted frames, fed back. The loss is MSE + L1 on the
predicted frames. Each epoch draws 54k/6k fresh train/val sequences; training uses Adam
1e-3 with cosine decay to 1e-5 (1 warmup epoch), grad clip 1, batch 32, for 50 epochs. The
best-val checkpoint is scored once at the end: test MSE on the first 10k sequences of seed
123, and the 100-frame rollout (seed 42) at $t=25$ and $t=100$. Per-sequence errors are
saved, so means, medians and MELSTM's weighting can all be recomputed.
(MELSTM's 4-worker loaders drew each fixed set 4 times over, so its 10k test sequences
were 4 copies of the first 2512 here.) The default model has 49,633 parameters
(MELSTM: 47,681). On an A100 a training step at batch 32 takes 265 ms and 10.7 GB, about
7 h for 50 epochs.

## Usage

```bash
python -m pytest -q tests                         # equivariance and exactness
python -m moving_mnist.check_velocity --n 2000    # velocity models, parameter-free baselines
python -m moving_mnist.train --lift motion --model_seed 1 --save_dir runs/met_ms1

# cluster: from the repo root with the environment active
SMOKE=1 RUN_TESTS=1 sbatch --job-name=met_smoke --time=00:30:00 submit_mm.sbatch met 1
sbatch --job-name=met_ms1 submit_mm.sbatch met 1   # arms: met, met_oracle, met_pair
```

`data` should point at a torchvision MNIST root, e.g. a link to FEConvLSTM's `data/`.
The dataset code is MELSTM's, copied unchanged
(`moving_mnist/time_dependent_moving_mnist_dataset.py`).

### Bridges-2 (PSC)

```bash
git clone https://github.com/VahidHosseinzadeh/METransformer.git && cd METransformer
bash scripts/setup_bridges2.sh   # PSC's PyTorch module + a thin .venv with wandb and pytest
# data/MNIST/raw: the four MNIST files (login nodes refuse scp; a tar over ssh works)
SMOKE=1 RUN_TESTS=1 sbatch --job-name=met_smoke --time=00:30:00 submit_bridges2.sbatch met 1
sbatch --job-name=met_ms1 submit_bridges2.sbatch met 1      # H100; --gres=gpu:v100-32:1 for a V100
```

The PyTorch stack is PSC's module (`pytorch/26.05-2.11-py3`: torch 2.11, cuDNN 9.10); the
setup adds only wandb and pytest. Runs write under the repo in `$HOME`, because the
allocation's shared `/ocean` space was full. An H100 hour costs 2 SU, a V100 or L40S hour 1 SU.
