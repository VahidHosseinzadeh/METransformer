"""Lifts: the moving frames in which the transformer attends.

A lift maps the context X_0..X_{T-1} (B, T, C, H, W) to K integer paths P (B, K, T, 2),
P[:, :, 0] = 0, and to the velocity u (B, K, 2) that continues them past the context,
P_{t+1} = P_t + u. Slot k sees frame t translated by -P^k_t, i.e. in its co-moving frame.
Also returned: the per-step velocities U (B, K, T, 2), U_t = P_t - P_{t-1}, U_0 = 0.

  NoLift      K = 1 and P = 0: a plain, translation-equivariant transformer.
  FlowLift    one slot per velocity v of the lattice V, P^v_t = t v: the flow equivariant
              transformer (Khaliliya & Keller, 2026), which enumerates velocities.
  MotionLift  K slots, P^k_t = sum_{s<=t} u^k_s with u estimated from the frames: the motion
              equivariant transformer, which estimates them.

MotionLift is motion equivariant when its velocity satisfies the gauge law of MELSTM,
u_t[psi X] = u_t[X] + psi_t - psi_{t-1} for translations. Then P_t[psi X] = P_t[X] + psi_t - psi_0,
so the co-moving tokens of psi X are those of X translated by psi_0, whatever the motion.
"""
import itertools

import torch
import torch.nn as nn

from .ops import torus_shift
from .phase_correlation import PhaseCorrelation


class NoLift(nn.Module):
    n_slots = 1

    def forward(self, x, gt_motion=None):
        B, T = x.shape[:2]
        U = torch.zeros(B, 1, T, 2, dtype=torch.long, device=x.device)
        return U, U[:, :, 0], U


class FlowLift(nn.Module):
    def __init__(self, v_range=2):
        super().__init__()
        r = range(-v_range, v_range + 1)
        self.register_buffer("v", torch.tensor(list(itertools.product(r, r)), dtype=torch.long),
                             persistent=False)
        self.n_slots = len(self.v)

    def forward(self, x, gt_motion=None):
        B, T = x.shape[:2]
        t = torch.arange(T, device=x.device)
        P = (t[None, :, None] * self.v[:, None, :]).expand(B, -1, -1, -1)
        u = self.v.expand(B, -1, -1)
        U = u[:, :, None].expand(-1, -1, T, -1).clone()
        U[:, :, 0] = 0
        return P, u, U


class MotionLift(nn.Module):
    """K slots moved by velocities that phase correlation estimates from the frames.

    velocity:
      'track'   slot tracking: t = 1 takes the K strongest peaks of PC(X_0, X_1) (the MELSTM
                bootstrap); from t = pair_steps on, each slot correlates its own template,
                carried to the coordinates of frame t-1, against X_t, and the steps between
                use frame-pair velocities as 'pair' does. The template is parameter-free: the
                slot's co-moving average of the raw frames so far (decay 1) or an exponential
                average, or their pixel-wise minimum ('min'). In the slot's frame its own object
                stands still and accumulates while the others move and blur, so each slot keeps
                to its object and the correspondence between frames never has to be solved.
                A template of only two frames still holds the other objects at half strength,
                which is why the first steps may use frame pairs.
      'pair'    MELSTM's frame-pair velocities: the K strongest peaks of PC(X_{t-1}, X_t),
                assigned to the slots by closeness to their previous velocities.
      'oracle'  the dataset's true velocities (a diagnostic; needs gt_motion and K digits).
      'state'   slot-state tracking, MELSTM's h-tracking: from t = pair_steps on (at least 2),
                the template of slot k is its own transformer output at t-1, rendered by
                state_template, so the paths depend on the network and METransformer.forward
                drives this lift one frame at a time through step(). t = 1 is the bootstrap and
                the steps between use frame pairs, as for 'track'.
    state_template ('state' only), how a slot output becomes a one-channel template:
      'dec'     the decoder applied to the slot alone (after the read-out norm), as it would
                render the slot's content without the others
      'nmean'   the channel mean after the read-out norm (identically zero at initialisation,
                when the norm's affine is the identity)
      'mean'    the raw channel mean, as MELSTM tracks with h
    """

    def __init__(self, n_slots=2, velocity="track", template="min", decay=1.0, pair_steps=4,
                 state_template="dec"):
        super().__init__()
        if velocity not in ("track", "pair", "oracle", "state"):
            raise ValueError(f"unknown velocity model {velocity!r}")
        if template not in ("mean", "min"):
            raise ValueError(f"unknown template {template!r}")
        if state_template not in ("dec", "nmean", "mean"):
            raise ValueError(f"unknown state template {state_template!r}")
        self.n_slots = n_slots
        self.velocity = velocity
        self.template = template
        self.decay = decay
        self.pair_steps = pair_steps
        self.state_template = state_template
        self.pc_boot = PhaseCorrelation(n_modes=n_slots)
        self.pc_track = PhaseCorrelation(n_modes=1)
        self.register_buffer("perms", torch.tensor(list(itertools.permutations(range(n_slots)))),
                             persistent=False)

    def forward(self, x, gt_motion=None):
        B, T = x.shape[:2]
        if self.velocity == "state":
            raise RuntimeError("velocity 'state' needs the slot outputs: METransformer.forward "
                               "drives it through step()")
        if self.velocity == "oracle":
            U = self._oracle(gt_motion, B, T, x.device)
        elif T < 2:
            U = torch.zeros(B, self.n_slots, T, 2, dtype=torch.long, device=x.device)
        elif self.velocity == "pair":
            U = self._pair(x)
        else:
            U = self._track(x)[0]
        return U.cumsum(dim=2), U[:, :, -1], U

    def _oracle(self, gt_motion, B, T, device):
        if gt_motion is None or gt_motion.shape[2] != self.n_slots:
            raise ValueError("velocity 'oracle' needs gt_motion with one digit per slot")
        U = torch.zeros(B, self.n_slots, T, 2, dtype=torch.long, device=device)
        U[:, :, 1:] = gt_motion[:, :T - 1].transpose(1, 2).to(device).long()
        return U

    def _bootstrap(self, x, U):
        U[:, :, 1] = self.pc_boot(x[:, 0], x[:, 1])[0]

    def _pair(self, x):
        B, T = x.shape[:2]
        U = torch.zeros(B, self.n_slots, T, 2, dtype=torch.long, device=x.device)
        self._bootstrap(x, U)
        for t in range(2, T):
            U[:, :, t] = self._pair_step(x, U, t)
        return U

    def state_steps_from(self):
        """The first step whose velocity comes from the slot templates ('state')."""
        return max(2, self.pair_steps)

    def step(self, x, U, t, P, tmpl=None):
        """'state': the velocities of step t >= 1 -> (B, K, 2). The bootstrap at t = 1; frame
        pairs before state_steps_from(); then each slot's template tmpl (B, K, C, H, W, in the
        slot frames, from its output at t-1) placed at P = P_{t-1} and phase-correlated against
        X_t. U (B, K, T, 2) holds the velocities of the steps before t."""
        if t == 1:
            return self.pc_boot(x[:, 0], x[:, 1])[0]
        if t < self.state_steps_from():
            return self._pair_step(x, U, t)
        B, K = tmpl.shape[:2]
        H, W = x.shape[-2:]
        placed = torus_shift(tmpl.mean(dim=2, keepdim=True), P).reshape(B * K, 1, H, W)
        frame = x[:, t].mean(dim=1, keepdim=True)[:, None].expand(B, K, 1, H, W)
        return self.pc_track(placed, frame.reshape(B * K, 1, H, W))[0].view(B, K, 2)

    def _pair_step(self, x, U, t):
        """The K peaks of PC(X_{t-1}, X_t), each slot given the one closest to its U_{t-1}."""
        cand = self.pc_boot(x[:, t - 1], x[:, t])[0]
        cost = (cand[:, self.perms] - U[:, None, :, t - 1]).pow(2).sum(dim=(2, 3))
        best = self.perms[cost.argmin(dim=1)]
        return torch.gather(cand, 1, best[..., None].expand(*best.shape, 2))

    def _track(self, x):
        """-> U (B, K, T, 2) and the final templates (B, K, 1, H, W) in the slot frames."""
        B, T, C, H, W = x.shape
        K = self.n_slots
        f = x.mean(dim=2, keepdim=True)
        frame = lambda t: f[:, t, None].expand(B, K, 1, H, W)
        U = torch.zeros(B, K, T, 2, dtype=torch.long, device=x.device)
        self._bootstrap(x, U)
        P = U[:, :, 1]
        acc, weight = self._accumulate(None, 0.0, frame(0))
        acc, weight = self._accumulate(acc, weight, torus_shift(frame(1), -P))
        for t in range(2, T):
            if t < self.pair_steps:
                U[:, :, t] = self._pair_step(x, U, t)
            else:
                tmpl = torus_shift(acc / weight, P).reshape(B * K, 1, H, W)
                U[:, :, t] = self.pc_track(tmpl, frame(t).reshape(B * K, 1, H, W))[0].view(B, K, 2)
            P = P + U[:, :, t]
            acc, weight = self._accumulate(acc, weight, torus_shift(frame(t), -P))
        return U, acc / weight

    def _accumulate(self, acc, weight, canon):
        if acc is None:
            return canon, 1.0
        if self.template == "min":
            return torch.minimum(acc, canon), 1.0
        return self.decay * acc + canon, self.decay * weight + 1.0


def comoving_template(x, P, kind="mean"):
    """The pixel-wise mean or minimum over time of the frames in each slot's co-moving frame.

    x: (B, T, C, H, W), P: (B, K, T, 2) -> (B, K, C, H, W)."""
    B, T = x.shape[:2]
    K = P.shape[1]
    canon = torus_shift(x[:, None].expand(B, K, *x.shape[1:]), -P)
    return canon.mean(dim=2) if kind == "mean" else canon.amin(dim=2)


def make_lift(name, n_slots=2, v_range=2, velocity="track", template="min", decay=1.0,
              pair_steps=4, state_template="dec"):
    if name == "none":
        return NoLift()
    if name == "flow":
        return FlowLift(v_range)
    if name == "motion":
        return MotionLift(n_slots, velocity, template, decay, pair_steps, state_template)
    raise ValueError(f"unknown lift {name!r}")
