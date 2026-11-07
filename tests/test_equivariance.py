"""Motion equivariance of the lifts and of the transformer, checked numerically."""
import pytest
import torch

from metransformer.lift import FlowLift, MotionLift, NoLift
from metransformer.model import METransformer, TemporalAttention
from metransformer.ops import torus_shift

H = W = 20
T = 7


def moving_objects(B=3, T=T, n=2, seed=0):
    """Textured squares moving with random, changing velocities on the torus, plus noise."""
    g = torch.Generator().manual_seed(seed)
    x = torch.zeros(B, T, 1, H, W, dtype=torch.float64)
    for _ in range(n):
        obj = torch.zeros(B, 1, H, W, dtype=torch.float64)
        obj[..., :6, :6] = torch.rand(B, 1, 6, 6, generator=g, dtype=torch.float64)
        vel = torch.randint(-2, 3, (B, T, 2), generator=g)
        pos = torch.randint(0, H, (B, 1, 2), generator=g) + vel.cumsum(1) - vel[:, :1]
        for t in range(T):
            x[:, t] += torus_shift(obj, pos[:, t])
    return x.clamp(0, 1) + 0.05 * torch.rand(x.shape, generator=g, dtype=torch.float64)


def random_motion(B, T, seed=1, v_max=6):
    """psi_t (B, T, 2): an arbitrary integer path, psi_0 included."""
    g = torch.Generator().manual_seed(seed)
    return torch.randint(-v_max, v_max + 1, (B, T, 2), generator=g).cumsum(1)


def act(psi, x):
    """(psi x)_t = psi_t . x_t, x: (B, T, C, H, W)."""
    return torus_shift(x, psi)


def continued(psi, steps):
    """psi continued past its end at its last velocity: psi_T .. psi_{T+steps-1}."""
    v = psi[:, -1] - psi[:, -2]
    return psi[:, -1:] + v[:, None] * torch.arange(1, steps + 1)[None, :, None]


def equal_on_torus(a, b):
    """Displacements are elements of Z_H x Z_W; phase correlation reports them in (-W/2, W/2]."""
    return bool(((a - b) % H == 0).all())


@pytest.mark.parametrize("pair_steps", [2, 4])
@pytest.mark.parametrize("velocity,template", [("track", "mean"), ("track", "min"), ("pair", "mean")])
def test_velocity_obeys_the_gauge_law(velocity, template, pair_steps):
    x = moving_objects()
    psi = random_motion(x.shape[0], T)
    lift = MotionLift(2, velocity, template, decay=0.8, pair_steps=pair_steps)
    P, u, U = lift(x)
    Pm, um, Um = lift(act(psi, x))
    dpsi = psi[:, 1:] - psi[:, :-1]
    assert equal_on_torus(Um[:, :, 1:], U[:, :, 1:] + dpsi[:, None])
    assert equal_on_torus(Pm, P + (psi - psi[:, :1])[:, None])
    assert equal_on_torus(um, u + dpsi[:, None, -1])


def make_model(lift, attn="pixel"):
    torch.manual_seed(0)
    model = METransformer(lift, dim=16, depth=2, heads=4, mlp_hidden=16, attn=attn, max_dist=4)
    return model.double().eval()


@pytest.mark.parametrize("attn", ["pixel", "frame"])
@pytest.mark.parametrize("velocity", ["track", "pair"])
def test_motion_equivariant_transformer(velocity, attn):
    x = moving_objects()
    psi = random_motion(x.shape[0], T, seed=2)
    model = make_model(MotionLift(2, velocity), attn)
    with torch.no_grad():
        pred, _ = model(x, pred_len=5)
        pred_m, _ = model(act(psi, x), pred_len=5)
    expected = act(continued(psi, 5), pred)
    assert (pred_m - expected).abs().max() < 1e-10


@pytest.mark.parametrize("state_template", ["dec", "nmean", "mean"])
@pytest.mark.parametrize("pair_steps", [2, 4])
def test_state_tracking_is_motion_equivariant(state_template, pair_steps):
    """Slot-state velocities obey the gauge law and the predictions transform with the motion.
    The norm's affine is randomised: at initialisation the 'nmean' template is zero."""
    x = moving_objects()
    psi = random_motion(x.shape[0], T, seed=2)
    model = make_model(MotionLift(2, "state", pair_steps=pair_steps, state_template=state_template))
    with torch.no_grad():
        model.norm.weight.normal_()
        model.norm.bias.normal_()
        pred, aux = model(x, pred_len=5)
        pred_m, aux_m = model(act(psi, x), pred_len=5)
    dpsi = psi[:, 1:] - psi[:, :-1]
    assert equal_on_torus(aux_m["U"][:, :, 1:], aux["U"][:, :, 1:] + dpsi[:, None])
    assert (pred_m - act(continued(psi, 5), pred)).abs().max() < 1e-10


def test_state_lift_without_state_steps_is_the_pair_lift():
    """With every step before pair_steps, the frame-by-frame context pass reproduces the
    parallel pass of the 'pair' lift: same velocities, same slot outputs and predictions."""
    x = moving_objects()
    with torch.no_grad():
        p1, a1 = make_model(MotionLift(2, "state", pair_steps=T + 1))(x, 4, return_slots=True)
        p2, a2 = make_model(MotionLift(2, "pair"))(x, 4, return_slots=True)
    assert torch.equal(a1["U"], a2["U"])
    assert (a1["slot_h"] - a2["slot_h"]).abs().max() < 1e-10
    assert (p1 - p2).abs().max() < 1e-10


def test_state_velocity_correlates_the_previous_slot_output():
    """U_t of every slot is the phase correlation of its output at t-1, placed at P_{t-1},
    against X_t."""
    x = moving_objects()
    model = make_model(MotionLift(2, "state", pair_steps=2))
    with torch.no_grad():
        _, aux = model(x, pred_len=2, return_slots=True)
        h, P, U = aux["slot_h"], aux["paths"], aux["U"]
        for t in range(2, T):
            u = model.lift.step(x, U, t, P[:, :, t - 1], model.state_template(h[:, :, t - 1]))
            assert torch.equal(u, U[:, :, t])


def test_state_tracking_backward():
    """Training through the frame-by-frame context: every parameter gets a gradient, and
    checkpointing the blocks does not change it."""
    x = moving_objects()
    model = make_model(MotionLift(2, "state", pair_steps=2)).train()
    grads = []
    for checkpoint in (False, True):
        model.checkpoint = checkpoint
        model.zero_grad()
        pred, _ = model(x[:, :5], pred_len=2)
        (pred - x[:, 5:7]).pow(2).mean().backward()
        grads.append([p.grad.clone() for p in model.parameters()])
    assert all(g.abs().sum() > 0 for g in grads[0])
    assert all((a - b).abs().max() < 1e-12 for a, b in zip(*grads))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_motion_equivariance_on_cuda_float32():
    """In float32 proper. With TF32 convolutions and matmuls (the CUDA default on Ampere,
    used in training) the two sides agree to about 1e-4 instead."""
    flags = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        x = moving_objects().float().cuda()
        psi = random_motion(x.shape[0], T, seed=2).cuda()
        model = make_model(MotionLift(2)).float().cuda()
        with torch.no_grad():
            pred, _ = model(x, pred_len=5)
            pred_m, _ = model(act(psi, x), pred_len=5)
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = flags
    v = psi[:, -1] - psi[:, -2]
    future = psi[:, -1:] + v[:, None] * torch.arange(1, 6, device=x.device)[None, :, None]
    assert (pred_m - act(future, pred)).abs().max() < 1e-5


@pytest.mark.parametrize("lift", [NoLift(), FlowLift(1), MotionLift(2)])
def test_global_translation_equivariance(lift):
    x = moving_objects()
    g = torch.tensor([[5, -3], [0, 7], [-11, 2]])[:, None].expand(-1, T, -1)
    model = make_model(lift)
    with torch.no_grad():
        pred, _ = model(x, pred_len=3)
        pred_g, _ = model(act(g, x), pred_len=3)
    assert (pred_g - act(g[:, :3], pred)).abs().max() < 1e-10


def test_plain_transformer_is_not_motion_equivariant():
    x = moving_objects()
    psi = random_motion(x.shape[0], T, seed=2)
    model = make_model(NoLift())
    with torch.no_grad():
        pred, _ = model(x, pred_len=3)
        pred_m, _ = model(act(psi, x), pred_len=3)
    assert (pred_m - act(continued(psi, 3), pred)).abs().max() > 1e-3


@pytest.mark.parametrize("lift", [NoLift(), FlowLift(1), MotionLift(2),
                                  MotionLift(2, "state", pair_steps=2)])
def test_cached_rollout_matches_parallel_pass(lift):
    """The rollout (KV cache, one frame per step) equals one causal pass over the context
    followed by the fed-back predictions."""
    x = moving_objects()
    model = make_model(lift)
    with torch.no_grad():
        pred, aux = model(x, pred_len=4)
        P, u = aux["P"], aux["U"][:, :, -1]
        P_roll = P[:, :, -1:] + u[:, :, None] * torch.arange(1, 5)[None, None, :, None]
        frames = torch.cat([x, pred[:, :-1]], dim=1)
        paths = torch.cat([P, P_roll[:, :, :-1]], dim=2)
        h, _ = model.transformer(model.tokens(frames, paths))
        h = h.view(x.shape[0], P.shape[1], frames.shape[1], *h.shape[2:])
        parallel = model.read(h[:, :, T - 1:], P_roll)
    assert (parallel - pred).abs().max() < 1e-10


@pytest.mark.parametrize("attn", ["pixel", "frame"])
def test_chunked_cache_and_checkpointing(attn):
    """Training keeps the cache in chunks and may checkpoint the blocks; neither changes the
    predictions (merged cache without gradients) or the gradients."""
    x = moving_objects()
    model = make_model(MotionLift(2), attn)
    with torch.no_grad():
        ref, _ = model(x, pred_len=4)
    grads = []
    for checkpoint in (False, True):
        model.checkpoint = checkpoint
        model.zero_grad()
        pred, _ = model(x, pred_len=4)
        assert (pred - ref).abs().max() < 1e-12
        pred.pow(2).sum().backward()
        grads.append([p.grad.clone() for p in model.parameters()])
    assert all((a - b).abs().max() < 1e-12 for a, b in zip(*grads))


@pytest.mark.parametrize("mode", ["pixel", "frame"])
def test_attention_matches_reference(mode):
    """TemporalAttention against a direct implementation: causal softmax attention along
    time, per pixel or per frame, with the lag bias."""
    torch.manual_seed(0)
    att = TemporalAttention(8, 2, mode=mode, max_dist=3).double()
    with torch.no_grad():
        att.time_bias.normal_()
    x = torch.randn(2, 5, 8, 4, 3, dtype=torch.float64)
    N, T, C, H, W = x.shape
    q, k, v = att.qkv(x.reshape(N * T, C, H, W)).view(N, T, 3, 2, 4, H * W).unbind(2)
    bias = torch.stack([torch.stack([att.time_bias[:, min(t - s, 2)] if s <= t
                                     else torch.full((2,), float("-inf"), dtype=x.dtype)
                                     for s in range(T)], -1) for t in range(T)], -2)  # (h, T, T)
    if mode == "pixel":
        logits = torch.einsum("nthdp,nshdp->nhpts", q, k) / 2.0 + bias[:, None]
        out = torch.einsum("nhpts,nshdp->nthdp", logits.softmax(-1), v)
    else:
        logits = torch.einsum("nthdp,nshdp->nhts", q, k) / (4 * H * W) ** 0.5 + bias
        out = torch.einsum("nhts,nshdp->nthdp", logits.softmax(-1), v)
    ref = att.proj(out.reshape(N * T, C, H, W)).view(N, T, C, H, W)
    with torch.no_grad():
        got, _ = att(x)
    assert (got - ref).abs().max() < 1e-12


def test_attention_is_causal():
    x = moving_objects()
    model = make_model(NoLift())
    x2 = x.clone()
    x2[:, -1] = torch.rand_like(x2[:, -1])
    with torch.no_grad():
        _, aux = model(x, pred_len=1, context_predictions=True)
        _, aux2 = model(x2, pred_len=1, context_predictions=True)
    assert torch.equal(aux["context_pred"], aux2["context_pred"])


@pytest.mark.parametrize("lift", [NoLift(), MotionLift(2)])
def test_returned_slots_reproduce_the_rollout(lift):
    """slot_h at time t, read at paths[t + 1], is the prediction of frame t + 1."""
    x = moving_objects()
    model = make_model(lift)
    with torch.no_grad():
        pred, _ = model(x, pred_len=4)
        pred_s, aux = model(x, pred_len=4, return_slots=True)
        h, paths = aux["slot_h"], aux["paths"]
        assert h.shape[2] == T + 3 and paths.shape[2] == T + 4
        reread = model.read(h[:, :, T - 1:], paths[:, :, T:])
    assert torch.equal(pred, pred_s)
    assert (reread - pred).abs().max() < 1e-12
