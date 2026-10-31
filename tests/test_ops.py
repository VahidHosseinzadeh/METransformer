import torch

from metransformer.ops import torus_shift
from metransformer.phase_correlation import PhaseCorrelation


def test_torus_shift_is_roll():
    torch.manual_seed(0)
    x = torch.randn(3, 4, 2, 7, 9)
    for dx in range(-6, 7):
        for dy in range(-6, 7):
            d = torch.tensor([dx, dy]).expand(3, 4, 2)
            assert torch.equal(torus_shift(x, d), torch.roll(x, (dy, dx), (-2, -1)))


def test_torus_shift_per_map():
    torch.manual_seed(1)
    x = torch.randn(5, 3, 8, 8)
    d = torch.randint(-20, 20, (5, 2))
    out = torus_shift(x, d)
    for n in range(5):
        assert torch.equal(out[n], torch.roll(x[n], (int(d[n, 1]), int(d[n, 0])), (-2, -1)))


def test_torus_shift_gradient_is_inverse_shift():
    torch.manual_seed(2)
    x = torch.randn(4, 2, 6, 5, requires_grad=True)
    w = torch.randn(4, 2, 6, 5)
    d = torch.randint(-9, 9, (4, 2))
    (torus_shift(x, d) * w).sum().backward()
    assert torch.equal(x.grad, torus_shift(w, -d))


def test_phase_correlation_recovers_shift():
    torch.manual_seed(3)
    pc = PhaseCorrelation(n_modes=1)
    a = torch.rand(16, 1, 36, 36)
    d = torch.randint(-17, 18, (16, 2))
    b = torus_shift(a, d)
    v, _ = pc(a, b)
    assert torch.equal(v[:, 0], d)
