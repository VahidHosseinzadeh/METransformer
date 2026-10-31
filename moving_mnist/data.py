"""The Moving MNIST benchmark of the MELSTM paper (run_comparison.sh settings)."""
import torch

from moving_mnist.time_dependent_moving_mnist_dataset import TDMovingMNISTDataset

TEST_SEED = 123
LEN_GEN_SEED = 42


def make_dataset(root, train, seq_len=25, input_frames=15, random=True, seed=42, image_size=36,
                 v_range=2, motion_mode="piecewise", transition_mode="smooth", min_segment=3,
                 max_segment=6, p_change=0.25, smooth_probability=0.8, download=False):
    """Two digits on the 36x36 torus with piecewise-constant velocities, frozen from the
    last context step on (freeze_after = input_frames). random=False draws from a fixed
    stream (seed), which reset_rng() restarts."""
    return TDMovingMNISTDataset(
        root=root, train=train, seq_len=seq_len, num_digits=2, image_size=image_size,
        max_speed=v_range, motion_mode=motion_mode, transition_mode=transition_mode,
        min_segment=min_segment, max_segment=max_segment, p_change=p_change,
        smooth_probability=smooth_probability, motion_difficulty=None,
        freeze_after=input_frames, min_center_distance=20, reject_overlap=True,
        require_distinct_velocities=True, return_motion=True, return_positions=False,
        transform=None, download=download, random=random, seed=seed, max_tries=300)


def fixed_set(dataset, n):
    """The first n sequences of a fixed (random=False) dataset's stream, drawn in this process:
    (frames (n, T, 1, H, W), motions (n, T, 2, 2)).

    MELSTM's loaders drew fixed sets in 4 workers that each restarted the stream, so its
    10k test sequences were 4 copies of the first 2512 here (see per-sequence results)."""
    dataset.reset_rng()
    items = [dataset[i] for i in range(n)]
    return torch.stack([s for s, _, _ in items]), torch.stack([m for _, _, m in items])
