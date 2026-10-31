"""Accuracy of slot velocities against the dataset's digit velocities."""
import itertools

import torch


def velocity_matches(U, gt_motion):
    """U (B, K, T, 2) slot velocities (U_t: frame t-1 -> t), gt_motion (B, >=T-1, N, 2)
    (motion t: frame t -> t+1). -> (B, T-1, N) bool: digit n's velocity at step t is matched
    by its slot, under the slot-digit assignment that matches most steps of the sequence."""
    T = U.shape[2]
    pred = U[:, :, 1:].transpose(1, 2).cpu()
    true = gt_motion[:, :T - 1].long().cpu()
    K, N = pred.shape[2], true.shape[2]
    perms = list(itertools.permutations(range(K), N))
    match = torch.stack([(pred[:, :, list(p)] == true).all(dim=-1) for p in perms])
    best = match.sum(dim=(2, 3)).argmax(dim=0)
    return match[best, torch.arange(len(best))]


def summarize(matches):
    """-> dict of accuracies from velocity_matches outputs concatenated over batches."""
    m = matches.float()
    return {
        "vel_acc": m.mean().item(),
        "vel_acc_boundary": m[:, -1].mean().item(),
        "vel_both_boundary": matches[:, -1].all(dim=1).float().mean().item(),
        "vel_acc_per_step": m.mean(dim=(0, 2)).tolist(),
    }
