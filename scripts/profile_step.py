"""Time and peak memory of METransformer training steps on random data (CUDA).

    python scripts/profile_step.py --batch 32 [--velocity track] [--table]
"""
import argparse
import time

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from metransformer.lift import make_lift
from metransformer.model import METransformer

p = argparse.ArgumentParser()
p.add_argument("--batch", type=int, default=32)
p.add_argument("--velocity", default="oracle")
p.add_argument("--steps", type=int, default=5)
p.add_argument("--table", action="store_true", help="also print the top ops of a profiled step")
args = p.parse_args()
dev = torch.device("cuda")
B = args.batch
x = torch.rand(B, 25, 1, 36, 36, device=dev)
motion = torch.randint(-2, 3, (B, 25, 2, 2), device=dev)

for checkpoint in (False, True):
    torch.manual_seed(0)
    model = METransformer(make_lift("motion", velocity=args.velocity), checkpoint=checkpoint).to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3)

    def step(pred_len=10):
        pred, _ = model(x[:, :15], pred_len, gt_motion=motion)
        loss = F.mse_loss(pred, x[:, 15:15 + pred_len])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for name, pred_len in (("context + 10-step rollout", 10), ("context + 1 predicted frame", 1)):
        for _ in range(3):
            step(pred_len)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(args.steps):
            step(pred_len)
        torch.cuda.synchronize()
        print(f"{name:28s} checkpoint={checkpoint!s:5s} {1000 * (time.time() - t0) / args.steps:7.1f} ms/step"
              f"  peak {torch.cuda.max_memory_allocated() / 1e9:5.1f} GB  (batch {B}, velocity "
              f"{args.velocity})", flush=True)
    if args.table and not checkpoint:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            step()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15,
                                        max_name_column_width=50))
    del model, opt
