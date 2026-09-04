"""Recompute SSIM two ways from stored posterior samples: E[SSIM(θ, θ*)] and SSIM(mean θ, θ*).

Usage: python scripts/ssim_variants.py [runs_dir]  (default: runs)
"""
import glob
import sys

import dm_pix
import jax
import jax.numpy as jnp
import numpy as np

S = sys.argv[1] if len(sys.argv) > 1 else "runs"
PAPER = {"CoDiff": [.227, .338, .528, .673, .789, .826], "Random": [.168, .275, .350, .391, .421, .463]}


def to01(x):
    return np.clip((x + 1) / 2, 0, 1)


for mode, label in [("optimized", "CoDiff"), ("random", "Random"), ("amortized", "Amortized")]:
    files = sorted(glob.glob(f"{S}/ssim_runs/ssim_{mode}_*.npz"))
    if not files:
        continue
    per_sample, on_mean = [], []
    for f in files:
        d = np.load(f)
        gt = jnp.asarray(to01(d["ground_truth"]))
        row_ps, row_om = [], []
        for k in range(d["samples"].shape[0]):
            s = jnp.asarray(to01(d["samples"][k]))
            row_ps.append(float(jnp.mean(jax.vmap(dm_pix.ssim, in_axes=(None, 0))(gt, s))))
            row_om.append(float(dm_pix.ssim(gt, jnp.mean(s, axis=0))))
        per_sample.append(row_ps)
        on_mean.append(row_om)
    ps = np.median(np.asarray(per_sample), axis=0)
    om = np.median(np.asarray(on_mean), axis=0)
    print(f"{label} ({len(files)} seeds)")
    print(f"  per-sample mean : {np.round(ps, 3).tolist()}")
    print(f"  on posterior mean: {np.round(om, 3).tolist()}")
    if label in PAPER:
        print(f"  paper Table 1    : {PAPER[label]}")
