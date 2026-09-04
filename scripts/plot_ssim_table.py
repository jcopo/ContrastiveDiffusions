"""Reproduce paper v2 Table 1: median SSIM over 20 digits, CoDiff vs random, 6 measurements.

Usage: python scripts/plot_ssim_table.py [runs_dir]  (default: runs)
Expects {runs_dir}/ssim_runs/ssim_{mode}_{seed}.npz from run_bed_amortized.py.
"""
import glob
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S = sys.argv[1] if len(sys.argv) > 1 else "runs"
PAPER = {
    "CoDiff": [0.227, 0.338, 0.528, 0.673, 0.789, 0.826],
    "Random": [0.168, 0.275, 0.350, 0.391, 0.421, 0.463],
}

ours = {}
for mode, label in [("optimized", "CoDiff"), ("random", "Random"), ("amortized", "Amortized")]:
    files = sorted(glob.glob(f"{S}/ssim_runs/ssim_{mode}_*.npz"))
    if not files:
        continue
    ssims = np.stack([np.load(f)["ssims"] for f in files])  # (n_seeds, 6)
    ours[label] = np.median(ssims, axis=0)
    print(f"{label}: {len(files)} seeds  median SSIM per measurement: {np.round(ours[label], 3)}")

k = np.arange(1, 7)
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for label, color in [("CoDiff", "C0"), ("Random", "C1"), ("Amortized", "C2")]:
    if label not in ours:
        continue
    if label not in PAPER:
        ax.plot(k, ours[label], "o-", color=color, label=f"{label} (ours, no FPS)")
        continue
    ax.plot(k, ours[label], "o-", color=color, label=f"{label} (ours)")
    ax.plot(k, PAPER[label], "s--", color=color, alpha=0.5, label=f"{label} (paper Table 1)")
ax.set_xlabel("measurement")
ax.set_ylabel("median SSIM (20 digits)")
ax.set_title("Paper v2 Table 1 reproduction")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{S}/paper_repro/figures/ssim_table_reproduction.png", dpi=150, bbox_inches="tight")
print("saved ssim_table_reproduction.png")
