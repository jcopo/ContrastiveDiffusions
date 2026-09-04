"""Measurement and posterior evolution for a few seeds of the ssim_runs experiment.

Usage: python scripts/plot_measurement_evolution.py [runs_dir] [mode] [seeds]
Defaults: runs optimized 1,4,9,16
Expects {runs_dir}/ssim_runs/ssim_{mode}_{seed}.npz from run_bed_amortized.py.
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S = sys.argv[1] if len(sys.argv) > 1 else "runs"
MODE = sys.argv[2] if len(sys.argv) > 2 else "optimized"
SEEDS = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "1,4,9,16").split(",")]
SIZE = 7

to01 = lambda x: np.clip((x + 1) / 2, 0, 1)

d0 = np.load(f"{S}/ssim_runs/ssim_{MODE}_{SEEDS[0]}.npz")
K = d0["designs"].shape[0]
R = len(SEEDS)

fig, axes = plt.subplots(2 * R, K + 1, figsize=(2.3 * (K + 1), 4.7 * R))
for r, seed in enumerate(SEEDS):
    d = np.load(f"{S}/ssim_runs/ssim_{MODE}_{seed}.npz")
    gt = to01(d["ground_truth"].squeeze())

    ax = axes[2 * r, 0]
    ax.imshow(gt, cmap="gray", vmin=0, vmax=1)
    ax.set_title("ground truth", fontsize=9)
    ax.set_ylabel(f"seed {seed}\nmeasurement", fontsize=9)
    axes[2 * r + 1, 0].axis("off")

    for k in range(K):
        # accumulated measurement with design boxes (red = newest)
        ax = axes[2 * r, k + 1]
        y = np.where(d["masks"][k].squeeze() > 0.5, to01(d["ys"][k].squeeze()), 0.0)
        ax.imshow(y, cmap="gray", vmin=0, vmax=1)
        for j in range(k + 1):
            color = "red" if j == k else "orange"
            ax.add_patch(
                plt.Rectangle(
                    (d["designs"][j, 0] - SIZE / 2, d["designs"][j, 1] - SIZE / 2),
                    SIZE, SIZE, fill=False, edgecolor=color, linewidth=1.8,
                )
            )
        ax.set_title(f"k={k + 1}   SSIM {d['ssims'][k]:.2f}", fontsize=9)
        ax.set_xlim(-0.5, 27.5)
        ax.set_ylim(27.5, -0.5)

        # posterior mean
        ax = axes[2 * r + 1, k + 1]
        ax.imshow(to01(d["samples"][k].squeeze()).mean(axis=0), cmap="gray", vmin=0, vmax=1)
        if k == 0:
            ax.set_ylabel("posterior mean", fontsize=9)

for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
# suptitle after tight_layout, above the axes; bbox_inches="tight" keeps it in frame
fig.suptitle(f"Amortized BED — measurement and posterior evolution ({R} of 20 digits, {MODE})", fontsize=14, y=1.012)
out = f"{S}/paper_repro/figures/measurement_evolution.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"saved {out}")
