"""Per-run panels (measurements+designs, posterior mean, design trajectories, EIG traces).

Usage: python scripts/plot_comparison.py [runs_dir]  (default: runs)
Plots every {runs_dir}/bed_*.npz that has design trajectories recorded.
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

S = sys.argv[1] if len(sys.argv) > 1 else "runs"
RUNS = [
    (os.path.basename(f), os.path.basename(f)[4:-4])
    for f in sorted(glob.glob(f"{S}/bed_*.npz"))
    if "design_trajs" in np.load(f)
]
SIZE = 7
R = len(RUNS)

fig, axes = plt.subplots(R, 4, figsize=(13.6, 3.2 * R), width_ratios=[1, 1, 1.25, 1.5], squeeze=False)

for r, (fname, label) in enumerate(RUNS):
    d = np.load(f"{S}/{fname}")
    gt = d["ground_truth"].squeeze()
    designs, ys, masks, samples = d["designs"], d["ys"], d["masks"], d["samples"]
    trajs, eigs = d["design_trajs"], d["eig_trajs"]
    K = designs.shape[0]

    # Model space is [-1,1] for correctly-scaled runs; map to [0,1] for display
    model_space = gt.min() < -0.5
    if model_space:
        gt = (gt + 1) / 2
        y_disp = np.where(masks[-1].squeeze() > 0.5, (ys[-1].squeeze() + 1) / 2, 0.0)
        samples = (samples + 1) / 2
    else:
        y_disp = ys[-1].squeeze()

    ax = axes[r, 0]
    ax.imshow(y_disp, cmap="gray", vmin=0, vmax=1)
    for j in range(K):
        ax.add_patch(
            plt.Rectangle(
                (designs[j, 0] - SIZE / 2, designs[j, 1] - SIZE / 2),
                SIZE, SIZE, fill=False, edgecolor=plt.get_cmap("autumn")(j / K), linewidth=1.5,
            )
        )
        ax.text(designs[j, 0], designs[j, 1], str(j + 1), color="cyan", fontsize=7, ha="center", va="center")
    ax.set_title("final measurements + designs", fontsize=8)
    ax.set_ylabel(label, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-0.5, 27.5); ax.set_ylim(27.5, -0.5)

    ax = axes[r, 1]
    pm = samples[-1].mean(axis=0).squeeze()
    mse = np.mean((np.clip(pm, 0, 1) - gt) ** 2)
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    ax.imshow(pm, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"posterior mean (PSNR {psnr:.1f} dB)", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[r, 2]
    ax.imshow(gt, cmap="gray", vmin=0, vmax=1, extent=[0, 28, 28, 0], alpha=0.5)
    for k in range(K):
        c = plt.get_cmap("autumn")(k / K)
        ax.plot(trajs[k, :, 0], trajs[k, :, 1], "-", color=c, lw=0.9, alpha=0.85)
        ax.scatter(*trajs[k, -1], marker="x", s=40, color=c)
    ax.add_patch(plt.Rectangle((0, 0), 28, 28, fill=False, edgecolor="black", lw=1, ls=":"))
    ax.set_title("design trajectories (x = final)", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-4, 33); ax.set_ylim(33, -4)
    ax.tick_params(labelsize=6)

    ax = axes[r, 3]
    for k in range(K):
        ax.plot(np.abs(eigs[k]) + 1e-12, lw=0.8, color=plt.get_cmap("autumn")(k / K), label=f"meas {k + 1}")
    ax.set_yscale("log")
    ax.set_title("|EIG surrogate| (log)", fontsize=8)
    ax.set_xlabel("opt step", fontsize=7)
    ax.tick_params(labelsize=6)
    if r == 0:
        ax.legend(fontsize=5, ncol=2)

fig.suptitle("CoDiff BED — run comparison", fontsize=12)
fig.tight_layout()
fig.savefig(f"{S}/paper_repro/figures/run_comparison.png", dpi=150, bbox_inches="tight")
print("saved run_comparison.png")
