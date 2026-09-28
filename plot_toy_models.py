#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot correlation dimension from MSC output files.
Reads MSC .txt files, computes CD = 3 + d(ln MSC)/d(ln r),
and plots all four toy model curves with 2D density insets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ========================================
# CONFIGURATION
# ========================================
input_dir = "."  # Change to wherever your MSC files are saved
particle_dir = "."  # Change to wherever your .npy files are

distributions = [
    ("uniform", "Uniform", "C0", "a)"),
    ("gaussian", "Gaussian", "C1", "b)"),
    ("void", "Giant Void", "C2", "c)"),
    ("fractal", "Swiss Cheese", "C3", "d)"),
]

# Radii at which to evaluate CD (midpoints between r_edges)
r_min_cd = 25.0  # Mpc/h
r_max_cd = 250.0  # Mpc/h
rebin = 8

# ========================================
# LAYOUT: 1 row of 4 density panels spanning full figure width on top,
# square CD panel below. Density panels placed manually so they fill
# edge-to-edge (overhanging the y-label area of the CD panel).
# ========================================
fig = plt.figure(figsize=(4.5, 5.5))

# CD panel via gridspec — sits in the lower portion of the figure
gs = fig.add_gridspec(1, 1, left=0.13, right=0.97, top=0.776, bottom=0.09)
ax_cd = fig.add_subplot(gs[0, 0])

# 4 density panels placed manually in figure coordinates.
# Width and height computed so panels are exactly square:
#   panel_width  = (1.00 + 0.05 - 3*0.01)/4 = 0.255  ->  1.1475 in
#   panel_height = 1.1475 / 5.5 (fig height)            = 0.2086 frac
panel_top = 1.00
panel_bot = 0.7914  # = panel_top - 0.1984
panel_width = (1.00 + 0.05 - 3 * 0.01) / 4  # 0.2425
inset_axes = []
for i in range(4):
    left = -0.05 + i * (panel_width + 0.01)
    ax = fig.add_axes([left, panel_bot, panel_width, panel_top - panel_bot])
    inset_axes.append(ax)

# ========================================
# MAIN LOOP
# ========================================
for idx, (dist_name, label, colour, panel) in enumerate(distributions):

    # --- Read MSC ---
    fname = f"{input_dir}/MSC_{dist_name}_000.txt"
    MSC = pd.read_csv(
        fname,
        sep=r"\s+",
        names=["r_edges", "MSC", "counts"],
        header=None,
        skiprows=1,
    )
    MSC = MSC.iloc[::rebin].reset_index(drop=True)
    # --- Compute correlation dimension ---
    r_edges = MSC["r_edges"].to_numpy()
    midpoint = np.exp(0.5 * (np.log(r_edges[:-1]) + np.log(r_edges[1:])))

    mask = (midpoint >= r_min_cd) & (midpoint <= r_max_cd)
    midpoint = midpoint[mask]

    # CD = 3 + d(ln MSC) / d(ln r)
    spline = CubicSpline(np.log(r_edges), np.log(MSC["MSC"].to_numpy()))
    CD = 3.0 + spline.derivative()(np.log(midpoint))

    # --- Plot CD curve ---
    ax_cd.plot(midpoint, CD, color=colour, linewidth=1.3, label=f"{panel} {label}")

    # --- Read MSC ---
    fname = f"{input_dir}/MSC_{dist_name}_zmatch_000.txt"
    MSC = pd.read_csv(
        fname,
        sep=r"\s+",
        names=["r_edges", "MSC", "counts"],
        header=None,
        skiprows=1,
    )
    MSC = MSC.iloc[::rebin].reset_index(drop=True)

    # --- Compute correlation dimension ---
    r_edges = MSC["r_edges"].to_numpy()
    midpoint = np.exp(0.5 * (np.log(r_edges[:-1]) + np.log(r_edges[1:])))

    mask = (midpoint >= r_min_cd) & (midpoint <= r_max_cd)
    midpoint = midpoint[mask]

    # CD = 3 + d(ln MSC) / d(ln r)
    spline = CubicSpline(np.log(r_edges), np.log(MSC["MSC"].to_numpy()))
    CD = 3.0 + spline.derivative()(np.log(midpoint))

    # --- Plot CD curve ---
    ax_cd.plot(midpoint, CD, color=colour, linestyle="--", linewidth=1.3)

    # --- 2D density panel ---
    ax_in = inset_axes[idx]
    particles = np.load(f"{particle_dir}/{dist_name}_particles.npy")

    # Cut based on radial distance from origin
    origin = np.array([0.0, 0.0, 0.0])
    # origin = np.array([-200.0, -200.0, -200.0])

    # Calculate maximum safe radius: distance from origin to nearest box edge for both observers
    box_size = 2000.0
    box_min = -box_size / 2
    box_max = box_size / 2
    r_max = min(
        origin[0] - box_min,  # distance to left edge
        box_max - origin[0],  # distance to right edge
        origin[1] - box_min,  # distance to front edge
        box_max - origin[1],  # distance to back edge
        origin[2] - box_min,  # distance to bottom edge
        box_max - origin[2],  # distance to top edge
    )
    print(f"Maximum radial distance for observer: {r_max:.2f} Mpc/h")

    # Apply radial cut to data
    # r_data = np.sqrt(np.sum((particles - origin[np.newaxis, :]) ** 2, axis=1))
    # data_mask = r_data <= r_max
    # particles = particles[data_mask]

    slice_tol = 100  # Mpc/h
    mask_slice = np.abs(particles[:, 2]) < slice_tol
    x = particles[mask_slice, 0]
    y = particles[mask_slice, 1]

    ax_in.hist2d(x, y, bins=200, range=[[-1000, 1000], [-1000, 1000]], cmap="viridis", shading="auto")
    ax_in.set_xlim(-1000, 1000)
    ax_in.set_ylim(-1000, 1000)
    ax_in.set_xticks([])
    ax_in.set_yticks([])

    # Add a dashed circle showing the observational radius for the second observer at origin
    ax_in.plot(origin[0], origin[1], color="w", marker="x", markersize=4, markeredgewidth=1.3)
    circle = plt.Circle((origin[0], origin[1]), r_max, color="w", linestyle="--", fill=False, linewidth=1.5)
    ax_in.add_patch(circle)

    # Thick black border matching reference style
    for spine in ax_in.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color("k")

    # Panel label inside
    ax_in.text(
        0.05,
        0.88,
        panel,
        transform=ax_in.transAxes,
        fontsize=11,
        fontweight="bold",
        color="white",
        bbox=dict(facecolor="k", alpha=0.5, pad=2.0, edgecolor="none"),
    )

# ========================================
# CD PANEL STYLING — match reference
# ========================================

# d=3 dashed reference line (like the 2.97 dashed line in reference)
ax_cd.axhline(3.00, color="k", linestyle="-", linewidth=1.5)
ax_cd.axhline(2.97, color="k", linestyle="--", linewidth=1.5)

# Legend inside the plot, lower-right, with box (like reference)
ax_cd.legend(fontsize=10, loc="lower right", framealpha=0.95, edgecolor="k", fancybox=False)

# Axis labels and ticks matching reference ~12-14pt
ax_cd.set_xlabel(r"$r\;[h^{-1}\,\mathrm{Mpc}]$", fontsize=12)
ax_cd.set_ylabel(r"$D_2(r)$", fontsize=12)
ax_cd.tick_params(labelsize=10)

# y-axis range similar to reference (2.80–3.00)
ax_cd.set_ylim(2.80, 3.01)
ax_cd.set_xlim(r_min_cd, r_max_cd)

text = ax_cd.text(
    0.54,
    0.02,
    # f"$\mathrm{{Observer\,at:}}$                    \n" + f"[-200,-200,-200] $h^{{{-1}}}\;\mathrm{{Mpc}}$",
    f"$\mathrm{{Observer\,at:}}$     \n" + f"[0,0,0] $h^{{{-1}}}\;\mathrm{{Mpc}}$",
    transform=ax_cd.transAxes,
    verticalalignment="bottom",
    horizontalalignment="right",
    fontsize=10,
)

plt.savefig("correlation_dimension_toy_models_000.png", dpi=300, bbox_inches="tight")
print("Saved correlation_dimension_toy_models.png")
