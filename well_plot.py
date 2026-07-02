"""
well_plot.py
Visualize Utah FORGE well 16B channel survey parameters.

Run:
    python well_plot.py
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

SURVEY_PATH = "/Users/sj201/Downloads/ChnlCoord16B_stimulation.csv"
SAVE_PATH   = "well_survey_16B.png"

# ── load ──────────────────────────────────────────────────────────────────────
rows = []
with open(SURVEY_PATH, newline="") as f:
    for r in csv.DictReader(f):
        rows.append({k: float(v) for k, v in r.items()})

ch    = np.array([r["Channel number"] for r in rows])
xN    = np.array([r["xN (m)"]        for r in rows])
yE    = np.array([r["yE (m)"]        for r in rows])
fd    = np.array([r["FD (m)"]        for r in rows])
md    = np.array([r["MD (m)"]        for r in rows])
tvd   = np.array([r["TVD (m)"]       for r in rows])

fd_offset = fd - fd[0]

# ── write reformatted CSV (channel, x_m, y_m, z_m) ───────────────────────────
OUT_CSV = "/Users/sj201/Downloads/survey.csv"
with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["channel", "x_m", "y_m", "z_m"])
    for i in range(len(ch)):
        writer.writerow([int(ch[i]), xN[i], yE[i], tvd[i]])
print(f"Survey CSV saved → {OUT_CSV}")

norm = plt.Normalize(md.min(), md.max())
cmap = cm.viridis

# ── layout: 2×2 ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    "Utah FORGE Well 16B — DAS Channel Survey",
    fontsize=13, fontweight="bold",
)
gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32,
                      left=0.07, right=0.96, top=0.93, bottom=0.06)

ax3d   = fig.add_subplot(gs[0, 0], projection="3d")
ax_map = fig.add_subplot(gs[0, 1])
ax_sec = fig.add_subplot(gs[1, 0])
ax_dep = fig.add_subplot(gs[1, 1])

# ── 1. 3D Trajectory (real UTM coords) ───────────────────────────────────────
sc3 = ax3d.scatter(yE, xN, -tvd, c=md, cmap=cmap, norm=norm, s=2, linewidths=0)
ax3d.scatter(yE[0],  xN[0],  -tvd[0],  color="limegreen", s=60, zorder=5,
             depthshade=False, label=f"Top  ch{int(ch[0])}")
ax3d.scatter(yE[-1], xN[-1], -tvd[-1], color="red",       s=60, zorder=5,
             depthshade=False, label=f"Bottom  ch{int(ch[-1])}")
ax3d.set_xlabel("Easting — yE (m)",  fontsize=7, labelpad=4)
ax3d.set_ylabel("Northing — xN (m)", fontsize=7, labelpad=4)
ax3d.set_zlabel("Depth (m)",         fontsize=7, labelpad=4)
ax3d.set_title("3D Trajectory", fontsize=9, fontweight="bold")
ax3d.tick_params(labelsize=6)
ax3d.legend(fontsize=7)
fig.colorbar(sc3, ax=ax3d, label="MD (m)", shrink=0.55, pad=0.12, aspect=20)

# ── 2. Plan View (real UTM coords) ───────────────────────────────────────────
sc_map = ax_map.scatter(yE, xN, c=md, cmap=cmap, norm=norm, s=3, zorder=3)
ax_map.plot(yE, xN, color="gray", lw=0.4, alpha=0.4, zorder=2)
ax_map.scatter(yE[0],  xN[0],  color="limegreen", s=80, zorder=5,
               label=f"Top  ch{int(ch[0])}")
ax_map.scatter(yE[-1], xN[-1], color="red",       s=80, zorder=5,
               label=f"Bottom  ch{int(ch[-1])}")
ax_map.set_xlabel("Easting — yE (m)",  fontsize=8)
ax_map.set_ylabel("Northing — xN (m)", fontsize=8)
ax_map.set_title("Plan View (Map)", fontsize=9, fontweight="bold")
ax_map.tick_params(labelsize=7)
ax_map.set_aspect("equal")
ax_map.legend(fontsize=7, loc="upper left")
ax_map.grid(True, alpha=0.25)
fig.colorbar(sc_map, ax=ax_map, label="MD (m)", aspect=25)

# ── 3. Section: TVD vs MD ─────────────────────────────────────────────────────
sc_sec = ax_sec.scatter(md, tvd, c=md, cmap=cmap, norm=norm, s=2)
ax_sec.plot([0, md.max()], [0, md.max()], "k--", lw=0.9, alpha=0.5,
            label="MD = TVD (vertical)")
ko_idx = np.argmax((md - tvd) > 5)
if ko_idx:
    ax_sec.annotate(
        f"Kick-off\nMD≈{md[ko_idx]:.0f} m",
        xy=(md[ko_idx], tvd[ko_idx]),
        xytext=(md[ko_idx] + 150, tvd[ko_idx] - 200),
        fontsize=7, color="crimson",
        arrowprops=dict(arrowstyle="->", color="crimson", lw=0.8),
    )
ax_sec.set_xlabel("Measured Depth — MD (m)", fontsize=8)
ax_sec.set_ylabel("True Vertical Depth — TVD (m)", fontsize=8)
ax_sec.set_title("Section: TVD vs MD", fontsize=9, fontweight="bold")
ax_sec.tick_params(labelsize=7)
ax_sec.legend(fontsize=7)
ax_sec.invert_yaxis()
ax_sec.grid(True, alpha=0.25)
fig.colorbar(sc_sec, ax=ax_sec, label="MD (m)", aspect=25)

# ── 4. Depth Parameters vs Channel ───────────────────────────────────────────
ax_dep.plot(ch, fd_offset, label="FD — Fiber Depth (relative)", lw=1.2, color="steelblue")
ax_dep.plot(ch, md,        label="MD — Measured Depth",          lw=1.2, color="darkorange", ls="--")
ax_dep.plot(ch, tvd,       label="TVD — True Vertical Depth",    lw=1.2, color="green",      ls="-.")
ax_dep.set_xlabel("Channel Number", fontsize=8)
ax_dep.set_ylabel("Depth (m)", fontsize=8)
ax_dep.set_title("Depth Parameters vs Channel", fontsize=9, fontweight="bold")
ax_dep.legend(fontsize=7, loc="upper left")
ax_dep.tick_params(labelsize=7)
ax_dep.grid(True, alpha=0.25)

# ── save & show ───────────────────────────────────────────────────────────────
plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {SAVE_PATH}")
plt.show()
