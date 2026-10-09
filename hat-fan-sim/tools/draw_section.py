"""Draw the centre-line section of the printable parts (for the report).

Run: python tools/draw_section.py configs/v1_23in.json out.png --set hat.droop_deg=20
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hatflow import cad  # noqa: E402
from hatflow.config import load_config  # noqa: E402
from hatflow.geometry import HeadShape  # noqa: E402

COLORS = {"crown": "#cbb994", "housing": "#8d969c", "top_grille": "#aab2b7", "rotor": "#16858c",
          "pinion": "#c9971c", "pod_tray": "#6f7b81", "pod_lid": "#9aa5aa"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    parts, info = cad.build_parts(cfg, 128)
    b = cad.Brim(cfg)
    mx, my = info["motor_xy"]
    parts["pinion"] = parts["pinion"].translate([mx, my, info["gear_z0"] + 2.0])

    fig, ax = plt.subplots(figsize=(13, 5.6))
    for n in COLORS:
        cs = parts[n].rotate([90, 0, 0]).slice(0.0)
        for poly in cs.to_polygons():
            p = np.array(poly)
            p[:, 1] *= -1
            ax.add_patch(Polygon(p, closed=True, fc=COLORS[n], ec="#2b3135", lw=0.4))

    # bought parts, drawn in the local brim frame at the back (x < 0)
    t = np.radians(b.droop)

    def local_rect(u0, w0, du, dw, **kw):
        # local (u, w) -> world (x, z) at azimuth 180 deg
        corners = np.array([[u0, w0], [u0 + du, w0], [u0 + du, w0 + dw], [u0, w0 + dw]])
        x = b.hinge + corners[:, 0] * np.cos(t) + corners[:, 1] * np.sin(t)
        z = b.zc - corners[:, 0] * np.sin(t) + corners[:, 1] * np.cos(t)
        ax.add_patch(Polygon(np.column_stack([-x, z]), closed=True, **kw))

    um = cad._u(b, info["motor_radius"])
    w_grille = cad.CHANNEL_HALF + cad.GRILLE_DEPTH
    local_rect(um - 7.5, w_grille - 2.4, 15, 25, fc="#d9dde0", ec="#2b3135", lw=0.6, hatch="////")
    uw = cad._u(b, info["brim_radius"])
    local_rect(uw + 6, -11.4, 65 * 0 + 18.5, 18.5, fc="#e8d27a", ec="#2b3135", lw=0.6)   # 18650 end-on
    local_rect(uw + 30, -12.4, 26, 1.6, fc="#3d8a52", ec="#2b3135", lw=0.4)             # charger board

    h = HeadShape(cfg)
    zs = np.linspace(h.zc - h.c, h.zc + h.c, 200)
    xf = np.array([h.A * float(h._profile(z)) * 1000 for z in zs])
    ax.plot(xf, zs * 1000, "--", color="#6b7378", lw=1)
    ax.plot(-xf, zs * 1000, "--", color="#6b7378", lw=1)
    ax.text(0, -60, "head\n(23 in)", ha="center", color="#6b7378", fontsize=9)

    def lab(x, y, text, tx, ty):
        ax.annotate(text, (x, y), xytext=(tx, ty), fontsize=9, color="#1d2226",
                    arrowprops=dict(arrowstyle="-", color="#1d2226", lw=0.6))

    lab(150, b.zm(150), "blade ring (rotor)", 150, 75)
    lab(178, b.zm(178) - 4, "356-tooth rim gear", 240, -5)
    lab(140, b.zm(140) - 14, "bottom grille", 190, -60)
    lab(140, b.zm(140) + 14, "top grille", 230, 45)
    lab(-60, 120, "crown", -150, 140)
    lab(-info["motor_radius"], b.zm(info["motor_radius"]) + 30, "130 motor (shaft down)", -250, 110)
    lab(-info["motor_radius"], info["gear_z0"] + 7, "16T pinion", -330, 10)
    lab(-(info["brim_radius"] + 20), b.zm(info["brim_radius"] + 20) - 2, "18650 cell", -330, -40)
    lab(-(info["brim_radius"] + 45), b.zm(info["brim_radius"] + 45) - 10, "USB-C charger board", -320, -85)
    lab(-(info["brim_radius"] + 40), b.zm(info["brim_radius"] + 40) + 40, "electronics pod", -300, 75)
    ax.text(330, -95, "face →", fontsize=10, color="#6b7378", ha="right")
    ax.set_aspect("equal")
    ax.set_xlim(-340, 330)
    ax.set_ylim(-100, 160)
    ax.axis("off")
    fig.savefig(a.out, dpi=130, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
