"""Figures: side/top/front flow slices and a face air-speed map."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from .geometry import FAN, SOLID  # noqa: E402

CMAP = "viridis"
SOLID_RGBA = ListedColormap([[0, 0, 0, 0], [0.62, 0.64, 0.66, 1.0]])
FAN_RGBA = ListedColormap([[0, 0, 0, 0], [0.0, 0.55, 0.6, 1.0]])


def _overlay(ax, flags2d, extent):
    ax.imshow((flags2d == SOLID).T.astype(int), origin="lower", extent=extent,
              cmap=SOLID_RGBA, interpolation="nearest", vmin=0, vmax=1, zorder=3)
    ax.imshow((flags2d == FAN).T.astype(int), origin="lower", extent=extent,
              cmap=FAN_RGBA, interpolation="nearest", vmin=0, vmax=1, zorder=4)


def _slice(ax, a, b, ua, ub, speed, flags2d, vmax, title, xlabel, ylabel, density=1.4):
    extent = (a[0], a[-1], b[0], b[-1])
    im = ax.imshow(speed.T, origin="lower", extent=extent, cmap=CMAP, vmin=0, vmax=vmax,
                   interpolation="bilinear")
    U = np.nan_to_num(ua.T)
    V = np.nan_to_num(ub.T)
    try:
        ax.streamplot(a, b, U, V, color=(1, 1, 1, 0.55), density=density, linewidth=0.6,
                      arrowsize=0.6, zorder=2)
    except ValueError:
        pass
    _overlay(ax, flags2d, extent)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_aspect("equal")
    return im


def _cm(v):
    return v * 100.0


def summary_figure(path, cfg, geo, u_mean, result):
    g = geo.grid
    speed = result["_speed"]
    vmax = max(cfg.fan.exit_speed, 0.5)
    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1], hspace=0.28, wspace=0.25)

    # side view through the middle of the face (y = 0)
    j = int(np.argmin(np.abs(g.y)))
    ax = fig.add_subplot(gs[:, 0])
    im = _slice(ax, _cm(g.x), _cm(g.z), u_mean[0, :, j, :], u_mean[2, :, j, :], speed[:, j, :],
                geo.flags[:, j, :], vmax, "Side view (centre of face): mean air speed",
                "x [cm]  (face is to the right)", "z [cm]  (0 = hat band)", density=2.0)
    for name, p in result["probes"].items():
        if abs(p["position_m"][1]) < 1e-6:
            ax.plot(_cm(p["position_m"][0]), _cm(p["position_m"][2]), "o", ms=4,
                    mfc="white", mec="k", zorder=6)
            ax.annotate(f"{name} {p['mean_speed']:.2f}", (_cm(p["position_m"][0]), _cm(p["position_m"][2])),
                        xytext=(6, 0), textcoords="offset points", fontsize=7, color="white",
                        zorder=7, va="center")
    ax.set_xlim(-25, _cm(g.x[-1]))
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="m/s")

    # top view at nose height
    zk = int(np.argmin(np.abs(g.z - geo.head.nose_z)))
    ax = fig.add_subplot(gs[0, 1])
    _slice(ax, _cm(g.x), _cm(g.y), u_mean[0, :, :, zk], u_mean[1, :, :, zk], speed[:, :, zk],
           geo.flags[:, :, zk], vmax, f"Top view at nose height (z={_cm(g.z[zk]):.0f} cm)",
           "x [cm]", "y [cm]")

    # front view just ahead of the face
    xf = result["probes"]["forehead"]["position_m"][0]
    i = int(np.argmin(np.abs(g.x - xf)))
    ax = fig.add_subplot(gs[0, 2])
    _slice(ax, _cm(g.y), _cm(g.z), u_mean[1, i, :, :], u_mean[2, i, :, :], speed[i, :, :],
           geo.flags[i, :, :], vmax, f"Front view, plane x={_cm(g.x[i]):.1f} cm",
           "y [cm]  (wearer's left ->)", "z [cm]")
    ax.invert_xaxis()  # looking at the wearer
    ax.set_ylim(-35, _cm(g.z[-1]))

    # face map
    ys, zs, fmap = result["_face_map"]
    fc = cfg.face
    ax = fig.add_subplot(gs[1, 1])
    im2 = ax.imshow(fmap, origin="lower", extent=(_cm(ys[0]), _cm(ys[-1]), _cm(zs[0]), _cm(zs[-1])),
                    cmap="magma", vmin=0, vmax=max(1.2, fc.strong * 1.2), interpolation="bilinear")
    Yc, Zc = np.meshgrid(_cm(ys), _cm(zs))
    fm = np.nan_to_num(fmap, nan=-1)
    cs = ax.contour(Yc, Zc, fm, levels=[fc.perceptible, fc.cooling], colors=["#9ecae1", "white"],
                    linewidths=1.0)
    ax.clabel(cs, fmt={fc.perceptible: f"{fc.perceptible} m/s", fc.cooling: f"{fc.cooling} m/s"},
              fontsize=7)
    for name, p in result["probes"].items():
        ax.plot(_cm(p["position_m"][1]), _cm(p["position_m"][2]), "o", ms=3, color="cyan")
        ax.annotate(f"{p['mean_speed']:.2f}", (_cm(p["position_m"][1]), _cm(p["position_m"][2])),
                    xytext=(4, 3), textcoords="offset points", fontsize=7, color="cyan")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.set_title(f"Air speed {fc.probe_offset*100:.1f} cm in front of the skin", fontsize=10)
    ax.set_xlabel("y [cm]")
    ax.set_ylabel("z [cm]")
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.02, label="m/s")

    # verdict text
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    r = result
    lines = [
        r["headline"], "",
        f"Mean speed over face:     {r['face_mean_speed']:.2f} m/s ({r['face_feel']})",
        f"Max speed over face:      {r['face_max_speed']:.2f} m/s",
        f"Face area > {fc.perceptible} m/s (felt): {100*r['face_fraction_perceptible']:.0f}%",
        f"Face area > {fc.cooling} m/s (cooling): {100*r['face_fraction_cooling']:.0f}%",
        "",
        f"Fan: {cfg.fan.exit_speed} m/s {cfg.fan.direction}, tilt {cfg.fan.tilt_inward_deg} deg inward,",
        f"     swirl {cfg.fan.swirl_ratio}, arc {cfg.fan.active_arc_deg} deg",
        f"     open area {r['fan_open_area_m2']*1e4:.0f} cm^2, flow {r['fan_flow_cfm']:.0f} CFM",
        f"Brim droop: {cfg.hat.droop_deg} deg" + ("" if cfg.hat.droop_arc_deg >= 360 else f" over front {cfg.hat.droop_arc_deg} deg"),
        f"Wind: {tuple(cfg.env.wind)} m/s",
        "", "Probe speeds (m/s):",
    ]
    for name, p in r["probes"].items():
        lines.append(f"  {name:<12} {p['mean_speed']:5.2f}  {p['feel']}")
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8.5)

    fig.suptitle(f"{cfg.name}: does the fan air reach the face?", fontsize=14)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def geometry_figure(path, cfg, geo):
    """Cross-sections of the voxel model (to check STL alignment before a run)."""
    g = geo.grid
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.5))
    j = int(np.argmin(np.abs(g.y)))
    band = (g.z > -0.08) & (g.z < cfg.hat.brim_thickness + 0.01)
    sub = geo.flags[:, :, band]
    plan = np.where((sub == FAN).any(2), FAN, np.where((sub == SOLID).any(2), SOLID, 0))
    kk = int(np.argmin(np.abs(g.z - geo.head.nose_z)))
    for ax, sl, ext, title, lab in [
        (axs[0], geo.flags[:, j, :], (g.x, g.z), "side (y=0)", ("x", "z")),
        (axs[1], plan, (g.x, g.y), "plan view of brim + fan ring", ("x", "y")),
        (axs[2], geo.flags[:, :, kk], (g.x, g.y), "top, nose height", ("x", "y")),
    ]:
        a, b = ext
        ax.imshow(np.zeros_like(sl, dtype=float).T, origin="lower", cmap="Greys", vmin=0, vmax=1,
                  extent=(_cm(a[0]), _cm(a[-1]), _cm(b[0]), _cm(b[-1])))
        _overlay(ax, sl, (_cm(a[0]), _cm(a[-1]), _cm(b[0]), _cm(b[-1])))
        ax.set_title(title)
        ax.set_xlabel(f"{lab[0]} [cm]")
        ax.set_ylabel(f"{lab[1]} [cm]")
        ax.set_aspect("equal")
    fig.suptitle(f"{cfg.name}: voxel geometry (grey = solid, teal = fan)")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def sweep_figure(path, param, rows, cfg):
    fc = cfg.face
    xs = [str(r["value"]) for r in rows]
    mean = [r["face_mean_speed"] for r in rows]
    cool = [100 * r["face_fraction_cooling"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(xs, mean, color="#2b8cbe")
    ax.axhline(fc.perceptible, ls=":", color="grey", label=f"felt ({fc.perceptible} m/s)")
    ax.axhline(fc.cooling, ls="--", color="k", label=f"cooling ({fc.cooling} m/s)")
    for xi, m, c in zip(xs, mean, cool):
        ax.annotate(f"{m:.2f} m/s\n{c:.0f}% cooled", (xi, m), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_xlabel(param)
    ax.set_ylabel("mean air speed over face [m/s]")
    ax.set_ylim(0, max(max(mean) * 1.3, fc.cooling * 1.3))
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"Sweep of {param}")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
