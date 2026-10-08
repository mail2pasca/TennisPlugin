"""Write results: JSON summary, Markdown report, ParaView-readable VTK."""
from __future__ import annotations

import json
import math

import numpy as np


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not math.isfinite(v) else round(v, 5)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def write_json(path, cfg, result):
    with open(path, "w") as fh:
        json.dump({"config": cfg.to_dict(), "result": _clean(result)}, fh, indent=2)


def write_report(path, cfg, result, figure_name="summary.png"):
    r, fc = result, cfg.face
    s = r["solver"]
    lines = [
        f"# {cfg.name}: airflow to the face", "",
        f"**{r['headline']}**", "",
        f"![summary]({figure_name})", "",
        "| metric | value |", "|---|---|",
        f"| mean air speed over face | {r['face_mean_speed']:.2f} m/s ({r['face_feel']}) |",
        f"| max air speed over face | {r['face_max_speed']:.2f} m/s |",
        f"| face area feeling > {fc.perceptible} m/s | {100*r['face_fraction_perceptible']:.0f}% |",
        f"| face area cooled > {fc.cooling} m/s | {100*r['face_fraction_cooling']:.0f}% |",
        f"| fan exit speed / open area | {cfg.fan.exit_speed} m/s / {r['fan_open_area_m2']*1e4:.0f} cm² |",
        f"| fan volume flow | {r['fan_flow_m3s']*1000:.0f} L/s ({r['fan_flow_cfm']:.0f} CFM) |",
        "", "## Probes", "",
        "| point | mean speed m/s | turbulence rms m/s | feel |", "|---|---|---|---|",
    ]
    for n, p in r["probes"].items():
        lines.append(f"| {n} | {p['mean_speed']:.2f} | {p['rms_fluctuation']:.2f} | {p['feel']} |")
    lines += [
        "", "## Solver", "",
        f"{s['cells']:,} cells at {cfg.domain.dx*1000:.0f} mm, {s['steps']} steps "
        f"({cfg.solver.sim_time} s simulated, averaged over the last "
        f"{s['averaged_steps']*s['dt']:.2f} s), runtime {s['runtime_s']:.0f} s.", "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def write_vtk(path, geo, u_mean, u_rms):
    """Legacy binary VTK structured points (open in ParaView)."""
    g = geo.grid
    nx, ny, nz = g.shape
    n = nx * ny * nz

    def be(a):  # VTK wants x fastest, big-endian float32
        return np.ascontiguousarray(np.nan_to_num(a).transpose(2, 1, 0), dtype=">f4").tobytes()

    vec = np.stack([np.nan_to_num(u_mean[c]).transpose(2, 1, 0).ravel() for c in range(3)], axis=1)
    with open(path, "wb") as fh:
        fh.write(b"# vtk DataFile Version 3.0\nhat fan flow\nBINARY\nDATASET STRUCTURED_POINTS\n")
        fh.write(f"DIMENSIONS {nx} {ny} {nz}\nORIGIN {g.x[0]} {g.y[0]} {g.z[0]}\n"
                 f"SPACING {g.dx} {g.dx} {g.dx}\nPOINT_DATA {n}\n".encode())
        fh.write(b"VECTORS velocity float\n")
        fh.write(np.ascontiguousarray(vec, dtype=">f4").tobytes())
        fh.write(b"\nSCALARS speed float 1\nLOOKUP_TABLE default\n")
        fh.write(be(np.sqrt((np.nan_to_num(u_mean) ** 2).sum(0))))
        fh.write(b"\nSCALARS turbulence_rms float 1\nLOOKUP_TABLE default\n")
        fh.write(be(u_rms))
        fh.write(b"\nSCALARS cell_type float 1\nLOOKUP_TABLE default\n")
        fh.write(be(geo.flags.astype(np.float32)))
        fh.write(b"\n")
