"""Command line: python -m hatflow {run,sweep,geometry} ..."""
from __future__ import annotations

import argparse
import csv
import os
import sys

from .config import clone, load_config, set_value

QUICK = ["domain.dx=0.012", "solver.sim_time=0.8", "solver.average_from=0.4"]


def _cfg(args):
    overrides = (QUICK if args.quick else []) + (args.set or [])
    return load_config(args.config, overrides)


def simulate(cfg, out_dir, vtk=False, log=print):
    from . import lbm
    from .analysis import analyse
    from .export import write_json, write_report, write_vtk
    from .geometry import build_geometry
    from .plots import summary_figure

    os.makedirs(out_dir, exist_ok=True)
    geo = build_geometry(cfg)
    u_mean, u_rms, info = lbm.run(geo, cfg, log=log)
    result = analyse(cfg, geo, u_mean, u_rms, info)
    summary_figure(os.path.join(out_dir, "summary.png"), cfg, geo, u_mean, result)
    write_json(os.path.join(out_dir, "results.json"), cfg, result)
    write_report(os.path.join(out_dir, "report.md"), cfg, result)
    if vtk:
        write_vtk(os.path.join(out_dir, "flow.vtk"), geo, u_mean, u_rms)
    log(f"\n{result['headline']}")
    log(f"mean face speed {result['face_mean_speed']:.2f} m/s, "
        f"{100*result['face_fraction_cooling']:.0f}% of face > {cfg.face.cooling} m/s")
    log(f"results in {out_dir}/ (summary.png, report.md, results.json{', flow.vtk' if vtk else ''})")
    return result


def cmd_run(args):
    cfg = _cfg(args)
    simulate(cfg, args.out or os.path.join("results", cfg.name), vtk=args.vtk)


def cmd_geometry(args):
    from .geometry import build_geometry
    from .plots import geometry_figure

    cfg = _cfg(args)
    out = args.out or os.path.join("results", cfg.name)
    os.makedirs(out, exist_ok=True)
    geo = build_geometry(cfg)
    path = os.path.join(out, "geometry.png")
    geometry_figure(path, cfg, geo)
    a, b = geo.head.a_band, geo.head.b_band
    print(f"grid {geo.grid.shape}, head semi-axes at band {a*100:.1f} x {b*100:.1f} cm, "
          f"fan open area {geo.fan_area*1e4:.0f} cm^2 -> {path}")


def cmd_sweep(args):
    from .plots import sweep_figure

    base = _cfg(args)
    out = args.out or os.path.join("results", f"{base.name}-sweep-{args.param}")
    rows = []
    for v in args.values:
        cfg = clone(base)
        set_value(cfg, f"{args.param}={v}")
        cfg.name = f"{base.name} {args.param}={v}"
        print(f"\n=== {args.param} = {v} ===")
        r = simulate(cfg, os.path.join(out, f"{args.param}={v}"), log=print)
        rows.append(dict(value=v, headline=r["headline"], face_mean_speed=r["face_mean_speed"],
                         face_max_speed=r["face_max_speed"],
                         face_fraction_perceptible=r["face_fraction_perceptible"],
                         face_fraction_cooling=r["face_fraction_cooling"],
                         **{f"probe {n}": p["mean_speed"] for n, p in r["probes"].items()}))
    with open(os.path.join(out, "sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sweep_figure(os.path.join(out, "sweep.png"), args.param, rows, base)
    print(f"\n{args.param:>24} | face mean m/s | cooled area | verdict")
    for r in rows:
        print(f"{r['value']:>24} | {r['face_mean_speed']:13.2f} | {100*r['face_fraction_cooling']:10.0f}% | "
              f"{r['headline']}")
    print(f"sweep results in {out}/ (sweep.png, sweep.csv)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="hatflow", description="Ring-fan hat airflow simulator")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("-c", "--config", help="JSON config (see configs/)")
        sp.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="override a setting, e.g. --set fan.exit_speed=3 (repeatable)")
        sp.add_argument("--quick", action="store_true", help="coarse 12 mm grid, ~30 s per run")
        sp.add_argument("-o", "--out", help="output directory")

    sp = sub.add_parser("run", help="simulate one design")
    common(sp)
    sp.add_argument("--vtk", action="store_true", help="also write flow.vtk for ParaView")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("sweep", help="simulate several values of one setting")
    common(sp)
    sp.add_argument("--param", required=True, help="e.g. fan.tilt_inward_deg")
    sp.add_argument("--values", required=True, nargs="+")
    sp.set_defaults(fn=cmd_sweep)

    sp = sub.add_parser("geometry", help="draw the voxel model only (check STL placement)")
    common(sp)
    sp.set_defaults(fn=cmd_geometry)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
