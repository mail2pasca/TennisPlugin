import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hatflow import lbm  # noqa: E402
from hatflow.analysis import analyse  # noqa: E402
from hatflow.config import load_config  # noqa: E402
from hatflow.geometry import FAN, SOLID, build_geometry  # noqa: E402


def tiny(*extra):
    return load_config(overrides=["domain.dx=0.02", "solver.sim_time=0.15",
                                  "solver.average_from=0.1", "solver.report_every=100000", *extra])


def test_head_girth_matches_hat_size():
    cfg = tiny()
    a, b = cfg.head_semi_axes()
    per = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
    assert abs(per - 0.584) < 1e-6


def test_geometry_has_fan_ring_and_head():
    geo = build_geometry(tiny())
    assert (geo.flags == FAN).sum() > 0
    i, j, k = geo.grid.index(0.0, 0.0, -0.05)
    assert geo.flags[i, j, k] == SOLID
    fan = geo.flags == FAN
    assert np.all(geo.fan_velocity[2][fan] < 0)  # blowing down


def test_run_produces_finite_face_speeds():
    cfg = tiny()
    geo = build_geometry(cfg)
    u, rms, info = lbm.run(geo, cfg, log=lambda *_: None)
    res = analyse(cfg, geo, u, rms, info)
    assert all(math.isfinite(p["mean_speed"]) for p in res["probes"].values())
    assert 0 <= res["face_fraction_cooling"] <= 1
    # air leaving the fan layer moves downward on average
    fan = geo.flags == FAN
    below = np.roll(fan, -1, axis=2)
    assert np.nanmean(u[2][below]) < 0
