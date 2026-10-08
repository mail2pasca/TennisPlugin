"""Turn the flow field into answers: how much air reaches the face?"""
from __future__ import annotations

import math

import numpy as np

from .config import SimConfig
from .geometry import Geometry

# Probe sites on the face: (y, z) on the skin, y>0 is the wearer's left.
PROBES = {
    "forehead": (0.0, -0.025),
    "left eye": (0.032, -0.050),
    "right eye": (-0.032, -0.050),
    "nose tip": (0.0, None),        # placed at the tip of the nose
    "left cheek": (0.045, -0.090),
    "right cheek": (-0.045, -0.090),
    "mouth": (0.0, -0.115),
    "chin": (0.0, -0.145),
}


def sample(field, grid, pts):
    """Trilinear interpolation that ignores NaN (solid) corners.

    field: (..., nx, ny, nz); pts: (n, 3). Returns (..., n).
    """
    pts = np.atleast_2d(pts)
    fx = (pts[:, 0] - grid.x[0]) / grid.dx
    fy = (pts[:, 1] - grid.y[0]) / grid.dx
    fz = (pts[:, 2] - grid.z[0]) / grid.dx
    i0 = np.clip(np.floor(fx).astype(int), 0, grid.x.size - 2)
    j0 = np.clip(np.floor(fy).astype(int), 0, grid.y.size - 2)
    k0 = np.clip(np.floor(fz).astype(int), 0, grid.z.size - 2)
    tx, ty, tz = fx - i0, fy - j0, fz - k0
    lead = field.shape[:-3]
    acc = np.zeros(lead + (len(pts),))
    wsum = np.zeros(len(pts))
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                w = ((tx if di else 1 - tx) * (ty if dj else 1 - ty) * (tz if dk else 1 - tz))
                v = field[..., i0 + di, j0 + dj, k0 + dk]
                ok = np.isfinite(v) if not lead else np.isfinite(v).all(axis=tuple(range(len(lead))))
                w = np.where(ok, w, 0.0)
                acc += np.where(ok, v, 0.0) * w
                wsum += w
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum > 1e-6, acc / np.maximum(wsum, 1e-12), np.nan)


def skin_x(head, y, z):
    """Forward-most skin x at (y, z) including the nose; nan if off the face."""
    x = head.front_x(y, z)
    q = 1 - (y / 0.016) ** 2 - ((z - head.nose_z) / 0.028) ** 2
    if q > 0:
        xn = head.nose_x + head.nose_len * math.sqrt(q)
        x = xn if math.isnan(x) else max(x, xn)
    return x


def probe_points(cfg: SimConfig, geo: Geometry):
    head = geo.head
    pts = {}
    for name, (y, z) in PROBES.items():
        if z is None:
            z = head.nose_z
        x = skin_x(head, y, z)
        pts[name] = (x + cfg.face.probe_offset, y, z)
    return pts


def face_map(cfg, geo, speed, res=0.004):
    """Speed on a surface 'probe_offset' in front of the skin, gridded in (y, z)."""
    ys = np.arange(-0.075, 0.075 + 1e-9, res)
    zs = np.arange(-0.165, 0.0 + 1e-9, res)
    pts, idx = [], []
    for a, z in enumerate(zs):
        for b, y in enumerate(ys):
            x = skin_x(geo.head, y, z)
            # stay on the front of the face (skip the sides of the head)
            if math.isnan(x) or x < 0.55 * geo.head.a_band:
                continue
            pts.append((x + cfg.face.probe_offset, y, z))
            idx.append((a, b))
    out = np.full((zs.size, ys.size), np.nan)
    if pts:
        vals = sample(speed, geo.grid, np.array(pts))
        for (a, b), v in zip(idx, vals):
            out[a, b] = v
    return ys, zs, out


def rate(v, fc):
    if not np.isfinite(v):
        return "n/a"
    if v < fc.perceptible:
        return "not felt"
    if v < fc.cooling:
        return "faint"
    if v < fc.strong:
        return "cooling breeze"
    return "strong breeze"


def analyse(cfg: SimConfig, geo: Geometry, u_mean, u_rms, info):
    fc = cfg.face
    speed = np.sqrt((u_mean.astype(np.float64) ** 2).sum(0))
    pts = probe_points(cfg, geo)
    names = list(pts)
    P = np.array([pts[n] for n in names])
    sp = sample(speed, geo.grid, P)
    rm = sample(u_rms, geo.grid, P)
    vel = sample(u_mean, geo.grid, P)
    probes = {}
    for n, p, s_, r_, v in zip(names, P, sp, rm, vel.T):
        probes[n] = dict(position_m=[round(float(c), 4) for c in p],
                         mean_speed=float(s_), rms_fluctuation=float(r_),
                         velocity=[float(c) for c in v], feel=rate(s_, fc))
    ys, zs, fmap = face_map(cfg, geo, speed)
    valid = np.isfinite(fmap)
    area = valid.sum()
    frac_perc = float((fmap[valid] >= fc.perceptible).sum() / max(area, 1))
    frac_cool = float((fmap[valid] >= fc.cooling).sum() / max(area, 1))
    face_mean = float(np.nanmean(fmap)) if area else math.nan
    # "effective" speed the skin feels: mean + turbulent fluctuation
    face_eff = float(np.nanmean(np.sqrt(sp ** 2 + 3 * rm ** 2)))
    fan_flow = geo.fan_area * cfg.fan.exit_speed
    verdict = rate(face_mean, fc)
    if frac_cool >= 0.5:
        headline = "YES - most of the face gets a cooling breeze"
    elif frac_perc >= 0.5:
        headline = "PARTLY - air reaches the face but it is weak"
    elif frac_perc > 0.1:
        headline = "MARGINAL - only part of the face feels the airflow"
    else:
        headline = "NO - the airflow is not reaching the face"
    return dict(
        name=cfg.name,
        headline=headline,
        face_mean_speed=face_mean,
        face_effective_speed=face_eff,
        face_max_speed=float(np.nanmax(fmap)) if area else math.nan,
        face_fraction_perceptible=frac_perc,
        face_fraction_cooling=frac_cool,
        face_feel=verdict,
        fan_exit_speed=cfg.fan.exit_speed,
        fan_open_area_m2=geo.fan_area,
        fan_flow_m3s=fan_flow,
        fan_flow_cfm=fan_flow * 2118.88,
        probes=probes,
        solver=info,
        _face_map=(ys, zs, fmap),
        _speed=speed,
    )
