"""D3Q19 lattice-Boltzmann solver with Smagorinsky LES (numba-accelerated).

Cell types (see geometry.py):
  FLUID     BGK collision with a Smagorinsky eddy viscosity
  SOLID     no-slip walls via half-way bounce-back
  FAN       actuator cells: populations are reset to equilibrium at the
            prescribed fan exit velocity every step (a velocity-forced disc)
  BOUNDARY  open far field at ambient pressure. Where the ambient wind blows
            into the box the velocity is the wind; elsewhere it is copied from
            the neighbouring interior cell, so the jet can leave and the fan
            can entrain room air freely.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit, prange

from ._kernel import _step
from .geometry import BOUNDARY, SOLID, Geometry

C = np.array([
    [0, 0, 0],
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
    [1, 0, 1], [-1, 0, -1], [1, 0, -1], [-1, 0, 1],
    [0, 1, 1], [0, -1, -1], [0, 1, -1], [0, -1, 1],
], dtype=np.int64)
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)


@njit(parallel=True, cache=True)
def _boundary(fn, cells, neighbour, inflow, u_out, rho_out, wind, cxyz, w):
    for n in prange(cells.size):
        c = cells[n]
        if inflow[n]:
            ux, uy, uz = wind[0], wind[1], wind[2]
        elif neighbour[n] < 0:  # neighbour is a wall
            ux, uy, uz = 0.0, 0.0, 0.0
        else:
            nb = neighbour[n]
            ux, uy, uz = u_out[0, nb], u_out[1, nb], u_out[2, nb]
        usq = 1.5 * (ux * ux + uy * uy + uz * uz)
        for q in range(19):
            cu = 3.0 * (cxyz[q, 0] * ux + cxyz[q, 1] * uy + cxyz[q, 2] * uz)
            fn[q, c] = w[q] * (1.0 + cu + 0.5 * cu * cu - usq)
        rho_out[c] = 1.0
        u_out[0, c] = ux
        u_out[1, c] = uy
        u_out[2, c] = uz


def _boundary_tables(flags, wind):
    nx, ny, nz = flags.shape
    I, J, K = np.nonzero(flags == BOUNDARY)
    nrm = np.stack([(I == 0).astype(int) - (I == nx - 1),
                    (J == 0).astype(int) - (J == ny - 1),
                    (K == 0).astype(int) - (K == nz - 1)])
    inflow = (wind[:, None] * nrm).sum(0) > 1e-12
    ni = np.clip(I + nrm[0], 1, nx - 2)
    nj = np.clip(J + nrm[1], 1, ny - 2)
    nk = np.clip(K + nrm[2], 1, nz - 2)
    cells = np.ravel_multi_index((I, J, K), flags.shape)
    nb = np.ravel_multi_index((ni, nj, nk), flags.shape)
    nb = np.where(flags[ni, nj, nk] == SOLID, -1, nb)
    return cells.astype(np.int64), nb.astype(np.int64), inflow


class Units:
    """Physical <-> lattice conversion. Velocities scale by dx/dt."""

    def __init__(self, dx, u_ref_phys, lattice_speed):
        self.dx = dx
        self.dt = lattice_speed * dx / u_ref_phys
        self.vel = dx / self.dt  # m/s per lattice unit


def run(geo: Geometry, cfg, log=print):
    """Run to cfg.solver.sim_time.

    Returns (u_mean, u_rms, info): time-averaged velocity (3, nx, ny, nz) in
    m/s, rms velocity fluctuation (nx, ny, nz) in m/s, and run statistics.
    """
    s, env = cfg.solver, cfg.env
    wind = np.array(env.wind, dtype=np.float64)
    fan_peak = float(np.sqrt((geo.fan_velocity.astype(np.float64) ** 2).sum(0)).max())
    u_ref = max(fan_peak, float(np.linalg.norm(wind)), 1e-6)
    units = Units(geo.grid.dx, u_ref, s.lattice_speed)

    nu_lat = max(env.kinematic_viscosity * units.dt / units.dx ** 2, s.min_lattice_viscosity)
    tau0 = 3.0 * nu_lat + 0.5
    nu_eff = nu_lat * units.dx ** 2 / units.dt
    steps = int(math.ceil(s.sim_time / units.dt))
    avg_start = min(int(s.average_from / units.dt), steps - 1)

    shape = geo.flags.shape
    nx, ny, nz = shape
    N = geo.flags.size
    flags = np.ascontiguousarray(geo.flags.ravel())
    ufan = (geo.fan_velocity.reshape(3, N) / units.vel).astype(np.float32)
    wind_l = wind / units.vel
    b_cells, b_nb, b_in = _boundary_tables(geo.flags, wind)

    rho = np.ones(N, np.float32)
    u = np.zeros((3, N), np.float32)
    u[:] = wind_l[:, None]
    u[:, flags == SOLID] = 0.0
    f = np.empty((19, N), np.float32)
    usq = 1.5 * (u ** 2).sum(0)
    for q in range(19):
        cu = 3.0 * (C[q, 0] * u[0] + C[q, 1] * u[1] + C[q, 2] * u[2])
        f[q] = W[q] * (1 + cu + 0.5 * cu * cu - usq)
    fn = f.copy()
    u_sum = np.zeros((3, N))
    u2_sum = np.zeros(N)
    n_avg = 0
    Wf = W.astype(np.float32)

    info = dict(cells=N, steps=steps, dt=units.dt, tau0=tau0, nu_effective=nu_eff,
                fan_peak_speed=fan_peak)
    log(f"grid {shape} = {N:,} cells, {steps} steps of {units.dt*1e3:.3f} ms; "
        f"base viscosity {nu_eff:.1e} m^2/s ({nu_eff/env.kinematic_viscosity:.0f}x air) + Smagorinsky LES")
    t0 = time.time()
    for n in range(steps):
        acc = n >= avg_start
        _step(f, fn, flags, ufan, tau0, s.smagorinsky ** 2, rho, u, nx, ny, nz, acc, u_sum, u2_sum)
        _boundary(fn, b_cells, b_nb, b_in, u, rho, wind_l, C, Wf)
        f, fn = fn, f
        n_avg += acc
        if (n + 1) % s.report_every == 0 or n == steps - 1:
            umax = float(np.sqrt((u.astype(np.float64) ** 2).sum(0)).max())
            if not math.isfinite(umax) or umax > 0.35:
                raise FloatingPointError(
                    f"solver went unstable at step {n+1} (max lattice speed {umax:.3f}); "
                    "raise solver.min_lattice_viscosity or lower solver.lattice_speed")
            el = time.time() - t0
            log(f"  t={(n+1)*units.dt:.3f}s  step {n+1}/{steps}  max|u|={umax*units.vel:.2f} m/s  "
                f"{(n+1)*N/el/1e6:.1f} MLUPS  ({el:.0f}s)")
    n_avg = max(n_avg, 1)
    u_mean = (u_sum / n_avg) * units.vel
    var = (u2_sum / n_avg) * units.vel ** 2 - (u_mean ** 2).sum(0)
    u_rms = np.sqrt(np.clip(var, 0, None) / 3.0)
    solid = flags == SOLID
    u_mean[:, solid] = np.nan
    u_rms[solid] = np.nan
    info["runtime_s"] = time.time() - t0
    info["averaged_steps"] = n_avg
    return (u_mean.reshape((3,) + shape).astype(np.float32),
            u_rms.reshape(shape).astype(np.float32), info)
