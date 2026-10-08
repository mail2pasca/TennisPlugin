"""Voxel geometry: wearer (head, neck, shoulders), hat, and the fan ring.

The parametric shapes stand in for the real CAD. If you have watertight STLs
(head scan / mannequin, hat assembly) pass them through the config and they
replace the parametric shapes; the fan annulus is always defined by
FanConfig so the solver knows where to push air.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import SimConfig

FLUID, SOLID, FAN, BOUNDARY = 0, 1, 2, 3


@dataclass
class Grid:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    dx: float

    @property
    def shape(self):
        return (self.x.size, self.y.size, self.z.size)

    def mesh(self):
        return np.meshgrid(self.x, self.y, self.z, indexing="ij")

    def index(self, px, py, pz):
        return (int(round((px - self.x[0]) / self.dx)),
                int(round((py - self.y[0]) / self.dx)),
                int(round((pz - self.z[0]) / self.dx)))


@dataclass
class Geometry:
    grid: Grid
    flags: np.ndarray        # uint8 cell type
    fan_velocity: np.ndarray  # (3, nx, ny, nz) physical m/s, non-zero only in FAN cells
    head: "HeadShape"
    fan_area: float           # open area of the active blade ring, m^2


def make_grid(cfg: SimConfig) -> Grid:
    d = cfg.domain
    n_xy = int(round(2 * d.half_width / d.dx)) + 1
    n_z = int(round((d.z_max - d.z_min) / d.dx)) + 1
    x = -d.half_width + d.dx * np.arange(n_xy)
    z = d.z_min + d.dx * np.arange(n_z)
    return Grid(x=x, y=x.copy(), z=z, dx=d.dx)


class HeadShape:
    """Mannequin head: ellipsoidal cranium, blunter lower face/jaw, a nose."""

    def __init__(self, cfg: SimConfig):
        h = cfg.head
        a_band, b_band = cfg.head_semi_axes()
        self.c = 0.5 * (h.top_above_band + h.chin_below_band)
        self.zc = 0.5 * (h.top_above_band - h.chin_below_band)
        # scale so the cross-section at the band (z=0) has the requested girth
        s0 = self._profile(0.0)
        self.A = a_band / s0
        self.B = b_band / s0
        self.a_band, self.b_band = a_band, b_band
        self.nose_len = h.nose_length
        # nose: centred on the face between eyes and mouth
        self.nose_z = -0.08
        self.nose_x = self.front_x(0.0, self.nose_z)

    def _profile(self, z):
        t = np.clip(np.abs((np.asarray(z, dtype=float) - self.zc) / self.c), 0, 1)
        p = np.where(np.asarray(z) >= self.zc, 2.0, 6.0)  # rounded skull, squarer jaw
        return np.sqrt(np.clip(1 - t ** p, 0, None))

    def contains(self, X, Y, Z):
        s = self._profile(Z)
        s = np.maximum(s, 1e-9)
        inside = (X / (self.A * s)) ** 2 + (Y / (self.B * s)) ** 2 <= 1.0
        inside &= (Z <= self.zc + self.c) & (Z >= self.zc - self.c)
        # nose: half-ellipsoid sticking out of the face
        nx = (X - self.nose_x) / self.nose_len
        ny = Y / 0.016
        nz = (Z - self.nose_z) / 0.028
        inside |= (nx ** 2 + ny ** 2 + nz ** 2 <= 1.0) & (X >= self.nose_x - 0.01)
        return inside

    def front_x(self, y, z):
        """x of the skin surface on the face side at (y, z); nan if off the head."""
        s = float(self._profile(z))
        if s <= 0:
            return math.nan
        q = 1 - (y / (self.B * s)) ** 2
        if q <= 0:
            return math.nan
        return self.A * s * math.sqrt(q)


def _torso(cfg: SimConfig, X, Y, Z):
    h = cfg.head
    neck = ((X + 0.01) ** 2 + Y ** 2 <= 0.055 ** 2) & (Z <= -0.05)
    ztop = -h.shoulder_top_below_band - 0.07 * (Y / h.shoulder_half_width) ** 2
    body = (((X + 0.02) / h.torso_half_depth) ** 2 + (Y / h.shoulder_half_width) ** 2 <= 1.0) & (Z <= ztop)
    return neck | body


def _stl_mask(path, scale, offset, X, Y, Z):
    try:
        import trimesh
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit("STL input needs trimesh: pip install trimesh") from exc
    mesh = trimesh.load(path, force="mesh")
    mesh.apply_scale(scale)
    mesh.apply_translation(offset)
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        if not mesh.is_watertight:
            print(f"warning: {path} is not watertight; inside/outside test may leak")
    lo, hi = mesh.bounds
    box = ((X >= lo[0]) & (X <= hi[0]) & (Y >= lo[1]) & (Y <= hi[1])
           & (Z >= lo[2]) & (Z <= hi[2]))
    mask = np.zeros(X.shape, dtype=bool)
    pts = np.column_stack([X[box], Y[box], Z[box]])
    if len(pts):
        mask[box] = mesh.contains(pts)
    return mask


def build_geometry(cfg: SimConfig) -> Geometry:
    grid = make_grid(cfg)
    X, Y, Z = grid.mesh()
    R = np.hypot(X, Y)
    head = HeadShape(cfg)
    hat, fan = cfg.hat, cfg.fan

    # --- wearer -------------------------------------------------------------
    if cfg.head.stl:
        solid = _stl_mask(cfg.head.stl, cfg.head.stl_scale, cfg.head.stl_offset, X, Y, Z)
    else:
        solid = head.contains(X, Y, Z)
        if cfg.head.include_torso:
            solid |= _torso(cfg, X, Y, Z)

    # --- hat ----------------------------------------------------------------
    in_ring = (R >= fan.inner_radius) & (R <= fan.outer_radius)
    in_brim_z = (Z >= 0.0) & (Z <= hat.brim_thickness)
    theta = np.degrees(np.arctan2(Y, X))  # 0 deg = straight ahead (face)
    in_arc = np.abs(theta) <= fan.active_arc_deg / 2 + 1e-9
    if cfg.hat.stl:
        hat_solid = _stl_mask(cfg.hat.stl, cfg.hat.stl_scale, cfg.hat.stl_offset, X, Y, Z)
    else:
        ca = head.a_band + hat.crown_clearance
        cb = head.b_band + hat.crown_clearance
        t = np.clip(Z / hat.crown_height, 0, 1)
        sc = np.sqrt(np.clip(1 - t ** 4, 0, None))
        crown = ((X / (ca * np.maximum(sc, 1e-9))) ** 2 + (Y / (cb * np.maximum(sc, 1e-9))) ** 2 <= 1) \
            & (Z >= 0) & (Z <= hat.crown_height)
        brim = (R <= hat.brim_outer_radius) & in_brim_z
        hat_solid = crown | brim
    # carve the active blade ring out of the brim so air can pass through it
    hat_solid &= ~(in_ring & in_arc & in_brim_z)
    solid |= hat_solid

    flags = np.where(solid, SOLID, FLUID).astype(np.uint8)

    # --- fan actuator layer -------------------------------------------------
    zmid = 0.5 * hat.brim_thickness
    kmid = int(np.argmin(np.abs(grid.z - zmid)))
    layer = np.zeros_like(solid)
    layer[:, :, kmid] = True
    fan_cells = layer & in_ring & in_arc & ~solid
    flags[fan_cells] = FAN

    fan_vel = np.zeros((3,) + grid.shape, dtype=np.float32)
    sign = -1.0 if fan.direction == "down" else 1.0
    if fan.direction not in ("down", "up"):
        raise ValueError("fan.direction must be 'down' or 'up'")
    tilt = math.radians(fan.tilt_inward_deg)
    u_ax = fan.exit_speed
    u_in = fan.exit_speed * math.tan(tilt)          # inward radial component
    u_sw = fan.swirl_ratio * fan.exit_speed          # tangential component
    Rs = np.where(R > 0, R, 1.0)
    er_x, er_y = X / Rs, Y / Rs
    et_x, et_y = -er_y, er_x
    fan_vel[0][fan_cells] = (-u_in * er_x + u_sw * et_x)[fan_cells]
    fan_vel[1][fan_cells] = (-u_in * er_y + u_sw * et_y)[fan_cells]
    fan_vel[2][fan_cells] = sign * u_ax

    # --- outer boundary layer -----------------------------------------------
    edge = np.zeros_like(solid)
    edge[[0, -1], :, :] = True
    edge[:, [0, -1], :] = True
    edge[:, :, [0, -1]] = True
    flags[edge & ~solid] = BOUNDARY
    # Solid cells on the boundary are kept: the torso exits through the floor.
    # The kernel never reads outside the grid because every interior cell's
    # neighbours are inside it.

    fan_area = float(fan_cells.sum()) * grid.dx ** 2
    return Geometry(grid=grid, flags=flags, fan_velocity=fan_vel, head=head, fan_area=fan_area)
