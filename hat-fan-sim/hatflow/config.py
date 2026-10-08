"""Simulation configuration.

All lengths are metres, speeds m/s, angles degrees. Coordinate frame:
  +x  = the direction the wearer faces (face is on the +x side of the head)
  +y  = wearer's left
  +z  = up; z = 0 is the underside of the brim (the hat band line)
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass


@dataclass
class HeadConfig:
    # 23-inch (58.4 cm) head circumference -> ellipse semi-axes at the band.
    circumference: float = 0.584
    length_width_ratio: float = 1.26   # typical adult head length / width
    top_above_band: float = 0.10       # crown of skull above the hat band
    chin_below_band: float = 0.16      # chin below the hat band
    nose_length: float = 0.022         # how far the nose tip sticks out of the face
    include_torso: bool = True         # neck + shoulders (they deflect a downward jet)
    shoulder_half_width: float = 0.22
    torso_half_depth: float = 0.12
    shoulder_top_below_band: float = 0.24
    stl: str | None = None             # optional watertight head/body STL
    stl_scale: float = 0.001           # STL units -> metres (mm by default)
    stl_offset: tuple = (0.0, 0.0, 0.0)  # added after scaling, metres


@dataclass
class HatConfig:
    crown_clearance: float = 0.012     # crown wall gap around the head at the band
    crown_height: float = 0.13         # crown top above the band
    brim_thickness: float = 0.035      # depth of the fan housing / brim
    brim_outer_radius: float = 0.19
    stl: str | None = None             # optional watertight hat STL (fan annulus is carved out)
    stl_scale: float = 0.001
    stl_offset: tuple = (0.0, 0.0, 0.0)


@dataclass
class FanConfig:
    inner_radius: float = 0.125        # inner edge of the blade ring
    outer_radius: float = 0.175        # outer edge of the blade ring
    exit_speed: float = 2.5            # mean axial speed through the blade ring, m/s
    direction: str = "down"            # "down" (blow onto wearer) or "up" (exhaust)
    tilt_inward_deg: float = 0.0       # louver/blade angle steering air toward the head
    swirl_ratio: float = 0.3           # tangential / axial speed from the rotating ring
    # Fraction of the ring that blows. 360 = full ring. Smaller values make a
    # sector centred on the face (e.g. 120 = front third only).
    active_arc_deg: float = 360.0


@dataclass
class EnvironmentConfig:
    # Air moving past the wearer, e.g. walking forward at 1.3 m/s is wind
    # of (-1.3, 0, 0). Zero = still air (worst case for the fan).
    wind: tuple = (0.0, 0.0, 0.0)
    kinematic_viscosity: float = 1.5e-5


@dataclass
class DomainConfig:
    dx: float = 0.008                  # cell size; 0.005 for a finer run
    half_width: float = 0.30           # x, y extent is [-half_width, half_width]
    z_min: float = -0.50
    z_max: float = 0.24


@dataclass
class SolverConfig:
    lattice_speed: float = 0.08        # lattice velocity the fastest flow maps to (<0.1)
    min_lattice_viscosity: float = 0.002  # stability floor (see README: effective Re)
    smagorinsky: float = 0.17
    sim_time: float = 1.2              # physical seconds simulated
    average_from: float = 0.5          # start time-averaging after this many seconds
    report_every: int = 200


@dataclass
class FaceConfig:
    probe_offset: float = 0.015        # how far in front of the skin the probes sit
    # Comfort bands for mean air speed at the face, m/s. ~0.2 m/s is about where
    # people start to notice air movement; ~0.5-1 m/s is a clearly cooling breeze.
    perceptible: float = 0.2
    cooling: float = 0.5
    strong: float = 1.0


@dataclass
class SimConfig:
    name: str = "ring-fan-hat"
    head: HeadConfig = field(default_factory=HeadConfig)
    hat: HatConfig = field(default_factory=HatConfig)
    fan: FanConfig = field(default_factory=FanConfig)
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    face: FaceConfig = field(default_factory=FaceConfig)

    # ---- derived head ellipse -------------------------------------------
    def head_semi_axes(self) -> tuple[float, float]:
        """Front-back (a) and side (b) semi-axes giving the configured circumference."""
        k = self.head.length_width_ratio
        b = 1.0
        a = k * b
        # Ramanujan's ellipse perimeter for a=k, b=1, then scale.
        per = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))
        s = self.head.circumference / per
        return a * s, b * s

    def to_dict(self) -> dict:
        return asdict(self)


def _apply(obj, data: dict, path: str = ""):
    names = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise KeyError(f"unknown config key '{path}{key}'")
        cur = getattr(obj, key)
        if is_dataclass(cur):
            if not isinstance(value, dict):
                raise TypeError(f"'{path}{key}' must be an object")
            _apply(cur, value, f"{path}{key}.")
        else:
            if isinstance(cur, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(obj, key, value)


def load_config(path: str | None = None, overrides: list[str] | None = None) -> SimConfig:
    cfg = SimConfig()
    if path:
        with open(path) as fh:
            _apply(cfg, json.load(fh))
    for item in overrides or []:
        set_value(cfg, item)
    return cfg


def set_value(cfg: SimConfig, assignment: str) -> SimConfig:
    """Apply 'fan.exit_speed=3' style overrides (value parsed as JSON when possible)."""
    key, _, raw = assignment.partition("=")
    if not _:
        raise ValueError(f"override '{assignment}' must look like section.key=value")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    parts = key.strip().split(".")
    data: dict = {}
    node = data
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    _apply(cfg, data)
    return cfg


def clone(cfg: SimConfig) -> SimConfig:
    return copy.deepcopy(cfg)
