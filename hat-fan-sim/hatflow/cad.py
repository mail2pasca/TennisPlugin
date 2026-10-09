"""3D-printable parts for the ring-fan hat, generated from the same config
the simulator uses (fan radii, brim size, cone droop, head size).

All output is in millimetres. The rotation axis is +z and the face is +x.
Parts:
  crown          elliptical dome that the head goes into; sits on the housing
  housing        brim tray: head opening plate, inner/outer channel walls,
                 bottom grille, bearing ledge for the rotor, motor notch
  top_grille     upper grille, screws onto the housing rim, has the motor pocket
  rotor          blade ring (hub ring + blades + tip ring) with an external
                 spur gear on the rim (module 1)
  pinion         16-tooth module-1 pinion for a "130" DC motor (2 mm shaft)
  pod_tray       electronics bay bolted to the back of the housing: 18650 cell,
                 USB-C charger board, slide switch
  pod_lid        covers the tray and the motor standing on the top grille

Large parts are also cut into 90-degree quadrants for 220-256 mm print beds.
This is a parametric starting point for a bench prototype: check fits in your
slicer and expect to tune clearances for your printer.
"""
from __future__ import annotations

import math
import os

import numpy as np
from manifold3d import CrossSection, Manifold

from .config import SimConfig

# fixed mechanical choices (mm)
WALL = 2.0
GRILLE_BAR = 1.6
GRILLE_DEPTH = 2.4
CHANNEL_HALF = 13.0     # half-height of the air channel around the rotor
RING_HALF = 7.0         # half-height of rotor hub/tip rings
BLADES = 24
BLADE_PITCH_DEG = 35.0
BLADE_THICK = 1.8
GEAR_MODULE = 1.0
PINION_TEETH = 16       # 356:16 = 22:1 -> ~350-500 rpm ring from a 130 motor
M3_CLEAR = 3.3
M3_PILOT = 2.5
MOTOR_BODY = (20.4, 15.4)  # "130" DC motor body (20 mm round, 15 mm across flats) + clearance
MOTOR_SHAFT = 1.95         # press fit on the 2.0 mm shaft
# electronics pod at the back of the brim (local brim frame, mm)
POD_HALF_WIDTH = 46.0      # clear of the rim screws at +/-15 deg
POD_OUT = 60.0             # how far the pod reaches past the brim edge
POD_LID_HEIGHT = 34.0      # clears the 25 mm motor body standing on the top grille


class Brim:
    """Cone geometry: brim mid-surface height as a function of radius (mm)."""

    def __init__(self, cfg: SimConfig):
        h, f = cfg.hat, cfg.fan
        self.zc = 0.5 * h.brim_thickness * 1000
        self.hinge = h.droop_hinge_radius * 1000
        self.tan = math.tan(math.radians(h.droop_deg))
        self.droop = h.droop_deg
        self.r_fan_in = f.inner_radius * 1000
        self.r_fan_out = f.outer_radius * 1000
        self.r_out = max(h.brim_outer_radius * 1000, self.r_fan_out + 17)
        a, b = cfg.head_semi_axes()
        self.crown_a = (a + h.crown_clearance) * 1000
        self.crown_b = (b + h.crown_clearance) * 1000
        self.crown_h = h.crown_height * 1000
        if h.droop_arc_deg < 360:
            print("note: printable parts always use a full cone (droop_arc_deg=360); "
                  "a rotating blade ring must be axisymmetric")

    def zm(self, r):
        return self.zc - max(r - self.hinge, 0.0) * self.tan


def _place(brim, m, theta_deg):
    """Map a part built in the local brim frame (u along the cone outward from
    the hinge, v tangential, w normal to the brim) onto the hat at azimuth theta."""
    return m.rotate([0, brim.droop, 0]).translate([brim.hinge, 0, brim.zc]).rotate([0, 0, theta_deg])


def _u(brim, r):
    """Local u coordinate of radius r on the cone."""
    return (r - brim.hinge) / math.cos(math.radians(brim.droop))


def _box(u0, u1, v0, v1, w0, w1):
    return Manifold.cube([u1 - u0, v1 - v0, w1 - w0]).translate([u0, v0, w0])


def _band(brim, r1, r2, zlo, zhi):
    """Revolvable (r, z) polygon between radii r1..r2 that follows the cone,
    spanning zlo..zhi relative to the mid-surface."""
    rs = sorted({r1, r2} | ({brim.hinge} if r1 < brim.hinge < r2 else set()))
    bottom = [(r, brim.zm(r) + zlo) for r in rs]
    top = [(r, brim.zm(r) + zhi) for r in reversed(rs)]
    return bottom + top


def _revolve(polys, seg):
    cs = CrossSection([np.array(p, dtype=float) for p in polys]).simplify(0.01)
    return Manifold.revolve(cs, seg)


def _cyl(r, h, x=0.0, y=0.0, z=0.0, seg=24):
    return Manifold.cylinder(h, r, r, seg).translate([x, y, z])


def _sector(theta0, theta1, rmax=400.0, zmin=-200.0, h=600.0):
    n = max(int(abs(theta1 - theta0) / 2), 2)
    ts = np.radians(np.linspace(theta0, theta1, n + 1))
    pts = [(0.0, 0.0)] + [(rmax * math.cos(t), rmax * math.sin(t)) for t in ts]
    return Manifold.extrude(CrossSection([np.array(pts)]), h).translate([0, 0, zmin])


def _gear_outline(teeth, module, inner_r=None):
    """Simple trapezoid-tooth spur outline (adequate for a slow module-1 drive)."""
    rp = teeth * module / 2
    ra, rf = rp + module, rp - 1.25 * module
    pts = []
    for i in range(teeth):
        t0 = 2 * math.pi * i / teeth
        dt = 2 * math.pi / teeth
        for frac, r in ((0.0, rf), (0.18, rf), (0.36, ra), (0.64, ra), (0.82, rf)):
            a = t0 + frac * dt
            pts.append((r * math.cos(a), r * math.sin(a)))
    contours = [np.array(pts)]
    if inner_r:
        n = max(teeth, 64)
        contours.append(np.array([(inner_r * math.cos(-2 * math.pi * k / n),
                                   inner_r * math.sin(-2 * math.pi * k / n)) for k in range(n)]))
    return CrossSection(contours)


def _grille(brim, r1, r2, zlo, zhi):
    """Concentric rings (as revolvable profiles) + radial spokes (3D) between r1 and r2.

    Spokes are inset 0.3 mm from the ring faces so no faces coincide exactly.
    """
    polys = []
    pitch = 11.0
    r = r1 + pitch
    while r <= r2 - 2.0:
        polys.append(_band(brim, r - GRILLE_BAR / 2, r + GRILLE_BAR / 2, zlo, zhi))
        r += pitch
    n_spokes = 36
    L = r2 - r1
    rm = 0.5 * (r1 + r2)
    tilt = brim.droop if r1 >= brim.hinge else 0.0
    spokes = []
    for i in range(n_spokes):
        spokes.append(Manifold.cube([L + 1.0, GRILLE_BAR, zhi - zlo - 0.6], center=True)
                      .rotate([0, tilt, 0])
                      .translate([rm, 0, brim.zm(rm) + 0.5 * (zlo + zhi)])
                      .rotate([0, 0, 360.0 * i / n_spokes]))
    return polys, Manifold.batch_boolean(spokes, _OP_ADD)


try:  # OpType moved between manifold3d releases
    from manifold3d import OpType
    _OP_ADD = OpType.Add
except ImportError:  # pragma: no cover
    _OP_ADD = None


def _union(parts):
    out = parts[0]
    for p in parts[1:]:
        out = out + p
    return out


def _bolt_circle(brim, r, n, z0, h, offset_deg=0.0):
    return _union([_cyl(M3_CLEAR / 2, h, r * math.cos(math.radians(offset_deg + 360 * k / n)),
                        r * math.sin(math.radians(offset_deg + 360 * k / n)), z0)
                   for k in range(n)])


def build_parts(cfg: SimConfig, seg: int = 256):
    b = Brim(cfg)
    rin, rout, R = b.r_fan_in, b.r_fan_out, b.r_out
    motor_theta = 180.0   # back of the hat, away from the face
    gear_teeth = int(round(2 * (rout + 3) / GEAR_MODULE))
    gear_rp = gear_teeth * GEAR_MODULE / 2
    pinion_rp = PINION_TEETH * GEAR_MODULE / 2
    motor_r = gear_rp + pinion_rp + 0.15        # centre distance + backlash
    wall_in = R - WALL
    if motor_r + pinion_rp + GEAR_MODULE + 1.0 > wall_in:
        R = motor_r + pinion_rp + GEAR_MODULE + 1.0 + WALL
        wall_in = R - WALL
        b.r_out = R
    mx = motor_r * math.cos(math.radians(motor_theta))
    my = motor_r * math.sin(math.radians(motor_theta))

    # ---------------- rotor -------------------------------------------------
    rings = _revolve([_band(b, rin + 2.5, rin + 2.5 + WALL * 1.5, -RING_HALF, RING_HALF),
                      _band(b, rout - 2.5 - WALL * 1.5, rout - 1.0, -RING_HALF, RING_HALF)], seg)
    blade_r1, blade_r2 = rin + 3.0, rout - 3.0
    L = blade_r2 - blade_r1 + 2.0
    chord = 2 * RING_HALF / math.sin(math.radians(BLADE_PITCH_DEG)) * 0.95
    rm = 0.5 * (blade_r1 + blade_r2)
    blades = []
    for i in range(BLADES):
        blades.append(Manifold.cube([L, chord, BLADE_THICK], center=True)
                      .rotate([BLADE_PITCH_DEG, 0, 0])
                      .rotate([0, b.droop, 0])
                      .translate([rm, 0, b.zm(rm)])
                      .rotate([0, 0, 360.0 * i / BLADES]))
    # keep blades inside the ring height band
    band = _revolve([_band(b, rin, rout + 5, -RING_HALF + 0.3, RING_HALF - 0.3)], seg)
    gear_z0 = b.zm(rout) - RING_HALF
    gear = Manifold.extrude(_gear_outline(gear_teeth, GEAR_MODULE, inner_r=rout - 2.0),
                            2 * RING_HALF).translate([0, 0, gear_z0])
    rotor = rings + (Manifold.batch_boolean(blades, _OP_ADD) ^ band) + gear

    # ---------------- housing (bottom tray) ---------------------------------
    plate_lo, plate_hi = CHANNEL_HALF, CHANNEL_HALF + 3.0
    head_hole = Manifold.cylinder(200, 1.0, 1.0, seg).scale([b.crown_a - WALL, b.crown_b - WALL, 1]) \
        .translate([0, 0, -100])
    g_polys, g_spokes = _grille(b, rin, wall_in, -CHANNEL_HALF - GRILLE_DEPTH, -CHANNEL_HALF)
    housing = _revolve([
        _band(b, 60.0, rin, plate_lo, plate_hi),                                  # head-opening plate
        _band(b, rin - WALL, rin, -CHANNEL_HALF - GRILLE_DEPTH, plate_hi),        # inner wall
        _band(b, rin - 1.0, rin + 5.0, -CHANNEL_HALF - GRILLE_DEPTH, -RING_HALF - 0.6),  # rotor ledge
        _band(b, wall_in, R, -CHANNEL_HALF - GRILLE_DEPTH, CHANNEL_HALF),         # outer wall
        _band(b, R - 8.0, R, CHANNEL_HALF - 3.0, CHANNEL_HALF),                   # rim flange
    ] + g_polys, seg)
    housing = (housing + g_spokes) - head_hole
    crown_bolts = [(0.5 * (b.crown_a + rin), 0.5 * (b.crown_b + rin))]
    for k in range(8):
        t = 2 * math.pi * k / 8 + math.pi / 8
        ax, ay = crown_bolts[0]
        housing = housing - _cyl(M3_CLEAR / 2, 60, ax * math.cos(t), ay * math.sin(t), -30)
    rim_holes = _bolt_circle(b, R - 4.0, 12, -60, 120, offset_deg=15)
    housing = housing - rim_holes
    # clearance pocket for the pinion through the bottom grille
    housing = housing - _cyl(pinion_rp + 1.5, 60, mx, my, b.zm(motor_r) - 40)

    # ---------------- top grille --------------------------------------------
    g_polys, g_spokes = _grille(b, rin, wall_in, CHANNEL_HALF, CHANNEL_HALF + GRILLE_DEPTH)
    top = _revolve([
        _band(b, rin - WALL, rin + 2.0, CHANNEL_HALF, CHANNEL_HALF + GRILLE_DEPTH),  # inner seat
        _band(b, R - 8.0, R, CHANNEL_HALF, CHANNEL_HALF + 3.0),                      # rim
    ] + g_polys, seg) + g_spokes
    # motor mount: rectangular through-pocket for a 130 motor (shaft down)
    pocket = _place(b, _box(_u(b, motor_r) - MOTOR_BODY[1] / 2, _u(b, motor_r) + MOTOR_BODY[1] / 2,
                            -MOTOR_BODY[0] / 2, MOTOR_BODY[0] / 2, -5, 40), motor_theta)
    top = top - pocket - rim_holes

    # ---------------- pinion -------------------------------------------------
    bore = CrossSection.circle(MOTOR_SHAFT / 2, 32)
    pinion = Manifold.extrude(_gear_outline(PINION_TEETH, GEAR_MODULE) - bore, 10.0)

    # ---------------- electronics pod (back of the brim) --------------------
    # Tray: bolts to the outside of the housing wall, holds an 18650 cell,
    # a USB-C charger board and a slide switch. Lid: covers the tray and
    # reaches inward over the motor standing on the top grille.
    W = POD_HALF_WIDTH
    w_bot, w_top = -CHANNEL_HALF - GRILLE_DEPTH, CHANNEL_HALF + 3.0
    u_wall, u_end = _u(b, R), _u(b, R + POD_OUT)
    tray = _box(_u(b, R - 10), u_end, -W, W, w_bot, w_top)
    tray = tray - _box(u_wall + 2.0, u_end - 2.0, -W + 2, W - 2, w_bot + 2.0, w_top + 1)
    posts, pilots = [], []
    post_uv = [(pu, pv) for pu in (u_wall + 5.0, u_end - 5.0) for pv in (-W + 5.0, W - 5.0)]
    for pu, pv in post_uv:  # posts overlap the walls by 1 mm so the union is clean
        posts.append(Manifold.cylinder(w_top - w_bot - 1.0, 4.0, 4.0, 24).translate([pu, pv, w_bot + 1.0]))
        pilots.append(Manifold.cylinder(14, M3_PILOT / 2, M3_PILOT / 2, 16).translate([pu, pv, w_top - 13]))
    tray = tray + _union(posts) - _union(pilots)
    # USB-C charge port and slide-switch slots in the back wall
    tray = tray - _box(u_end - 3, u_end + 1, 10, 21, -6, -1.5) - _box(u_end - 3, u_end + 1, -22, -12, -6, -1.5)
    # two radial M3 bolts through tray and housing wall
    bolts = _union([Manifold.cylinder(30, M3_CLEAR / 2, M3_CLEAR / 2, 16).rotate([0, 90, 0])
                    .translate([u_wall - 15, pv, 0.0]) for pv in (-28.0, 28.0)])
    tray = _place(b, tray - bolts, motor_theta)
    # the tray hugs the curved housing wall
    tray = tray - _revolve([_band(b, 0.0, R + 0.3, -60, 60)], seg)
    housing = housing - _place(b, bolts, motor_theta)

    lid = _box(_u(b, motor_r - 16), u_end, -W, W, w_top, w_top + POD_LID_HEIGHT)
    lid = lid - _box(_u(b, motor_r - 14), u_end - 2.0, -W + 2, W - 2, w_top - 1, w_top + POD_LID_HEIGHT - 2)
    for pu, pv in post_uv:
        lid = lid + Manifold.cylinder(POD_LID_HEIGHT - 1.0, 4.0, 4.0, 24).translate([pu, pv, w_top])
    for pu, pv in post_uv:
        lid = lid - Manifold.cylinder(POD_LID_HEIGHT + 2, M3_CLEAR / 2, M3_CLEAR / 2, 16).translate([pu, pv, w_top - 1])
    lid = _place(b, lid, motor_theta)

    # ---------------- crown --------------------------------------------------
    zs = np.linspace(0, 1, 40)
    s = np.sqrt(np.clip(1 - zs ** 4, 0, None))
    H = b.crown_h
    outer = [(1.0 * si, z * H) for si, z in zip(s, zs)]
    inner = [(si * (1 - WALL / b.crown_a), z * (H - WALL)) for si, z in zip(s, zs)]
    prof = [(0.0, 0.0)] + outer + [(0.0, H)]
    prof_in = [(0.0, 0.0)] + inner + [(0.0, H - WALL)]
    dome = Manifold.revolve(CrossSection([np.array(prof)]), seg).scale([b.crown_a, b.crown_b, 1])
    hollow = Manifold.revolve(CrossSection([np.array(prof_in)]), seg).scale([b.crown_a, b.crown_b, 1]) \
        .translate([0, 0, -0.01])
    flange = (Manifold.cylinder(3.0, 1.0, 1.0, seg).scale([rin - 1.0, rin - 1.0 - (b.crown_a - b.crown_b), 1])
              - Manifold.cylinder(3.0, 1.0, 1.0, seg).scale([b.crown_a - WALL, b.crown_b - WALL, 1]))
    crown = (dome - hollow) + flange
    for k in range(8):
        t = 2 * math.pi * k / 8 + math.pi / 8
        ax, ay = crown_bolts[0]
        crown = crown - _cyl(M3_CLEAR / 2, 10, ax * math.cos(t), ay * math.sin(t), -2)
    crown = crown.translate([0, 0, b.zm(rin) + plate_hi])

    return dict(crown=crown, housing=housing, top_grille=top, rotor=rotor, pinion=pinion,
                pod_tray=tray, pod_lid=lid), \
        dict(gear_teeth=gear_teeth, motor_radius=motor_r, brim_radius=R, droop=b.droop,
             motor_xy=(mx, my), gear_z0=gear_z0, pinion_teeth=PINION_TEETH)


def _to_trimesh(m):
    import trimesh
    # collapse sub-0.01 mm slivers so float32 STL output stays watertight
    mesh = m.simplify(0.01).to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]
    f = np.asarray(mesh.tri_verts)
    # manifold keeps seam duplicates apart; its merge vectors give true topology
    src, dst = np.asarray(mesh.merge_from_vert), np.asarray(mesh.merge_to_vert)
    if src.size:
        remap = np.arange(len(v))
        remap[src] = dst
        f = remap[f]
    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
    tm.remove_unreferenced_vertices()
    return tm


def _stl_watertight(tm):
    import io

    import trimesh
    buf = io.BytesIO()
    tm.export(buf, file_type="stl")
    buf.seek(0)
    return trimesh.load(buf, file_type="stl").is_watertight


def export_parts(cfg: SimConfig, out_dir: str, seg: int = 256, quadrants: bool = True, log=print):
    os.makedirs(out_dir, exist_ok=True)
    parts, info = build_parts(cfg, seg)
    written = []
    for name, m in parts.items():
        tm = _to_trimesh(m)
        path = os.path.join(out_dir, f"{name}.stl")
        tm.export(path)
        ext = tm.bounds[1] - tm.bounds[0]
        written.append((name, path, ext, tm.is_watertight, tm.volume))
        log(f"  {name:<12} {ext[0]:6.0f} x {ext[1]:6.0f} x {ext[2]:5.0f} mm  "
            f"{'watertight' if tm.is_watertight else 'NOT watertight'}  {tm.volume/1000:6.1f} cm^3")
    if quadrants:
        qdir = os.path.join(out_dir, "quadrants")
        os.makedirs(qdir, exist_ok=True)
        for name in ("housing", "top_grille", "rotor"):
            # seams near 22.5 deg (clear of the face and the motor); nudge the
            # angle if a seam slices a tooth or bar into an unprintable sliver
            for nudge in (0.0, 0.37, -0.41, 0.83, -0.79, 1.3):
                pieces = []
                for q in range(4):
                    a0 = 22.5 + nudge + 90 * q
                    piece = (parts[name] ^ _sector(a0, a0 + 90)).rotate([0, 0, -a0])
                    pieces.append(_to_trimesh(piece))
                if all(_stl_watertight(t) for t in pieces):
                    break
            for q, tm in enumerate(pieces):
                tm.export(os.path.join(qdir, f"{name}_q{q+1}.stl"))
        log(f"  quadrants (each fits a 200 x 200 mm bed) for housing, top_grille, rotor -> {qdir}/")
    import trimesh
    pin = parts["pinion"].translate([info["motor_xy"][0], info["motor_xy"][1], info["gear_z0"] + 2.0])
    asm = trimesh.util.concatenate([_to_trimesh(m) for m in
                                    (parts["crown"], parts["housing"], parts["top_grille"], parts["rotor"], pin,
                                     parts["pod_tray"], parts["pod_lid"])])
    asm.export(os.path.join(out_dir, "assembly_preview.stl"))
    return written, info
