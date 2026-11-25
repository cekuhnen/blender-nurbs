"""Headless integration test:  blender -b --factory-startup --python tests/test_headless.py"""
import json, math, os, sys, tempfile
import bpy
import numpy as np
from mathutils import Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, store, geom, history, jobs, steps
from blender_nurbs.kernel import load as _kl
ext = _kl().blender_ext

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

def bezier(name, pts, z=0.0):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('BEZIER'); sp.bezier_points.add(len(pts) - 1)
    for bp, p in zip(sp.bezier_points, pts):
        bp.co = (p[0], p[1], z); bp.handle_left_type = bp.handle_right_type = 'AUTO'
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def nurbs(name, pts, order=4, endpoint=True):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('NURBS'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.order_u = order; sp.use_endpoint_u = endpoint
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def ins(*pairs): return json.dumps([[o.name, r] for o, r in pairs])

A = bezier("A", [(0,0),(1,0.4),(2,0),(3,0.4)], 0.0)
B = nurbs("B", [(0,0,2),(1,-0.4,2),(2,0,2),(3,-0.4,2)])
C = nurbs("C", [(0,0,4),(1,0.3,4.3),(2,0,4),(3,0.3,4)], endpoint=False)

# --- Loft G0 (kernel skin), 3 curves
r = bpy.ops.nproto.loft(inputs=ins((A,"S0"),(B,"S0"),(C,"S0")))
L = ctx.active_object
check("loft skin 3 curves", r == {'FINISHED'} and store.is_nurbs(L) and len(L.data.polygons) > 0,
      f"{len(L.data.polygons)} tris")
check("loft has history", features.history(L) is not None)

# --- Extrude + Revolve + Planar
r = bpy.ops.nproto.extrude(inputs=ins((A,"S0")), distance=1.5, direction="Y")
E1 = ctx.active_object; check("extrude", r == {'FINISHED'} and len(E1.data.polygons) > 0)

# --- Extrude's live preview (valuedrag.py) draws what the command builds
from blender_nurbs import valuedrag
from blender_nurbs.ops import extrude_rows
check("value drag: mouse ray to arrow parameter",
      abs(valuedrag.axis_param((2, 5, 10), (0, 0, -1), (0, 0, 0), (1, 0, 0)) - 2.0) < 1e-12
      and abs(valuedrag.axis_param((0, 0, 10), (0, 0, -1), (1, 0, 0), (-1, 0, 0)) - 1.0) < 1e-12
      and valuedrag.axis_param((0, 0, 10), (0, 0, -1), (0, 0, 0), (0, 0, 1)) is None)
check("value drag: snap", valuedrag.snap(1.26, 0.25) == 1.25 and valuedrag.snap(-0.13, 0.1) == -0.1)

def _poly_dist(p, poly):
    best = 1e300
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a; L = ab @ ab
        t = 0.0 if L == 0 else min(1.0, max(0.0, (p - a) @ ab / L))
        best = min(best, np.linalg.norm(a + t * ab - p))
    return best

_pv_open = nurbs("PvOpen", [(0, 0, 0), (1, 0.6, 0), (2, -0.4, 0), (3, 0.2, 0)])
bpy.ops.curve.primitive_nurbs_circle_add(radius=0.7, location=(0, 4, 0))
_pv_circ = ctx.active_object
_pv_cu = bpy.data.curves.new("PvRect", 'CURVE'); _pv_cu.dimensions = '3D'
_pv_sp = _pv_cu.splines.new('POLY'); _pv_sp.points.add(3); _pv_sp.use_cyclic_u = True
for _q, _c in zip(_pv_sp.points, [(0, 8, 0), (2, 8, 0), (2, 9, 0), (0, 9, 0)]):
    _q.co = (*_c, 1.0)
_pv_rect = bpy.data.objects.new("PvRect", _pv_cu); ctx.collection.objects.link(_pv_rect)
_pv_worst = []
for _o, _p in [(_pv_open, dict(distance=1.3, direction="NORMAL")),
               (_pv_open, dict(distance=-0.8, direction="Y", tilt=20.0, tilt_azimuth=30.0)),
               (_pv_open, dict(distance=1.0, direction="Z", draft=10.0)),
               (_pv_open, dict(distance=0.6, direction="Z", draft=8.0, symmetric=True)),
               (_pv_circ, dict(distance=1.2, direction="NORMAL", draft=15.0)),
               (_pv_circ, dict(distance=1.2, direction="Z", draft=12.0, draft_flip=True)),
               (_pv_circ, dict(distance=0.9, direction="LOCAL_Z", symmetric=True)),
               (_pv_rect, dict(distance=0.7, direction="Z", draft=20.0)),
               (_pv_rect, dict(distance=0.5, direction="Z", draft=10.0, draft_flip=True, symmetric=True))]:
    _c = features.input_curve(_o, "S0")
    _cv = ext.curve_cvs(_c)
    _poly = _cv["degree"] == 1 and not _cv.get("weights")      # as the tool does: a polyline's corners
    _P = np.array(_cv["points"], float) if _poly else geom.sample_curve(_c, 97)
    _rows = extrude_rows(_P, features.extrude_direction(_o, _c, _p), _p["distance"], _p.get("draft", 0.0),
                         _p.get("draft_flip", False), _p.get("symmetric", False), smooth=not _poly)
    _s = features.run("extrude", [(_o, "S0")], _p)
    _edges = [np.array(geom.boundary_points(_s, w, 801)) for w in ("U0", "U1", "V0", "V1")]
    _far = [_rows[0], _rows[-1]]
    _worst = max(min(_poly_dist(q, e) for e in _edges) for r_ in _far for q in r_[::6])
    _pv_worst.append(round(float(_worst), 6))
# drawn edges of an extruded rectangle go through its corners (evenly spaced
# samples cut every corner but the seam's with a chord)
_rc_corner = []
for _dr in (0.0, 8.0):
    bpy.ops.nproto.extrude(inputs=ins((_pv_rect, "S0")), distance=0.5, direction="Z", draft=_dr)
    _rx = ctx.active_object
    _rP = [np.array(e, float) for e in store.display(_rx)["edges"]]
    _rbot = [e for e in _rP if np.ptp(e[:, 2]) < 1e-9 and abs(e[0, 2]) < 1e-9]
    _rc_corner.append(max(min(np.min(np.linalg.norm(b - np.array(c), axis=1)) for b in _rbot)
                          for c in [(0, 8, 0), (2, 8, 0), (2, 9, 0), (0, 9, 0)]))
    _rs = store.get_surfaces(_rx)[0]
    _rb = [np.array(geom.boundary_points(_rs, w, 48, corners=True)) for w in ("U0", "U1", "V0", "V1")]
    _rb = [b for b in _rb if np.ptp(b[:, 2]) < 1e-9 and abs(b[0, 2]) < 1e-9]
    _rc_corner.append(max(min(np.min(np.linalg.norm(b - np.array(c), axis=1)) for b in _rb)
                          for c in [(0, 8, 0), (2, 8, 0), (2, 9, 0), (0, 9, 0)]))
check("extruded rectangle: drawn edges and pick outlines hit every corner (plain, draft)",
      max(_rc_corner) < 1e-9, [round(float(x), 6) for x in _rc_corner])
# --- Revolve / Offset / Edge Fillet previews (valuedrag.py) land on what the commands build
from mathutils.bvhtree import BVHTree as _BVH
from blender_nurbs.valuedrag import rotate_points as _rot
from blender_nurbs.ops import offset_frame as _offr
from blender_nurbs.ops_trim import fillet_frames as _ffr, fillet_sections as _fsec

def _tree(o_):
    M_ = o_.matrix_world
    return _BVH.FromPolygons([M_ @ v_.co for v_ in o_.data.vertices], [p_.vertices[:] for p_ in o_.data.polygons])

_pv_prof = nurbs("PvProf", [(1, 0, 0), (1.3, 0, 0.5), (0.8, 0, 1), (1, 0, 1.5)])
_pv_prof.location = (0.5, 0.2, 9.0); ctx.view_layer.update()
_S = geom.sample_curve(features.input_curve(_pv_prof, "S0"), 97)
_rv_gap = []
for _ang, _ax, _pvt in ((120.0, "Z", "WORLD"), (-75.0, "LOCAL_Z", "OBJECT"), (360.0, "Y", "WORLD")):
    _z = np.array(geom.axis_vector(_ax, _pv_prof), float); _z /= np.linalg.norm(_z)
    _O = np.array(geom.world_matrix(_pv_prof).translation if _pvt == "OBJECT" else (0, 0, 0), float)
    _sf = features.run("revolve", [(_pv_prof, "S0")], {"angle": _ang, "axis": _ax, "pivot": _pvt})
    _sm = _kl()
    _cp = lambda q_: np.array((lambda c_: [c_.x, c_.y, c_.z] if hasattr(c_, "x") else list(c_))(
        _sm.surface_closest_point(_sf, tuple(float(x_) for x_ in q_))[0]), float)
    _rv_gap.append(max(np.linalg.norm(_cp(q_) - q_)
                       for f_ in (0.5, 1.0) for q_ in _rot(_S[::8], _O, _z, _ang * f_)))
check("revolve preview: rotated profiles lie on the built surface (world / local axis, object pivot, 360)",
      max(_rv_gap) < 1e-6, [float(f"{x:.2g}") for x in _rv_gap])
# a picked line as the axis: the surface turns about it, and the Feature box keeps it when the angle changes
_ap, _ad = (3.0, 1.0, 9.0), (0.3, 0.2, 1.0)
r = bpy.ops.nproto.revolve(inputs=ins((_pv_prof, "S0")), angle=150.0, axis="PICKED", axis_point=_ap, axis_dir=_ad)
_rp = ctx.active_object
_z = np.array(_ad) / np.linalg.norm(_ad)
_sf = store.get_surfaces(_rp)[0]
_Mrp = np.array(_rp.matrix_world, float)
_gap_p = max(np.linalg.norm(_cp((np.linalg.inv(_Mrp) @ np.append(q_, 1.0))[:3]) - (np.linalg.inv(_Mrp) @ np.append(q_, 1.0))[:3])
             for q_ in _rot(_S[::8], np.array(_ap), _z, 150.0))
_rp.nproto_feature.angle = 90.0           # edit in the Feature box: rebuilds from the history
_hp = features.history(_rp)["params"]
# the user's case: profile and axis line both selected, the line picked as the axis -> the line is
# left out (turned about itself it has no surface: the kernel refused the face), the profile revolves,
# and a failed object is not left behind
_ucv = nurbs("UCurve", [(0.6425, 0, 0.3741), (0.3886, 0, 0.2942), (0.3517, 0, 0.0999), (0.4261, 0, -0.1368)])
_uln = bpy.data.curves.new("ULine", 'CURVE'); _uln.dimensions = '3D'
_usp = _uln.splines.new('POLY'); _usp.points.add(1)
_usp.points[0].co = (0.8344, 0, 0.293, 1.0); _usp.points[1].co = (0.8344, 0, -0.1146, 1.0)
_ulo = bpy.data.objects.new("ULine", _uln); ctx.collection.objects.link(_ulo)
_n0 = len(bpy.data.objects)
r_u = bpy.ops.nproto.revolve(inputs=ins((_ucv, "S0"), (_ulo, "S0")), angle=331.75, axis="PICKED",
                             axis_point=(0.8344, 0, 0.293), axis_dir=(0, 0, -1))
_umade = [o_ for o_ in bpy.data.objects if o_.name.startswith("Revolve") and o_ not in (_rp,)]
_n1 = len(bpy.data.objects)
_bad_g = features.run("revolve", [(_ulo, "S0")], {"angle": 90.0, "axis": "PICKED", "axis_point": (0.8344, 0, 0.293),
                                                  "axis_dir": (0, 0, -1)})
_nb = len(bpy.data.objects)
try:
    store.new_object(ctx, "ShouldNotStay", _bad_g); _raised = False
except Exception:
    _raised = True
check("revolve: a selected curve on the picked axis is left out, the profile revolves; a failed object is removed",
      r_u == {'FINISHED'} and _n1 == _n0 + 1 and _raised and len(bpy.data.objects) == _nb
      and "ShouldNotStay" not in bpy.data.objects,
      (r_u, _n1 - _n0, _raised, len(bpy.data.objects) - _nb))
# the last selected decides: a straight active line is Revolve's axis; Sweep's active curve is the path and,
# on two rails, the profile is the curve touching both others
from blender_nurbs.ops import _straight_line
from blender_nurbs.ops_sweep import sweep_profile
_sl = _straight_line(features.input_curve(_ulo, "S0"))
_sw_r1 = nurbs("SwR1", [(0, 20, 0), (0, 20.5, 1), (0, 21, 2), (0, 21.5, 3)])
_sw_r2 = nurbs("SwR2", [(2, 20, 0), (2, 20.5, 1), (2, 21, 2), (2, 21.5, 3)])
_sw_pr = nurbs("SwPr", [(0, 20, 0), (0.7, 19.6, 0), (1.3, 19.6, 0), (2, 20, 0)])
check("two-element tools: a straight line is found (a curve is not); Sweep's profile among three curves",
      _sl is not None and np.allclose(_sl[1], (0, 0, -1), atol=1e-6) and _straight_line(features.input_curve(_ucv, "S0")) is None
      and sweep_profile([_sw_r1, _sw_pr, _sw_r2], _sw_r2) is _sw_pr and sweep_profile([_sw_pr, _sw_r1], _sw_r1) is _sw_pr,
      (_sl, sweep_profile([_sw_r1, _sw_pr, _sw_r2], _sw_r2).name))
check("revolve about a picked line: on the preview's rotation; a Feature box edit keeps the line",
      r == {'FINISHED'} and _gap_p < 1e-5 and _hp["axis"] == "PICKED" and abs(_hp["angle"] - 90.0) < 1e-9
      and np.allclose(_hp["axis_point"], _ap) and np.allclose(_hp["axis_dir"], _ad, atol=1e-6),
      (r, float(f"{_gap_p:.2g}"), _hp.get("axis"), _hp.get("angle"), _hp.get("axis_point")))

bpy.ops.nproto.extrude(inputs=ins((_pv_open, "S0")), distance=1.5, direction="Z"); _osh = ctx.active_object
bpy.ops.nproto.add_primitive(kind="BOX", width=2, depth=1.5, height=1); _obx = ctx.active_object
_obx.location = (0, -12, 0); ctx.view_layer.update()
_of_gap = []
for _o, _d in ((_osh, 0.2), (_osh, -0.15), (_obx, 0.2)):
    bpy.ops.object.select_all(action='DESELECT'); _o.select_set(True); ctx.view_layer.objects.active = _o
    _V, _W, _T, _sh = _offr(_o)
    bpy.ops.nproto.offset(distance=_d); _res = ctx.active_object; ctx.view_layer.update()
    _t = _tree(_res)
    _of_gap.append(max(_t.find_nearest(Vector(q_))[3] for q_ in _V + _d * _W))
    bpy.data.objects.remove(_res)
check("offset preview: the offset mesh lies on the result (sheet both sides, box with extended corners)",
      max(_of_gap) < 2e-3, [round(float(x), 5) for x in _of_gap])

_fl_gap = []
for _shape in ("CIRCULAR", "CHAMFER"):
    bpy.ops.nproto.add_primitive(kind="BOX", width=2, depth=1.5, height=1); _fb = ctx.active_object
    _fb.location = (0, -16, 0); ctx.view_layer.update()
    _ei = store.shared_edges(_fb)[0]
    _P, _D1, _D2 = _ffr(_fb, _ei)
    _X = _fsec(_P, _D1, _D2, 0.2, _shape)
    bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_fb.name, f"E{_ei}"]]), radius=0.2, shape=_shape)
    ctx.view_layer.update()
    _t = _tree(_fb)
    _fl_gap.append(max(_t.find_nearest(Vector(q_))[3] for q_ in _X[1:-1].reshape(-1, 3)))
check("edge fillet preview: rolling-ball arc and chamfer line lie on the kernel's fillet (box edge)",
      max(_fl_gap) < 1e-3, [round(float(x), 6) for x in _fl_gap])

# --- Surface Fillet / Offset Curve / Extend Surface previews and their arrows
from blender_nurbs import valuedrag as _vd, curveoffset as _cof, curvejoin as _cj, primitives as _prims
from blender_nurbs.ops_trim import (NPROTO_OT_fillet as _OpSF, surface_fillet_frames as _sff,
                                    surface_fillet_sections as _sfs)
from blender_nurbs.ops_sweep import NPROTO_OT_extend as _OpEx

def _pl_dist(q_, R_):
    """Distance from point q_ to the polyline R_ (vectorized)."""
    a_, ab_ = R_[:-1], np.diff(R_, axis=0)
    t_ = np.clip(np.einsum("ij,ij->i", q_ - a_, ab_) / np.maximum(np.einsum("ij,ij->i", ab_, ab_), 1e-300), 0, 1)
    return float(np.min(np.linalg.norm(a_ + t_[:, None] * ab_ - q_, axis=1)))

_v3k = lambda c_: np.array([c_.x, c_.y, c_.z] if hasattr(c_, "x") else list(c_), float)

def _plain(cls, **attrs):
    """An object with the operator's methods (no Blender operator), for its preview code."""
    body = {k: v for k, v in cls.__dict__.items() if not k.startswith("__") and not k.startswith("bl_")}
    o_ = type("_Plain" + cls.__name__, (_vd.ValueDrag,), body)()
    for k, v in attrs.items():
        setattr(o_, k, v)
    return o_

from blender_nurbs.ops import NPROTO_OT_thicken as _OpTk
_tk_la = nurbs("TkLa", [(0, -20, 0), (2, -20, 0)], order=2)
_tk_lb = nurbs("TkLb", [(2, -20, 0), (2, -20, 1.5)], order=2)
bpy.ops.nproto.extrude(inputs=ins((_tk_la, "S0")), distance=1.0, direction="Y"); _tka = ctx.active_object
bpy.ops.nproto.extrude(inputs=ins((_tk_lb, "S0")), distance=1.0, direction="Y"); _tkb = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); _tka.select_set(True); _tkb.select_set(True)
ctx.view_layer.objects.active = _tka
bpy.ops.nproto.join(); _tkl = ctx.active_object
_tk_gap = []
for _base, _d, _sym in ((_osh, 0.2, False), (_osh, -0.15, False), (_osh, 0.3, True), (_tkl, 0.1, False),
                        (_tkl, -0.1, True)):
    _src = _base.copy(); _src.data = _base.data.copy(); ctx.collection.objects.link(_src)
    for k_ in _base.keys():
        _src[k_] = _base[k_]
    bpy.ops.object.select_all(action='DESELECT'); _src.select_set(True); ctx.view_layer.objects.active = _src
    _x = _plain(_OpTk, distance=_d, symmetric=_sym)
    _ok = _x.drag_setup(ctx, None)
    _pv = _x.drag_preview()
    _h = _x.drag_handles()[0]
    _tip = _h["origin"] + _h["param"] * _h["dir"]
    _pts = np.vstack([_pv["tris"]] + list(_pv["lines"]))
    bpy.ops.nproto.thicken(distance=_d, symmetric=_sym); _res = ctx.active_object; ctx.view_layer.update()
    _t = _tree(_res)
    _g = max(_t.find_nearest(Vector(q_))[3] for q_ in _pts)
    _tk_gap.append(round(float(max(_g, _t.find_nearest(Vector(_tip))[3])) if _ok else 9.0, 5))
    bpy.data.objects.remove(_res)
    if _src.name in bpy.data.objects:
        bpy.data.objects.remove(_src)
for o_ in (_tk_la, _tk_lb, _tka, _tkb, _tkl):
    if o_.name in bpy.data.objects:
        bpy.data.objects.remove(o_)
check("thicken preview: skins, side walls and the arrow tip lie on the solid (sheet both sides, symmetric, "
      "joined L with its corner extended)", max(_tk_gap) < 2e-3, _tk_gap)

_sf_gap, _sf_arrow = [], []
for _kind, _fa, _fb2, _shape, _ra, _rb in ((60, False, False, "CIRCULAR", 0.2, 0.2), (60, True, True, "CIRCULAR", 0.2, 0.2),
                                          (125, False, True, "CHAMFER", 0.15, 0.25), ("cyl", False, True, "CIRCULAR", 0.2, 0.2)):
    bpy.ops.nproto.add_primitive(kind="PLANE"); _sa_o = ctx.active_object; _sa_o.location = (40, 0, 0)
    if _kind == "cyl":
        bpy.ops.nproto.add_primitive(kind="CYLINDER"); _sb_o = ctx.active_object; _sb_o.location = (40.2, 0.1, -0.3)
    else:
        bpy.ops.nproto.add_primitive(kind="PLANE"); _sb_o = ctx.active_object
        _sb_o.rotation_euler = (0, math.radians(_kind), math.radians(20)); _sb_o.location = (40.3, 0, 0.2)
    ctx.view_layer.update()
    _fr = _sff(_sa_o, _sb_o)
    _sg = (-1.0 if _fa else 1.0), (-1.0 if _fb2 else 1.0)
    # the arrows stand on the rails and give back the value they show
    _f = _plain(_OpSF, _sf=_fr, flip_a=_fa, flip_b=_fb2, rail_type="RADIUS", same_radius=_ra == _rb, radius=_ra,
                radius_b=_rb, shape=_shape)
    # (a chamfer's values are its distances: the preview's ball radii come from them)
    _Xs = [_sfs(P_, A_, B_, *_f._radii(P_, A_, B_), *_sg, _shape) for P_, A_, B_ in _fr]
    _j = len(_fr[0][0]) // 2
    # one radius: on rail A; Radius A / B: each on the rail its value moves (A's on rail B, B's on rail A);
    # a chamfer's Distance A / B: each on its own rail
    _rails = (_Xs[0][_j, 0], _Xs[0][_j, -1]) if (_ra == _rb or _shape == "CHAMFER") else (_Xs[0][_j, -1], _Xs[0][_j, 0])
    for _h, _rail in zip(_f.drag_handles(), _rails):
        _sf_arrow.append(max(float(np.linalg.norm(_vd.ValueDrag._pos(_h) - _rail)),
                             abs(_h["to_value"](_h["param"]) - getattr(_f, _h["field"]))))
    r = bpy.ops.nproto.fillet(surf_a=_sa_o.name, surf_b=_sb_o.name, radius=_ra, radius_b=_rb, same_radius=_ra == _rb,
                              shape=_shape, flip_a=_fa, flip_b=_fb2, auto_flip=False, trim_bases=False)
    _t = _tree(ctx.active_object)
    _sf_gap.append(max(_t.find_nearest(Vector(q_))[3] for X_ in _Xs for q_ in X_[2:-2, 1:-1].reshape(-1, 3)))
    for o_ in (_sa_o, _sb_o, ctx.active_object):
        bpy.data.objects.remove(o_)
check("surface fillet preview: ball arcs / chamfers on the kernel's fillet on every side (planes, cylinder, "
      "two chamfer distances); the arrows sit on the rails", max(_sf_gap) < 2e-3 and max(_sf_arrow) < 1e-9,
      ([round(float(x), 5) for x in _sf_gap], max(_sf_arrow)))
bpy.ops.nproto.add_primitive(kind="PLANE"); _sa_o = ctx.active_object; _sa_o.location = (40, 0, 0)
bpy.ops.nproto.add_primitive(kind="PLANE"); _sb_o = ctx.active_object; _sb_o.location = (40, 0, 3)
ctx.view_layer.update()
check("surface fillet: surfaces that do not cross have no preview (the dialog asks)", _sff(_sa_o, _sb_o) is None)

# two extruded lines meeting at their ends (a tent): only the inside has room, so
# the preview starts there (not on the default side, off the surfaces' ends) and
# A / B can't move it; the kernel makes it on that side with Auto Side off
from blender_nurbs.ops_trim import surface_fillet_sides as _sfsides, part_size as _psize
_tl = nurbs("TentL", [(39.7, 0, 0), (40, 0, 0.33)], order=2)
_tr = nurbs("TentR", [(40, 0, 0.33), (40.3, 0, 0)], order=2)
bpy.ops.nproto.extrude(inputs=ins((_tl, "S0")), distance=0.2, direction="Y"); _ta = ctx.active_object
bpy.ops.nproto.extrude(inputs=ins((_tr, "S0")), distance=0.2, direction="Y"); _tb = ctx.active_object
_fr = _sff(_ta, _tb)
_fits = _sfsides(_ta, _tb, _fr, _psize([_ta, _tb]))
_f = _plain(_OpSF, surf_a=_ta.name, surf_b=_tb.name, flip_a=True, flip_b=True, rail_type="RADIUS",
            same_radius=True, radius=0.03, radius_b=0.03, shape="CIRCULAR", _msg="")
_f.drag_setup(ctx, None)
_side = (_f.flip_a, _f.flip_b)
_f._toggle_side("a")
_kept = (_f.flip_a, _f.flip_b) == _side and bool(_f._msg)
_X = _sfs(*_fr[0], 0.03, 0.03, -1.0 if _side[0] else 1.0, -1.0 if _side[1] else 1.0)
_zmax = float(_X[..., 2].max())
r = bpy.ops.nproto.fillet(surf_a=_ta.name, surf_b=_tb.name, radius=0.03, flip_a=_side[0], flip_b=_side[1],
                          auto_flip=False, trim_bases=False)
_t = _tree(ctx.active_object) if r == {'FINISHED'} else None
_gap = _t and max(_t.find_nearest(Vector(q_))[3] for q_ in _X[2:-2, 1:-1].reshape(-1, 3))
check("surface fillet on a tent: one side has room, the preview starts there (inside, under the apex), A can't "
      "leave it, the kernel's fillet matches it",
      [k for k, v in _fits.items() if v > 0] == [_side] and _fits[_side] == 1.0 and _kept
      and _zmax < 0.33 and _gap is not None and _gap < 2e-3, (_fits, _side, _kept, round(_zmax, 4), r, _gap))
for o_ in (_tl, _tr, _ta, _tb) + ((ctx.active_object,) if r == {'FINISHED'} else ()):
    bpy.data.objects.remove(o_)

_oc_free = nurbs("OcFree", [(0, 50, 0), (1, 51, 0), (2, 49.5, 0), (3, 50.8, 0), (4, 50, 0)])
_oc_l = bpy.data.curves.new("OcL", 'CURVE'); _oc_l.dimensions = '3D'
_oc_ls = _oc_l.splines.new('POLY'); _oc_ls.points.add(4); _oc_ls.use_cyclic_u = True
for _q, _c in zip(_oc_ls.points, [(0, 54, 0), (3, 54, 0), (3, 55, 0), (1, 55, 0), (0, 56.5, 0)]):
    _q.co = (*_c, 1.0)
_oc_lo = bpy.data.objects.new("OcL", _oc_l); ctx.collection.objects.link(_oc_lo)
_oc_dev = []
for _o, _prm in ((_oc_free, {"distance": 0.3}), (_oc_free, {"distance": -0.2, "both_sides": True}),
                 (_oc_lo, {"distance": 0.25, "oc_corner": "SHARP"}), (_oc_lo, {"distance": 0.25, "oc_corner": "ROUND"}),
                 (_oc_lo, {"distance": -0.2, "oc_corner": "CHAMFER", "flip": True})):
    _pcs = [_cj.input_spans(_o, "S0")]
    _L = _cof.preview_polylines(_pcs, _prm, 1e-3)
    _R = []
    for sp_ in _cof.offset_curves(_pcs, _prm):  # noqa
        k_ = _prims.spec_to_kernel_curve(sp_)
        _R.append(np.array([k_.evaluate(t_) for t_ in np.linspace(*k_.parameter_range(), 3000)]))
    _Rl = _R
    _dev = max(min(_pl_dist(q_, R_) for R_ in _Rl) for L_ in _L for q_ in L_)
    _R = np.vstack(_R)
    _cov = max(min(_poly_dist(q_, L_) for L_ in _L) for q_ in _R[::25])     # and the preview covers all of it
    # the arrow: its foot on the curve, a positive distance along it lands on the offset
    _sp, _cl, _N, _sgn = _cof._chains(_pcs, dict(_prm, distance=1.0, both_sides=False), _cof._new_info())[0]
    _B, _v = _cof.side_direction(_sp, _cl, _N, _sgn, _cof._start(_sp) + 0.3)
    _arw = min(_pl_dist(_B + _prm["distance"] * _v, R_) for R_ in _Rl)
    _oc_dev.append(round(max(_dev, _cov, _arw), 5))
check("offset curve preview: the offset (corners sharp / round / chamfer, both sides, flipped) without the fit, "
      "and its arrow points to the side the distance goes", max(_oc_dev) < 2e-3, _oc_dev)

_ex_w = nurbs("ExW", [(0, 60, 0), (1, 60.5, 0), (2, 59.6, 0), (3, 60.1, 0)])
bpy.ops.nproto.extrude(inputs=ins((_ex_w, "S0")), distance=1.5, direction="Z"); _ex_s = ctx.active_object
_ex_s.scale = (1.5, 1.5, 1.5); ctx.view_layer.update()
_ex_dev = []
for _edge in ("U0", "U1", "V0", "V1"):
    for _d in (0.4, -0.3):
        _x = _plain(_OpEx, inputs=json.dumps([[_ex_s.name, _edge]]), distance=_d)
        _ok = _x.drag_setup(ctx, None)
        _rows, _path = _x._strip()
        _h = _x.drag_handles()[0]
        _sv = _ex_s.copy(); _sv.data = _ex_s.data.copy(); ctx.collection.objects.link(_sv)
        for k_ in _ex_s.keys():
            _sv[k_] = _ex_s[k_]
        bpy.ops.nproto.extend(inputs=json.dumps([[_sv.name, _edge]]), distance=_d)
        _rs = store.get_surfaces(_sv)[0]
        _Mi = np.array(_sv.matrix_world.inverted(), float)
        _loc = lambda q_: _Mi[:3, :3] @ q_ + _Mi[:3, 3]
        _cpd = lambda q_: float(np.linalg.norm(_v3k(_sm.surface_closest_point(_rs, tuple(map(float, _loc(q_))))[0])
                                               - _loc(q_)))
        _new = _rows[-1] if _d > 0 else _rows[0]
        _on = max(_cpd(q_) for r_ in (_rows if _d > 0 else [_new]) for q_ in r_[::4])
        # the new edge is the result's boundary there (its far side: nothing beyond)
        _bd = np.array(geom.boundary_points(_rs, _edge, 33)) @ np.array(_sv.matrix_world, float)[:3, :3].T \
            + np.array(_sv.matrix_world.translation)
        _edge_gap = max(min(_poly_dist(q_, _bd), _poly_dist(q_, _bd[::-1])) for q_ in _new[::4])
        _ex_dev.append(round(max(_on, _edge_gap, float(np.linalg.norm(_vd.ValueDrag._pos(_h) - _new[16])))
                             * (1 if _ok else 1e9), 7))
        bpy.data.objects.remove(_sv)
check("extend preview: the band lies on the extended surface, its far row is the new edge (shrink too), "
      "the arrow head on it (U0 / U1 / V0 / V1, scaled object)", max(_ex_dev) < 1e-5, _ex_dev)
# drafts of smooth curves: the result is a fit within ~3e-4 of the true draft here
check("extrude preview rows lie on the built surface's edges (plain, tilt, draft, symmetric, cone, polyline)",
      max(_pv_worst) < 1e-3 and max(_pv_worst[i] for i in (0, 1, 6, 7, 8)) < 1e-5, _pv_worst)
prof = nurbs("Prof", [(1,0,0),(1.3,0,0.5),(0.8,0,1),(1,0,1.5)])
r = bpy.ops.nproto.revolve(inputs=ins((prof,"S0")), axis="Z", pivot="WORLD", angle=360)
R = ctx.active_object; check("revolve", r == {'FINISHED'} and len(R.data.polygons) > 0)
bpy.ops.curve.primitive_bezier_circle_add(radius=1.0, location=(5,0,0))
circ = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); circ.select_set(True)
r = bpy.ops.nproto.planar(); P = ctx.active_object
check("planar", r == {'FINISHED'} and store.is_nurbs(P) and len(P.data.polygons) > 0,
      f"area={store.get_brep(P).area():.4f} (pi={math.pi:.4f})")
bpy.ops.object.select_all(action='DESELECT'); R.select_set(True)
r = bpy.ops.nproto.offset(distance=0.1); O = ctx.active_object
check("offset", r == {'FINISHED'} and len(O.data.polygons) > 0)

# --- Two surfaces facing each other, G1/G2 blend between their edges
s1c = nurbs("S1c", [(0,0,0),(1,0.3,0),(2,-0.2,0),(3,0,0)])
s2c = nurbs("S2c", [(0,0,3),(1,-0.3,3),(2,0.2,3),(3,0,3)])
bpy.ops.nproto.extrude(inputs=ins((s1c,"S0")), distance=1.0, direction="Z"); S1 = ctx.active_object
# extrude downward from z=0: flip direction by negative distance
bpy.ops.nproto.extrude(inputs=ins((s2c,"S0")), distance=-1.0, direction="Z"); S2 = ctx.active_object
# S1 spans z 0..1 (u along Z); S2 spans z 3..2; find edges facing each other
def edge_z(obj, which):
    s = store.get_surfaces(obj)[0]
    return np.mean([p[2] for p in geom.boundary_points(s, which, 8)])
ea = max(("U0","U1"), key=lambda w: edge_z(S1, w))
eb = min(("U0","U1"), key=lambda w: edge_z(S2, w))
check("facing edges found", abs(edge_z(S1, ea) - 1) < 1e-6 and abs(edge_z(S2, eb) - 2) < 1e-6, f"{ea} {eb}")

def cross_check(blend, src, src_edge, cont):
    """Compare normals (G1) across the shared edge at a few points."""
    sb = store.get_surfaces(blend)[0]; ss = store.get_surfaces(src)[0]
    worst = 0.0
    for t in (0.2, 0.5, 0.8):
        es = geom.sample_surface_edge(ss, src_edge, src.matrix_world, 11)
        p = es.P[int(t*10)]
        # nearest point on blend boundary rows u=0 / u=1
        (u0,u1),(v0,v1) = sb.u_domain(), sb.v_domain()
        best = None
        for u in (u0, u1):
            for k in range(201):
                v = v0 + (v1-v0)*k/200
                q = np.array(sb.evaluate(u, v)); d = np.linalg.norm(q - p)
                if best is None or d < best[0]: best = (d, u, v)
        d, u, v = best
        nb = np.array(sb.normal(u, v)); 
        # source normal at that point
        (a0,a1),(b0,b1) = ss.u_domain(), ss.v_domain()
        uu = a0 if src_edge=="U0" else a1 if src_edge=="U1" else None
        bestv = min(((np.linalg.norm(np.array(ss.evaluate(uu, b0+(b1-b0)*k/200)) - p), b0+(b1-b0)*k/200) for k in range(201)))
        ns = np.array(ss.normal(uu, bestv[1]))
        ang = math.degrees(math.acos(min(1, abs(float(np.dot(nb, ns)) / (np.linalg.norm(nb)*np.linalg.norm(ns))))))
        worst = max(worst, ang)
    return d, worst

for cont in ("G0", "G1", "G2"):
    r = bpy.ops.nproto.loft(inputs=ins((S1,ea),(S2,eb)), cont_a=cont, cont_b=cont, tan_a=1.0, tan_b=1.0)
    BL = ctx.active_object
    gap, ang = cross_check(BL, S1, ea, cont)
    d = store.display(BL)["nets"][0]
    if cont == "G0":
        check(f"blend {cont}", r == {'FINISHED'} and gap < 1e-3, f"gap={gap:.2e} normal angle={ang:.2f}deg nu={d['nu']}")
    else:
        check(f"blend {cont}", r == {'FINISHED'} and gap < 1e-3 and ang < 0.5,
              f"gap={gap:.2e} normal angle={ang:.3f}deg rows={d['nu']}")
G2 = BL

def cross_curvature(surf, u, v, axis, M=None):
    """Normal curvature along the u (axis=0) or v (axis=1) iso direction via finite differences."""
    h = 1e-4
    def P(uu, vv): return np.array(surf.evaluate(uu, vv))
    if axis == 0:
        d1 = (P(u + h, v) - P(u - h, v)) / (2*h); d2 = (P(u + h, v) - 2*P(u, v) + P(u - h, v)) / (h*h)
    else:
        d1 = (P(u, v + h) - P(u, v - h)) / (2*h); d2 = (P(u, v + h) - 2*P(u, v) + P(u, v - h)) / (h*h)
    n = np.array(surf.normal(u, v)); n /= np.linalg.norm(n)
    return float(np.dot(d2, n) / np.dot(d1, d1)), n
sb = store.get_surfaces(G2)[0]; ss = store.get_surfaces(S1)[0]
# source edge ea is U1 or U0 of S1 (u = extrusion direction); blend's u runs across.
us = ss.u_domain()[1] - 1e-3 if ea == "U1" else ss.u_domain()[0] + 1e-3
k_src, n_src = cross_curvature(ss, us, 0.5, 0)
# blend side touching S1: u = 0 (a side), probe just inside
k_bl, n_bl = cross_curvature(sb, 1e-3, 0.5, 0)
k_bl *= np.sign(np.dot(n_src, n_bl))
check("blend G2 curvature match", abs(k_src - k_bl) < 0.05 + 0.05 * abs(k_src), f"k_src={k_src:.4f} k_blend={k_bl:.4f}")

# --- G2 against a source that is curved across the edge (the 3-curve skin L)
top = nurbs("Top", [(0,0,6.5),(1,0.2,6.5),(2,0,6.5),(3,0.2,6.5)])
sL = store.get_surfaces(L)[0]
eL = max(("U0","U1"), key=lambda w: np.mean([p[2] for p in geom.boundary_points(sL, w, 8)]))
uL = sL.u_domain()[1] - 2e-3 if eL == "U1" else sL.u_domain()[0] + 2e-3
res = {}
for cont in ("G1", "G2"):
    bpy.ops.nproto.loft(inputs=ins((L, eL), (top, "S0")), cont_a=cont, tan_a=1.0,
                        edges_a="FREE", edges_b="FREE")        # measured along the loft's u lines: square to the edge
    blend = ctx.active_object; sbl = store.get_surfaces(blend)[0]
    ks = []; kb = []
    for v in (0.3, 0.5, 0.7):
        k1, n1 = cross_curvature(sL, uL, v, 0)
        k2, n2 = cross_curvature(sbl, 2e-3, v, 0)
        ks.append(k1); kb.append(k2 * np.sign(np.dot(n1, n2)))
    res[cont] = (ks, kb)
    bpy.data.objects.remove(blend)
ks, kb2 = res["G2"]; _, kb1 = res["G1"]
err2 = max(abs(a - b) for a, b in zip(ks, kb2)); err1 = max(abs(a - b) for a, b in zip(ks, kb1))
check("G2 matches cross curvature of curved source", err2 < 0.1 * max(abs(k) for k in ks) + 0.02 and err2 < err1,
      f"k_src={[round(k,3) for k in ks]} G2={[round(k,3) for k in kb2]} G1={[round(k,3) for k in kb1]}")

# --- History: move input curve B, process the timer, loft must change
before = sorted(tuple(round(c, 5) for c in v.co) for v in L.data.vertices)
B.location.x += 0.5
ctx.view_layer.update()
history._process()  # in UI the timer does this
after = sorted(tuple(round(c, 5) for c in v.co) for v in L.data.vertices)
check("history follows input move", before != after)
# history chain: move S1 -> G2 blend rebuilds
rev = G2.get("nproto_rev")
S1.location.y += 0.2; ctx.view_layer.update(); history._process()
check("history rebuilds blend from moved surface", G2.get("nproto_rev") != rev)

# --- Delete history
bpy.ops.object.select_all(action='DESELECT'); L.select_set(True)
bpy.ops.nproto.delete_history()
check("delete history", features.history(L) is None)

# --- Trim / Split / Untrim / Fillet / Chamfer (worker process) -----------
def poly(name, pts, loc=(0, 0, 0)):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    o = bpy.data.objects.new(name, cu); o.location = loc; ctx.collection.objects.link(o); return o
def area(o): return store.get_brep(o).area()
def make_floor(name, loc=(0, 0, 0)):
    ln = poly(name + "_ln", [(0, -2, 0), (0, 2, 0)], loc)
    bpy.ops.nproto.extrude(inputs=ins((ln, "S0")), distance=4.0, direction="X")
    o = ctx.active_object; o.name = name; return o
floor = make_floor("Floor")
check("floor area 16", abs(area(floor) - 16) < 1e-6, f"{area(floor):.4f}")
bpy.ops.curve.primitive_bezier_circle_add(radius=0.5, location=(1, 0, 1)); cc = ctx.active_object
tr = dict(target="Floor", cutters=ins((cc, "S0")), ref_point=(1, 0, 0), normal=(0, 0, 1), view_dir=(0, 0, -1))
r = bpy.ops.nproto.trim(action="KEEP", **tr)
check("trim keep circle", r == {'FINISHED'} and abs(area(floor) - math.pi / 4) < 2e-3, f"{area(floor):.4f} vs {math.pi/4:.4f}")
check("trimmed edges pickable (E refs)", not store.is_natural(floor) and len(store.display(floor)["edges"]) >= 1)
r = bpy.ops.nproto.untrim()
check("untrim", r == {'FINISHED'} and abs(area(floor) - 16) < 1e-6 and store.is_natural(floor))
bpy.ops.object.select_all(action='DESELECT')
r = bpy.ops.nproto.trim(action="DELETE", **tr)
check("trim discard circle", abs(area(floor) - (16 - math.pi / 4)) < 2e-3, f"{area(floor):.4f}")
floor.select_set(True); ctx.view_layer.objects.active = floor; bpy.ops.nproto.untrim()
r = bpy.ops.nproto.trim(action="SPLIT", **tr)
parts = [o for o in bpy.data.objects if o.name.startswith("Floor") and store.is_nurbs(o)]
check("split -> 2 objects", len(parts) == 2 and abs(sum(area(o) for o in parts) - 16) < 2e-3,
      ", ".join(f"{o.name}={area(o):.3f}" for o in parts))
for o in parts:
    if o.name != "Floor": bpy.data.objects.remove(o)
floor.select_set(True); ctx.view_layer.objects.active = floor; bpy.ops.nproto.untrim()

# trim on a moved object (local/world handling)
f2 = make_floor("FloorMoved", loc=(10, 0, 0))
f2.location.z += 0.5; ctx.view_layer.update()
bpy.ops.curve.primitive_bezier_circle_add(radius=0.5, location=(11, 0, 2)); cc2 = ctx.active_object
r = bpy.ops.nproto.trim(target="FloorMoved", cutters=ins((cc2, "S0")), ref_point=(11, 0, 0.5), normal=(0, 0, 1), action="KEEP")
ws = [f2.matrix_world @ v.co for v in f2.data.vertices]
cx = sum(v.x for v in ws) / len(ws); cz = sum(v.z for v in ws) / len(ws)
check("trim moved object", r == {'FINISHED'} and abs(area(f2) - math.pi / 4) < 2e-3 and abs(cx - 11) < 0.05 and abs(cz - 0.5) < 1e-4,
      f"area={area(f2):.4f} centre=({cx:.3f}, {cz:.3f})")

# trim by an intersecting surface
wl = poly("Wall_ln", [(2, -2, -1), (2, 2, -1)])
bpy.ops.nproto.extrude(inputs=ins((wl, "S0")), distance=3.0, direction="Z"); wall = ctx.active_object; wall.name = "Wall"
r = bpy.ops.nproto.trim(target="Floor", cutters=json.dumps([["Wall", "SURF"]]), ref_point=(1, 0, 0), normal=(0, 0, 1), action="KEEP")
check("trim by surface intersection", r == {'FINISHED'} and abs(area(floor) - 8) < 2e-3, f"{area(floor):.4f}")
floor.select_set(True); ctx.view_layer.objects.active = floor; bpy.ops.object.select_all(action='DESELECT')
floor.select_set(True); bpy.ops.nproto.untrim()

# fillet floor/wall, trimming both
wall_area0 = area(wall)
r = bpy.ops.nproto.fillet(surf_a="Floor", surf_b="Wall", radius=0.3, shape="CIRCULAR", trim_bases=True)
fil = ctx.active_object
check("fillet circular", r == {'FINISHED'} and abs(area(fil) - 0.5 * math.pi * 0.3 * 4) < 2e-3,
      f"fillet={area(fil):.4f} (expect {0.5*math.pi*0.3*4:.4f})")
fa, wa = area(floor), area(wall)
check("fillet trimmed both bases", abs(fa - 1.7 * 4) < 2e-3 and abs(wa - (wall_area0 - 0.3 * 4)) < 2e-3 or abs(wa - (wall_area0 - 1.3*4)) < 2e-3,
      f"floor={fa:.4f} wall={wa:.4f} (wall before {wall_area0:.2f})")
bpy.ops.ed.undo_push(message="x")
for o in (floor, wall):
    bpy.ops.object.select_all(action='DESELECT'); o.select_set(True); bpy.ops.nproto.untrim()
r = bpy.ops.nproto.fillet(surf_a="Floor", surf_b="Wall", radius=0.3, shape="CHAMFER", trim_bases=False)
ch = ctx.active_object
check("chamfer", r == {'FINISHED'} and abs(area(ch) - 0.3 * math.sqrt(2) * 4) < 2e-3 and abs(area(floor) - 16) < 1e-6,
      f"chamfer={area(ch):.4f} (expect {0.3*math.sqrt(2)*4:.4f}), floor untouched={area(floor):.1f}")

# no solution / crash safety: parallel far-apart surfaces
far = make_floor("FarFloor", loc=(0, 0, 50))
try:
    r = bpy.ops.nproto.fillet(surf_a="Floor", surf_b="FarFloor", radius=0.3)
except RuntimeError as exc:
    r = str(exc)
check("fillet with no solution fails cleanly", r != {'FINISHED'}, str(r)[:90])

# loft from a trimmed B-rep edge (G0)
bpy.ops.nproto.trim(target="FloorMoved", cutters=ins((cc2, "S0")), ref_point=(11, 0, 0.5), normal=(0, 0, 1), action="KEEP")
eidx = 0
r = bpy.ops.nproto.loft(inputs=ins((f2, f"E{eidx}"), (cc2, "S0")))
check("loft from trimmed edge", r == {'FINISHED'} and len(ctx.active_object.data.polygons) > 0)

# --- Planar from picked edges / loops ------------------------------------
from blender_nurbs import ops as nops
from mathutils import Vector as V
pf = make_floor("PlanarSrc", loc=(0, 20, 0))
r = bpy.ops.nproto.planar(inputs=ins((pf, "U0"), (pf, "U1"), (pf, "V0"), (pf, "V1")))
po = ctx.active_object
check("planar from 4 surface edges", r == {'FINISHED'} and abs(area(po) - 16) < 1e-3, f"{area(po):.4f}")
check("chain natural surface edge -> 4 edges", nops.edge_loop_refs(pf, "V1") == ["U0", "U1", "V0", "V1"])
# trimmed hole: pick ONE edge of the hole, chain gives the hole loop
bpy.ops.curve.primitive_bezier_circle_add(radius=0.5, location=(1, 20, 1)); hc = ctx.active_object
bpy.ops.nproto.trim(target="PlanarSrc", cutters=ins((hc, "S0")), ref_point=(1, 20, 0), normal=(0, 0, 1), action="DELETE")
edges = store.display(pf)["edges"]
def centroid(pl): return V([sum(p[i] for p in pl) / len(pl) for i in range(3)])
hole_i = min(range(len(edges)), key=lambda i: (pf.matrix_world @ centroid(edges[i]) - V((1, 20, 0))).length if edges[i] else 1e9)
loop = nops.edge_loop_refs(pf, f"E{hole_i}")
outer_i = next(i for i in range(len(edges)) if f"E{i}" not in loop and edges[i])
outer = nops.edge_loop_refs(pf, f"E{outer_i}")
check("chain trimmed edges -> hole loop / outer loop", len(loop) >= 1 and len(outer) == 4 and not set(loop) & set(outer),
      f"hole={loop} outer={outer}")
r = bpy.ops.nproto.planar(inputs=ins(*[(pf, e) for e in loop]))
check("planar fills the hole", r == {'FINISHED'} and abs(area(ctx.active_object) - math.pi / 4) < 3e-3,
      f"{area(ctx.active_object):.4f}")
r = bpy.ops.nproto.planar(inputs=ins(*[(pf, e) for e in outer + loop]))
check("planar outer + hole loops", r == {'FINISHED'} and abs(area(ctx.active_object) - (16 - math.pi / 4)) < 3e-3,
      f"{area(ctx.active_object):.4f}")
# closure detection on world polylines
M = pf.matrix_world
polys = [[M @ V(p) for p in edges[int(e[1:])]] for e in outer]
check("closure: 4 outer edges closed, 3 open", nops.polylines_closed(polys, 1e-4) and not nops.polylines_closed(polys[:3], 1e-4))
# mixed: 3 surface edges + one curve closing the 4th side (curve on the U0? side)
src = make_floor("PlanarSrc2", loc=(0, 30, 0))
bb = [store.display(src)["nets"][0]["p"][i] for i in (0, -1)]
side = {}
for w in ("U0", "U1", "V0", "V1"):
    pl = [src.matrix_world @ V(p) for p in geom.boundary_points(store.get_surfaces(src)[0], w, 2)]
    side[w] = pl
w_last = "V1"
a_, b_ = side[w_last]
cl = poly("CloseLine", [tuple(a_), tuple(b_)])
r = bpy.ops.nproto.planar(inputs=ins((src, "U0"), (src, "U1"), (src, "V0"), (cl, "S0")))
check("planar from 3 edges + 1 curve", r == {'FINISHED'} and abs(area(ctx.active_object) - 16) < 1e-3, f"{area(ctx.active_object):.4f}")
# non-planar: lift the closing curve
cl.location.z += 1.0; ctx.view_layer.update()
try:
    r = bpy.ops.nproto.planar(inputs=ins((src, "U0"), (src, "U1"), (src, "V0"), (cl, "S0")))
except RuntimeError as exc:
    r = str(exc)
check("non-planar profile fails cleanly", r != {'FINISHED'}, str(r)[:80])

# --- Planar cap on a revolve rim (closed edge = its own loop) -------------
rp = nurbs("CapProf", [(31, 0, 0), (31.4, 0, 0.6), (30.7, 0, 1.4), (31.2, 0, 2.0)])
rp.location = (0, 0, 0)
bpy.ops.nproto.revolve(inputs=ins((rp, "S0")), axis="Z", pivot="WORLD", angle=360)
rv = ctx.active_object
s_rv = store.get_surfaces(rv)[0]
rims = {w: geom.boundary_points(s_rv, w, 8) for w in ("U0", "U1", "V0", "V1")}
top = max(rims, key=lambda w: sum(p[2] for p in rims[w]) / 8 if abs(rims[w][0][2] - rims[w][-1][2]) < 1e-9 else -1e9)
check("revolve rim chains to itself only", nops.edge_loop_refs(rv, top) == [top], f"{top} -> {nops.edge_loop_refs(rv, top)}")
r = bpy.ops.nproto.planar(inputs=ins((rv, top)))
check("planar cap on revolve rim", r == {'FINISHED'} and abs(area(ctx.active_object) - math.pi * 31.2 ** 2) < 0.5,
      f"{area(ctx.active_object):.2f} vs {math.pi*31.2**2:.2f}")

# --- Blender NURBS circle primitive: exact conversion + stale matrices ---
bpy.ops.curve.primitive_nurbs_circle_add(radius=0.8, location=(40, 0, 1.5), rotation=(math.pi / 2, 0, 0))
nc = ctx.active_object
sp = nc.data.splines[0]
kc = features.input_curve(nc, "S0")
P = geom.sample_curve(kc, 200)
c = np.array([40, 0, 1.5])
rad = np.linalg.norm(P - c, axis=1)
check("NURBS circle primitive converts to an exact circle", rad.max() - rad.min() < 1e-5 and abs(rad.mean() - 0.8) < 1e-5,
      f"order={sp.order_u} cyclic={sp.use_cyclic_u} endpoint={sp.use_endpoint_u} bezier={sp.use_bezier_u} radius {rad.min():.6f}..{rad.max():.6f}")
r = bpy.ops.nproto.extrude(inputs=ins((nc, "S0")), distance=2.0, direction="NORMAL", symmetric=True)
ex = ctx.active_object
ex_area = store.get_brep(ex).area(1e-7)
check("extruded NURBS circle = cylinder, area 2*pi*r*h", abs(ex_area - 2 * math.pi * 0.8 * 4.0) < 1e-4,
      f"{ex_area:.6f} vs {2*math.pi*0.8*4:.6f}")
# stale runtime matrix (what F9 redo sees right after its undo): move without updating
nc.location = (45, 3, 2.5)
kc2 = features.input_curve(nc, "S0")
_s = geom.sample_curve(kc2, 400); ctr = 0.5 * (_s.min(axis=0) + _s.max(axis=0))
check("inputs use stored transform, not stale matrix_world", np.linalg.norm(ctr - np.array([45, 3, 2.5])) < 1e-4,
      f"centre={np.round(ctr, 4)} matrix_world.t={tuple(round(x, 2) for x in nc.matrix_world.translation)}")
ctx.view_layer.update()
# a Blender spline that Blender itself cannot evaluate gives a clear error
bad = nurbs("BadBez", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0), (2, 2, 0), (2, 0, 0)])
bs = bad.data.splines[0]; bs.order_u = 4; bs.use_cyclic_u = True; bs.use_bezier_u = True
try:
    features.input_curve(bad, "S0"); msg = "no error"
except ValueError as exc:
    msg = str(exc)
check("unevaluable Blender spline -> clear error", "cannot evaluate" in msg, msg[:80])

# --- Trim on a closed revolve (vase): region picking + curve-normal projection
vp = nurbs("VaseProf", [(1.2, 0, 0), (1.5, 0, 0.8), (0.8, 0, 1.8), (1.1, 0, 3.0)])
vp.location = (60, 0, 0); ctx.view_layer.update()
bpy.ops.nproto.revolve(inputs=ins((vp, "S0")), axis="Z", pivot="OBJECT", angle=360)
vase = ctx.active_object; vase.name = "VaseT"
vase_area = store.get_brep(vase).area(1e-6)
def vcircle(name, x, r):
    bpy.ops.curve.primitive_nurbs_circle_add(radius=r, location=(x, 0, 1.5), rotation=(math.pi / 2, 0, 0))
    o = ctx.active_object; o.name = name; return o
small = vcircle("VaseSmall", 60.0, 0.3)          # inside the outline: hits front AND back wall
rad_front = 1.25  # vase radius near z=1.5 is ~1.1-1.3; pick point in front (y>0) on the wall
front_pt = (60.0, rad_front, 1.5)
def vtrim(action, cutter, pt, scope="PICKED"):
    vase.select_set(True); ctx.view_layer.objects.active = vase
    return bpy.ops.nproto.trim(target="VaseT", cutters=ins((cutter, "S0")), ref_point=pt,
                               normal=(0, 1, 0), view_dir=(0, -1, 0), action=action, scope=scope)
r = vtrim("DELETE", small, front_pt)
a1 = store.get_brep(vase).area(1e-6)
disc = vase_area - a1
check("vase: discard removes only the picked (front) region", r == {'FINISHED'} and 0.2 < disc < 0.4,
      f"removed {disc:.4f} (one wall disc ~0.29, both walls would be ~0.58)")
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()
r = vtrim("KEEP", small, front_pt)
check("vase: keep leaves only the picked disc", r == {'FINISHED'} and abs(store.get_brep(vase).area(1e-6) - disc) < 1e-3,
      f"kept {store.get_brep(vase).area(1e-6):.4f}")
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()
side = vcircle("VaseSide", 61.25, 0.5)           # straddles the outline: one notch through the side
on_wall, _ = store.get_brep(vase).faces()[0].closest_point((61.25, 0.0, 1.5))
r = vtrim("DELETE", side, on_wall)
notch = vase_area - store.get_brep(vase).area(1e-6)
check("vase: circle across the outline cuts one notch (both seam halves)", r == {'FINISHED'} and 1.2 < notch < 2.0,
      f"removed {notch:.4f}, faces={store.get_brep(vase).face_count()}")
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()
# curve-normal is the default: same result as forcing world Y for these circles
r = bpy.ops.nproto.trim(target="VaseT", cutters=ins((small, "S0")), ref_point=front_pt, normal=(1, 0, 0),
                        view_dir=(1, 0, 0), action="DELETE", scope="PICKED")
check("projection defaults to the curve normal (ignores surface normal/view)",
      r == {'FINISHED'} and abs((vase_area - store.get_brep(vase).area(1e-6)) - disc) < 1e-3)
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()

# --- Trim: regions marked before committing (the preview stage) -------------
regs, picked_i, _ = steps.trim_regions(vase, {}, curves=[steps.cutter_curve_json(features.input_curve(small, "S0"),
                                       "CURVE", Vector((0, -1, 0)))], direction=[0, 1, 0], ref_point=[60.0, 0.0, 3.0],
                                       extend=True)
tri_ok = all(len(steps.region_mesh(rg["text"], vase.matrix_world)[0]) for rg in regs)
check("trim preview: three regions (outside + a disc on each wall), each drawable", len(regs) == 3 and tri_ok,
      f"{len(regs)} regions")
front_i = max(range(len(regs)), key=lambda i: regs[i]["point"][1])
marks = json.dumps([[rg["point"], i != front_i] for i, rg in enumerate(regs)])
r = bpy.ops.nproto.trim(target="VaseT", cutters=ins((small, "S0")), ref_point=(60.0, 0.0, 3.0), normal=(0, 1, 0),
                        view_dir=(0, -1, 0), action="KEEP", scope="THROUGH", marks=marks)
cut = vase_area - store.get_brep(vase).area(1e-6)
check("trim marks: keep outside + back disc removes only the front disc", r == {'FINISHED'} and abs(cut - disc) < 1e-3,
      f"removed {cut:.4f} vs {disc:.4f}")
st = features.stack(vase)
if st:
    features.rebuild(vase, ctx, force=True)
    check("trim marks survive a history rebuild",
          abs((vase_area - store.get_brep(vase).area(1e-6)) - disc) < 1e-3 and bool(st[-1]["params"].get("marks_local")))
marked_edges = store.get_brep(vase).edge_count()
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()
# one-sided: the circle's Z points to -Y, so Against Normal cuts only the +Y (front_pt) wall
r = bpy.ops.nproto.trim(target="VaseT", cutters=ins((small, "S0")), ref_point=front_pt, normal=(0, 1, 0),
                        view_dir=(0, -1, 0), action="DELETE", scope="THROUGH", reach="BACK")
cut = vase_area - store.get_brep(vase).area(1e-6)
check("trim reach: against the sketch normal cuts one wall only", r == {'FINISHED'} and abs(cut - disc) < 1e-3,
      f"removed {cut:.4f} vs {disc:.4f}")
one_sided_edges = store.get_brep(vase).edge_count()
check("trim marks: the kept wall has no seam left from the cut (as many edges as a one-sided cut)",
      marked_edges == one_sided_edges, f"{marked_edges} vs {one_sided_edges} edges")
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()
r = bpy.ops.nproto.trim(target="VaseT", cutters=ins((small, "S0")), ref_point=front_pt, normal=(0, 1, 0),
                        view_dir=(0, -1, 0), action="KEEP", scope="THROUGH", reach="FRONT")
check("trim reach: along the normal only the -Y wall loses its disc (keep outside)", r == {'FINISHED'}
      and abs(store.get_brep(vase).area(1e-6) - (vase_area - disc)) < 1e-3,
      f"{store.get_brep(vase).area(1e-6):.4f} vs {vase_area - disc:.4f}")
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()

# marks made with one Cut direction keep their side of the cutter when Cut is flipped (F9 / F)
regs, picked_i, _ = steps.trim_regions(vase, {}, curves=[steps.cutter_curve_json(features.input_curve(small, "S0"),
                                       "CURVE", Vector((0, -1, 0)), steps.curve_up(small))], direction=[0, 1, 0],
                                       ref_point=list(front_pt), extend=True, reach="FRONT")
marks = json.dumps([[rg["point"], not any(rg["side"]), rg["side"]] for rg in regs])   # keep the outside
r = bpy.ops.nproto.trim(target="VaseT", cutters=ins((small, "S0")), ref_point=front_pt, normal=(0, 1, 0),
                        view_dir=(0, -1, 0), action="KEEP", reach="BACK", marks=marks)
cut = vase_area - store.get_brep(vase).area(1e-6)
check("trim marks: flipping Cut keeps the same side (the outside), not the clicked region",
      r == {'FINISHED'} and abs(cut - disc) < 1e-3, f"removed {cut:.4f} vs {disc:.4f}")
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()

# --- Untrim Region: give back one trimmed-away area -----------------------
r = vtrim("DELETE", small, front_pt, scope="THROUGH")          # a hole in each wall
def hole_point(sign):
    for e in store.get_brep(vase).edges():
        t0, t1 = e.parameter_range(); m = e.point_at(0.5 * (t0 + t1))
        if sign * m[1] > 0.5 and abs(m[0] - 60.0) < 0.5 and 1.0 < m[2] < 2.0:
            return m
prev = steps.untrim_preview(vase[store.PROP_BREP], features.base_text(vase), jobs.xform(vase))
check("untrim preview: both holes show as removed areas", len(prev) == 2
      and all(abs(store.brep_from_text(t).area(1e-6) - disc) < 1e-3 for t, _ in prev), f"{len(prev)} areas")
r = bpy.ops.nproto.untrim_region(target="VaseT", point=hole_point(+1))
a = store.get_brep(vase).area(1e-6)
check("untrim region: the +Y hole is filled, the -Y hole stays", r == {'FINISHED'} and abs((vase_area - a) - disc) < 1e-3
      and hole_point(+1) is None and hole_point(-1) is not None, f"missing {vase_area - a:.4f} vs {disc:.4f}")
check("untrim region: one face, no seam where the hole was", store.get_brep(vase).face_count() == 1)
if features.PROP_STACK in vase:
    check("untrim region is a history step", [e["op"] for e in features.stack(vase)][-1] == "untrim_region")
    features.rebuild(vase, ctx, force=True)
    check("untrim region survives a rebuild", abs((vase_area - store.get_brep(vase).area(1e-6)) - disc) < 1e-3)
try:
    r = bpy.ops.nproto.untrim_region(target="VaseT", point=(58.8, 0.0, 0.0))     # the vase's own bottom rim
except RuntimeError:
    r = {'CANCELLED'}
check("untrim region refuses the surface's own edge", r == {'CANCELLED'})
prev = steps.untrim_preview(vase[store.PROP_BREP], features.base_text(vase), jobs.xform(vase))
r = bpy.ops.nproto.untrim_region(points=json.dumps({"VaseT": [p for _, p in prev]}))     # picked in the preview
check("untrim region: the last hole gives the whole surface back (trim steps gone)",
      r == {'FINISHED'} and abs(store.get_brep(vase).area(1e-6) - vase_area) < 1e-3
      and not any(e["op"] in ("trim", "untrim_region") for e in features.stack(vase)))
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()

# --- Split with pieces marked in the preview (on a copy of the vase) -------
vc = store.new_object_text(ctx, "VaseCopy", vase[store.PROP_BREP], geom.world_matrix(vase))
info = steps.split_info([(vc, "SURF"), (small, "S0")], {"projection": "CURVE", "view_dir": (0, -1, 0)})
plus = max(range(len(info["texts"])), key=lambda i: info["points"][i][1] if any(info["sides"][i]) else -1e9)
marks = json.dumps({"VaseCopy": [[pt, i != plus, sd] for i, (pt, sd) in enumerate(zip(info["points"], info["sides"]))]})
bpy.ops.object.select_all(action='DESELECT')
r = bpy.ops.nproto.split(targets=json.dumps(["VaseCopy"]), curves=ins((small, "S0")), view_dir=(0, -1, 0), marks=marks)
made = list(ctx.selected_objects)
pieces = sorted((store.get_brep(o).area(1e-6), store.get_brep(o).face_count()) for o in made)
check("split marks: two pieces -- the +Y disc, and the rest as one face (no seam at the -Y disc)",
      r == {'FINISHED'} and len(pieces) == 2 and abs(pieces[0][0] - disc) < 1e-3
      and abs(pieces[1][0] - (vase_area - disc)) < 1e-3 and pieces[1][1] == 1, str(pieces))
for o in made + [vc]:
    if o.name in bpy.data.objects:
        bpy.data.objects.remove(o)
vc = store.new_object_text(ctx, "VaseCopy", vase[store.PROP_BREP], geom.world_matrix(vase))
before = vc[store.PROP_BREP]
info = steps.split_info([(vc, "SURF"), (small, "S0")], {"projection": "CURVE", "view_dir": (0, -1, 0)})
marks = json.dumps({"VaseCopy": [[pt, True, sd] for pt, sd in zip(info["points"], info["sides"])]})
try:
    r = bpy.ops.nproto.split(targets=json.dumps(["VaseCopy"]), curves=ins((small, "S0")), view_dir=(0, -1, 0),
                             marks=marks)
except RuntimeError:
    r = {'CANCELLED'}
check("split marks: a surface with every region joined is left as it is (no seam, no new object)",
      r == {'CANCELLED'} and vc[store.PROP_BREP] == before and "VaseCopy_split1" not in bpy.data.objects)
bpy.data.objects.remove(vc)
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); ctx.view_layer.objects.active = vase

# --- a cutting surface extended through the part ----------------------------
bpy.ops.curve.primitive_nurbs_circle_add(radius=0.3, location=(60.0, 1.0, 1.5), rotation=(math.pi / 2, 0, 0))
tube_c = ctx.active_object; tube_c.name = "TubeCirc"
bpy.ops.nproto.extrude(inputs=ins((tube_c, "S0")), distance=0.8, direction="Y")
tube = ctx.active_object; tube.name = "ShortTube"
n_as_is = len(steps.split_info([(vase, "SURF"), (tube, "SURF")], {"view_dir": (0, -1, 0)})["texts"])
n_through = len(steps.split_info([(vase, "SURF"), (tube, "SURF")], {"view_dir": (0, -1, 0), "through": True})["texts"])
check("surfaces through: a short tube through one wall cuts both when extended", n_as_is == 2 and n_through == 3,
      f"{n_as_is} regions as is, {n_through} through")
for o in (tube, tube_c):
    bpy.data.objects.remove(o)
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); ctx.view_layer.objects.active = vase

# --- Trim: THROUGH scope, revolve seam, surface cutters -------------------
r = vtrim("DELETE", small, front_pt, scope="THROUGH")
both = vase_area - store.get_brep(vase).area(1e-6)
check("through: circle cuts both walls in one trim", r == {'FINISHED'} and abs(both - 2 * disc) < 2e-3,
      f"removed {both:.4f} (2 x {disc:.4f})")
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()
# click OUTSIDE the circle -> the complement: keep only the two discs... with DELETE: remove everything outside
r = vtrim("KEEP", small, front_pt, scope="THROUGH")
check("through keep: keeps both discs", r == {'FINISHED'} and abs(store.get_brep(vase).area(1e-6) - 2 * disc) < 2e-3,
      f"kept {store.get_brep(vase).area(1e-6):.4f}")
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()
# seam: circle centred on the revolve seam (the profile plane y=0, x=61.x side faces +x) -> the
# projected hole is split by the seam into two faces; it must still be removed as ONE region
seam_c = vcircle("SeamCircle", 61.0, 0.3)
seam_c.rotation_euler = (0, math.pi / 2, 0); ctx.view_layer.update()   # normal along X: projects onto the seam side
on_seam, _ = store.get_brep(vase).faces()[0].closest_point((61.5, 0.0, 1.5))
r = vtrim("DELETE", seam_c, on_seam, scope="PICKED")
seam_hole = vase_area - store.get_brep(vase).area(1e-6)
check("seam: hole across the revolve seam removed completely", r == {'FINISHED'} and 0.2 < seam_hole < 0.4,
      f"removed {seam_hole:.4f} (full disc ~{math.pi*0.3**2:.3f} on a curved wall)")
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()
# surface cutter: a tube along Y through the vase (exact imprint)
tc = vcircle("TubeC", 60.0, 0.4); tc.location.y = -3; ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((tc, "S0")), distance=6.0, direction="Y"); tube = ctx.active_object; tube.name = "TubeT"
tube_area0 = store.get_brep(tube).area(1e-6)
r = bpy.ops.nproto.trim(target="VaseT", cutters=json.dumps([["TubeT", "SURF"]]), ref_point=front_pt,
                        normal=(0, 1, 0), view_dir=(0, -1, 0), action="DELETE", scope="THROUGH")
holes = vase_area - store.get_brep(vase).area(1e-6)
check("surface cutter: tube punches both walls exactly", r == {'FINISHED'} and 0.6 < holes < 1.6,
      f"removed {holes:.4f}, faces={store.get_brep(vase).face_count()}")
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); bpy.ops.nproto.untrim()
# trim the tube by the vase: discard the part inside the vase
inside_pt = (60.0, 0.4, 1.5)   # on the tube, inside the vase
r = bpy.ops.nproto.trim(target="TubeT", cutters=json.dumps([["VaseT", "SURF"]]), ref_point=inside_pt,
                        normal=(0, 0, 1), view_dir=(0, -1, 0), action="DELETE", scope="THROUGH")
t_after = store.get_brep(tube).area(1e-6)
check("surface cutter: tube trimmed back to the vase", r == {'FINISHED'} and 0.5 < tube_area0 - t_after < tube_area0 * 0.6,
      f"tube {tube_area0:.3f} -> {t_after:.3f}, faces={store.get_brep(tube).face_count()}")

# --- Split ---------------------------------------------------------------
def pieces(prefix):     # (with history the split surface stays as a hidden input: not a piece)
    return sorted((o for o in bpy.data.objects if o.name.startswith(prefix) and store.is_nurbs(o) and not o.hide_get()),
                  key=lambda o: o.name)
sf = make_floor("SpF", loc=(0, 80, 0))
swl = poly("SpW_ln", [(2, 78, -1), (2, 82, -1)])
bpy.ops.nproto.extrude(inputs=ins((swl, "S0")), distance=3.0, direction="Z"); sw = ctx.active_object; sw.name = "SpW"
bpy.ops.object.select_all(action='DESELECT'); sf.select_set(True); sw.select_set(True); ctx.view_layer.objects.active = sw
r = bpy.ops.nproto.split()
fl, wl = pieces("SpF"), pieces("SpW")
check("split: two surfaces split each other", r == {'FINISHED'} and len(fl) == 2 and len(wl) == 2,
      f"floor={[round(area(o), 3) for o in fl]} wall={[round(area(o), 3) for o in wl]}")
check("split: areas preserved", abs(sum(area(o) for o in fl) - 16) < 1e-3 and abs(sum(area(o) for o in wl) - 12) < 1e-3
      and sorted(round(area(o), 3) for o in wl) == [4.0, 8.0])
bpy.ops.object.select_all(action='DESELECT'); fl[-1].select_set(True); ctx.view_layer.objects.active = fl[-1]
bpy.ops.nproto.untrim()
check("split: untrim a piece restores the whole surface", abs(area(fl[-1]) - 16) < 1e-6)
features.rebuild(fl[-1], ctx, force=True)       # (a split piece's feature gives the piece: the untrim is a step)
check("split: the untrimmed piece stays whole through a rebuild", abs(area(fl[-1]) - 16) < 1e-6 and store.is_natural(fl[-1]))
_sp = fl[0]
_sp.location = (0.5, -0.25, 0.75); ctx.view_layer.update()      # moved: Untrim Region counts the cuts in world space
_prev = steps.untrim_preview(_sp[store.PROP_BREP], steps.untrim_reference(_sp), jobs.xform(_sp))
r = bpy.ops.nproto.untrim_region(points=json.dumps({_sp.name: [p for _, p in _prev]}))
features.rebuild(_sp, ctx, force=True)
check("split: Untrim Region on a moved piece gives the whole plain surface, kept through a rebuild",
      r == {'FINISHED'} and len(_prev) == 1 and abs(area(_sp) - 16) < 1e-6 and store.is_natural(_sp),
      f"{area(_sp):.4f} natural={store.is_natural(_sp)}")
_sp.location = (0, 0, 0); ctx.view_layer.update()
# curve splits a surface
cf = make_floor("SpC", loc=(0, 90, 0))
bpy.ops.curve.primitive_bezier_circle_add(radius=0.5, location=(1, 90, 1)); scc = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); cf.select_set(True); scc.select_set(True)
r = bpy.ops.nproto.split()
cp = pieces("SpC")
check("split: curve splits a surface", r == {'FINISHED'} and len(cp) == 2 and abs(min(area(o) for o in cp) - math.pi / 4) < 2e-3,
      f"{[round(area(o), 4) for o in cp]}")
# vase + tube split each other (seam aware): vase -> remainder + 2 holes, tube -> inside + outside parts
for o in (vase, tube):
    bpy.ops.object.select_all(action='DESELECT'); o.select_set(True); bpy.ops.nproto.untrim()
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); tube.select_set(True)
r = bpy.ops.nproto.split()
vp_, tp_ = pieces("VaseT"), pieces("TubeT")
check("split: vase and tube split each other", r == {'FINISHED'} and len(vp_) == 3 and len(tp_) >= 2
      and abs(sum(store.get_brep(o).area(1e-7) for o in vp_) - vase_area) < 2e-3,
      f"vase pieces={[round(area(o), 3) for o in vp_]} tube pieces={[round(area(o), 3) for o in tp_]}")

# cutters that stop short are extended until they cut across (floors: x -50..-46, y -2..2 around y0)
def _split_sel(*objs, **kw):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = objs[0]
    try:
        return bpy.ops.nproto.split(**kw)
    except RuntimeError:
        return {'CANCELLED'}
_xf = make_floor("SxA", loc=(-50, 200, 0)); _xl = poly("SxA_cut", [(-48, 197, 1), (-48, 201, 1)])
r0 = _split_sel(_xf, _xl, extend=False)
r = _split_sel(_xf, _xl)
check("split: a line that stops inside the surface is extended to cut across (8 + 8); without Extend nothing splits",
      r0 == {'CANCELLED'} and r == {'FINISHED'} and sorted(round(area(o), 4) for o in pieces("SxA")) == [8.0, 8.0],
      f"{r0} {r} {[round(area(o), 4) for o in pieces('SxA')]}")
_xf = make_floor("SxB", loc=(-50, 210, 0)); _xl = poly("SxB_cut", [(-49, 209.5, 1), (-49, 210.5, 1)])
r = _split_sel(_xf, _xl)
check("split: a line inside the surface is extended at both ends (4 + 12)",
      r == {'FINISHED'} and sorted(round(area(o), 4) for o in pieces("SxB")) == [4.0, 12.0],
      f"{[round(area(o), 4) for o in pieces('SxB')]}")
_xf = make_floor("SxC", loc=(-50, 220, 0)); _xwl = poly("QwC_ln", [(-48, 217, -1), (-48, 221, -1)])
bpy.ops.nproto.extrude(inputs=ins((_xwl, "S0")), distance=3.0, direction="Z"); _xw = ctx.active_object; _xw.name = "QwC"
r = _split_sel(_xf, _xw)
check("split: a wall and a floor that each stop inside the other both get extended (floor 8 + 8, wall 4 + 8)",
      r == {'FINISHED'} and sorted(round(area(o), 4) for o in pieces("SxC")) == [8.0, 8.0]
      and sorted(round(area(o), 4) for o in pieces("QwC")) == [4.0, 8.0],
      f"{[round(area(o), 4) for o in pieces('SxC')]} {[round(area(o), 4) for o in pieces('QwC')]}")
# an edge picked in Edge mode cuts like a curve; its surface is not split
from blender_nurbs import subsel as _ss
_xf = make_floor("SxD", loc=(-50, 230, 0))
_xpl = poly("QpD_ln", [(-48, 229, 1), (-48, 231, 1)])
bpy.ops.nproto.extrude(inputs=ins((_xpl, "S0")), distance=1.0, direction="X"); _xp = ctx.active_object; _xp.name = "QpD"
_xpe = store.get_brep(_xp).edges()
_xat = _xp.matrix_world.inverted() @ Vector((-48, 230, 1))
_ei = min(range(len(_xpe)), key=lambda i: (Vector(_xpe[i].point_at(sum(_xpe[i].parameter_range()) / 2)) - _xat).length)
_xp.select_set(True); _ss.set_picks(_xp, "EDGE", [_ei])
r = _split_sel(_xf, _xp)
_ss.set_picks(_xp, "EDGE", [])
check("split: a picked surface edge splits like a line (extended, 8 + 8), its own surface stays whole",
      r == {'FINISHED'} and sorted(round(area(o), 4) for o in pieces("SxD")) == [8.0, 8.0]
      and len(pieces("QpD")) == 1 and abs(area(_xp) - 2.0) < 1e-6,
      f"{[round(area(o), 4) for o in pieces('SxD')]} {[o.name for o in pieces('QpD')]}")

# curves lying on the surface cut along themselves, not projected (cylinder r=1, z 0..2 at x=-60)
_xcp = poly("SxE_p", [(1, 0, 0), (1, 0, 2)], loc=(-60, 240, 0))
bpy.ops.nproto.revolve(inputs=ins((_xcp, "S0")), axis="Z", pivot="OBJECT", angle=360); _xc = ctx.active_object
_xc.name = "SxE"
_xc0 = area(_xc)
bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(-60, 240, 0.5)); _xring = ctx.active_object
r = _split_sel(_xc, _xring, view_dir=(0, -1, 0))      # projected along this view it would cut both walls
check("split: a ring lying on a cylinder cuts along itself, not projected through both walls (1/4 + 3/4)",
      r == {'FINISHED'} and len(pieces("SxE")) == 2
      and abs(min(area(o) for o in pieces("SxE")) - 0.25 * _xc0) < 0.01 * _xc0,
      f"{[round(area(o), 4) for o in pieces('SxE')]} of {_xc0:.4f}")
# an exact 120 deg arc on a half cylinder, extended both ways (exactly round) to its straight edges
_xhp = poly("SxF_p", [(1, 0, 0), (1, 0, 2)], loc=(-70, 240, 0))
bpy.ops.nproto.revolve(inputs=ins((_xhp, "S0")), axis="Z", pivot="OBJECT", angle=180); _xh = ctx.active_object
_xh.name = "SxF"
_xh0 = area(_xh)
_cu = bpy.data.curves.new("SxF_arc", 'CURVE'); _cu.dimensions = '3D'
_sp = _cu.splines.new('NURBS'); _sp.points.add(2)
_c30 = math.cos(math.pi / 6)
_hb = store.get_brep(_xh).bounding_box()                # which half the revolve made: the arc goes round its middle
_side = 1.0 if (_hb[0][1] + _hb[1][1]) * 0.5 > 0 else -1.0
for _p, _c in zip(_sp.points, [(_c30, 0.5 * _side, 1, 1), (0, 2 * _side, 1, 0.5), (-_c30, 0.5 * _side, 1, 1)]):
    _p.co = _c
_sp.order_u = 3; _sp.use_endpoint_u = True
_xarc = bpy.data.objects.new("SxF_arc", _cu); _xarc.location = (-70, 240, 0); ctx.collection.objects.link(_xarc)
ctx.view_layer.update()
r = _split_sel(_xh, _xarc, view_dir=(0, -1, 0))
check("split: an arc on a half cylinder that stops short is extended round it to the edges (half + half)",
      r == {'FINISHED'} and len(pieces("SxF")) == 2 and all(abs(area(o) - 0.5 * _xh0) < 2e-3 * _xh0 for o in pieces("SxF")),
      f"{[round(area(o), 4) for o in pieces('SxF')]} of {_xh0:.4f}")
# Trim extends its cutters too, and cuts along a curve on the surface whatever the view
_xt = make_floor("SxG", loc=(-50, 250, 0)); _xtl = poly("SxG_cut", [(-48, 247, 1), (-48, 251, 1)])
r = bpy.ops.nproto.trim(target="SxG", cutters=ins((_xtl, "S0")), ref_point=(-49, 250, 0), normal=(0, 0, 1),
                        view_dir=(0, 0, -1), action="KEEP")
check("trim: a line that stops inside the surface is extended to cut across (16 -> 8)",
      r == {'FINISHED'} and abs(area(_xt) - 8.0) < 1e-4, f"{area(_xt):.4f}")
_xt = make_floor("SxH", loc=(-50, 260, 0)); _xtl = poly("SxH_cut", [(-47, 259, 0), (-47, 260.5, 0)])
r = bpy.ops.nproto.trim(target="SxH", cutters=ins((_xtl, "S0")), ref_point=(-49, 260, 0), normal=(0, 0, 1),
                        view_dir=(1, 0, 0), action="KEEP", projection="SURFACE")
check("trim On Surface: a line lying in the surface cuts along itself (a view along the surface cannot project it), "
      "extended both ways (16 -> 12)", r == {'FINISHED'} and abs(area(_xt) - 12.0) < 1e-4, f"{area(_xt):.4f}")

# split keeps history: the regions follow the split surface and what cut it
_sl = poly("SpH_ln", [(0, 120, 0), (0, 124, 0)])
bpy.ops.nproto.extrude(inputs=ins((_sl, "S0")), distance=4.0, direction="X"); _shf = ctx.active_object; _shf.name = "SpH"
_scl = poly("SpHW_ln", [(2, 119, -1), (2, 125, -1)])
bpy.ops.nproto.extrude(inputs=ins((_scl, "S0")), distance=3.0, direction="Z"); _shw = ctx.active_object; _shw.name = "SpHW"
bpy.ops.object.select_all(action='DESELECT'); _shf.select_set(True); _shw.select_set(True); ctx.view_layer.objects.active = _shf
r = bpy.ops.nproto.split(extend=False)    # (extended, the floor would cut the wall too)
_hp = pieces("SpH_")
_hp_hist = [features.history(o) for o in _hp]
_kept = _shf.hide_get() and not _shw.hide_get()       # (the wall is not split by the floor: it stays, as the cutter)
_a0 = sorted(round(area(o), 4) for o in _hp)
_shw.location.x += 1.0; ctx.view_layer.update()          # move the cutting wall: the regions follow
for o in _hp:
    features.rebuild(o, ctx, force=True)
_a1 = sorted(round(area(o), 4) for o in _hp)
check("split with history: the split surface is kept as a hidden input, every region is its own object with a Split "
      "feature, and moving the cutter re-splits them (8 + 8 -> 12 + 4)",
      r == {'FINISHED'} and len(_hp) == 2 and all(h is not None and h["op"] == "split_region" for h in _hp_hist)
      and _kept and _a0 == [8.0, 8.0] and _a1 == [4.0, 12.0], f"{[o.name for o in _hp]} {_a0} -> {_a1} kept {_kept}")

# --- Join / Unjoin ---------------------------------------------------------
def patch(name, p0, p1, d, dist):
    ln = poly(name + "_ln", [p0, p1])
    bpy.ops.nproto.extrude(inputs=ins((ln, "S0")), distance=dist, direction=d)
    o = ctx.active_object; o.name = name; return o
JA = patch("JA", (0, 100, 0), (0, 104, 0), "X", 2.0)
JB = patch("JB", (2, 100, 0), (2, 104, 0), "X", 2.0)
JC = patch("JC", (2, 104, 0), (4, 104, 0), "Z", 2.0)
bpy.ops.object.select_all(action='DESELECT')
for o in (JA, JB, JC): o.select_set(True)
ctx.view_layer.objects.active = JA
r = bpy.ops.nproto.join()
J = ctx.active_object
jb = store.get_brep(J)
shared = sum(1 for e in jb.edges() if len(e.faces()) == 2)
check("join: 3 touching surfaces -> 1 joined object, 2 stitched edges, sources hidden", r == {'FINISHED'} and jb.face_count() == 3
      and shared == 2 and all(o.hide_get() for o in (JA, JB, JC)) and features.history(J)["op"] == "join",
      f"faces={jb.face_count()} shared={shared} area={jb.area(1e-6):.3f}")
jname = J.name
r = bpy.ops.nproto.restore_inputs('EXEC_DEFAULT')
check("restore inputs (history join): the 3 sources come back", r == {'FINISHED'} and jname not in bpy.data.objects
      and not any(o.hide_get() for o in (JA, JB, JC)))
# history off: the old one-shot behaviour (merge into active, sources removed; unjoin per face)
ctx.scene.nproto_settings.construction_history = False
bpy.ops.object.select_all(action='DESELECT')
for o in (JA, JB, JC): o.select_set(True)
ctx.view_layer.objects.active = JA
r = bpy.ops.nproto.join()
check("join without history merges into the active object", r == {'FINISHED'} and ctx.active_object.name == "JA"
      and "JB" not in bpy.data.objects and store.get_brep(JA).face_count() == 3)
r = bpy.ops.nproto.unjoin()
parts = [o for o in bpy.data.objects if o.name.startswith("JA") and store.is_nurbs(o)]
check("unjoin without history: one object per face", r == {'FINISHED'} and len(parts) == 3
      and abs(sum(store.get_brep(o).area(1e-6) for o in parts) - 20) < 1e-4)
ctx.scene.nproto_settings.construction_history = True
# gaps and tolerance
G1 = patch("G1", (0, 110, 0), (0, 114, 0), "X", 2.0)
G2 = patch("G2", (2.0005, 110, 0), (2.0005, 114, 0), "X", 2.0)
bpy.ops.object.select_all(action='DESELECT'); G1.select_set(True); G2.select_set(True); ctx.view_layer.objects.active = G1
r = bpy.ops.nproto.join()
check("join: 0.0005 gap stitched at default tolerance", r == {'FINISHED'} and store.get_brep(ctx.active_object).face_count() == 2)
G3 = patch("G3", (0, 120, 0), (0, 124, 0), "X", 2.0)
G4 = patch("G4", (2.01, 120, 0), (2.01, 124, 0), "X", 2.0)
bpy.ops.object.select_all(action='DESELECT'); G3.select_set(True); G4.select_set(True); ctx.view_layer.objects.active = G3
try:
    r = bpy.ops.nproto.join()
except RuntimeError as exc:
    r = str(exc)
check("join: 0.01 gap refused at default tolerance", r != {'FINISHED'} and "G4" in bpy.data.objects, str(r)[:80])
r = bpy.ops.nproto.join(tolerance=0.02)
check("join: 0.01 gap stitched with tolerance 0.02", r == {'FINISHED'} and store.get_brep(ctx.active_object).face_count() == 2)
# closed solid: tube + two planar caps
bpy.ops.curve.primitive_nurbs_circle_add(radius=0.5, location=(0, 130, 0)); cb = ctx.active_object
bpy.ops.nproto.extrude(inputs=ins((cb, "S0")), distance=2.0, direction="Z"); tube2 = ctx.active_object
bpy.ops.nproto.planar(inputs=ins((cb, "S0"))); cap1 = ctx.active_object
bpy.ops.curve.primitive_nurbs_circle_add(radius=0.5, location=(0, 130, 2)); ct = ctx.active_object
bpy.ops.nproto.planar(inputs=ins((ct, "S0"))); cap2 = ctx.active_object
bpy.ops.object.select_all(action='DESELECT')
for o in (cap1, cap2, tube2): o.select_set(True)
ctx.view_layer.objects.active = tube2
r = bpy.ops.nproto.join()
tube2 = ctx.active_object
sb = store.get_brep(tube2)
vol = sb.volume(1e-6) if sb.is_manifold_solid() else 0.0
check("join: tube + 2 caps = closed solid", r == {'FINISHED'} and sb.is_manifold_solid() and abs(abs(vol) - math.pi * 0.25 * 2) < 1e-3,
      f"solid={sb.is_manifold_solid()} volume={vol:.5f} (pi*r^2*h={math.pi*0.25*2:.5f})")
# a trimmed upright cylinder and an elbow built G2 into it along the trim arch
# (from a user's boot model): faces that touch tangentially along their edge.
# Join copies the faces in and stitches; intersecting them first (a MERGE
# boolean) failed outright or came back with the cylinder's face destroyed
import gzip
with gzip.open(os.path.join(ROOT, "tests", "data", "join_tangent_elbow.json.gz"), "rt") as _f:
    _tg = json.load(_f)
for _rev in (False, True):           # either one active (receives the result)
    _perm = [store.new_object_text(ctx, "TG_" + k, t) for k, t in _tg.items()]
    _perm = _perm[::-1] if _rev else _perm
    _tga = sorted(round(store.get_brep(o).faces()[0].area(), 4) for o in _perm)
    bpy.ops.object.select_all(action='DESELECT')
    for o in _perm:
        o.select_set(True)
    ctx.view_layer.objects.active = _perm[-1]
    r = bpy.ops.nproto.join()
    _tb = store.get_brep(ctx.active_object)
    _sh = sum(1 for e in _tb.edges() if len(e.faces()) == 2)
    check("join: tangent elbow + trimmed cylinder keep both faces whole, arch stitched",
          r == {'FINISHED'} and _tb.face_count() == 2 and sorted(round(f.area(), 4) for f in _tb.faces()) == _tga
          and _sh >= 4, f"faces={_tb.face_count()} shared={_sh} areas={[round(f.area(), 4) for f in _tb.faces()]} vs {_tga}")
    bpy.data.objects.remove(ctx.active_object)

# --- Edge fillet -----------------------------------------------------------
def make_L(tag, loc=(0, 0, 0)):
    a = patch(tag + "a", (0, 140, 0), (0, 144, 0), "X", 2.0)
    b = patch(tag + "b", (2, 140, 0), (2, 144, 0), "Z", 2.0)
    bpy.ops.object.select_all(action='DESELECT'); a.select_set(True); b.select_set(True); ctx.view_layer.objects.active = a
    bpy.ops.nproto.join()
    o = ctx.active_object; o.name = tag
    o.location = loc; ctx.view_layer.update()
    return o
def fillet_all(o, **kw):
    return bpy.ops.nproto.edge_fillet(inputs=json.dumps([[o.name, f"E{i}"] for i in store.shared_edges(o)]), **kw)
def A(o): return store.get_brep(o).area(1e-7)
L1 = make_L("EF1")
check("edge fillet: L has one shared edge", len(store.shared_edges(L1)) == 1)
r = fillet_all(L1, shape="CIRCULAR", radius=0.3)
check("edge fillet circular", r == {'FINISHED'} and abs(A(L1) - (16 - 2.4 + 4 * math.pi * 0.3 / 2)) < 1e-3,
      f"{A(L1):.4f} vs {16 - 2.4 + 4*math.pi*0.3/2:.4f}")
L2 = make_L("EF2")
r = fillet_all(L2, shape="CHAMFER", radius=0.3)
check("edge chamfer", r == {'FINISHED'} and abs(A(L2) - (16 - 2.4 + 4 * 0.3 * math.sqrt(2))) < 1e-3, f"{A(L2):.4f}")
L3 = make_L("EF3")
r = fillet_all(L3, shape="BLEND", radius=0.3, continuity="2")
check("edge blend G2", r == {'FINISHED'} and A(L3) > 16 - 2.4 + 4 * math.pi * 0.3 / 2, f"{A(L3):.4f} (> circular)")
L4 = make_L("EF4")
r = fillet_all(L4, shape="CIRCULAR", radius=0.1, variable=True, end_radius=0.4)
check("edge fillet variable radius", r == {'FINISHED'} and store.get_brep(L4).face_count() == 3, f"area {A(L4):.4f}")
L5 = make_L("EF5", loc=(5, -3, 2))
r = fillet_all(L5, shape="CIRCULAR", radius=0.3)
check("edge fillet on a moved object", r == {'FINISHED'} and abs(A(L5) - (16 - 2.4 + 4 * math.pi * 0.3 / 2)) < 1e-3)
# solid: fillet only the top cap edge of the healed cylinder (tube2 from the join tests)
sb = store.get_brep(tube2)
v0 = abs(sb.volume(1e-7))
M = geom.world_matrix(tube2)
top_i = [i for i in store.shared_edges(tube2) if abs((M @ Vector(store.display(tube2)["edges"][i][20]))[2] - (M @ Vector((0, 0, 2)))[2]) < 1e-4]
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[tube2.name, f"E{i}"] for i in top_i]), shape="CIRCULAR", radius=0.1)
sb = store.get_brep(tube2)
check("edge fillet on a solid keeps it solid", r == {'FINISHED'} and len(top_i) == 1 and sb.is_manifold_solid()
      and 0 < v0 - abs(sb.volume(1e-7)) < 0.02, f"volume {v0:.5f} -> {abs(sb.volume(1e-7)):.5f}")
# a non-shared edge is refused cleanly
open_i = [i for i in range(len(store.display(L1)["edges"])) if i not in store.shared_edges(L1)][0]
try:
    r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[L1.name, f"E{open_i}"]]), radius=0.1)
except RuntimeError as exc:
    r = str(exc)
check("edge fillet refuses a boundary edge", r != {'FINISHED'}, str(r)[:80])

# --- CV editing -------------------------------------------------------------
import bmesh
from blender_nurbs import cvs as ncvs
from blender_nurbs import history as nhist
cvc = nurbs("CVcurve", [(0, 150, 0), (1, 150.5, 0), (2, 149.5, 0), (3, 150, 0)])
bpy.ops.nproto.extrude(inputs=ins((cvc, "S0")), distance=2.0, direction="Z"); cs = ctx.active_object; cs.name = "CVSurf"
# a G0 loft from the surface's top edge to a curve (has history -> must follow the CV edit)
topc = nurbs("CVtop", [(0, 151, 4), (1, 151, 4), (2, 151, 4), (3, 151, 4)])
s_cs = store.get_surfaces(cs)[0]
top_edge = max(("U0", "U1"), key=lambda w: geom.boundary_points(s_cs, w, 4)[0][2])
bpy.ops.nproto.loft(inputs=ins((cs, top_edge), (topc, "S0")))
lofted = ctx.active_object
area_before = store.get_brep(cs).area(1e-7)
bpy.ops.object.select_all(action='DESELECT'); cs.select_set(True); ctx.view_layer.objects.active = cs
r = bpy.ops.nproto.edit_cvs()
cage = ctx.active_object
d0 = ext.surface_cvs(store.get_surfaces(cs)[0])
check("edit CVs opens a cage in edit mode", r == {'FINISHED'} and ncvs.is_cage(cage) and cage.mode == 'EDIT'
      and len(cage.data.vertices) == d0["count_u"] * d0["count_v"], f"{d0['count_u']}x{d0['count_v']} CVs, mode={cage.mode}")
from blender_nurbs import diagnostics as _ndiag
_vw = ctx.window_manager.nproto_view; _so = _vw.diag_selected_only; _vw.diag_selected_only = True
check("Selected Only analysis keeps the surface whose CVs are being edited",
      not cs.select_get() and cs in _ndiag.analysis_objects(ctx, _vw) and lofted not in _ndiag.analysis_objects(ctx, _vw))
_vw.diag_selected_only = _so
# move the last CV (a surface corner on the top edge) up by 0.5 like a G-grab would
bm = bmesh.from_edit_mesh(cage.data); bm.verts.ensure_lookup_table()
moved_idx = d0["count_u"] * d0["count_v"] - 1
bm.verts[moved_idx].co.z += 0.5
bmesh.update_edit_mesh(cage.data)
ctx.view_layer.update(); ncvs._process()
new_corner = ext.surface_cvs(store.get_surfaces(cs)[0])["points"][moved_idx]
s_new = store.get_surfaces(cs)[0]
(u0, u1), (v0, v1) = s_new.u_domain(), s_new.v_domain()
corner = s_new.evaluate(u1, v1)
check("moving a cage vertex moves the CV and the surface corner live",
      abs(new_corner[2] - (d0["points"][moved_idx][2] + 0.5)) < 1e-9 and np.linalg.norm(np.array(corner) - np.array(new_corner)) < 1e-9,
      f"CV z {d0['points'][moved_idx][2]:.3f} -> {new_corner[2]:.3f}, corner {np.round(corner, 3)}")
loft_rev = lofted.get("nproto_rev")
ctx.view_layer.update(); nhist._process()
check("loft built from the surface follows the CV edit (history)", lofted.get("nproto_rev") != loft_rev)
r = bpy.ops.nproto.cvs_done()
check("Done keeps the edit and removes the cage", r == {'FINISHED'} and ncvs.cage_for(cs) is None
      and abs(store.get_brep(cs).area(1e-7) - area_before) > 1e-4 and store.is_natural(cs))
# Tab out of Edit Mode on the cage = Done
bpy.ops.object.select_all(action='DESELECT'); cs.select_set(True); ctx.view_layer.objects.active = cs
bpy.ops.nproto.edit_cvs(); cage = ctx.active_object; cname = cage.name
ctx.view_layer.update()
bm = bmesh.from_edit_mesh(cage.data); bm.verts.ensure_lookup_table()
bm.verts[0].co.z -= 0.25; bmesh.update_edit_mesh(cage.data)
ctx.view_layer.update(); ncvs._process()
a_tab = store.get_brep(cs).area(1e-7)
bpy.ops.object.mode_set(mode='OBJECT')        # Tab
ctx.view_layer.update()
pending = cname in ncvs._finish
ncvs._finish_cages()
check("Tab out of the CV cage keeps the edit and removes the cage", pending and cname not in bpy.data.objects
      and ncvs.cage_for(cs) is None and ctx.view_layer.objects.active == cs
      and abs(store.get_brep(cs).area(1e-7) - a_tab) < 1e-9, f"pending={pending}")
# Cancel restores
bpy.ops.object.select_all(action='DESELECT'); cs.select_set(True); ctx.view_layer.objects.active = cs
area_mid = store.get_brep(cs).area(1e-7)
bpy.ops.nproto.edit_cvs(); cage = ctx.active_object
bm = bmesh.from_edit_mesh(cage.data); bm.verts.ensure_lookup_table(); bm.verts[0].co.x -= 1.0; bmesh.update_edit_mesh(cage.data)
ctx.view_layer.update(); ncvs._process()
changed = abs(store.get_brep(cs).area(1e-7) - area_mid) > 1e-4
r = bpy.ops.nproto.cvs_cancel()
check("Cancel restores the surface", r == {'FINISHED'} and changed and abs(store.get_brep(cs).area(1e-7) - area_mid) < 1e-7
      and ncvs.cage_for(cs) is None)
# trimmed surfaces: trims stay fixed in UV while the CVs move
def edge_gap(o):
    b = store.get_brep(o); worst = 0.0
    for e in b.edges():
        t0, t1 = e.parameter_range()
        for k in range(9):
            p = tuple(e.point_at(t0 + (t1 - t0) * k / 8))
            worst = max(worst, min(f.closest_point(p)[1] for f in b.faces()))
    return worst
bpy.ops.object.select_all(action='DESELECT'); floor.select_set(True); ctx.view_layer.objects.active = floor
bpy.ops.nproto.trim(target="Floor", cutters=ins((cc, "S0")), ref_point=(1, 0, 0), normal=(0, 0, 1), action="DELETE")
holed_area = store.get_brep(floor).area(1e-7)
r = bpy.ops.nproto.edit_cvs(); cage = ctx.active_object
ok_open = r == {'FINISHED'} and ncvs.is_cage(cage)
bm = bmesh.from_edit_mesh(cage.data); bm.verts.ensure_lookup_table()
bm.verts[len(bm.verts) - 1].co.z += 1.0          # lift one corner: the floor tilts/warps
bmesh.update_edit_mesh(cage.data); ctx.view_layer.update(); ncvs._process()
fb = store.get_brep(floor)
hole_mid = [np.array(e.point_at(sum(e.parameter_range()) / 2)) for e in fb.edges()]
hole_z = max(abs(p[2]) for p in hole_mid if abs(p[0] - 1) < 0.6 and abs(p[1]) < 0.6) if hole_mid else 0
check("trimmed CV edit: hole stays, edges ride on the new surface", ok_open and fb.face_count() == 1 and edge_gap(floor) < 1e-6
      and hole_z > 0.01 and not store.is_natural(floor),
      f"faces={fb.face_count()} gap={edge_gap(floor):.1e} hole edge lifted to z={hole_z:.3f}")
r = bpy.ops.nproto.cvs_done()
edited_untrimmed_ok = store.PROP_UNTRIM in floor
bpy.ops.nproto.untrim()
plain = store.get_brep(floor).area(1e-7)
check("trimmed CV edit: Done, then Untrim gives the EDITED plain surface", r == {'FINISHED'} and edited_untrimmed_ok
      and plain > 16 + 1e-3, f"untrimmed area {plain:.4f} (flat was 16)")
# cancel on a trimmed surface restores it exactly
bpy.ops.nproto.trim(target="Floor", cutters=ins((cc, "S0")), ref_point=(1, 0, 0), normal=(0, 0, 1), action="DELETE")
before = floor[store.PROP_BREP]; before_untrim = floor.get(store.PROP_UNTRIM)
bpy.ops.nproto.edit_cvs(); cage = ctx.active_object
bm = bmesh.from_edit_mesh(cage.data); bm.verts.ensure_lookup_table(); bm.verts[0].co.z -= 2.0
bmesh.update_edit_mesh(cage.data); ctx.view_layer.update(); ncvs._process()
r = bpy.ops.nproto.cvs_cancel()
check("trimmed CV edit: Cancel restores brep and untrim data", r == {'FINISHED'} and floor[store.PROP_BREP] == before
      and floor.get(store.PROP_UNTRIM) == before_untrim and not store.is_natural(floor))
bpy.ops.nproto.untrim()
# vase with a hole across the revolve seam: scale the whole CV net about the axis
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); ctx.view_layer.objects.active = vase
if store.PROP_UNTRIM in vase: bpy.ops.nproto.untrim()
on_seam, _ = store.get_brep(vase).faces()[0].closest_point((61.5, 0.0, 1.5))
r = vtrim("DELETE", seam_c, on_seam, scope="PICKED")
a_before = store.get_brep(vase).area(1e-7)
faces_before = store.get_brep(vase).face_count()
bpy.ops.object.select_all(action='DESELECT'); vase.select_set(True); ctx.view_layer.objects.active = vase
r2 = bpy.ops.nproto.edit_cvs(); cage = ctx.active_object
M = geom.world_matrix(cage); Mi = M.inverted()
bm = bmesh.from_edit_mesh(cage.data)
for v in bm.verts:
    w = M @ v.co
    w.x = 60 + (w.x - 60) * 1.15; w.y = w.y * 1.15
    v.co = Mi @ w
bmesh.update_edit_mesh(cage.data); ctx.view_layer.update(); ncvs._process()
bpy.ops.nproto.cvs_done()
vb = store.get_brep(vase)
check("trimmed CV edit on a revolve (hole across the seam)", r == {'FINISHED'} and r2 == {'FINISHED'} and vb.face_count() == faces_before
      and edge_gap(vase) < 1e-6 and 1.10 < vb.area(1e-7) / a_before < 1.20,
      f"faces {faces_before}, area {a_before:.4f} -> {vb.area(1e-7):.4f} (x{vb.area(1e-7)/a_before:.3f}), gap {edge_gap(vase):.1e}")
bpy.ops.nproto.untrim()

# --- Blend curve ---------------------------------------------------------
def kappa(d):
    c1, c2 = np.array(d["tangent"]), np.array(d["curvature"])
    return np.cross(c1, c2) / (np.linalg.norm(c1) ** 3)
BA = nurbs("BlendA", [(0, 160, 0), (1, 160.8, 0.2), (2, 159.4, 0.4), (3, 160, 0)])
BB = nurbs("BlendB", [(5, 161, 1.0), (6, 162, 0.6), (7, 160.5, 0.9), (8, 161.5, 1.5)])
smk = _kl()
for cont in ("G0", "G1", "G2"):
    bpy.ops.object.select_all(action='DESELECT')
    r = bpy.ops.nproto.blend_curve(inputs=ins((BA, "S0"), (BB, "S0")), end_a=1, end_b=0, cont_a=cont, cont_b=cont)
    bc = ctx.active_object
    kb = features.input_curve(bc, "S0")
    ka_ = features.input_curve(BA, "S0"); kbb = features.input_curve(BB, "S0")
    t0, t1 = kb.parameter_range()
    s0, s1 = smk.evaluate_curve(kb, t0), smk.evaluate_curve(kb, t1)
    ea = smk.evaluate_curve(ka_, ka_.parameter_range()[1]); eb = smk.evaluate_curve(kbb, kbb.parameter_range()[0])
    g0 = np.linalg.norm(np.array(s0["point"]) - ea["point"]) + np.linalg.norm(np.array(s1["point"]) - eb["point"])
    u = lambda v: np.array(v) / np.linalg.norm(v)
    ang_a = math.degrees(math.acos(np.clip(np.dot(u(s0["tangent"]), u(ea["tangent"])), -1, 1)))
    ang_b = math.degrees(math.acos(np.clip(np.dot(u(s1["tangent"]), u(eb["tangent"])), -1, 1)))
    dk = np.linalg.norm(kappa(s0) - kappa(ea)) + np.linalg.norm(kappa(s1) - kappa(eb))
    sp = bc.data.splines[0]
    if cont == "G0":
        okc = g0 < 1e-5 and sp.order_u == 2
    elif cont == "G1":
        okc = g0 < 1e-5 and ang_a < 1e-3 and ang_b < 1e-3 and sp.order_u == 4
    else:
        okc = g0 < 1e-5 and ang_a < 1e-3 and ang_b < 1e-3 and dk < 1e-3 and sp.order_u == 6
    check(f"blend curve {cont}", r == {'FINISHED'} and okc,
          f"order={sp.order_u} gap={g0:.1e} tangent angles=({ang_a:.4f},{ang_b:.4f})deg curvature diff={dk:.1e}")
# weight = tangent length (|B'(0)| = w * L)
bpy.ops.nproto.blend_curve(inputs=ins((BA, "S0"), (BB, "S0")), end_a=1, end_b=0, cont_a="G1", cont_b="G1", w_a=2.0)
bc = ctx.active_object; kb = features.input_curve(bc, "S0")
L = np.linalg.norm(np.array(ka_.end_point()) - np.array(kbb.start_point()))
d0 = np.linalg.norm(smk.evaluate_curve(kb, kb.parameter_range()[0])["tangent"]) / (kb.parameter_range()[1] - kb.parameter_range()[0])
check("blend curve weight sets the tangent length", abs(d0 - 2.0 * L) < 1e-4, f"|B'(0)|={d0:.4f} vs 2*L={2*L:.4f}")
# history: move curve B -> the blend follows
start_before = tuple(bc.data.splines[0].points[-1].co)
BB.location.z += 0.5; ctx.view_layer.update(); history._process()
check("blend curve follows its inputs (history)", tuple(bc.data.splines[0].points[-1].co) != start_before)
# selection-based: the closest pair of ends is used
bpy.ops.object.select_all(action='DESELECT'); BA.select_set(True); BB.select_set(True); ctx.view_layer.objects.active = BB
r = bpy.ops.nproto.blend_curve('INVOKE_DEFAULT')
bc2 = ctx.active_object
p_first = np.array(bc2.matrix_world @ Vector(bc2.data.splines[0].points[0].co[:3]))
check("blend curve from selection picks the nearest ends", r == {'FINISHED'} and np.linalg.norm(p_first - np.array(ka_.end_point())) < 1e-5)
# its CV hull shows outside Edit mode: while selected (a Blend Curve feature), with CVs / Hulls off
from blender_nurbs import draw as _cdrw
_vv = ctx.window_manager.nproto_view; _old_cv = _vv.show_cvs; _vv.show_cvs = False
_h, _c, _hd = _cdrw.curve_cage(bc2)
_shown = _cdrw.curve_cvs_shown(ctx, bc2); bc2.select_set(False); _hid = _cdrw.curve_cvs_shown(ctx, bc2); bc2.select_set(True)
_npts = len(bc2.data.splines[0].points)
BA.select_set(True); _ba_off = _cdrw.curve_cvs_shown(ctx, BA)
BA.select_set(True); _vv.show_cvs = True; _ba_on = _cdrw.curve_cvs_shown(ctx, BA); BA.select_set(False); _vv.show_cvs = _old_cv
_bc2_feat = "nproto_hist" in bc2
check("curve CVs: a selected Blend Curve (a feature) shows its CV hull with CVs / Hulls off only while its F9 panel is up; "
      "curves with CVs / Hulls on",
      _bc2_feat and _shown == _cdrw.curve_preview_active(ctx) and not _hid and len(_c) == _npts and len(_h) == 2 * (_npts - 1) and not _hd and not _ba_off and _ba_on
      and np.allclose(np.array(_c[0]), p_first, atol=1e-6), f"{_shown} {_cdrw.curve_preview_active(ctx)} {_hid} {len(_c)}/{_npts} {_ba_off} {_ba_on}")
bpy.ops.curve.primitive_bezier_circle_add(radius=1.0, location=(0, 0, 30)); _bz = ctx.active_object
_h, _c, _hd = _cdrw.curve_cage(_bz)
check("curve CVs: a Bezier circle's cage is its 4 points with their handles", len(_c) == 4 and len(_hd) == 8 and len(_h) == 16
      and abs(np.linalg.norm(np.array(_c[0]) - (0, 0, 30)) - 1.0) < 1e-6)
bpy.data.objects.remove(_bz)
# a blend curve is a normal curve input: loft from it
r = bpy.ops.nproto.extrude(inputs=ins((bc2, "S0")), distance=1.0, direction="Z")
check("blend curve usable as a tool input (extrude)", r == {'FINISHED'})
_vv.show_cvs = False; bc2.select_set(True)
check("curve CVs: after another tool a selected Blend Curve hides its CVs (CVs / Hulls off)",
      not _cdrw.curve_preview_active(ctx) and not _cdrw.curve_cvs_shown(ctx, bc2))
_vv.show_cvs = _old_cv

# --- Primitives + live Feature editor -------------------------------------
ctx.scene.cursor.location = (0, 200, 0)
def prim(kind, **kw):
    r = bpy.ops.nproto.add_primitive(kind=kind, **kw)
    return r, ctx.active_object
def Ar(o): return store.get_brep(o).area(1e-7)
def Vol(o):
    b = store.get_brep(o); return abs(b.volume(1e-7)) if b.is_manifold_solid() else 0.0
def radii(o, c=(0, 200, 0)):
    P = geom.sample_curve(features.input_curve(o, "S0"), 400)
    d = np.linalg.norm(P - np.array(c), axis=1); return d.min(), d.max()
r, o = prim("CIRCLE", radius=1.5, degree="0"); lo, hi = radii(o)
check("prim circle exact", r == {'FINISHED'} and hi - lo < 1e-6 and abs(lo - 1.5) < 1e-6, f"{lo:.7f}..{hi:.7f}")
r, o = prim("CIRCLE", radius=1.5, degree="3", spans_u=8); lo, hi = radii(o)
check("prim circle degree 3 / 8 spans", r == {'FINISHED'} and hi - lo < 0.02 and o.data.splines[0].order_u == 4, f"{lo:.4f}..{hi:.4f}")
r, o = prim("ARC", radius=2.0, angle=135.0, degree="0"); lo, hi = radii(o)
kc = features.input_curve(o, "S0")
check("prim arc exact 135deg", r == {"FINISHED"} and hi - lo < 1e-5 and abs(kc.length(1e-7) - 2.0 * math.radians(135)) < 1e-5,
      f"length {kc.length(1e-7):.6f} vs {2*math.radians(135):.6f}")
r, o = prim("SPHERE", radius=1.0, degree="0")
check("prim sphere exact", r == {'FINISHED'} and abs(Ar(o) - 4 * math.pi) < 1e-4, f"area {Ar(o):.6f} vs {4*math.pi:.6f}")
sph = o
r, o = prim("SPHERE", radius=1.0, degree="3", spans_u=8, spans_v=4)
check("prim sphere degree 3", r == {'FINISHED'} and abs(Ar(o) - 4 * math.pi) / (4 * math.pi) < 0.01, f"area {Ar(o):.4f}")
r, o = prim("CYLINDER", radius=0.5, height=2.0, degree="0", caps=True)
check("prim cylinder exact + caps = solid", r == {'FINISHED'} and abs(Vol(o) - math.pi * 0.25 * 2) < 1e-4, f"volume {Vol(o):.6f}")
r, o = prim("CONE", radius=1.0, radius2=0.0, height=3.0, degree="0", caps=True)
check("prim cone exact + cap = solid", r == {'FINISHED'} and abs(Vol(o) - math.pi * 3 / 3) < 1e-3, f"volume {Vol(o):.5f} vs {math.pi:.5f}")
r, o = prim("TORUS", radius=2.0, radius2=0.5, degree="0")
check("prim torus exact", r == {'FINISHED'} and abs(Ar(o) - 4 * math.pi ** 2 * 2 * 0.5) < 1e-3, f"area {Ar(o):.5f}")
r, o = prim("BOX", width=2, depth=3, height=4)
check("prim box = solid", r == {'FINISHED'} and abs(Vol(o) - 24) < 1e-6 and store.get_brep(o).face_count() == 6, f"volume {Vol(o):.6f}")
r, o = prim("PLANE", width=3, depth=2, degree="3", spans_u=2, spans_v=2)
check("prim plane degree 3", r == {'FINISHED'} and abs(Ar(o) - 6) < 1e-6)
r, o = prim("DISK", radius=1.0, degree="0")
check("prim disk exact", r == {'FINISHED'} and abs(Ar(o) - math.pi) < 1e-5)
for kind in ("LINE", "RECTANGLE", "POLYGON", "HELIX", "ELLIPSE"):
    r, o = prim(kind)
    check(f"prim {kind.lower()}", r == {'FINISHED'} and o.type == 'CURVE' and features.input_curve(o, "S0") is not None)
# feature editor: change the exact sphere's radius / degree live
sph.nproto_feature.radius = 2.0
check("feature edit: sphere radius 1 -> 2", abs(Ar(sph) - 16 * math.pi) < 1e-3, f"area {Ar(sph):.5f}")
sph.nproto_feature.degree = "3"
ext_ = _kl().blender_ext
check("feature edit: sphere degree exact -> 3", ext_.surface_cvs(store.get_surfaces(sph)[0])["degree_u"] == 3)
# feature editor on a blend curve: weight change rebuilds it and its dependent extrusion
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.blend_curve(inputs=ins((BA, "S0"), (BB, "S0")), end_a=1, end_b=0, cont_a="G2", cont_b="G2")
bcf = ctx.active_object
bpy.ops.nproto.extrude(inputs=ins((bcf, "S0")), distance=1.0, direction="Z"); dep = ctx.active_object
pts0 = [tuple(p.co) for p in bcf.data.splines[0].points]; rev0 = dep.get("nproto_rev")
check("feature panel synced for blend curve", bcf.nproto_feature.cont_a == "G2" and abs(bcf.nproto_feature.w_a - 1.0) < 1e-9)
bcf.nproto_feature.w_a = 2.5
ctx.view_layer.update(); history._process()
check("feature edit: blend weight rebuilds curve and its dependents",
      [tuple(p.co) for p in bcf.data.splines[0].points] != pts0 and dep.get("nproto_rev") != rev0)
bcf.nproto_feature.cont_b = "G1"
check("feature edit: blend continuity G2 -> G1 (degree 5 -> 4)", bcf.data.splines[0].order_u == 5)
# Blend Curve G3: the curves' third derivatives carried on -- curvature, its rate along the curve and the
# torsion match at both ends (also with Curvature moved, and to a rational curve); labels and panel offer G3
def _bc_inv(c_, t_, s_=1.0):
    D_ = geom.curve_derivs(c_, t_, 3)
    a_, b_, d3_ = s_ * D_[1], D_[2], s_ * D_[3]
    ab_ = np.cross(a_, b_)
    k_ = np.linalg.norm(ab_) / np.linalg.norm(a_) ** 3
    dk_ = ((ab_ @ np.cross(a_, d3_)) / np.linalg.norm(ab_) / np.linalg.norm(a_) ** 3
           - 3 * k_ * (a_ @ b_) / (a_ @ a_)) / np.linalg.norm(a_)
    return np.array([k_, dk_, (ab_ @ d3_) / (ab_ @ ab_)])
_g3A = nurbs("BC3_A", [(0, 30, 0), (1, 30.3, 0.1), (2, 29.8, 0.4), (3, 30.5, 0.2), (4, 30.1, 0.9)], order=5)
_g3B = nurbs("BC3_B", [(6, 31, 1), (7, 31.5, 0.3), (8, 30.6, -0.4), (9, 31.2, 0.1)])
for _i_, _w_ in enumerate((1, 1.3, 0.8, 1)):
    _g3B.data.splines[0].points[_i_].co[3] = _w_
_ca3, _cb3 = features.input_curve(_g3A, "S0"), features.input_curve(_g3B, "S0")
_bc3 = []
for _cc in ((0.0, 0.0), (0.7, -0.4)):
    _, _dg, _pt = features.run("blend_curve", [(_g3A, "S0"), (_g3B, "S0")],
                               {"end_a": 1, "end_b": 0, "cont_a": "G3", "cont_b": "G3", "c_a": _cc[0], "c_b": _cc[1]})
    _bl = ext.curve_from_cvs(_dg, [tuple(q_) for q_ in _pt], [0.0] * (_dg + 1) + [1.0] * (_dg + 1), None)
    _ia, _ib = _bc_inv(_ca3, _ca3.parameter_range()[1]), _bc_inv(_cb3, _cb3.parameter_range()[0])
    _bc3.append((_dg, float(np.max(np.abs(_bc_inv(_bl, 0.0) - _ia) / np.abs(_ia))),
                 float(np.max(np.abs(_bc_inv(_bl, 1.0) - _ib) / np.abs(_ib)))))
_, _dg2, _pt2 = features.run("blend_curve", [(_g3A, "S0"), (_g3B, "S0")], {"end_a": 1, "end_b": 0, "cont_a": "G2", "cont_b": "G3"})
_bl2 = ext.curve_from_cvs(_dg2, [tuple(q_) for q_ in _pt2], [0.0] * (_dg2 + 1) + [1.0] * (_dg2 + 1), None)
_g2open = float(np.max(np.abs(_bc_inv(_bl2, 0.0) - _bc_inv(_ca3, _ca3.parameter_range()[1]))[1:]))
r = bpy.ops.nproto.blend_curve(inputs=ins((_g3A, "S0"), (_g3B, "S0")), end_a=1, end_b=0, cont_a="G3", cont_b="G3")
_bco = ctx.active_object
from blender_nurbs import contlabels as _clbc, feature_ui as _fuibc
_clbc._cache.clear()
_fuibc.sync_from_history(_bco)
check("blend curve G3: degree 7; curvature, its rate and the torsion match at both ends (Curvature moved, rational "
      "B too); a G2 end leaves them open; labels and Feature panel offer G3",
      all(d_ == 7 and ea_ < 1e-8 and eb_ < 1e-8 for d_, ea_, eb_ in _bc3) and _dg2 == 6 and _g2open > 1e-3
      and r == {'FINISHED'} and _bco.data.splines[0].order_u == 8
      and [L_["values"] for L_ in _clbc.labels(_bco)] == [["G0", "G1", "G2", "G3"]] * 2
      and _bco.nproto_feature.cont_a == "G3"
      and [i_[0] for i_ in _fuibc._side_cont_items(_bco.nproto_feature, ctx)] == ["G0", "G1", "G2", "G3"],
      f"{_bc3} G2 end {_g2open:.3g}")

# --- History chains: Trim / Split / Surface Fillet / Join / Edge Fillet ---
def settle():
    for _ in range(4):
        ctx.view_layer.update(); history._process()
def A7(o): return store.get_brep(o).area(1e-7)
def hole_center(o):
    b = store.get_brep(o)
    best = None
    for e in b.edges():
        P = np.array([e.point_at(t) for t in np.linspace(*e.parameter_range(), 24)])
        if np.linalg.norm(P[0] - P[-1]) < 1e-6:   # the closed hole edge
            best = P.mean(axis=0)
    return best
# 1) extrude (history) -> trim with a circle (history step)
hl = poly("HTline", [(0, 250, 0), (0, 254, 0)])
bpy.ops.nproto.extrude(inputs=ins((hl, "S0")), distance=4.0, direction="X"); hs = ctx.active_object; hs.name = "HTsurf"
bpy.ops.curve.primitive_bezier_circle_add(radius=0.5, location=(1, 251, 1)); hc = ctx.active_object; hc.name = "HTcut"
r = bpy.ops.nproto.trim(target="HTsurf", cutters=ins((hc, "S0")), ref_point=(1, 251, 0), normal=(0, 0, 1), action="DELETE")
st = features.stack(hs)
check("trim records a history step on a history surface", r == {'FINISHED'} and len(st) == 1 and st[0]["op"] == "trim"
      and features.history(hs)["op"] == "extrude" and abs(A7(hs) - (16 - math.pi / 4)) < 2e-3)
hc.location.x += 1.5; settle()
c = hole_center(hs)
check("moving the cutting curve moves the hole", c is not None and abs(c[0] - 2.5) < 0.05 and abs(A7(hs) - (16 - math.pi / 4)) < 2e-3,
      f"hole centre x={c[0] if c is not None else None}")
hl.location.y += 0.5; settle()
bb = store.get_brep(hs).bounding_box()
check("moving the extrude's curve rebuilds surface + trim", abs(bb[0][1] - 250.5) < 1e-3 and abs(A7(hs) - (16 - math.pi / 4)) < 2e-3,
      f"ymin={bb[0][1]:.3f}")
# remove the trim step -> plain extrusion again
bpy.ops.object.select_all(action='DESELECT'); hs.select_set(True); ctx.view_layer.objects.active = hs
r = bpy.ops.nproto.remove_step(index=0)
check("remove history step", r == {'FINISHED'} and features.PROP_STACK not in hs and abs(A7(hs) - 16) < 1e-6
      and store.is_natural(hs))
# 2) split action keeps history on both pieces
r = bpy.ops.nproto.trim(target="HTsurf", cutters=ins((hc, "S0")), ref_point=(2.5, 251.5, 0), normal=(0, 0, 1), action="SPLIT")
piece = bpy.data.objects.get("HTsurf_split")
check("split action: both pieces carry history", r == {'FINISHED'} and piece is not None and len(features.stack(piece)) == 1
      and features.stack(piece)[0]["params"]["action"] == "DELETE")
hc.location.x -= 1.0; settle()
check("split pieces both follow the cutter", abs(A7(hs) + A7(piece) - 16) < 2e-3 and abs(A7(hs) - math.pi / 4) < 2e-3
      and abs(0.5 * (store.get_brep(hs).bounding_box()[0][0] + store.get_brep(hs).bounding_box()[1][0]) - 1.5) < 0.05,
      f"{A7(hs):.4f} + {A7(piece):.4f}")
bpy.data.objects.remove(piece)
bpy.ops.object.select_all(action='DESELECT'); hs.select_set(True); ctx.view_layer.objects.active = hs
r = bpy.ops.nproto.untrim()
check("untrim removes trim steps", r == {'FINISHED'} and features.PROP_STACK not in hs and abs(A7(hs) - 16) < 1e-6)
# 3) surface fillet with history: fillet surface + base trims follow the wall's curve
fl_ = poly("HFfloor_ln", [(0, 260, 0), (0, 264, 0)])
bpy.ops.nproto.extrude(inputs=ins((fl_, "S0")), distance=4.0, direction="X"); hfl = ctx.active_object; hfl.name = "HFfloor"
wl_ = poly("HFwall_ln", [(2, 260, -1), (2, 264, -1)])
bpy.ops.nproto.extrude(inputs=ins((wl_, "S0")), distance=3.0, direction="Z"); hwl = ctx.active_object; hwl.name = "HFwall"
r = bpy.ops.nproto.fillet(surf_a="HFfloor", surf_b="HFwall", radius=0.3, trim_bases=True)
hf = ctx.active_object
check("surface fillet with history", r == {'FINISHED'} and features.history(hf)["op"] == "surface_fillet"
      and features.stack(hfl)[0]["op"] == "fillet_trim" and features.stack(hwl)[0]["op"] == "fillet_trim"
      and abs(A7(hfl) - 6.8) < 2e-3, f"floor {A7(hfl):.4f}")
wl_.location.x += 0.5; settle()
fb_ = store.get_brep(hf).bounding_box()
check("moving the wall moves the fillet and re-trims the floor", abs(A7(hfl) - (2.2 * 4)) < 2e-3
      and abs(fb_[0][0] - 2.2) < 0.02, f"floor {A7(hfl):.4f} (expect 8.8), fillet xmin {fb_[0][0]:.3f}")
# 4) chain: curves -> extrudes -> join (history) -> edge fillet (step)
ja_ = poly("CHa_ln", [(0, 270, 0), (0, 274, 0)])
bpy.ops.nproto.extrude(inputs=ins((ja_, "S0")), distance=2.0, direction="X"); cha = ctx.active_object
jb_ = poly("CHb_ln", [(2, 270, 0), (2, 274, 0)])
bpy.ops.nproto.extrude(inputs=ins((jb_, "S0")), distance=2.0, direction="Z"); chb = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); cha.select_set(True); chb.select_set(True); ctx.view_layer.objects.active = cha
bpy.ops.nproto.join(); chj = ctx.active_object
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[chj.name, f"E{i}"] for i in store.shared_edges(chj)]), radius=0.3)
a_fil = A7(chj)
check("edge fillet on a history join records a step", r == {'FINISHED'} and features.stack(chj)[0]["op"] == "edge_fillet"
      and abs(a_fil - (16 - 2.4 + 4 * math.pi * 0.3 / 2)) < 1e-3)
# move both source curves together (the joint stays closed): the whole chain follows
ja_.location.y += 1.0; jb_.location.y += 1.0; settle()
bbj = store.get_brep(chj).bounding_box()
check("curve move ripples through extrude -> join -> edge fillet", abs(bbj[0][1] - 271.0) < 1e-3
      and abs(A7(chj) - a_fil) < 1e-3 and store.get_brep(chj).face_count() == 3, f"ymin {bbj[0][1]:.3f}")
chb_rev = chj.get("nproto_rev")
bpy.ops.object.select_all(action='DESELECT')
cha.hide_set(False); chb.hide_set(False)
# lengthen the wall via its feature panel: height 2 -> 3, the joined + filleted result follows
chb.nproto_feature.distance = 3.0; settle()
check("feature edit on a hidden join source updates the filleted join", chj.get("nproto_rev") != chb_rev
      and abs(A7(chj) - (8 + 12 - 2.4 + 4 * math.pi * 0.3 / 2)) < 1e-3, f"area {A7(chj):.4f}")

# 5) editable history steps: radius of the edge fillet step, mute, tree
from blender_nurbs import history_ui
area_r3 = A7(chj)
ctx.view_layer.objects.active = chj
history_ui.sync_step(chj, 0)
chj.nproto_step.radius = 0.5
check("editing a step's radius rebuilds the fillet", abs(A7(chj) - (8 + 12 - 4.0 + 4 * math.pi * 0.5 / 2)) < 1e-3
      and features.stack(chj)[0]["params"]["radius"] == 0.5, f"area {A7(chj):.4f}")
r = bpy.ops.nproto.mute_step(index=0)
check("muting the step removes the fillet (step kept)", r == {'FINISHED'} and store.get_brep(chj).face_count() == 2
      and len(features.stack(chj)) == 1 and abs(A7(chj) - 20) < 1e-3, f"area {A7(chj):.4f}")
bpy.ops.nproto.mute_step(index=0)
check("un-muting re-applies it", store.get_brep(chj).face_count() == 3)
kids = [o.name for o, _ in history_ui._children(chj)]
check("tree: the join lists both sources; the source is used by the join",
      sorted(kids) == sorted([cha.name, chb.name]) and chj in history_ui.dependents(cha, ctx.scene)
      and "Join" in history_ui._summary(chj) and "Edge" in history_ui._summary(chj), f"{kids} {[cha.name, chb.name]} {[d.name for d in history_ui.dependents(cha, ctx.scene)]}")
import importlib
importlib.reload(features)
check("feature builders survive an add-on module reload", "join" in features.BUILDERS and "prim_cylinder" in features.BUILDERS)
cha.hide_set(True)
bpy.ops.nproto.history_focus(name=cha.name, root=chj.name)
check("tree: clicking a hidden input keeps the tree on the join and makes the input editable",
      ctx.view_layer.objects.active == cha and history_ui.tree_root(ctx, cha) == chj and cha.hide_get())
bpy.ops.nproto.history_focus(name=chj.name, root=chj.name)
# Unjoin keeps the current (filleted) shape; Restore Inputs is the separate undo-the-join
a_now = A7(chj)
bpy.ops.object.select_all(action='DESELECT'); chj.select_set(True); ctx.view_layer.objects.active = chj
jn = chj.name
r = bpy.ops.nproto.unjoin()
pieces = [o for o in bpy.data.objects if (o.name == jn or o.name.startswith(jn + "_face")) and store.is_nurbs(o)]
check("unjoin keeps the fillet: one object per face, same total area", r == {'FINISHED'} and len(pieces) == 3
      and abs(sum(A7(o) for o in pieces) - a_now) < 1e-3 and not features.has_history(chj),
      f"{len(pieces)} pieces, {sum(A7(o) for o in pieces):.4f} vs {a_now:.4f}")

# --- Sweep / Pipe / Boundary / curves / Rebuild / Continuity ---------------
from blender_nurbs import sweeps as nsw
def circle_at(name, r, loc, rot=(0, 0, 0)):
    bpy.ops.curve.primitive_nurbs_circle_add(radius=r, location=loc, rotation=rot)
    o = ctx.active_object; o.name = name; return o
def SA(o): return store.get_brep(o).area(1e-7)
Y0 = 300
# 1-rail straight: circle r=0.3 swept 5 along Z -> cylinder
pr = circle_at("SWprof", 0.3, (0, Y0, 0))
rl = poly("SWrail", [(0, Y0, 0), (0, Y0, 5)])
r = bpy.ops.nproto.sweep(inputs=ins((pr, "S0"), (rl, "S0")))
sw = ctx.active_object
check("sweep 1 rail (straight)", r == {'FINISHED'} and abs(SA(sw) - 2 * math.pi * 0.3 * 5) < 2e-3, f"{SA(sw):.5f} vs {2*math.pi*0.3*5:.5f}")
# curved rail: quarter arc radius 3 in the XZ plane, starting at the profile
rc = nurbs("SWarc", [(0, Y0 + 5, 0), (0, Y0 + 5, 1.6568542), (1.3431458, Y0 + 5, 3.0), (3, Y0 + 5, 3)])
rc.data.splines[0].order_u = 3
sp0 = rc.data.splines[0]; sp0.points.add(0)
# exact quarter circle as rational quadratic
sp0.points[0].co = (0, Y0 + 5, 0, 1); sp0.points[1].co = (0, Y0 + 5, 3, 0.70710678); sp0.points[2].co = (3, Y0 + 5, 3, 1)
for i in range(len(sp0.points) - 1, 2, -1):
    pass
rc2 = nurbs("SWarc2", [(0, Y0 + 5, 0), (0, Y0 + 5, 3), (3, Y0 + 5, 3)])
rc2.data.splines[0].order_u = 3
rc2.data.splines[0].points[1].co = (0, Y0 + 5, 3, 0.70710678)
pr2 = circle_at("SWprof2", 0.3, (0, Y0 + 5, 0))
r = bpy.ops.nproto.sweep(inputs=ins((pr2, "S0"), (rc2, "S0")))
sw2 = ctx.active_object
expect = 2 * math.pi * 0.3 * (3 * math.pi / 2)
check("sweep 1 rail (curved, Pappus)", r == {'FINISHED'} and abs(SA(sw2) - expect) / expect < 2e-3, f"{SA(sw2):.4f} vs {expect:.4f}")
# end scale: frustum r 0.3 -> 0.6 over 5
r = bpy.ops.nproto.sweep(inputs=ins((pr, "S0"), (rl, "S0")), scale_end=2.0)
sl = math.sqrt(5 ** 2 + 0.3 ** 2)
check("sweep end scale (frustum)", r == {'FINISHED'} and abs(SA(ctx.active_object) - math.pi * (0.3 + 0.6) * sl) < 3e-3,
      f"{SA(ctx.active_object):.5f} vs {math.pi*(0.3+0.6)*sl:.5f}")
# twist: rectangle profile, 90 deg twist -> the end section is rotated
rect = poly("SWrect", [(-0.5, Y0 + 10 - 0.2, 0), (0.5, Y0 + 10 - 0.2, 0), (0.5, Y0 + 10 + 0.2, 0), (-0.5, Y0 + 10 + 0.2, 0), (-0.5, Y0 + 10 - 0.2, 0)])
rl3 = poly("SWrail3", [(0, Y0 + 10, 0), (0, Y0 + 10, 4)])
r = bpy.ops.nproto.sweep(inputs=ins((rect, "S0"), (rl3, "S0")), twist=90.0)
bt = store.get_brep(ctx.active_object).bounding_box()
check("sweep twist 90", r == {'FINISHED'} and bt[1][1] - bt[0][1] > 0.9, f"y extent {bt[1][1]-bt[0][1]:.3f} (untwisted 0.4)")
# two rails: diverging lines, an arc profile spanning them
r1 = poly("SW2r1", [(0, Y0 + 15, 0), (0, Y0 + 15 + 6, 0)])
r2 = poly("SW2r2", [(2, Y0 + 15, 0), (4, Y0 + 15 + 6, 0)])
arcp = nurbs("SW2prof", [(0, Y0 + 15, 0), (1, Y0 + 15, 1.5), (2, Y0 + 15, 0)]); arcp.data.splines[0].order_u = 3
r = bpy.ops.nproto.sweep(inputs=ins((arcp, "S0"), (r1, "S0"), (r2, "S0")))
s2r = store.get_surfaces(ctx.active_object)[0]
(u0, u1), (v0, v1) = s2r.u_domain(), s2r.v_domain()
end_gap = max(np.linalg.norm(np.array(s2r.evaluate(u1, v)) - np.array(p)) for v, p in ((v0, (0, Y0 + 21, 0)), (v1, (4, Y0 + 21, 0))))
check("sweep 2 rails: edges ride on both rails", r == {'FINISHED'} and end_gap < 1e-4, f"end gap {end_gap:.2e}")
# tangent sweep along the top edge of a curved wall (G1 to the wall)
wc = nurbs("TWcurve", [(0, Y0 + 25, 0), (1.5, Y0 + 26, 0), (3, Y0 + 24.5, 0), (4.5, Y0 + 25.5, 0)])
bpy.ops.nproto.extrude(inputs=ins((wc, "S0")), distance=2.0, direction="Z"); wall2 = ctx.active_object; wall2.name = "TWall"
s_w = store.get_surfaces(wall2)[0]
top = max(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([q[2] for q in geom.boundary_points(s_w, w, 8)]))
p0 = np.array(geom.boundary_points(s_w, top, 2)[0])
prof3 = nurbs("TWprof", [tuple(p0), tuple(p0 + np.array([0.0, 0.6, 0.6])), tuple(p0 + np.array([0.0, 1.4, 0.4]))])
prof3.data.splines[0].order_u = 3
r = bpy.ops.nproto.sweep(inputs=ins((prof3, "S0"), (wall2, top)), tangent=True)
tsw = ctx.active_object
ss_ = store.get_surfaces(tsw)[0]
worst = 0.0
for qpt in geom.boundary_points(s_w, top, 9)[1:-1]:
    na = nsw._normal_near(s_w, np.array(qpt)); nb = nsw._normal_near(ss_, np.array(qpt))
    worst = max(worst, math.degrees(math.acos(min(1.0, abs(float(na @ nb))))))
check("tangent sweep along a surface edge is G1", r == {'FINISHED'} and worst < 0.5, f"max normal angle {worst:.3f} deg")
bpy.ops.object.select_all(action='DESELECT'); wall2.select_set(True); tsw.select_set(True)
r = bpy.ops.nproto.continuity()
rep = json.loads(ctx.scene["nproto_last_continuity"])
check("continuity check on the tangent sweep", r == {'FINISHED'} and rep["gap"] < 1e-3 and rep["angle"] < 1.5, str(rep))
# pipe along the quarter arc
bpy.ops.object.select_all(action='DESELECT'); rc2.select_set(True); ctx.view_layer.objects.active = rc2
r = bpy.ops.nproto.pipe(inputs=ins((rc2, "S0")), radius=0.2)
check("pipe (Pappus)", r == {'FINISHED'} and abs(SA(ctx.active_object) - 2 * math.pi * 0.2 * 3 * math.pi / 2) < 5e-3)
# boundary: 4 lines of a 2x3 rectangle (in any order / direction) -> planar patch
b1 = poly("BD1", [(0, Y0 + 30, 0), (2, Y0 + 30, 0)]); b2 = poly("BD2", [(2, Y0 + 33, 0), (2, Y0 + 30, 0)])
b3 = poly("BD3", [(0, Y0 + 33, 0), (2, Y0 + 33, 0)]); b4 = poly("BD4", [(0, Y0 + 30, 0), (0, Y0 + 33, 0)])
r = bpy.ops.nproto.boundary(inputs=ins((b1, "S0"), (b3, "S0"), (b2, "S0"), (b4, "S0")))
check("boundary surface from 4 curves", r == {'FINISHED'} and abs(SA(ctx.active_object) - 6) < 1e-4, f"{SA(ctx.active_object):.6f}")
# curves from surfaces
ctx.scene.cursor.location = (0, Y0 + 40, 0)
bpy.ops.nproto.add_primitive(kind="SPHERE", radius=1.0, degree="0"); sp_ = ctx.active_object
bpy.ops.nproto.iso_curve(iso_dir="V", param=0.5); iso = ctx.active_object
L_iso = features.input_curve(iso, "S0").length(1e-6)
check("iso curve (sphere equator / meridian)", abs(L_iso - 2 * math.pi) < 0.01 or abs(L_iso - math.pi) < 0.01, f"length {L_iso:.4f}")
# Duplicate Edge / Iso Curve keep the surface's own CVs: an exact cylinder's rim
# comes back as its rational circle (cyclic, the ring's CV count), not a dense fit
ctx.scene.cursor.location = (0, Y0 + 40, 0)
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0, degree="0"); cy_ = ctx.active_object
_cyd = ext.surface_cvs(store.get_surfaces(cy_)[0])
_rims = []
for _ref in ("U0", "U1", "V0", "V1"):
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.nproto.dup_edge(inputs=ins((cy_, _ref)))
    _sp = ctx.active_object.data.splines[0]
    _q = geom.sample_curve(features.input_curve(ctx.active_object, "S0"), 64)
    _rr = np.linalg.norm(_q[:, :2] - [0, Y0 + 40], axis=1)
    if np.ptp(_q[:, 2]) < 1e-9:
        _rims.append((_ref, len(_sp.points), _sp.use_cyclic_u, min(p.co[3] for p in _sp.points), float(np.abs(_rr - 1).max())))
_ring_n = max(_cyd["count_u"], _cyd["count_v"])
check("duplicate edge: exact cylinder rim keeps its rational CVs (cyclic, no refit, on the circle)",
      len(_rims) == 2 and all(n_ == _ring_n - 1 and cyc and w_ < 0.99 and e_ < 1e-6 for _, n_, cyc, w_, e_ in _rims),
      f"{_rims} ring {_ring_n}")
_isos = []
for _dir in ("U", "V"):
    bpy.ops.object.select_all(action='DESELECT'); cy_.select_set(True); ctx.view_layer.objects.active = cy_
    bpy.ops.nproto.iso_curve(iso_dir=_dir, param=0.3)
    _q = geom.sample_curve(features.input_curve(ctx.active_object, "S0"), 64)
    _isos.append((len(ctx.active_object.data.splines[0].points), float(np.ptp(_q[:, 2])),
                  float(np.abs(np.linalg.norm(_q[:, :2] - [0, Y0 + 40], axis=1) - 1).max())))
check("iso curve: exact isoparm (circle with the ring's CVs, line with the profile's), both on the cylinder",
      any(z_ < 1e-9 and n_ == _ring_n - 1 for n_, z_, _ in _isos) and all(e_ < 2e-5 for _, _, e_ in _isos)
      and any(abs(z_ - 2.0) < 1e-6 and n_ <= min(_cyd["count_u"], _cyd["count_v"]) for n_, z_, _ in _isos),
      f"{_isos} grid {_cyd['count_u']}x{_cyd['count_v']}")
ctx.scene.cursor.location = (0, Y0 + 45, 0)
bpy.ops.nproto.add_primitive(kind="PLANE", width=4, depth=4, pl_degree_u=3, pl_degree_v=3, spans_u=1, spans_v=1); pl_ = ctx.active_object
bpy.ops.curve.primitive_bezier_circle_add(radius=1.0, location=(0, Y0 + 45, 3)); pc = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); pc.select_set(True); pl_.select_set(True); ctx.view_layer.objects.active = pl_
r = bpy.ops.nproto.project_curve(direction="Z")
pcs = geom.sample_curve(features.input_curve(ctx.active_object, "S0"), 64)
check("project curve onto a surface", r == {'FINISHED'} and np.abs(pcs[:, 2]).max() < 1e-4
      and abs(np.linalg.norm(pcs[:, :2] - [0, Y0 + 45], axis=1).mean() - 1.0) < 1e-3)
# a shallow view sends the circle's shadow past the plane (kernel SM_ERR 1001): a warning,
# FINISHED with nothing made, so Adjust Last Operation stays open to change the direction
bpy.ops.object.select_all(action='DESELECT'); pc.select_set(True); pl_.select_set(True); ctx.view_layer.objects.active = pl_
_nobj = len(bpy.data.objects)
_miss = [bpy.ops.nproto.project_curve(direction="VIEW", view_dir=(-0.79, 0.41, -0.45)),
         bpy.ops.nproto.project_curve(direction="X")]
check("project curve: a miss along the view or parallel to the plane makes nothing, keeps the redo panel",
      _miss == [{'FINISHED'}] * 2 and len(bpy.data.objects) == _nobj)
# the default direction is the active work plane's normal: World Z with the World active,
# a tilted plane's normal (the circle's centre lands along it) when that plane is active
from blender_nurbs import workplanes as _wpp
_pjd = []
for _tilt in (False, True):
    if _tilt:
        _wpP = _wpp.add_plane(ctx.scene, "ProjTilt", _wpp.frame_from_normal(np.zeros(3), np.array([0.3, 0.0, 1.0])))
        _nP = _wpp.active_frame(ctx.scene)[3]
    bpy.ops.object.select_all(action='DESELECT'); pc.select_set(True); pl_.select_set(True); ctx.view_layer.objects.active = pl_
    _r = bpy.ops.nproto.project_curve()
    _q = geom.sample_curve(features.input_curve(ctx.active_object, "S0"), 64)
    _want = np.array([0.0, Y0 + 45, 0.0]) - (3.0 * _nP / _nP[2] if _tilt else 0.0)
    _pjd.append((_r, float(np.abs(_q[:, 2]).max()), float(np.linalg.norm((_q[:, :2].min(0) + _q[:, :2].max(0)) / 2 - _want[:2]))))
ctx.scene.nproto_workplanes.active = -1
ctx.scene.nproto_workplanes.planes.remove(len(ctx.scene.nproto_workplanes.planes) - 1)
check("project curve: defaults to the work plane normal (World Z, then a tilted plane's)",
      all(r_ == {'FINISHED'} and z_ < 1e-4 and c_ < 1e-3 for r_, z_, c_ in _pjd), f"{_pjd}")
# onto a plane the projection is exact: the curve's own CVs moved along the direction
# (same degree and CV count), cut exactly where it runs over the plane's edge (x = 2);
# a curved surface still gets a fitted curve on it
def _proj(curves, surf, **kw):
    bpy.ops.object.select_all(action='DESELECT')
    for c_ in curves: c_.select_set(True)
    surf.select_set(True); ctx.view_layer.objects.active = surf
    r_ = bpy.ops.nproto.project_curve(**kw)
    return r_, ctx.active_object
_dP = np.array([0.2, -0.1, -1.0])
_src = [(-1, Y0 + 44, 2), (0, Y0 + 46, 2.5), (1, Y0 + 44, 1.5), (1.5, Y0 + 46, 2)]
_pa = nurbs("PrjA", _src)
_rA, _oA = _proj([_pa], pl_, direction="VIEW", view_dir=tuple(_dP))
_want = np.array([np.array(p_) - _dP * p_[2] / _dP[2] for p_ in _src])
_got = np.array([p_.co[:3] for p_ in _oA.data.splines[0].points])
_pb = nurbs("PrjB", [(1, Y0 + 45, 1), (1.7, Y0 + 45.5, 1), (2.4, Y0 + 44.5, 1), (3, Y0 + 45, 1)])
_rB, _oB = _proj([_pb], pl_, direction="Z")
_qB = geom.sample_curve(features.input_curve(_oB, "S0"), 200)
_rC, _oC = _proj([pc], pl_, direction="Z")
_qC = geom.sample_curve(features.input_curve(_oC, "S0"), 200)
_sC = geom.sample_curve(features.input_curve(pc, "S0"), 4000)      # Blender's Bezier circle: not a true circle
def _polar(Q):
    a_ = np.arctan2(Q[:, 1] - (Y0 + 45), Q[:, 0]); o_ = np.argsort(a_)
    return a_[o_], np.linalg.norm(Q[:, :2] - [0, Y0 + 45], axis=1)[o_]
_aS, _rS = _polar(_sC); _aC, _rCC = _polar(_qC)
_eC = float(np.abs(_rCC - np.interp(_aC, _aS, _rS, period=2 * np.pi)).max())
bpy.ops.curve.primitive_bezier_circle_add(radius=0.5, location=(0, Y0 + 40, 3)); _pd = ctx.active_object
_rD, _oD = _proj([_pd], sp_, direction="Z")
_qD = np.vstack([geom.sample_curve(features.input_curve(_oD, f"S{i}"), 100) for i in range(len(_oD.data.splines))])
_rr = np.linalg.norm(_qD - [0, Y0 + 40, 0], axis=1)
check("project curve: exact onto a plane (CVs moved, cut on the edge), fitted onto a sphere",
      [_rA, _rB, _rC, _rD] == [{'FINISHED'}] * 4 and _got.shape == (4, 3) and np.abs(_got - _want).max() < 5e-5
      and len(_oB.data.splines[0].points) == 4 and abs(_qB[:, 0].max() - 2.0) < 1e-6 and np.abs(_qB[:, 2]).max() < 1e-9
      and np.linalg.norm(_qB[np.argmin(_qB[:, 0])] - [1, Y0 + 45, 0]) < 1e-5
      and _eC < 1e-6 and np.abs(_qC[:, 2]).max() < 1e-9
      and np.abs(_rr - 1.0).max() < 1e-3,
      f"A {np.abs(_got - _want).max():.1e}, B {len(_oB.data.splines[0].points)} CVs x {_qB[:, 0].max():.9f}, "
      f"circle off its shadow {_eC:.1e}, sphere {np.abs(_rr - 1.0).max():.1e}")
bpy.ops.object.select_all(action='DESELECT'); sp_.select_set(True)
ctx.scene.cursor.location = (0, Y0 + 40, 0.5)
bpy.ops.nproto.add_primitive(kind="PLANE", width=4, depth=4, pl_degree_u=1, pl_degree_v=1, spans_u=1, spans_v=1); cutp = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); sp_.select_set(True); cutp.select_set(True)
r = bpy.ops.nproto.intersect()
ic = geom.sample_curve(features.input_curve(ctx.active_object, "S0"), 64)
rr = np.linalg.norm(ic[:, :2] - [0, Y0 + 40], axis=1)
check("intersect sphere x plane = circle r=sqrt(0.75)", r == {'FINISHED'} and abs(rr.mean() - math.sqrt(0.75)) < 2e-3
      and np.abs(ic[:, 2] - 0.5).max() < 1e-3, f"r={rr.mean():.5f}")
# rebuild surface: exact sphere -> degree 3, 6x6 spans
bpy.ops.object.select_all(action='DESELECT'); sp_.select_set(True); ctx.view_layer.objects.active = sp_
r = bpy.ops.nproto.rebuild_surface(degree_u=3, degree_v=3, spans_u=8, spans_v=6)
d_ = _kl().blender_ext.surface_cvs(store.get_surfaces(sp_)[0])
check("rebuild surface degree/spans", r == {'FINISHED'} and (d_["degree_u"], d_["count_u"], d_["count_v"]) == (3, 11, 9)
      and abs(SA(sp_) - 4 * math.pi) / (4 * math.pi) < 0.01, f"{d_['count_u']}x{d_['count_v']} CVs, area {SA(sp_):.4f}")
# rebuild a plane with evenly spaced cubic CVs (5x5 spans) to 1 span: CVs evenly at thirds
bpy.ops.nproto.add_primitive(kind="PLANE", width=3, depth=3, pl_degree_u=3, pl_degree_v=3, spans_u=5, spans_v=5); rp_ = ctx.active_object
rp_.location = (0, 0, -40)
bpy.ops.nproto.rebuild_surface(degree_u=3, degree_v=3, spans_u=1, spans_v=1)
d_ = _kl().blender_ext.surface_cvs(store.get_surfaces(rp_)[0])
xs = sorted({round(p[0], 4) for p in d_["points"]}); ys = sorted({round(p[1], 4) for p in d_["points"]})
check("rebuild 5x5-span plane to 1 span: CVs evenly spaced, flat, same size",
      d_["count_u"] == 4 and np.allclose(xs, [-1.5, -0.5, 0.5, 1.5], atol=2e-3) and np.allclose(ys, [-1.5, -0.5, 0.5, 1.5], atol=2e-3)
      and max(abs(p[2]) for p in d_["points"]) < 1e-9 and abs(SA(rp_) - 9) < 1e-6, f"x {xs} y {ys} n {d_['count_u']}x{d_['count_v']} deg {d_['degree_u']} area {SA(rp_)}")
from blender_nurbs import draw as ndraw, cvs as ncvs2
tp = ndraw.topology(rp_)
check("rebuild preview topology: 1x1 spans, 4x4 CVs, 4 isoparms", tp is not None and tp["spans"] == (1, 1)
      and tp["count"] == (4, 4) and len(tp["iso"]) == 4 and len(tp["hull"]) == 8, str(tp and (tp["spans"], tp["count"])))
# Spans: the interior knot lines of every face, clipped to the trims
bpy.ops.nproto.add_primitive(kind="PLANE", width=4, depth=4, pl_degree_u=3, pl_degree_v=3, spans_u=4, spans_v=4); spp = ctx.active_object
spp.name = "SpansPlane"; spp.location = (0, 0, -60); ctx.view_layer.update()
def _span_len(o):
    S = ndraw.span_lines(o).astype(float)
    return float(np.linalg.norm(S[1::2] - S[0::2], axis=1).sum()), S
_l0, _S0 = _span_len(spp)
check("spans: a 4 x 4 span plane shows 3 + 3 lines across it", abs(_l0 - 24.0) < 1e-6, f"length {_l0}")
bpy.ops.curve.primitive_bezier_circle_add(radius=0.5, location=(0, 0, -59)); _hc = ctx.active_object
r = bpy.ops.nproto.trim(target="SpansPlane", cutters=ins((_hc, "S0")), ref_point=(0, 0, -60), normal=(0, 0, 1),
                        view_dir=(0, 0, -1), action="DELETE")
import time as _tm
_t0 = _tm.time(); _l1, _S1 = _span_len(spp); _dt = _tm.time() - _t0
_rad = np.linalg.norm(_S1[:, :2], axis=1)
_near = _rad[_rad < 0.51]
check("spans: a hole clips the lines through it, ends on the trim", r == {'FINISHED'} and abs(_l1 - 22.0) < 2e-3
      and len(_near) >= 4 and np.allclose(_near, 0.5, atol=2e-3) and _rad.min() > 0.498, f"length {_l1:.4f} ends {_near} {_dt * 1000:.0f} ms")
bpy.data.objects.remove(_hc); bpy.data.objects.remove(spp)
bpy.ops.object.select_all(action='DESELECT'); rp_.select_set(True); ctx.view_layer.objects.active = rp_
r = bpy.ops.nproto.tab_edit()
cg = ctx.active_object
check("Tab on a NURBS surface opens its CV cage", r == {"FINISHED"} and ncvs2.is_cage(cg) and cg.mode == "EDIT", f"{r} {cg.name} {cg.mode} {ctx.mode}")
bpy.ops.nproto.cvs_cancel()
# --- Extend surface (exact natural continuation) ---------------------------
bpy.ops.nproto.add_primitive(kind="PLANE", width=3, depth=3, pl_degree_u=3, pl_degree_v=3, spans_u=5, spans_v=5); ex_ = ctx.active_object
ex_.location = (0, 0, -60)
bb0 = store.get_brep(ex_).bounding_box()
xedge = max(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([p[0] for p in geom.boundary_points(store.get_surfaces(ex_)[0], w, 8)]))
r = bpy.ops.nproto.extend(inputs=json.dumps([[ex_.name, xedge]]), distance=1.0)
bb1 = store.get_brep(ex_).bounding_box()
check("extend plane edge by 1: +1 in x, flat, area 12, history step", r == {'FINISHED'} and abs(bb1[1][0] - bb0[1][0] - 1.0) < 1e-4
      and abs(bb1[0][0] - bb0[0][0]) < 1e-9 and abs(bb1[1][2] - bb0[1][2]) < 1e-9 and abs(SA(ex_) - 12) < 1e-4
      and features.stack(ex_)[-1]["op"] == "extend_surface", f"dx {bb1[1][0] - bb0[1][0]:.6f} area {SA(ex_):.5f} {bb0} {bb1} {[e['op'] for e in features.stack(ex_)]}")
from blender_nurbs import history_ui as _hu
_hu.sync_step(ex_, len(features.stack(ex_)) - 1)
ex_.nproto_step.distance = -0.5
bb2 = store.get_brep(ex_).bounding_box()
check("history: distance -0.5 shrinks the edge exactly", abs(bb2[1][0] - bb0[1][0] + 0.5) < 1e-4 and abs(SA(ex_) - 7.5) < 1e-4,
      f"dx {bb2[1][0] - bb0[1][0]:.6f} area {SA(ex_):.5f}")
# exact cylinder: extending across its seam keeps it a perfect circle (rational continuation)
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0, degree="0", spans_u=1, spans_v=1, caps=False); ecy = ctx.active_object
ecy.location = (10, 0, -60)
s0 = store.get_surfaces(ecy)[0]
seam = [w for w in ("U0", "U1", "V0", "V1") if np.ptp([p[2] for p in geom.boundary_points(s0, w, 8)]) > 1.0][0]
a0 = SA(ecy)
r = bpy.ops.nproto.extend(inputs=json.dumps([[ecy.name, seam]]), distance=0.5)
vv = np.array([v.co for v in ecy.data.vertices])
rad = np.linalg.norm(vv[:, :2], axis=1)
check("extend exact cylinder across the seam: stays on radius 1, area grows by 0.5 x 2", r == {'FINISHED'}
      and np.abs(rad - 1).max() < 2e-3 and abs(SA(ecy) - a0 - 1.0) < 1e-3, f"r {rad.min():.5f}..{rad.max():.5f} dA {SA(ecy) - a0:.5f}")

# freeform: the original surface is untouched, the extension continues it smoothly
from blender_nurbs import extend as _ext
_pts = [(i * 0.6, j * 0.6, 0.35 * math.sin(i * 1.1) * math.cos(j * 0.8) + 0.1 * i) for i in range(5) for j in range(5)]
_s = ext.surface_from_cvs(3, 3, _pts, geom.clamped_uniform_knots(5, 3), geom.clamped_uniform_knots(5, 3))
_t0 = store.brep_to_text(ext.brep_from_surface(_s))
_s0 = store.brep_from_text(_t0).faces()[0].surface()
_s1 = store.brep_from_text(_ext.extend_surface_text(_t0, "U1", 0.8)).faces()[0].surface()
# old u in [0,1] maps to new [0, c]; find c from the knots (the interior knots are unchanged in old units)
_d1 = ext.surface_cvs(_s1); _d0 = ext.surface_cvs(_s0)
c = _d1["knots_u"][len(_d1["knots_u"]) - 4 - 1]   # last interior knot of new = 2/3 of old (old units /(b'))
c = c / _d0["knots_u"][len(_d0["knots_u"]) - 4 - 1]
dev = max(np.linalg.norm(np.array(_s0.evaluate(u, v)) - np.array(_s1.evaluate(u * c, v)))
          for u in np.linspace(0, 1, 9) for v in np.linspace(0, 1, 9))
# smooth across the old end: compare second derivatives just inside / outside via finite differences
def _p(u, v): return np.array(_s1.evaluate(u, v))
h = 1e-3; kink = max(np.linalg.norm((_p(c + h, v) - 2 * _p(c, v) + _p(c - h, v))) / h**2 -
                     np.linalg.norm((_p(c, v) - 2 * _p(c - h, v) + _p(c - 2 * h, v))) / h**2 for v in (0.2, 0.5, 0.8))
check("extend freeform: original part unchanged, curvature continuous across the old edge",
      dev < 1e-9 and abs(kink) < 1e-1, f"dev {dev:.2e} d2 jump {kink:.3e} c {c:.4f}")

# trimmed surfaces: the edges still on the surface's own boundary extend, the trim edges are refused
from blender_nurbs import extend as _ext
bpy.ops.nproto.add_primitive(kind="PLANE", width=4, depth=4, pl_degree_u=3, pl_degree_v=3, spans_u=4, spans_v=4)
etr = ctx.active_object; etr.name = "ExTrimmed"; etr.location = (0, 40, -80); ctx.view_layer.update()
_xe = max(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([p[0] for p in geom.boundary_points(store.get_surfaces(etr)[0], w, 8)]))
bpy.ops.curve.primitive_bezier_circle_add(radius=1.0, location=(2, 40, -79)); _nc = ctx.active_object
r = bpy.ops.nproto.trim(target=etr.name, cutters=ins((_nc, "S0")), ref_point=(2, 40, -80), normal=(0, 0, 1),
                        view_dir=(0, 0, -1), action="DELETE")
_a_notch = SA(etr)
_be = _ext.boundary_edges(etr)
_eb = store.get_brep(etr).edges()
_mid = lambda i: np.array(_eb[i].point_at(sum(_eb[i].parameter_range()) / 2))
_runs = sorted([i for i, v in _be.items() if v[0] == _xe], key=lambda i: _mid(i)[1])
_arc = [i for i in range(len(_eb)) if i not in _be]
check("extend trimmed: a notch in the +x edge leaves two boundary pieces there and the arc as trim edges",
      r == {'FINISHED'} and abs(_a_notch - (16 - math.pi / 2)) < 2e-3 and len(_runs) == 2 and len(_arc) >= 1
      and not store.is_natural(etr), f"{r} area {_a_notch:.4f} runs {_runs} trims {_arc} {_be}")
_up = f"E{_runs[-1]}"
check("extend trimmed: the trim edge is refused (why), the boundary piece taken",
      _ext.refusal(etr, f"E{_arc[0]}") == _ext.TRIM_EDGE and _ext.refusal(etr, _up) == ""
      and _ext.resolve(etr, _up)[0] == _xe, f"{_ext.refusal(etr, f'E{_arc[0]}')!r} {_ext.resolve(etr, _up)}")
# the preview band spans only the picked piece (y 1..2) and lies on the result
_x = _plain(_OpEx, inputs=json.dumps([[etr.name, _up]]), distance=1.0)
_ok = _x.drag_setup(ctx, None)
_rows, _path = _x._strip()
_R = np.vstack(_rows)
_sv = etr.copy(); _sv.data = etr.data.copy(); ctx.collection.objects.link(_sv)
for k_ in etr.keys():
    _sv[k_] = etr[k_]
r = bpy.ops.nproto.extend(inputs=json.dumps([[_sv.name, _up]]), distance=1.0)
def _xmax(o):       # local x extent of the edges (the brep's box is the carrier's, a little loose)
    return max(e.point_at(float(t))[0] for e in store.get_brep(o).edges() for t in np.linspace(*e.parameter_range(), 33))
_bb = _xmax(_sv)
_on = max(store.get_brep(_sv).faces()[0].closest_point(tuple(map(float, q_ - np.array([0, 40, -80]))))[1] for q_ in _R[::7])
check("extend trimmed: one boundary piece grows by 1 (area +1), the rest of the trim stays, still trimmed; "
      "the preview band covers just that piece and lies on the result",
      _ok and r == {'FINISHED'} and abs(SA(_sv) - _a_notch - 1.0) < 2e-3 and abs(_bb - 3.0) < 1e-3
      and not store.is_natural(_sv) and abs(_R[:, 1].min() - 41) < 1e-3 and abs(_R[:, 1].max() - 42) < 1e-3
      and _on < 1e-4 and features.stack(_sv)[-1]["params"].get("at") is not None,
      f"{_ok} {r} dA {SA(_sv) - _a_notch:.5f} x {_bb:.4f} band y {_R[:, 1].min():.3f}..{_R[:, 1].max():.3f} on {_on:.1e}")
# history: the step's distance edited to a shrink; the whole face is cut back at x = 1.5
_hu.sync_step(_sv, len(features.stack(_sv)) - 1)
_sv.nproto_step.distance = -0.5
_cut = 16 - 2.0 - (math.pi / 2 - (0.5 * math.sqrt(0.75) + math.asin(0.5)))
check("extend trimmed, history: distance -0.5 cuts the trimmed surface back at x = 1.5 (through the notch)",
      abs(SA(_sv) - _cut) < 3e-3 and abs(_xmax(_sv) - 1.5) < 1e-3
      and store.get_brep(_sv).face_count() == 1, f"area {SA(_sv):.5f} want {_cut:.5f} x {_xmax(_sv):.4f} faces {store.get_brep(_sv).face_count()}")
# Untrim: the trim goes, the same step extends the plain plane's whole edge
_sv.nproto_step.distance = 1.0
bpy.ops.object.select_all(action='DESELECT'); _sv.select_set(True); ctx.view_layer.objects.active = _sv
r = bpy.ops.nproto.untrim()
check("extend trimmed, then Untrim: the plain plane with its whole edge extended (area 20), untrimmed",
      r == {'FINISHED'} and abs(SA(_sv) - 20.0) < 2e-3 and store.is_natural(_sv), f"{r} area {SA(_sv):.5f}")
bpy.data.objects.remove(_sv)
# joined surfaces are refused with their reason
bpy.ops.nproto.add_primitive(kind="PLANE", width=1, depth=1); _ja = ctx.active_object; _ja.location = (10, 40, -80)
bpy.ops.nproto.add_primitive(kind="PLANE", width=1, depth=1); _jb = ctx.active_object; _jb.location = (11, 40, -80)
ctx.view_layer.update()
bpy.ops.object.select_all(action='DESELECT'); _ja.select_set(True); _jb.select_set(True); ctx.view_layer.objects.active = _ja
bpy.ops.nproto.join(); _jj = ctx.active_object
check("extend: an edge of joined surfaces is refused (Unjoin first)",
      store.get_brep(_jj).face_count() == 2 and _ext.refusal(_jj, "E0") == _ext.JOINED, _ext.refusal(_jj, "E0"))
bpy.data.objects.remove(_jj)
# Edge mode names an untrimmed surface's edges E<i>; the pick tools call them U0 / U1 / V0 / V1
from blender_nurbs.ops import natural_side as _nside
bpy.ops.nproto.add_primitive(kind="PLANE", width=2, depth=2); _np_ = ctx.active_object
_sides = [_nside(_np_, f"E{i}") for i in range(store.get_brep(_np_).edge_count())]
_bd = {w: np.array(geom.boundary_points(store.get_surfaces(_np_)[0], w, 5)) for w in ("U0", "U1", "V0", "V1")}
_eb2 = store.get_brep(_np_).edges()
_same = all(min(np.linalg.norm(_bd[w] - np.array(_eb2[i].point_at(sum(_eb2[i].parameter_range()) / 2)), axis=1)) < 1e-6
            for i, w in enumerate(_sides) if w)
check("pre-picked edges: each edge of an untrimmed plane maps to the side it lies on (Edge mode E<i> -> U0..V1); "
      "a trimmed surface's edges keep their E names",
      sorted(_sides) == ["U0", "U1", "V0", "V1"] and _same and _nside(etr, "E0") is None, f"{_sides}")
bpy.data.objects.remove(_np_)
bpy.data.objects.remove(etr); bpy.data.objects.remove(_nc)

# --- Square (G0/G1/G2 four-sided patch) -------------------------------------
from blender_nurbs import square as _sq
def _qF(s1, s2, t1, t2):
    x = (s1 + s2) / 2; y = (t1 + t2) / 2
    return (x, y + 300, 0.2 * s1 * s2 - 0.15 * t1 * t2 + 0.05 * x * y)
def _qpatch(name, x0, x1, y0, y1):
    A = [(x0, x0), (x0, x1), (x1, x1)]; B_ = [(y0, y0), (y0, y1), (y1, y1)]
    srf = ext.surface_from_cvs(2, 2, [_qF(a[0], a[1], b[0], b[1]) for a in A for b in B_], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1])
    return store.new_object(ctx, name, srf)
qL, qR = _qpatch("SQ_L", -2, -1, -1, 1), _qpatch("SQ_R", 1, 2, -1, 1)
qB, qT = _qpatch("SQ_B", -1, 1, -2, -1), _qpatch("SQ_T", -1, 1, 1, 2)
def _near_edge(o, pt):
    sf = store.get_surfaces(o)[0]
    return min(("U0", "U1", "V0", "V1"), key=lambda w: np.linalg.norm(np.mean(geom.boundary_points(sf, w, 9), axis=0) - np.array(pt)))
sq_in = [(qL, _near_edge(qL, (-1, 300, 0))), (qB, _near_edge(qB, (0, 299, 0))), (qR, _near_edge(qR, (1, 300, 0))), (qT, _near_edge(qT, (0, 301, 0)))]
for cont, deg in (("G0", 2), ("G1", 3), ("G2", 5)):   # neighbours' degree 2, raised to what one span needs
    r = bpy.ops.nproto.square(inputs=json.dumps([[o.name, rf] for o, rf in sq_in]), e0=cont, e1=cont, e2=cont, e3=cont)
    sqo = ctx.active_object
    ds = ext.surface_cvs(store.get_surfaces(sqo)[0])
    srf = store.get_surfaces(sqo)[0]
    dev = max(abs(srf.evaluate(u, v)[2] - (lambda x, y: 0.2 * x * x - 0.15 * y * y + 0.05 * x * y)(srf.evaluate(u, v)[0], srf.evaluate(u, v)[1] - 300))
              for u in np.linspace(0, 1, 7) for v in np.linspace(0, 1, 7))
    qi = json.loads(sqo["nproto_square_info"])
    check(f"square {cont} in a curved quadratic hole: exact surface, degree matched to the neighbours", r == {'FINISHED'}
          and ds["degree_u"] == deg and (ds["count_u"], ds["count_v"]) == (deg + 1, deg + 1) and dev < 1e-7 and qi["angle"] < 1e-3,
          f"deg {ds['degree_u']} {ds['count_u']}x{ds['count_v']} dev {dev:.1e} angle {qi['angle']:.2e}")
for cont, deg in (("G1", 3), ("G2", 5)):
    r = bpy.ops.nproto.square(inputs=json.dumps([[o.name, rf] for o, rf in sq_in]), e0=cont, e1=cont, e2=cont, e3=cont,
                              sq_degree_u=str(deg), sq_degree_v=str(deg))
    ds = ext.surface_cvs(store.get_surfaces(ctx.active_object)[0])
    check(f"square {cont} with degree {deg} set: one span, {deg + 1}x{deg + 1}", r == {'FINISHED'} and ds["degree_u"] == deg
          and (ds["count_u"], ds["count_v"]) == (deg + 1, deg + 1), f"{ds['count_u']}x{ds['count_v']}")
# separate U / V degrees, the G0 default, Free sides, older files' single degree
r = bpy.ops.nproto.square(inputs=json.dumps([[o.name, rf] for o, rf in sq_in]), e0="G1", e1="G1", e2="G1", e3="G1",
                          sq_degree_u="3", sq_degree_v="5", sq_spans_u=1, sq_spans_v=1)
ds = ext.surface_cvs(store.get_surfaces(ctx.active_object)[0])
check("square: degree U 3 x V 5", r == {'FINISHED'} and (ds["degree_u"], ds["degree_v"]) == (3, 5)
      and (ds["count_u"], ds["count_v"]) == (4, 6), f"{ds['degree_u']}x{ds['degree_v']} {ds['count_u']}x{ds['count_v']}")
from blender_nurbs import ops_sweep as _ops_sw
check("square: edges start at G0", all(_ops_sw.NPROTO_OT_square.__annotations__[f"e{i}"].keywords["default"] == "G0"
                                       for i in range(4)))
_g = features.run("square", [(o, rf) for o, rf in sq_in], {"e0": "G1", "e1": "G1", "e2": "G1", "e3": "G1",
                                                         "square_degree": 5, "sq_spans_u": 1, "sq_spans_v": 1})
check("square: an older file's single Degree still applies to both directions", tuple(_sq.LAST_INFO["degree"]) == (5, 5))
# Free: the side isn't held to its boundary -- here a wavy curve on top of the hole
from blender_nurbs import curvetrim as _ct, curvejoin as _cj, curvetools as _cvt
wav = nurbs("SQ_Wave", [(-1.0, 301.0, 0.0), (-0.5, 301.0, 0.3), (0.0, 301.0, -0.3), (0.5, 301.0, 0.3), (1.0, 301.0, 0.1)])
_fin = [(qL, sq_in[0][1]), (qB, sq_in[1][1]), (qR, sq_in[2][1]), (wav, "S0")]
gaps = {}
for c3 in ("G0", "FREE"):
    r = bpy.ops.nproto.square(inputs=json.dumps([[o.name, rf] for o, rf in _fin]), e0="G0", e1="G0", e2="G0", e3=c3)
    srf = store.get_surfaces(ctx.active_object)[0]
    _ts = np.linspace(0, 1, 401)
    top = np.array([srf.evaluate(u, 1.0) for u in _ts] + [srf.evaluate(u, 0.0) for u in _ts]
                   + [srf.evaluate(0.0, v) for v in _ts] + [srf.evaluate(1.0, v) for v in _ts])
    W = _ct.sample_spans(_cj.transform_spans(_cj.spline_spans(wav.data.splines[0]), geom.world_matrix(wav)), 200)[0]
    gaps[c3 + "_maxdev"] = float(max(_cvt._seg_dist(W, top).max(), 0)) if r == {'FINISHED'} else None
check("square: a Free side doesn't follow its curve (G0 does)", gaps["G0_maxdev"] is not None and gaps["G0_maxdev"] < 1e-3
      and gaps["FREE_maxdev"] is not None and gaps["FREE_maxdev"] > 0.05, f"{gaps}")
# history: change one side to G0 through the Feature panel
from blender_nurbs import feature_ui as _fui
_fui.sync_from_history(sqo)
sqo.nproto_feature.e0 = "G0"
check("square feature edit rebuilds (edge 1 -> G0)", features.history(sqo)["params"]["e0"] == "G0"
      and store.get_brep(sqo).face_count() == 1)
# clickable continuity labels: one per side; a click steps the side's setting and rebuilds
from blender_nurbs import contlabels as _cl
_lb = _cl.labels(sqo)
check("continuity labels: 4 on a square, surface edges offer Free/G0/G1/G2", len(_lb) == 4
      and all(L["values"] == ["FREE", "G0", "G1", "G2"] for L in _lb), f"{[(L['param'], L['value']) for L in _lb]}")
_before = features.history(sqo)["params"]["e1"]
r = bpy.ops.nproto.cycle_continuity(obj_name=sqo.name, index=1)
_after = features.history(sqo)["params"]["e1"]
r2 = bpy.ops.nproto.cycle_continuity(obj_name=sqo.name, index=1, step=-1)
check("continuity labels: click steps the side and the square rebuilds; step back restores",
      r == {'FINISHED'} and _after == _cl.next_value(_lb[1]["values"], _before, 1) and r2 == {'FINISHED'}
      and features.history(sqo)["params"]["e1"] == _before and store.get_brep(sqo).face_count() == 1,
      f"{_before} -> {_after} -> {features.history(sqo)['params']['e1']}")
# the label's click operator must not be an UNDO operator: Blender's redo of the last operator (the
# feature's Adjust Last Operation panel) run inside one pushes no step for the redone feature, and the
# next click's redo then restores the scene from before an OLDER step of the same name (all later
# work gone: "Input object 'Extrude.001' not found"). The history edit is its own UNDO operator.
_gz_ops = set(_cl.NPROTO_GGT_cont_labels._sync.__code__.co_consts)
_before = features.history(sqo)["params"]["e1"]
r = bpy.ops.nproto.click_continuity(obj_name=sqo.name, index=1)
_after = features.history(sqo)["params"]["e1"]
bpy.ops.nproto.cycle_continuity(obj_name=sqo.name, index=1, step=-1)
check("continuity labels: the click operator has no UNDO flag, the history edit has; the gizmos run the click",
      'UNDO' not in _cl.NPROTO_OT_click_continuity.bl_options and 'UNDO' in _cl.NPROTO_OT_cycle_continuity.bl_options
      and "nproto.click_continuity" in _gz_ops and "nproto.cycle_continuity" not in _gz_ops
      and r == {'FINISHED'} and _after == _cl.next_value(_lb[1]["values"], _before, 1)
      and features.history(sqo)["params"]["e1"] == _before, f"{r} {_before} -> {_after}")
_c1 = nurbs("CL_A", [(0, 500, 0), (1, 500, 0.3), (2, 500, 0)]); _c2 = nurbs("CL_B", [(0, 502, 0), (1, 502, -0.3), (2, 502, 0)])
bpy.ops.nproto.loft(inputs=ins((_c1, "S0"), (_c2, "S0"))); _lo = ctx.active_object
_lf = _cl.labels(_lo)
check("continuity labels: a loft of curves has 2 labels that stay G0", len(_lf) == 2 and all(x["values"] == ["G0"] for x in _lf))
# three-sided: kernel fillet corner replaced by a G2 square
bpy.ops.nproto.add_primitive(kind="BOX", width=2, depth=2, height=2, degree="0"); sbx = ctx.active_object
sbx.location = (0, 320, 0)
_b = store.get_brep(sbx)
def _emid(b, i):
    e = b.edges()[i]; t0, t1 = e.parameter_range(); return np.array(e.point_at(0.5 * (t0 + t1)))
_pick = [i for i in store.shared_edges(sbx) if sum(abs(_emid(_b, i) - 1) < 1e-6) >= 2]
bpy.ops.nproto.edge_fillet(inputs=json.dumps([[sbx.name, f"E{i}"] for i in _pick]), shape="BLEND", radius=0.4, continuity="2")
bpy.ops.object.select_all(action='DESELECT'); sbx.select_set(True); ctx.view_layer.objects.active = sbx
bpy.ops.nproto.unjoin()
_parts = [o for o in bpy.data.objects if o.name.startswith(sbx.name) and store.is_nurbs(o)]
_corner = [o for o in _parts if store.get_brep(o).edge_count() == 3][0]
_cb = store.get_brep(_corner); _cm = [_emid(_cb, i) for i in range(3)]
_ins = []
for o in _parts:
    if o is _corner:
        continue
    bo = store.get_brep(o)
    for i in range(bo.edge_count()):
        if min(np.linalg.norm(_emid(bo, i) - c) for c in _cm) < 1e-6:
            _ins.append([o.name, f"E{i}"])
bpy.data.objects.remove(_corner)
r = bpy.ops.nproto.square(inputs=json.dumps(_ins), e0="G2", e1="G2", e2="G2")
sq3 = ctx.active_object; qi3 = json.loads(sq3.get("nproto_square_info", "{}"))
check("square, 3 sides (fillet corner): degree 5 single span like the blends, G2 within 0.01 deg", r == {'FINISHED'}
      and len(_ins) == 3 and sq3 is not sqo and tuple(qi3.get("cvs", ())) == (6, 6)
      and qi3.get("gap", 1) < 1e-5 and qi3.get("angle", 9) <= 0.01 and qi3.get("curv", 9) <= 0.01,
      f"{r} {len(_ins)} " + str({k: (round(v, 4) if isinstance(v, float) else v) for k, v in qi3.items()}))

# --- Refit to Trim: trimmed 4-sided wall -> untrimmed surface in the boundary --
bpy.ops.nproto.add_primitive(kind="ARC", radius=1.0, angle=120.0, degree="0"); rarc = ctx.active_object
rarc.location = (0, 0, -120)
bpy.ops.nproto.extrude(inputs=ins((rarc, "S0")), distance=4.0, direction="Z"); rwall = ctx.active_object
rwall.name = "RF_Wall"
_wpts = [(x, -3.0, -120 + 3.0 + 0.3 * math.sin(3 * x)) for x in np.linspace(-1.6, 1.6, 41)]
_cu = bpy.data.curves.new("RF_Cut", 'CURVE'); _cu.dimensions = '3D'
_sp = _cu.splines.new('NURBS'); _sp.points.add(len(_wpts[::4]) - 1)
for _p, _c in zip(_sp.points, _wpts[::4]): _p.co = (*_c, 1.0)
_sp.order_u = 4; _sp.use_endpoint_u = True
rcut = bpy.data.objects.new("RF_Cut", _cu); ctx.collection.objects.link(rcut)
_ws = store.get_surfaces(rwall)[0]
_ref = geom.world_matrix(rwall) @ Vector(_ws.evaluate(sum(_ws.u_domain()) / 2, sum(_ws.v_domain()) / 2))
bpy.ops.object.select_all(action='DESELECT')
r = bpy.ops.nproto.trim(action="KEEP", target="RF_Wall", cutters=ins((rcut, "S0")), ref_point=tuple(_ref), projection="Y",
                        normal=(0, 0, 1), view_dir=(0, 1, 0))
a_trim = store.get_brep(rwall).area(1e-7)
check("refit setup: wall trimmed along a wavy curve", r == {'FINISHED'} and not store.is_natural(rwall), f"area {a_trim:.4f}")
bpy.ops.object.select_all(action='DESELECT'); rwall.select_set(True); ctx.view_layer.objects.active = rwall
r = bpy.ops.nproto.refit_trim()
from blender_nurbs import refit as _rf
rs = store.get_surfaces(rwall)[0]
# every refit point lies on the original cylinder wall (radius 1 about the arc centre)
_M = geom.world_matrix(rwall)
_pts = np.array([np.array(_M @ Vector(rs.evaluate(u, v))) for u in np.linspace(0, 1, 9) for v in np.linspace(0, 1, 9)])
_rad = np.linalg.norm(_pts[:, :2] - np.array(rarc.location[:2]), axis=1)
check("refit to trim: untrimmed, area kept, on the original wall, light net", r == {'FINISHED'} and store.is_natural(rwall)
      and store.get_brep(rwall).face_count() == 1 and abs(store.get_brep(rwall).area(1e-7) - a_trim) / a_trim < 1e-3
      and np.abs(_rad - 1).max() < 2e-3 and max(_rf.LAST_INFO["cvs"]) <= 24
      and features.stack(rwall)[-1]["op"] == "refit_trim",
      f"area {store.get_brep(rwall).area(1e-7):.4f} vs {a_trim:.4f}, r {_rad.min():.5f}..{_rad.max():.5f}, {_rf.LAST_INFO}")

# --- Loft G1/G2 from TRIMMED edges, light net ------------------------------
lf = make_floor("LT_Floor", (0, 400, 0))
_c1 = poly("LT_c1", [(1.5, 395, 0), (1.5, 405, 0)])
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.trim(action="KEEP", target="LT_Floor", cutters=ins((_c1, "S0")), ref_point=(0.5, 400, 0), normal=(0, 0, 1),
                    view_dir=(0, 0, -1), projection="Z")
_wl = poly("LT_wl", [(3, 398, 1), (3, 402, 1)])
bpy.ops.nproto.extrude(inputs=ins((_wl, "S0")), distance=2.0, direction="Z"); lw = ctx.active_object; lw.name = "LT_Wall"
_c2 = poly("LT_c2", [(2.0, 395, 1.5), (2.0, 405, 1.5)])
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.trim(action="KEEP", target="LT_Wall", cutters=ins((_c2, "S0")), ref_point=(3, 400, 2.5), normal=(1, 0, 0),
                    view_dir=(1, 0, 0), projection="X")
def _trim_edge(o, test):
    b = store.get_brep(o); M = geom.world_matrix(o)
    for i, e in enumerate(b.edges()):
        t0, t1 = e.parameter_range(); m = M @ Vector(e.point_at(0.5 * (t0 + t1)))
        if test(m):
            return f"E{i}"
ea = _trim_edge(lf, lambda m: abs(m.x - 1.5) < 1e-4)
eb = _trim_edge(lw, lambda m: abs(m.z - 1.5) < 1e-4)
for cont in ("G1", "G2"):
    r = bpy.ops.nproto.loft(inputs=json.dumps([[lf.name, ea], [lw.name, eb]]), cont_a=cont, cont_b=cont)
    lo = ctx.active_object
    dl = ext.surface_cvs(store.get_surfaces(lo)[0])
    sl = store.get_surfaces(lo)[0]
    # tangent to the floor (normal = z) along the floor edge, tangent to the wall (normal = x) along the wall edge
    Ml = np.array(geom.world_matrix(lo))
    def _nrm(u, v):
        _, su, sv = sl.derivatives(u, v); n = np.cross(su, sv); n = Ml[:3, :3] @ n; return n / np.linalg.norm(n)
    (u0, u1), (v0, v1) = sl.u_domain(), sl.v_domain()
    if (Ml @ np.append(sl.evaluate(u0, 0.5 * (v0 + v1)), 1))[2] > 0.5:   # u may run either way (orientation)
        u0, u1 = u1, u0
    angA = max(np.degrees(np.arccos(min(1, abs(_nrm(u0, v) @ np.array([0, 0, 1]))))) for v in np.linspace(v0, v1, 9))
    angB = max(np.degrees(np.arccos(min(1, abs(_nrm(u1, v) @ np.array([1, 0, 0]))))) for v in np.linspace(v0, v1, 9))
    check(f"loft {cont} between two trimmed edges: tangent to both faces, light net", r == {'FINISHED'} and ea and eb
          and angA < 0.01 and angB < 0.01 and dl["count_v"] <= 4,
          f"{ea} {eb} angles {angA:.4f}/{angB:.4f} CVs {dl['count_u']}x{dl['count_v']}")

# a straight (degree-1) edge lofted G2 to a CURVED trimmed edge (a spoon's handle to its bowl):
# cubic along the edge, one smooth face -- not a polyline through the samples split into strips
_sb = [nurbs(f"SB{k}", [(x, -w, 600.0), (x, 0, 600.5 + 0.3 * (k == 1)), (x, w, 600.0)], order=3)
       for k, (x, w) in enumerate(((-2.0, 1.0), (0.0, 1.5), (2.0, 1.0)))]
bpy.ops.nproto.loft(inputs=ins(*[(c_, "S0") for c_ in _sb])); _bowl = ctx.active_object; _bowl.name = "SB_Bowl"
_cut = poly("SB_cut", [(1.2, -3, 605), (1.2, 3, 605)])
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.trim(action="KEEP", target="SB_Bowl", cutters=ins((_cut, "S0")), ref_point=(0, 0, 600.6),
                    normal=(0, 0, 1), view_dir=(0, 0, -1), projection="Z")
_eb = _trim_edge(_bowl, lambda m: abs(m.x - 1.2) < 1e-3)
_hl = poly("SB_hl", [(3, -0.4, 600.4), (3, 0.4, 600.4)])
bpy.ops.nproto.extrude(inputs=ins((_hl, "S0")), distance=3.0, direction="X"); _hd = ctx.active_object
_er = min(("U0", "U1", "V0", "V1"), key=lambda e_: abs(np.mean(np.array(geom.boundary_points(store.get_surfaces(_hd)[0], e_, 5))[:, 0]
                                                               + geom.world_matrix(_hd).translation.x) - 3))
for _smp in (5, 7, 8, 9, 12, 24):
    r = bpy.ops.nproto.loft(inputs=json.dumps([[_hd.name, _er], [_bowl.name, _eb]]), cont_a="G2", cont_b="G2", samples=_smp)
    _sp = ctx.active_object
    _d = ext.surface_cvs(store.get_surfaces(_sp)[0])
    _labs = __import__("blender_nurbs.contlabels", fromlist=["x"]).labels(_sp)
    check(f"loft G2 straight edge -> curved trimmed edge ({_smp} samples): cubic along it, one smooth face, holds",
          r == {'FINISHED'} and _eb and store.get_brep(_sp).face_count() == 1 and _d["degree_v"] >= 3
          and store._crease_pieces(store.get_surfaces(_sp)[0], 1.0) is None
          and all(L_["check"] and L_["check"]["ok"] for L_ in _labs),
          f"{_eb} {_er} faces {store.get_brep(_sp).face_count()} degree {_d['degree_u']}x{_d['degree_v']} "
          f"CVs {_d['count_u']}x{_d['count_v']} {[L_['check'] and L_['check']['text'] for L_ in _labs]}")

_ok_or = True
_s3 = store.get_surfaces(sq3)[0]; _M3 = np.array(geom.world_matrix(sq3))
for _nm, _rf in _ins:
    _sd = _sq.sample_side(bpy.data.objects[_nm], _rf, 9)
    for _i in (2, 4, 6):
        _u, _v = _sq._foot(_s3, np.linalg.solve(_M3[:3, :3], _sd.P[_i] - _M3[:3, 3]))
        _, _su, _sv = _s3.derivatives(_u, _v)
        _n = _M3[:3, :3] @ np.cross(_su, _sv)
        _ok_or &= float(_n @ _sd.Nout[_i]) > 0
check("square normal agrees with the neighbouring faces' outward normals", _ok_or)
# display meshes of neighbouring objects meet without visible slits (fine edge sampling)
import bmesh as _bmesh
from mathutils.bvhtree import BVHTree as _BVH
def _bnd_world(o):
    bm = _bmesh.new(); bm.from_mesh(o.data)
    out = [o.matrix_world @ v.co for v in bm.verts if any(e.is_boundary for e in v.link_edges)]; bm.free(); return out
def _bvh_world(o):
    o.data.calc_loop_triangles()
    return _BVH.FromPolygons([o.matrix_world @ v.co for v in o.data.vertices], [tuple(t.vertices) for t in o.data.loop_triangles])
_nbs = [bpy.data.objects[n] for n, _ in _ins]
_tr = {o.name: _bvh_world(o) for o in _nbs}
_gap = 0.0
for _p in _bnd_world(sq3):
    _gap = max(_gap, min(_tr[o.name].find_nearest(_p)[3] for o in _nbs))
check("display meshes of the square and its neighbours meet (no visible slits)", _gap < 1e-4 * 10, f"max gap {_gap:.2e}")
# re-trim a planar face to a clean curve (exact other edges kept)
rtf = make_floor("RT_Floor", (0, 420, 0))
_c3 = poly("RT_c", [(1.5, 415, 0), (1.5, 425, 0)])
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.trim(action="KEEP", target="RT_Floor", cutters=ins((_c3, "S0")), ref_point=(0.5, 420, 0), normal=(0, 0, 1),
                    view_dir=(0, 0, -1), projection="Z")
_e = _trim_edge(rtf, lambda m: abs(m.x - 1.5) < 1e-4)
_a0 = store.get_brep(rtf).area(1e-7)
# clean curve: degree 5 Bezier bulging 0.1 outwards in +x (end points on the old trim line)
_ys = np.linspace(418, 422, 6)
_cv = [(1.5 + (0.1 * 16 / 5 * (i * (5 - i)) / 6.25 / 4 if 0 < i < 5 else 0.0), y, 0.0) for i, y in enumerate(_ys)]
_kn = [0.0] * 6 + [1.0] * 6
_new = _sq.retrim_planar(rtf, _e, _cv, _kn, 5)
_cc = ext.curve_from_cvs(5, _cv, _kn, None)
_bulge = np.trapezoid([_cc.evaluate(t)[0] - 1.5 for t in np.linspace(0, 1, 401)], [(_cc.evaluate(t)[1]) for t in np.linspace(0, 1, 401)])
check("re-trim planar face to a clean curve: new edge, area grows by the bulge, other edges kept",
      store.get_brep(rtf).face_count() == 1 and _new.startswith("E")
      and abs(store.get_brep(rtf).area(1e-7) - (_a0 + _bulge)) < 1e-4 and store.get_brep(rtf).edge_count() == 4,
      f"{_a0:.4f} + {_bulge:.4f} -> {store.get_brep(rtf).area(1e-7):.4f}, edges {store.get_brep(rtf).edge_count()}")

# --- Diagnostics: curvature comb ---------------------------------------------
from blender_nurbs import diagnostics as _dg
bpy.ops.curve.primitive_nurbs_circle_add(radius=2.0, location=(0, 0, -150)); dc = ctx.active_object
_d = _dg.comb_data(dc, 64)
_P, _K = _d[0]
_kap = np.linalg.norm(_K, axis=1)
_toc = np.array([0, 0, -150]) - _P
_cosang = np.sum(_K * _toc, axis=1) / (_kap * np.linalg.norm(_toc, axis=1))
check("curvature comb: NURBS circle r=2 -> kappa 0.5 everywhere, pointing to the centre",
      np.abs(_kap - 0.5).max() < 1e-4 and _cosang.min() > 0.9999, f"kappa {_kap.min():.6f}..{_kap.max():.6f}")
# an S-curve: the comb changes side at the inflection (curvature vector flips)
_sc = poly("DG_S", [(0, 0, -160), (1, 0, -160)])
_sc.data.splines.clear(); _sp = _sc.data.splines.new('BEZIER'); _sp.bezier_points.add(1)
_sp.bezier_points[0].co = (0, 0, -160); _sp.bezier_points[0].handle_right = (1, 1, -160); _sp.bezier_points[0].handle_left = (-1, -1, -160)
_sp.bezier_points[1].co = (3, 0, -160); _sp.bezier_points[1].handle_left = (2, -1, -160); _sp.bezier_points[1].handle_right = (4, 1, -160)
_P2, _K2 = _dg.comb_data(_sc, 64)[0]
_side = np.sign(np.cross(np.gradient(_P2, axis=0), _K2)[:, 2])
check("curvature comb: S-curve flips side once (inflection)", int(np.sum(np.abs(np.diff(_side[2:-2])) > 0)) == 1)

# zebra uses its own fine tessellation with exact normals
_zp, _zn, _zt = _dg.zebra_mesh(sq3, 2.0, 0.0005)
_ss = store.get_surfaces(sq3)[0]
_errs = []
# the 3-sided patch has one corner collapsed to a point: its normal is undefined there
_corners = [np.array(_ss.evaluate(a, b)) for a in _ss.u_domain() for b in _ss.v_domain()]
_pole = [c for i, c in enumerate(_corners) if any(np.linalg.norm(c - d) < 1e-9 for d in _corners[i + 1:])]
_rad = 0.15 * float(np.linalg.norm(np.ptp(_zp, axis=0)))
for _k_ in range(0, len(_zp), max(1, len(_zp) // 40)):
    if any(np.linalg.norm(_zp[_k_] - c) < _rad for c in _pole):
        continue
    _u, _v = _sq._foot(_ss, _zp[_k_].astype(float))
    _, _su, _sv = _ss.derivatives(_u, _v); _ne = np.cross(_su, _sv); _ne /= np.linalg.norm(_ne)
    _errs.append(np.degrees(np.arccos(min(1, abs(float(_ne @ (_zn[_k_] / np.linalg.norm(_zn[_k_]))))))))
check("zebra tessellation: fine and with exact normals", len(_zt) > 200 and max(_errs) < 0.05,
      f"{len(_zt)} tris, max normal error {max(_errs):.4f} deg")

# --- STEP export -------------------------------------------------------------
_stp = os.path.join(tempfile.gettempdir(), "nproto_test.step")
bpy.ops.object.select_all(action='DESELECT')
for _o in (sbx_parts if False else []): pass
_cyl = [o for o in bpy.data.objects if o.name.startswith("NURBS Cylinder") and store.is_nurbs(o)]
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0, degree="0", caps=True); _sc_ = ctx.active_object
_sc_.location = (0, 0, -200)
bpy.ops.object.select_all(action='DESELECT'); _sc_.select_set(True); lf.select_set(True)
r = bpy.ops.nproto.export_step(filepath=_stp, use_selection=True, unit="MM")
_txt = open(_stp).read()
check("STEP export: valid Part 21 file, one solid + one sheet, the exact cylinder a true cylinder, mm",
      r == {'FINISHED'} and _txt.startswith("ISO-10303-21;") and _txt.rstrip().endswith("END-ISO-10303-21;")
      and _txt.count("MANIFOLD_SOLID_BREP(") == 1 and _txt.count("SHELL_BASED_SURFACE_MODEL(") == 1
      and "CYLINDRICAL_SURFACE(" in _txt and "SI_UNIT(.MILLI.,.METRE.)" in _txt
      and _txt.count("ADVANCED_FACE(") == 3 + store.get_brep(lf).face_count(),
      f"faces {_txt.count('ADVANCED_FACE(')}")
# every #ref used is defined
import re as _re
_defs = set(_re.findall(r"^#(\d+)=", _txt, _re.M)); _uses = set(_re.findall(r"#(\d+)", _txt))
check("STEP export: every referenced entity is defined", _uses <= _defs, str(sorted(_uses - _defs)[:5]))

# --- Diagnostics: curvature map + draft angle --------------------------------
bpy.ops.nproto.add_primitive(kind="SPHERE", radius=2.0, degree="0"); _sph = ctx.active_object; _sph.location = (0, 0, -170)
_P, _N, _T, _V = _dg.face_curvature_mesh(_sph, 8.0, 0.004)
check("curvature map: sphere r=2 -> Gaussian 0.25, mean +0.5 (convex, outward), k1 = k2 at every vertex",
      len(_T) > 50 and np.abs(_V["GAUSS"] - 0.25).max() < 1e-3 and np.abs(_V["MEAN"] - 0.5).max() < 1e-3
      and np.abs(_V["MAX"] - _V["MIN"]).max() < 1e-3,
      f"K {_V['GAUSS'].min():.5f}..{_V['GAUSS'].max():.5f} H {_V['MEAN'].min():.5f}..{_V['MEAN'].max():.5f}")
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0, degree="0", caps=False); _cy2 = ctx.active_object
_cy2.location = (6, 0, -170)
_P, _N, _T, _V = _dg.face_curvature_mesh(_cy2, 8.0, 0.004)
_sg = np.sign(_V["MEAN"].mean())   # open sheet: its face normal decides convex / concave
check("curvature map: cylinder r=1 -> Gaussian 0, mean +-0.5, principal 1 and 0 (consistent sign)",
      np.abs(_V["GAUSS"]).max() < 1e-3 and np.abs(_V["MEAN"] - 0.5 * _sg).max() < 1e-3
      and np.abs((_V["MAX"] if _sg > 0 else -_V["MIN"]) - 1).max() < 1e-3
      and np.abs(_V["MIN"] if _sg > 0 else _V["MAX"]).max() < 1e-3,
      f"H {_V['MEAN'].min():.4f}..{_V['MEAN'].max():.4f} k1 {_V['MAX'].min():.4f}..{_V['MAX'].max():.4f} "
      f"k2 {_V['MIN'].min():.4f}..{_V['MIN'].max():.4f}")
_rngs = {k: _dg.curvature_range([_sph, _cy2], k, 8.0, 0.004) for k in _dg.CURV_KINDS}
check("curvature map: auto range is plain Python floats for every kind (numpy float32 breaks the GPU uniform)",
      all(type(x) is float for r in _rngs.values() for x in r), str({k: [type(x).__name__ for x in r] for k, r in _rngs.items()}))
check("draft angle: vertical wall 0, 5 deg tilt towards pull +5, undercut -5",
      abs(_dg.draft_angle_deg((1, 0, 0), (0, 0, 1))) < 1e-9
      and abs(_dg.draft_angle_deg((math.cos(math.radians(5)), 0, math.sin(math.radians(5))), (0, 0, 1)) - 5) < 1e-9
      and abs(_dg.draft_angle_deg((math.cos(math.radians(5)), 0, -math.sin(math.radians(5))), (0, 0, 1)) + 5) < 1e-9)

# --- Extrude: tilt, vector, draft (Multi-Draft) -------------------------------
_ln = poly("EX_ln", [(0, 0, -190), (2, 0, -190)])
r = bpy.ops.nproto.extrude(inputs=ins((_ln, "S0")), distance=1.0, direction="Z", tilt=30.0, tilt_azimuth=90.0)
_bb = store.get_brep(ctx.active_object).bounding_box()
check("extrude tilted 30 deg (towards +Y): far edge at (0, sin30, cos30)", r == {'FINISHED'}
      and abs(_bb[1][1] - 0.5) < 1e-4 and abs(_bb[1][2] - (-190 + math.cos(math.radians(30)))) < 1e-4, str(_bb))
r = bpy.ops.nproto.extrude(inputs=ins((_ln, "S0")), distance=2.0, direction="VECTOR", dir_vec=(0, 1, 1))
_bb = store.get_brep(ctx.active_object).bounding_box()
check("extrude along a vector (0,1,1): length 2", abs(_bb[1][1] - math.sqrt(2)) < 1e-4)
# draft: closed square 2x2, height 1, 10 deg -> exact mitred trapezoids
_sq_c = bpy.data.curves.new("EX_sq", 'CURVE'); _sq_c.dimensions = '3D'
_sp = _sq_c.splines.new('POLY'); _sp.points.add(3)
for _pt, _c in zip(_sp.points, [(0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)]): _pt.co = (*_c, 1.0)
_sp.use_cyclic_u = True
_sqo = bpy.data.objects.new("EX_sq", _sq_c); ctx.collection.objects.link(_sqo); _sqo.location = (10, 0, -190)
r = bpy.ops.nproto.extrude(inputs=ins((_sqo, "S0")), distance=1.0, direction="Z", draft=10.0)
_dr = ctx.active_object
_t = math.tan(math.radians(10))
_area = 4 * ((2 + (2 + 2 * _t)) / 2) * math.sqrt(1 + _t * _t)
_bb = store.get_brep(_dr).bounding_box()
check("extrude square with 10 deg draft: outward, exact trapezoids (mitred corners)", r == {'FINISHED'}
      and abs(store.get_brep(_dr).area(1e-7) - _area) < 1e-6 and abs((_bb[1][0] - _bb[0][0]) - (2 + 2 * _t)) < 1e-4,
      f"area {store.get_brep(_dr).area(1e-7):.6f} vs {_area:.6f}")
r = bpy.ops.nproto.extrude(inputs=ins((_sqo, "S0")), distance=1.0, direction="Z", draft=10.0, draft_flip=True)
_bb = store.get_brep(ctx.active_object).bounding_box()
check("flip draft: top shrinks inwards", abs((_bb[1][0] - _bb[0][0]) - 2.0) < 1e-4
      and abs(store.get_brep(ctx.active_object).area(1e-7) - 4 * ((2 + (2 - 2 * _t)) / 2) * math.sqrt(1 + _t * _t)) < 1e-6)
# draft on a circle: cone frustum
bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(20, 0, -190)); _ci = ctx.active_object
r = bpy.ops.nproto.extrude(inputs=ins((_ci, "S0")), distance=1.0, direction="Z", draft=10.0)
_r2 = 1 + _t
_fr = math.pi * (1 + _r2) * math.sqrt(1 + _t * _t)
check("extrude circle with 10 deg draft: cone frustum area", r == {'FINISHED'}
      and abs(store.get_brep(ctx.active_object).area(1e-7) - _fr) / _fr < 1e-4,
      f"{store.get_brep(ctx.active_object).area(1e-7):.5f} vs {_fr:.5f}")
r = bpy.ops.nproto.extrude(inputs=ins((_ci, "S0")), distance=1.0, direction="Z", draft=10.0, symmetric=True)
_sy = store.get_surfaces(ctx.active_object)[0]
_M = np.array(geom.world_matrix(ctx.active_object))
_pts = np.array([_M[:3, :3] @ np.array(_sy.evaluate(u, v)) + _M[:3, 3] for u in (0.0, 0.5, 1.0) for v in np.linspace(0, 1, 13)])
_rad = np.linalg.norm(_pts[:, :2] - np.array([20, 0]), axis=1).reshape(3, -1)
check("symmetric draft: both ends open away from the parting circle (r 1 -> 1+tan10 at z -1 and +1)", r == {'FINISHED'}
      and np.abs(_rad[1] - 1).max() < 1e-4 and np.abs(_rad[0] - _r2).max() < 1e-4 and np.abs(_rad[2] - _r2).max() < 1e-4
      and abs(_pts[:, 2].max() - _pts[:, 2].min() - 2) < 1e-6, f"{_rad.min(axis=1)} {_rad.max(axis=1)}")
# a drafted exact circle stays exact: the wall is a true cone on the circle's own 9 rational CVs
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _ec = ctx.active_object
_ec.location = (30, 0, -190); ctx.view_layer.update()
r = bpy.ops.nproto.extrude(inputs=ins((_ec, "S0")), distance=1.0, direction="Z", draft=10.0)
for _sym in (False, True):
    if _sym:
        r = bpy.ops.nproto.extrude(inputs=ins((_ec, "S0")), distance=1.0, direction="Z", draft=10.0, symmetric=True)
    _cs_ = store.get_surfaces(ctx.active_object)[0]
    _cd = ext.surface_cvs(_cs_)
    _M = np.array(geom.world_matrix(ctx.active_object))
    (_u0, _u1), (_v0, _v1) = _cs_.u_domain(), _cs_.v_domain()
    _rows = [np.array([_M[:3, :3] @ np.array(_cs_.evaluate(_u, _v)) + _M[:3, 3] for _v in np.linspace(_v0, _v1, 721)])
             for _u in (_u0, _u1)]
    _rr = [np.linalg.norm(_q[:, :2] - np.array([30, 0]), axis=1) for _q in _rows]
    check(f"drafted circle{' (symmetric)' if _sym else ''}: an exact cone on the circle's own rational CVs "
          "(9 round, radius constant to 1e-6 at both ends, 1 -> 1 + tan 10)", r == {'FINISHED'} and bool(_cd["weights"])
          and _cd["count_v"] == 9 and _cd["count_u"] == (3 if _sym else 2) and max(np.ptp(_x) for _x in _rr) < 1e-6
          and abs(_rr[0].mean() - (1 + _t if _sym else 1.0)) < 1e-6 and abs(_rr[1].mean() - (1 + _t)) < 1e-6,
          f"cvs {_cd['count_u']}x{_cd['count_v']} rational {bool(_cd['weights'])} radii "
          f"{[(round(float(_x.mean()), 7), float(np.ptp(_x))) for _x in _rr]}")
# a drafted rational profile with double knots (C0 basis there) stays G1 at them: the far row's CVs either
# side of such a knot lie on the profile's tangent line (a free fit creased an ellipse's rim and wall by 0.1 deg)
def _kink(_a, _b):
    return math.degrees(math.acos(max(-1.0, min(1.0, _a @ _b / (np.linalg.norm(_a) * np.linalg.norm(_b))))))
_dk = {}
for _i, (_shape, _dr) in enumerate((("circle", -8.0), ("circle", 5.0), ("ellipse", -8.0), ("ellipse", 5.0),
                                    ("rounded rect", -8.0), ("rounded rect", 5.0))):
    if _shape == "rounded rect":
        bpy.ops.nproto.add_primitive(kind="RECTANGLE", width=2.0, depth=1.2, corner=0.3)
    else:
        bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0)
    _ep = ctx.active_object
    if _shape == "ellipse":
        _h = json.loads(_ep[features.PROP_HIST]); _h["params"]["elliptic"] = True; _h["params"]["radius2"] = 0.6
        _ep[features.PROP_HIST] = json.dumps(_h); features.rebuild(_ep, ctx, force=True)
    _ep.location = (50 + 4 * _i, 0, -190); ctx.view_layer.update()
    r = bpy.ops.nproto.extrude(inputs=ins((_ep, "S0")), distance=2.0, direction="Z", draft=_dr)
    _eo = ctx.active_object
    _es = store.get_surfaces(_eo)[0]
    _kv = ext.surface_cvs(_es)["knots_v"]
    _side = max(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([q[2] for q in geom.boundary_points(_es, w, 8)]))
    _kc = features.input_curve(_eo, _side)
    _E = lambda t: np.array(_kc.evaluate(t), float)
    _t0, _t1 = _kc.parameter_range()
    _ks = [(_k - _kv[0]) / (_kv[-1] - _kv[0]) for _k in sorted(set(_kv)) if _kv[0] < _k < _kv[-1]]
    _rim = [_kink(_E(_t0 + (_t1 - _t0) * _s) - _E(_t0 + (_t1 - _t0) * _s - 1e-7),
                  _E(_t0 + (_t1 - _t0) * _s + 1e-7) - _E(_t0 + (_t1 - _t0) * _s)) for _s in _ks]
    _rim.append(_kink(_E(_t1) - _E(_t1 - 1e-7), _E(_t0 + 1e-7) - _E(_t0)))          # the seam
    (_u0, _u1), (_v0, _v1) = _es.u_domain(), _es.v_domain()
    def _nrm(_u, _v):
        _S = lambda a, b: np.array(_es.evaluate(a, b), float)
        return np.cross(_S(min(_u + 1e-6, _u1), _v) - _S(max(_u - 1e-6, _u0), _v), _S(_u, _v + 1e-8) - _S(_u, _v - 1e-8))
    _wall = [_kink(_nrm(_u, _v0 + (_v1 - _v0) * _s - 2e-7), _nrm(_u, _v0 + (_v1 - _v0) * _s + 2e-7))
             for _s in _ks for _u in np.linspace(_u0, _u1, 5)]
    _dk[f"{_shape} {_dr:+g}"] = (r == {'FINISHED'}, round(max(_rim), 5), round(max(_wall), 5))
check("drafted circle, ellipse and rounded rectangle (-8 and +5 deg): top rim and wall G1 at every knot and the "
      "seam (< 0.001 deg)", all(_v[0] and _v[1] < 1e-3 and _v[2] < 1e-3 for _v in _dk.values()), str(_dk))
# a smooth profile with a sharp corner still drafts (sampled, mitred path)
_bzc = bezier("EX_bz", [(40, 0), (42, 0), (42, 2), (40, 2)], -190.0)
_bzc.data.splines[0].use_cyclic_u = True
for _bp in _bzc.data.splines[0].bezier_points:
    _bp.handle_left_type = _bp.handle_right_type = 'VECTOR'
r = bpy.ops.nproto.extrude(inputs=ins((_bzc, "S0")), distance=1.0, direction="Z", draft=10.0)
_bb = store.get_brep(ctx.active_object).bounding_box()
check("drafted smooth profile with sharp corners: still drafts with mitred corners",
      r == {'FINISHED'} and abs((_bb[1][0] - _bb[0][0]) - (2 + 2 * _t)) < 1e-3, str(_bb))

# sharp shading at B-rep creases (cap rim of a capped cylinder)
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0, degree="0", caps=True); _cc = ctx.active_object
_cc.location = (0, 0, -230)
_me = _cc.data; _me.calc_loop_triangles()
_worst = 0.0
for _t in _me.loop_triangles:
    _tn = _t.normal
    for _l in _t.loops:
        _worst = max(_worst, math.degrees(_me.corner_normals[_l].vector.angle(_tn)))
check("capped cylinder: corners at the rim shade with their own face's normal (sharp crease)", _worst < 20, f"max corner-vs-triangle angle {_worst:.1f}")

# no long sliver fans on ruled / revolved sides (they shade as dark bands)
_me = _cc.data; _me.calc_loop_triangles()
_asp = []
for _t in _me.loop_triangles:
    _p = [Vector(_me.vertices[v].co) for v in _t.vertices]
    _L = max((_p[k] - _p[(k + 1) % 3]).length for k in range(3))
    _asp.append(_L * _L / (2 * _t.area + 1e-30))
check("display mesh of a capped cylinder: no sliver fans (95% of triangles aspect < 12)",
      float(np.percentile(_asp, 95)) < 12, f"p95 aspect {np.percentile(_asp, 95):.1f}")

# revolve seam angle: same surface, seam moved around the axis
_prof = poly("RV_prof", [(1.0, 0, 0), (1.3, 0, 1.0), (0.9, 0, 2.0)], loc=(0, 0, -260))
bpy.ops.nproto.revolve(inputs=ins((_prof, "S0")), angle=360.0, axis="Z", pivot="OBJECT", seam_angle=0.0); _r0 = ctx.active_object
bpy.ops.nproto.revolve(inputs=ins((_prof, "S0")), angle=360.0, axis="Z", pivot="OBJECT", seam_angle=120.0); _r1 = ctx.active_object
_s1 = store.get_surfaces(_r1)[0]
_seam = np.array(_s1.evaluate(_s1.u_domain()[0], sum(_s1.v_domain()) / 2)) if True else None
_ang = math.degrees(math.atan2(*(_seam[[1, 0]])))
_d = store.get_surfaces(_r0)[0]
_ar0, _ar1 = store.get_brep(_r0).area(1e-7), store.get_brep(_r1).area(1e-7)
check("revolve seam angle: same area, seam turned to 120 deg", abs(_ar0 - _ar1) < 1e-5 * max(_ar0, 1.0)
      and (abs(_ang - 120) < 0.5 or abs(_ang - 120 + 360) < 0.5 or abs(_ang + 240) < 0.5 or abs(np.degrees(np.arctan2(_seam[1], _seam[0])) - 120) < 0.5),
      f"seam at {np.degrees(np.arctan2(_seam[1], _seam[0])):.1f} deg, areas {_ar0:.9f} / {_ar1:.9f}")

# revolve of a MULTI-SPAN profile covers the whole profile (kernel revolve only did span 1)
_cu = bpy.data.curves.new("RV_ms", 'CURVE'); _cu.dimensions = '3D'
_sp = _cu.splines.new('NURBS'); _pp = [(1.0, 0, 0), (1.4, 0, 0.8), (0.9, 0, 1.6), (1.0, 0, 3.0)]; _sp.points.add(3)
for _q, _c in zip(_sp.points, _pp): _q.co = (*_c, 1.0)
_sp.order_u = 3; _sp.use_endpoint_u = True
_ms = bpy.data.objects.new("RV_ms", _cu); ctx.collection.objects.link(_ms); _ms.location = (0, 0, -300)
bpy.ops.nproto.revolve(inputs=ins((_ms, "S0")), angle=360.0, axis="Z", pivot="OBJECT"); _rv = ctx.active_object
_kc = features.input_curve(_ms, "S0"); _t0, _t1 = _kc.parameter_range()
_Q = np.array([_kc.evaluate(t) for t in np.linspace(_t0, _t1, 4001)]); _Q[:, :2] -= np.array([0, 0])
_r = np.hypot(_Q[:, 0], _Q[:, 1]); _ds = np.linalg.norm(np.diff(_Q, axis=0), axis=1)
_pappus = float(np.sum(2 * np.pi * 0.5 * (_r[1:] + _r[:-1]) * _ds))
_bbr = store.get_brep(_rv).bounding_box()
check("revolve of a 2-span profile: full height and Pappus area", abs(_bbr[1][2] - _bbr[0][2] - 3.0) < 1e-3
      and abs(store.get_brep(_rv).area(1e-7) - _pappus) / _pappus < 1e-4,
      f"height {_bbr[1][2] - _bbr[0][2]:.4f} area {store.get_brep(_rv).area(1e-7):.4f} vs {_pappus:.4f}")
bpy.ops.nproto.revolve(inputs=ins((_ms, "S0")), angle=90.0, axis="Z", pivot="OBJECT"); _rq = ctx.active_object
check("revolve 90 deg: a quarter of the area", abs(store.get_brep(_rq).area(1e-7) - _pappus / 4) / _pappus < 1e-4)

# vase with a cylinder opening across its seam: trim, join, edge fillet the opening
bpy.ops.nproto.revolve(inputs=ins((_ms, "S0")), angle=360.0, axis="Z", pivot="OBJECT"); _va = ctx.active_object; _va.name = "VO_Vase"
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=0.5, height=3.0, degree="0", caps=False); _cy = ctx.active_object; _cy.name = "VO_Cyl"
_cy.rotation_euler = (0, math.radians(90), 0); _cy.location = (0, 0, -298.5); ctx.view_layer.update()
_bbc = store.get_brep(_cy).bounding_box(); _Mc = np.array(_cy.matrix_world)
_xs = [(_Mc[:3, :3] @ np.array([x, y, z]) + _Mc[:3, 3])[0] for x in (_bbc[0][0], _bbc[1][0]) for y in (_bbc[0][1], _bbc[1][1]) for z in (_bbc[0][2], _bbc[1][2])]
_cy.location.x += 0.3 - min(_xs); ctx.view_layer.update()
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.trim(action="DELETE", scope="PICKED", target="VO_Vase", cutters=ins((_cy, "SURF")), ref_point=(0.97, 0, -298.5),
                    normal=(1, 0, 0), view_dir=(-1, 0, 0))
bpy.ops.nproto.trim(action="KEEP", scope="PICKED", target="VO_Cyl", cutters=ins((_va, "SURF")), ref_point=(0.5, 0, -298.0),
                    normal=(0, 0, 1), view_dir=(0, 0, -1))
bpy.ops.object.select_all(action='DESELECT'); _va.select_set(True); _cy.select_set(True); ctx.view_layer.objects.active = _va
bpy.ops.nproto.join(tolerance=1e-3); _vj = ctx.active_object
_a_before = store.get_brep(_vj).area(1e-6); _f_before = store.get_brep(_vj).face_count()
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_vj.name, f"E{i}"] for i in store.shared_edges(_vj)]), shape="CIRCULAR", radius=0.08)
check("vase opening across the seam: trim + join + edge fillet succeed", r == {'FINISHED'} and features.stack(_vj)
      and features.stack(_vj)[-1]["op"] == "edge_fillet" and store.get_brep(_vj).face_count() > _f_before
      and abs(store.get_brep(_vj).area(1e-6) - _a_before) > 1e-3,
      f"faces {_f_before} -> {store.get_brep(_vj).face_count()}")

# tangent chain picking: a rim split into 3 arcs is one chain; a 90 deg edge is not part of it
from blender_nurbs.ops_trim import tangent_chain as _tch
def _arc(a0, a1): return [Vector((math.cos(t), math.sin(t), 0)) for t in np.linspace(a0, a1, 20)]
_cands = [("O", "E0", _arc(0, 2)), ("O", "E1", _arc(2, 4)), ("O", "E2", _arc(4, 2 * math.pi)),
          ("O", "E3", [Vector((1, 0, 0)), Vector((1, 0, 1))])]
_ch = sorted(c[1] for c in _tch(_cands[1], _cands))
check("tangent chain: a rim in 3 pieces is picked together, the 90 deg edge is not", _ch == ["E0", "E1", "E2"], str(_ch))

# revolve + planar cap from its rim: exact rim -> clean join (no slivers / chunks); solid edge fillet;
# surface fillet trims only the strip under the fillet on the closed revolve
_rim = max(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([p[2] for p in geom.boundary_points(store.get_surfaces(_rv)[0], w, 8)]))
r = bpy.ops.nproto.planar(inputs=ins((_rv, _rim))); _cap = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); _rv.select_set(True); _cap.select_set(True); ctx.view_layer.objects.active = _rv
bpy.ops.nproto.join(tolerance=1e-3); _rj = ctx.active_object
_bj = store.get_brep(_rj)
check("join revolve + planar cap: 2 faces, one shared rim edge, no slivers",
      _bj.face_count() == 2 and len(store.shared_edges(_rj)) == 1 and min(f.area() for f in _bj.faces()) > 1e-3,
      f"faces {[round(f.area(), 4) for f in _bj.faces()]} shared {len(store.shared_edges(_rj))}")
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_rj.name, f"E{i}"] for i in store.shared_edges(_rj)]), radius=0.1)
check("edge fillet on the revolve + cap rim succeeds", r == {'FINISHED'} and store.get_brep(_rj).face_count() == 3)
_ident = {"loc": [0, 0, 0], "axis": [0, 0, 1], "angle": 0.0, "scale": [1, 1, 1]}
_a0 = store.get_brep(_rv).area(1e-6)
_info, _br = jobs.run("fillet", {"a": _rv, "b": _cap}, ["fillet", "a", "b"], radius_a=-0.1, radius_b=-0.1, xsect=1,
                      tolerance=1e-4, trim_bases=True)
_ta = store.brep_from_text(_br["a"]).area(1e-6)
check("surface fillet on a closed revolve trims only the strip under the fillet", 0.9 * _a0 < _ta < _a0,
      f"{_a0:.4f} -> {_ta:.4f}")

# surface fillet operator with Auto Side finds the working side even if other sides raise kernel errors
bpy.ops.object.select_all(action='DESELECT')
_rv2 = [o for o in bpy.data.objects if o.name.startswith("Revolve") and store.is_nurbs(o) and features.history(o)
        and features.history(o)["op"] == "revolve"][-1]
_rv2.hide_set(False); _cap.hide_set(False)
_rv2.select_set(True); _cap.select_set(True); ctx.view_layer.objects.active = _cap
r = bpy.ops.nproto.fillet(surf_a=_rv2.name, surf_b=_cap.name, radius=0.1, flip_a=False, flip_b=True, auto_flip=True,
                          trim_bases=True)
check("surface fillet (revolve + cap) with Auto Side succeeds from a failing start side", r == {'FINISHED'})

# surface fillet where the intersection runs across a revolve's seam (seam moved internally)
bpy.ops.nproto.revolve(inputs=ins((_ms, "S0")), angle=360.0, axis="Z", pivot="OBJECT"); _sv = ctx.active_object
_seam_pt = np.array(geom.world_matrix(_sv) @ Vector(store.get_surfaces(_sv)[0].evaluate(0.0, 0.5)))
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=0.35, height=3.0, degree="0", caps=False); _sc2 = ctx.active_object
_d = _seam_pt[:2] / np.linalg.norm(_seam_pt[:2] - np.array([0, 0])) if np.linalg.norm(_seam_pt[:2]) > 0 else np.array([1, 0])
_sc2.rotation_euler = (0, math.radians(90), math.atan2(_d[1], _d[0])); _sc2.location = (0, 0, _seam_pt[2]); ctx.view_layer.update()
_ok = False
for _sa in (1, -1):
    for _sb in (1, -1):
        try:
            jobs.run("fillet", {"a": _sc2, "b": _sv}, ["fillet"], radius_a=_sa * 0.08, radius_b=_sb * 0.08, xsect=2,
                     tolerance=1e-4, trim_bases=False)
            _ok = True
        except jobs.JobError:
            pass
check("surface fillet across a revolve's seam is found (seam moved internally)", _ok)

# --- Align Curve ---------------------------------------------------------------
from blender_nurbs import align as _al
bpy.ops.nproto.add_primitive(kind="ARC", radius=2.0, angle=90.0, degree="0"); _aa = ctx.active_object; _aa.location = (0, 0, -330)
bpy.ops.nproto.extrude(inputs=ins((_aa, "S0")), distance=1.5, direction="Z"); _asf = ctx.active_object; _asf.name = "AL_Surf"
_ss = store.get_surfaces(_asf)[0]
_edges = [w for w in ("U0", "U1", "V0", "V1")
          if np.ptp([p[2] for p in geom.boundary_points(_ss, w, 8)]) < 1e-6]       # the two arc edges (z const)
_lo, _hi = sorted(_edges, key=lambda w: geom.boundary_points(_ss, w, 2)[0][2])
# a drawn NURBS curve near the arc's end at (0, 2): from below the low arc end to the high arc end
_cu = bpy.data.curves.new("AL_c", 'CURVE'); _cu.dimensions = '3D'; _sp = _cu.splines.new('NURBS')
_pp = [(-0.2, 2.3, -330.0), (-0.8, 2.6, -330.0), (-1.4, 2.6, -329.6), (-1.4, 2.6, -328.9), (-0.8, 2.6, -328.5), (-0.2, 2.3, -328.5)]
_sp.points.add(len(_pp) - 1)
for _q, _c in zip(_sp.points, _pp): _q.co = (*_c, 1.0)
_sp.order_u = 4; _sp.use_endpoint_u = True
_ac = bpy.data.objects.new("AL_c", _cu); ctx.collection.objects.link(_ac)
r1 = bpy.ops.nproto.align_curve(inputs=json.dumps([[_ac.name, "S0"], [_asf.name, _lo]]), end=0, cont="G2", mode="EDGE")
r2 = bpy.ops.nproto.align_curve(inputs=json.dumps([[_ac.name, "S0"], [_asf.name, _hi]]), end=1, cont="G2", mode="EDGE")
_m0 = _al.measure(_ac, 0, _asf, _lo); _m1 = _al.measure(_ac, 1, _asf, _hi)
check("align curve: both ends G2 to the surface's arc edges (gap, angle, curvature)", r1 == r2 == {'FINISHED'}
      and _m0[0] < 1e-6 and _m1[0] < 1e-6 and _m0[1] < 1e-4 and _m1[1] < 1e-4 and _m0[2] < 1e-4 and _m1[2] < 1e-4,
      f"{_m0} {_m1}")
check("align curve: history holds both ends", features.history(_ac)["op"] == "align_curve"
      and set(features.history(_ac)["params"]["ends"]) == {"0", "1"})
_mid_before = np.array(_ac.data.splines[0].points[len(_ac.data.splines[0].points) // 2].co[:3])
_asf.location.x += 0.5; settle()
_m0 = _al.measure(_ac, 0, _asf, _lo); _m1 = _al.measure(_ac, 1, _asf, _hi)
check("align curve follows the surface (history), same 6 CVs", _m0[0] < 1e-6 and _m1[0] < 1e-6 and _m0[1] < 1e-4
      and _m0[2] < 1e-4 and len(_ac.data.splines[0].points) == 6, f"{_m0} {_m1}")
# tangent to surface: a curve that leaves the low arc edge end; its start tangent lies in the surface's tangent plane
_cu2 = bpy.data.curves.new("AL_d", 'CURVE'); _cu2.dimensions = '3D'; _sp2 = _cu2.splines.new('NURBS')
_pp2 = [(0.5, 2.5, -330.2), (0.3, 2.0, -329.5), (0.6, 1.5, -329.0), (1.2, 1.0, -328.6), (1.6, 0.8, -328.2)]
_sp2.points.add(len(_pp2) - 1)
for _q, _c in zip(_sp2.points, _pp2): _q.co = (*_c, 1.0)
_sp2.order_u = 4; _sp2.use_endpoint_u = True
_ad = bpy.data.objects.new("AL_d", _cu2); ctx.collection.objects.link(_ad)
r = bpy.ops.nproto.align_curve(inputs=json.dumps([[_ad.name, "S0"], [_asf.name, _lo]]), end=0, cont="G1", mode="SURFACE")
_kc = features.input_curve(_ad, "S0"); _P, _D, _ = geom.curve_end_frame(_kc, 0)
_n, _ = _al.surface_frame(_asf, _lo, _P)
check("align curve, Tangent to Surface: start on the edge end, tangent in the surface's tangent plane", r == {'FINISHED'}
      and abs(float(_n @ (_D / np.linalg.norm(_D)))) < 1e-6, f"n.t = {float(_n @ (_D / np.linalg.norm(_D))):.2e}")

_cu3 = bpy.data.curves.new("AL_e", 'CURVE'); _cu3.dimensions = '3D'; _sp3 = _cu3.splines.new('NURBS')
_pp3 = [(-0.2, 2.3, -330.0), (-1.4, 2.6, -329.6), (-1.4, 2.6, -328.9), (-0.2, 2.3, -328.5)]
_sp3.points.add(3)
for _q, _c in zip(_sp3.points, _pp3): _q.co = (*_c, 1.0)
_sp3.order_u = 4; _sp3.use_endpoint_u = True
_ae = bpy.data.objects.new("AL_e", _cu3); ctx.collection.objects.link(_ae)
bpy.ops.nproto.align_curve(inputs=json.dumps([[_ae.name, "S0"], [_asf.name, _lo]]), end=0, cont="G2")
r = bpy.ops.nproto.align_curve(inputs=json.dumps([[_ae.name, "S0"], [_asf.name, _hi]]), end=1, cont="G2")
_m0 = _al.measure(_ae, 0, _asf, _lo); _m1 = _al.measure(_ae, 1, _asf, _hi)
check("align a 4-CV curve G2 at both ends: refit to enough CVs, both ends G2", r == {'FINISHED'}
      and len(_ae.data.splines[0].points) >= 6 and max(_m0[0], _m1[0]) < 1e-6 and max(_m0[1], _m1[1]) < 1e-4
      and max(_m0[2], _m1[2]) < 1e-4, f"{len(_ae.data.splines[0].points)} CVs {_m0} {_m1}")
bpy.ops.curve.primitive_bezier_curve_add(location=(0, 0, 0)); _ab = ctx.active_object; _ab.name = "AL_b"
_bpts = _ab.data.splines[0].bezier_points; _ab.data.splines[0].bezier_points.add(1)
for _b, (_co, _hl, _hr) in zip(_ab.data.splines[0].bezier_points, [((-0.2, 2.3, -330), (0, 2, -330), (-0.8, 2.6, -330)),
                                                                  ((-1.4, 2.6, -329.2), (-1.4, 2.6, -329.8), (-1.4, 2.6, -328.7)),
                                                                  ((-0.2, 2.3, -328.5), (-0.8, 2.6, -328.5), (0, 2, -328.5))]):
    _b.co, _b.handle_left, _b.handle_right = _co, _hl, _hr
bpy.ops.nproto.align_curve(inputs=json.dumps([[_ab.name, "S0"], [_asf.name, _lo]]), end=0, cont="G2")
r = bpy.ops.nproto.align_curve(inputs=json.dumps([[_ab.name, "S0"], [_asf.name, _hi]]), end=1, cont="G2")
_m0 = _al.measure(_ab, 0, _asf, _lo); _m1 = _al.measure(_ab, 1, _asf, _hi)
check("align a Bezier curve G2 at both ends (handles), stays Bezier", r == {'FINISHED'} and _ab.data.splines[0].type == 'BEZIER'
      and max(_m0[0], _m1[0]) < 1e-5 and max(_m0[1], _m1[1]) < 1e-3 and max(_m0[2], _m1[2]) < 1e-3, f"{_m0} {_m1}")

# continuity check catches a 90 degree corner
cA = patch("CCa", (0, Y0 + 50, 0), (0, Y0 + 54, 0), "X", 2.0)      # floor 0..2 in x
cB = patch("CCb", (2, Y0 + 50, 0), (2, Y0 + 54, 0), "Z", 2.0)      # wall standing on its edge x=2
bpy.ops.object.select_all(action='DESELECT'); cA.select_set(True); cB.select_set(True)
if "nproto_last_continuity" in ctx.scene: del ctx.scene["nproto_last_continuity"]
r = bpy.ops.nproto.continuity()
rep2 = json.loads(ctx.scene.get("nproto_last_continuity", "{}"))
check("continuity check reports a 90 deg crease", r == {'FINISHED'} and rep2.get("angle", 0) > 80, str(rep2))
# history: move the sweep's rail -> the sweep follows
a_before = SA(sw)
rl.data.splines[0].points[1].co = (0, Y0, 6, 1); ctx.view_layer.update(); history._process()
check("sweep follows its rail (history)", abs(SA(sw) - 2 * math.pi * 0.3 * 6) < 2e-3, f"{a_before:.4f} -> {SA(sw):.4f}")

# --- display/brep caches survive undo-like revision reuse
cc = [o for o in bpy.data.objects if store.is_nurbs(o)][0]
txt_a = cc[store.PROP_BREP]; rev_a = cc.get(store.PROP_REV)
store.display(cc); store.get_brep(cc)          # caches hold state A
store.set_geometry_text(cc, sw[store.PROP_BREP], ctx, keep_untrimmed=False)   # state B
store._disp_cache[cc.session_uid] = (cc[store.PROP_REV], {"edges": []})      # stale entry with B's rev
cc[store.PROP_REV] = rev_a                      # "undo" back to A's revision
store._disp_cache.pop(cc.session_uid, None); store.display(cc)
store.set_geometry_text(cc, txt_a, ctx, keep_untrimmed=False)                 # new edit after undo
check("revision tokens never repeat after undo", cc[store.PROP_REV] != rev_a)
check("display cache follows the geometry", len(store.display(cc).get("edges", [])) == store.get_brep(cc).edges().__len__())



# --- Loft keeps the inputs' layout along the edge (planes from lines: no refit to degree 3)
la = nurbs("LyA", [(-1, 90, 0), (1, 90, 0)], order=2); lb = nurbs("LyB", [(-1, 93, 0), (1, 93, 0)], order=2)
EA = store.new_object(ctx, "LyEA", features.run("extrude", [(la, "S0")], {"distance": 1.0, "direction": "Z", "dir_vec": [0, 0, 1]}))
EB = store.new_object(ctx, "LyEB", features.run("extrude", [(lb, "S0")], {"distance": 1.0, "direction": "Z", "dir_vec": [0, 0, 1]}))
top = max(("U0", "U1", "V0", "V1"), key=lambda r: geom.sample_curve(features.input_curve(EA, r), 3)[:, 2].min())
lay = {}
for cont in ("G0", "G1", "G2"):
    d = ext.surface_cvs(features.run("loft", [(EA, top), (EB, top)], {"cont_a": cont, "cont_b": cont}))
    lay[cont] = (d["degree_v"], d["count_v"], d["count_u"])
check("loft between degree-1 edges stays degree 1 along the edge", lay == {"G0": (1, 2, 2), "G1": (1, 2, 4), "G2": (1, 2, 6)}, str(lay))


# --- Sweep: a profile standing on the rail's END sweeps from there
srail = nurbs("SwRail", [(0, 100, 0), (0, 101.5, 0), (2, 101.5, 0), (2, 100, 0)], order=4)
sp_end = nurbs("SwProfEnd", [(2, 100, 0), (2, 100, 1)], order=2)
sp_start = nurbs("SwProfStart", [(0, 100, 0), (0, 100, 1)], order=2)
bb = {}
for pr in (sp_end, sp_start):
    d = ext.surface_cvs(features.run("sweep", [(pr, "S0"), (srail, "S0")], {"samples": 32}))
    X = np.array(d["points"]); bb[pr.name] = (X.min(0), X.max(0))
check("sweep from the rail's end matches the sweep from its start",
      all(np.allclose(bb["SwProfEnd"][i], bb["SwProfStart"][i], atol=1e-3) for i in (0, 1)) and bb["SwProfEnd"][1][1] < 101.6,
      str({k: (np.round(v[0], 3).tolist(), np.round(v[1], 3).tolist()) for k, v in bb.items()}))

# --- Sweep: an exact circle along a path that runs straight, then bends (a user's tube): the row fit
# used 28 CVs on 32 rows with a first knot span holding no row; the tube folded back behind its profile
from blender_nurbs import primitives as _prm
_tr = nurbs("SwTubePath", [(0, 200, 0), (0, 200.3048, 0), (0, 200.9144, 0), (0.3573, 201.47, 0), (1.2798, 201.648, 0),
                           (1.4625, 200.701, 0), (1.8627, 200.6642, 0)], order=4)
_tc = bpy.data.objects.new("SwTubeProf", bpy.data.curves.new("SwTubeProf", 'CURVE')); ctx.collection.objects.link(_tc)
_prm.write_spline_object(_tc, _prm.exact_circle_spec(0.1016))
_tc.location = (0, 200, 0); _tc.rotation_euler = (math.pi / 2, 0, math.pi); ctx.view_layer.update()
_ts = features.run("sweep", [(_tc, "S0"), (_tr, "S0")], {"samples": 32})
_tk = features.input_curve(_tr, "S0")
_RP = np.array([_tk.evaluate(t) for t in np.linspace(*_tk.parameter_range(), 6000)])
(_u0, _u1), (_v0, _v1) = _ts.u_domain(), _ts.v_domain()
_X = np.array([_ts.evaluate(u, v) for u in np.linspace(_u0, _u1, 161) for v in np.linspace(_v0, _v1, 16)])
_rerr = max(abs(float(np.min(np.linalg.norm(_RP - x, axis=1))) - 0.1016) for x in _X)
check("sweep: a circle along a straight-then-bent path stays a round tube, nothing behind the profile",
      _rerr < 1e-3 and _X[:, 1].min() > 200 - 1e-6, f"radius error {_rerr:.5f}, min y {_X[:, 1].min() - 200:.5f}")


# --- Loft Rails: profiles + 0 / 1 / 2 rails, G0 / G1 / G2
from blender_nurbs import loft_rails, square as _sqm
def _lr_dev(surf, objs):
    out = []
    for o, r in objs:
        for x in geom.sample_curve(features.input_curve(o, r), 21):
            u, v = _sqm._foot(surf, x)
            out.append(float(np.linalg.norm(np.array(surf.evaluate(u, v)) - x)))
    return max(out)
lp0 = nurbs("LRp0", [(0, 120, 0), (4, 120, 0)], order=2); lp1 = nurbs("LRp1", [(0, 124, 0), (4, 124, 0)], order=2)
lr0 = nurbs("LRr0", [(0, 120, 0), (0, 121, 0.6), (0, 122, 0.9), (0, 123, 0.6), (0, 124, 0)])
lr1 = nurbs("LRr1", [(4, 120, 0), (4, 121, 0.6), (4, 122, 0.9), (4, 123, 0.6), (4, 124, 0)])
zm = geom.sample_curve(features.input_curve(lr0, "S0"), 201)[100][2]
lpm = nurbs("LRpm", [(0, 122, zm), (1.3, 122, zm + 1.0), (2.7, 122, zm + 1.0), (4, 122, zm)])
ins5 = [(lp0, "S0"), (lr1, "S0"), (lpm, "S0"), (lr0, "S0"), (lp1, "S0")]      # any pick order after the 1st profile
g = features.run("loft_rails", ins5, {})
inf = dict(loft_rails.LAST_INFO)
check("loft rails: 3 profiles + 2 rails, through every input", inf["profiles"] == 3 and inf["rails"] == 2 and
      _lr_dev(g, ins5) < 1e-4 and inf["folds"] == 0, f"{inf['cvs']} dev {_lr_dev(g, ins5):.1e}")
lg = nurbs("LRguide", [(2, 120, 0), (2, 121.3, 1.0), (2, 122.7, 1.0), (2, 124, 0)])
g = features.run("loft_rails", [(lp0, "S0"), (lp1, "S0"), (lg, "S0")], {})
check("loft rails: 1 guide rail through the middle", loft_rails.LAST_INFO["rails"] == 1 and
      _lr_dev(g, [(lp0, "S0"), (lp1, "S0"), (lg, "S0")]) < 1e-4, f"{_lr_dev(g, [(lp0, 'S0'), (lp1, 'S0'), (lg, 'S0')]):.1e}")
g = features.run("loft_rails", [(lp0, "S0"), (lpm, "S0"), (lp1, "S0")], {})
check("loft rails: 3 profiles, no rails", loft_rails.LAST_INFO["rails"] == 0 and
      _lr_dev(g, [(lp0, "S0"), (lpm, "S0"), (lp1, "S0")]) < 1e-4)
gs = features.run("square", [(lp0, "S0"), (lr1, "S0"), (lp1, "S0"), (lr0, "S0")], {})
g = features.run("loft_rails", [(lp0, "S0"), (lp1, "S0"), (lr0, "S0"), (lr1, "S0")], {})
pts = np.array([gs.evaluate(*(np.array(gs.u_domain()) @ [1 - a, a], np.array(gs.v_domain()) @ [1 - b, b]))
                for a in np.linspace(0, 1, 7) for b in np.linspace(0, 1, 7)])
dmax = max(float(np.linalg.norm(np.array(g.evaluate(*_sqm._foot(g, x))) - x)) for x in pts)
check("loft rails: 2 profiles + 2 rails = Square", dmax < 1e-3, f"{dmax:.1e}")
def _lr_wall(name, a, b):
    c = nurbs(name + "_c", [a, b], order=2)
    o = store.new_object(ctx, name, features.run("extrude", [(c, "S0")], {"distance": 1.0, "direction": "Z", "dir_vec": [0, 0, 1]}))
    return o, max(("U0", "U1", "V0", "V1"), key=lambda r: geom.sample_curve(features.input_curve(o, r), 3)[:, 2].min())
wa = _lr_wall("LRwa", (0, 130, 0), (4, 130, 0)); wb = _lr_wall("LRwb", (0, 134, 0), (4, 134, 0))
wg = nurbs("LRwg", [(2, 130, 1), (2, 130, 1.8), (2, 134, 1.8), (2, 134, 1)])
for cont, lim in (("G1", 0.1), ("G2", 0.1)):
    g = features.run("loft_rails", [wa, wb], {"lr_start": cont, "lr_end": cont})
    inf = dict(loft_rails.LAST_INFO)
    check(f"loft rails: {cont} to both walls", inf["angle"] < 0.1 and inf["curv"] <= lim and inf["folds"] == 0,
          f"{inf['cvs']} angle {inf['angle']:.4f} curv {inf['curv']:.3f}")
g = features.run("loft_rails", [wa, wb, (wg, "S0")], {"lr_start": "G1", "lr_end": "G1"})
inf = dict(loft_rails.LAST_INFO)
check("loft rails: G1 to both walls + guide rail", inf["rails"] == 1 and inf["angle"] < 0.1 and
      _lr_dev(g, [wa, wb, (wg, "S0")]) < 1e-3, f"{inf['cvs']} angle {inf['angle']:.4f}")
r = bpy.ops.nproto.loft_rails(inputs=ins(*ins5))
check("loft rails operator + history", r == {'FINISHED'} and features.history(ctx.active_object)["op"] == "loft_rails")


# the inputs' own layout: degree-1 profiles x the rails' layout, no densifying
d = ext.surface_cvs(features.run("loft_rails", [(lp0, "S0"), (lp1, "S0"), (lr0, "S0"), (lr1, "S0")], {}))
check("loft rails keeps the inputs' layout (degree-1 profiles, rails' CVs)",
      (d["degree_u"], d["count_u"], d["degree_v"], d["count_v"]) == (1, 2, 3, 5), str((d["degree_u"], d["count_u"], d["degree_v"], d["count_v"])))
# one rail: the free edge follows the rail; G1 to two walls keeps the net even
wv1 = _lr_wall("LRwv1", (-2, 150, 0), (0, 150, 0)); wv2 = _lr_wall("LRwv2", (-2, 152, 0), (0, 152, 0))
def _vedge(o):
    return min(("U0", "U1", "V0", "V1"), key=lambda r: np.linalg.norm(geom.sample_curve(features.input_curve(o, r), 2)[:, :2].mean(axis=0) - (0, 151)))
bc = nurbs("LRbend", [(0, 150, 0), (0.6, 150, 0), (1.2, 150, 0), (1.2, 152, 0), (0.6, 152, 0), (0, 152, 0)], order=6)   # G2 to the walls, like a Blend Curve
for cont in ("G0", "G1", "G2"):
    g = features.run("loft_rails", [(wv1[0], _vedge(wv1[0])), (wv2[0], _vedge(wv2[0])), (bc, "S0")], {"lr_start": cont, "lr_end": cont})
    d = ext.surface_cvs(g); X = np.array(d["points"]).reshape(d["count_u"], d["count_v"], 3)
    zs = np.ptp(X[:, :, 2], axis=1)            # every CV row along the rail stays level
    inf = dict(loft_rails.LAST_INFO)
    check(f"loft rails, 1 rail at the edge, {cont}: light even net, free edge follows the rail",
          d["count_u"] * d["count_v"] <= 12 and zs.max() < 1e-6 and inf["angle"] < 0.1,
          f"{(d['count_u'], d['count_v'])} level {zs.max():.1e} angle {inf['angle']:.3f}")

# Planar from a curved surface's edge + a closing edge: a picked edge of a curved
# surface no longer drags its whole (non-planar) boundary loop along
from blender_nurbs import ops as _ops
bend_s = store.new_object(ctx, "PLbend", features.run("extrude", [(bc, "S0")], {"distance": 1.0, "direction": "Z", "dir_vec": [0, 0, 1]}))
b_ = store.get_brep(bend_s)
top_ref = max((f"E{i}" for i in range(len(b_.edges()))),
              key=lambda r: min(e[2] for e in [b_.edges()[int(r[1:])].point_at(t) for t in b_.edges()[int(r[1:])].parameter_range()]))
loop = _ops.edge_loop_refs(bend_s, top_ref)
polys = [[Vector(b_.edges()[int(r[1:])].point_at(t)) for t in np.linspace(*b_.edges()[int(r[1:])].parameter_range(), 9)] for r in loop]
check("planar picking: a curved surface's boundary loop is not planar (single edge is taken)", len(loop) > 1 and not _ops._polylines_planar(polys))
close = nurbs("PLclose", [(0, 150, 1), (0, 152, 1)], order=2)
g = features.run("planar", [(bend_s, top_ref), (close, "S0")], {})
check("planar from a curved surface's edge + a closing curve", g is not None)

# picking: an edge that continues the profile is taken alone, even on a planar face
import types as _types
cap = store.new_object(ctx, "PLcap", g)
def _cands_of(o):
    b = store.get_brep(o); out = []
    for i, e in enumerate(b.edges()):
        out.append((o, f"E{i}", [Vector(e.point_at(t)) for t in np.linspace(*e.parameter_range(), 9)]))
    return out
cands = _cands_of(bend_s) + _cands_of(cap)
fake = _types.SimpleNamespace(chain=True, _cands=cands, _picks=[])
fake._touches_picked = lambda c: _ops.NPROTO_OT_planar._touches_picked(fake, c)
cap_edges = [c for c in cands if c[0] is cap]
straight = min(cap_edges, key=lambda c: (c[2][0] - c[2][-1]).length - sum((a - b).length for a, b in zip(c[2][:-1], c[2][1:])) * -1)
first = _ops.NPROTO_OT_planar.expand_pick(fake, straight)
fake._picks = [(bend_s, top_ref)]
second = _ops.NPROTO_OT_planar.expand_pick(fake, straight)
check("planar picking: a planar face's loop on the first pick, a single edge when continuing a profile",
      len(first) == len(cap_edges) and len(second) == 1, f"{len(first)} / {len(second)}")

# --- Loft Rails, Centreline mode: profiles follow a spine they don't touch
import math as _m
cl_spine = nurbs("CLspine", [(0, 140, 0), (0, 143, 0), (3, 146, 0), (6, 146, 0)])
def _circle_at(name, c, nrm, r):
    from blender_nurbs import primitives
    o = bpy.data.objects.new(name, bpy.data.curves.new(name, 'CURVE')); ctx.collection.objects.link(o)
    primitives.write_spline_object(o, primitives.exact_circle_spec(r))
    o.location = c
    o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(Vector(nrm))
    return o
c0 = _circle_at("CLc0", (0, 140, 0), (0, 1, 0), 0.5)
c1 = _circle_at("CLc1", (6, 146, 0), (1, 0, 0), 0.3)
bpy.context.view_layer.update()
cin = [(c0, "S0"), (c1, "S0"), (cl_spine, "S0")]
g = features.run("loft_rails", cin, {"lr_mode": "CENTRELINE"})
inf = dict(loft_rails.LAST_INFO)
# the sections follow the spine: the spine's middle point is (nearly) at the centre of the section there
mid = geom.sample_curve(features.input_curve(cl_spine, "S0"), 3)[1]
(u0, u1), (v0, v1) = g.u_domain(), g.v_domain()
best = min(range(41), key=lambda i: np.linalg.norm(np.mean([g.evaluate(u0 + (u1 - u0) * i / 40, v0 + (v1 - v0) * j / 24)
                                                              for j in range(24)], axis=0) - mid))
ring = np.array([g.evaluate(u0 + (u1 - u0) * best / 40, v0 + (v1 - v0) * j / 24) for j in range(24)])
rad = np.linalg.norm(ring - mid, axis=1)
check("centreline loft: through both profiles, exact rational profiles", inf["gap"] < 1e-4 and inf["exact_profiles"],
      f"gap {inf['gap']:.1e} {inf['cvs']}")
check("centreline loft follows the spine (round section around it)", np.ptp(rad) < 0.05 and 0.29 < rad.mean() < 0.51,
      f"radius {rad.mean():.3f} +- {np.ptp(rad):.3f}")
rads = []
for i in range(41):
    rg = np.array([g.evaluate(u0 + (u1 - u0) * i / 40, v0 + (v1 - v0) * j / 24) for j in range(24)])
    rads.append(np.linalg.norm(rg - rg.mean(axis=0), axis=1).mean())
check("centreline loft: the section radius goes evenly from 0.5 to 0.3 (no shrink from turned seams)",
      min(rads) > 0.29 and max(rads) < 0.51 and all(b <= a + 1e-3 for a, b in zip(rads[:-1], rads[1:])),
      f"{min(rads):.3f}..{max(rads):.3f}")

sp_pts = geom.sample_curve(features.input_curve(cl_spine, "S0"), 201)
q = sp_pts[100]
tq = sp_pts[101] - sp_pts[99]; tq /= np.linalg.norm(tq)
e1 = np.cross(tq, (0, 0, 1)); e1 /= np.linalg.norm(e1); e2 = np.cross(tq, e1)
sq_pts = [tuple(q + 0.4 * (a * e1 + b * e2)) for a, b in ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 0))]
lsq = nurbs("CLdiamond", sq_pts, order=2)
g = features.run("loft_rails", [(c0, "S0"), (lsq, "S0"), (c1, "S0"), (cl_spine, "S0")], {"lr_mode": "CENTRELINE"})
inf = dict(loft_rails.LAST_INFO)
check("centreline loft: 3 mixed profiles (circle, diamond, circle) refitted to one layout",
      inf["profiles"] == 3 and not inf["exact_profiles"] and inf["gap"] < 1e-2, f"gap {inf['gap']:.1e} {inf['cvs']}")

r = bpy.ops.nproto.loft_rails(inputs=ins(*cin), lr_mode="CENTRELINE")
check("centreline loft operator + history", r == {'FINISHED'} and
      features.history(ctx.active_object)["params"]["lr_mode"] == "CENTRELINE")

# --- Join Curves / Unjoin Curves
from blender_nurbs import curvejoin, primitives
ctx.scene.nproto_settings.construction_history = True
def _samples(o, n=300):
    return np.vstack([geom.sample_curve(features.input_curve(o, f"S{i}"), n) for i in range(len(o.data.splines))])
def _dev(src, dst):
    D = _samples(dst, 20000)
    return max(np.min(np.linalg.norm(D - x, axis=1)) for x in _samples(src, 120))
jl1 = nurbs("JL1", [(-4, 60, 0), (0, 60, 0)], order=2)
jb = nurbs("JB", [(0, 60, 0), (1.5, 60, 0), (2.2, 61, 0), (2.2, 62, 0), (1.5, 63, 0), (0, 63, 0)], order=4)
jl2 = nurbs("JL2", [(-4, 63, 0), (0.0005, 63, 0)], order=2)       # points the wrong way, small gap
srcs = [jl1, jb, jl2]
ref_pts = {o.name: _samples(o) for o in srcs}
for o in ctx.selected_objects: o.select_set(False)
for o in srcs: o.select_set(True)
r = bpy.ops.nproto.join_curves(names=json.dumps([o.name for o in srcs]), tolerance=0.001)
J = ctx.active_object
info = dict(curvejoin.LAST_INFO)
check("join curves: one spline, 2 joints", r == {'FINISHED'} and len(J.data.splines) == 1 and info["joints"] == 2,
      str(info))
dev = max(_dev(o, J) for o in srcs)
check("join curves exact: shape unchanged", dev < 1e-3, f"{dev:.2e}")
check("join curves: history + sources hidden", features.history(J) is not None and all(o.hide_get() for o in srcs))
jb.data.splines[0].points[2].co = (2.6, 61, 0, 1); jb.data.splines[0].points[3].co = (2.6, 62, 0, 1)
features.rebuild(J, ctx)
check("join curves follows its inputs", _dev(jb, J) < 1e-3, f"{_dev(jb, J):.2e}")
r = bpy.ops.nproto.join_curves(names=json.dumps([o.name for o in srcs]), tolerance=0.001, jc_mode="REBUILD")
JR = ctx.active_object
sp = JR.data.splines[0]
check("join curves rebuild: one uniform spline", r == {'FINISHED'} and not sp.use_bezier_u and
      curvejoin.LAST_INFO["fit_error"] < 1e-3 * 7, str(dict(curvejoin.LAST_INFO)))
for o in ctx.selected_objects: o.select_set(False)
J.select_set(True); ctx.view_layer.objects.active = J
r = bpy.ops.nproto.unjoin_curves()
parts = [o for o in ctx.selected_objects]
degs = sorted(o.data.splines[0].order_u - 1 for o in parts)
check("unjoin curves: back to 3 pieces", r == {'FINISHED'} and len(parts) == 3 and degs == [1, 1, 3], str(degs))
devs = [min(max(np.min(np.linalg.norm(_samples(p, 20000) - x, axis=1)) for x in _samples(o, 60)) for p in parts) for o in srcs]
check("unjoin curves: pieces match the sources", max(devs) < 1e-3, str([f"{d:.1e}" for d in devs]))
# rational arc + line: exact join keeps the arc
arc = bpy.data.objects.new("JArc", bpy.data.curves.new("JArc", 'CURVE')); ctx.collection.objects.link(arc)
primitives.write_spline_object(arc, primitives.exact_arc_spec(1.0, 90.0))
arc.location = (0, 70, 0)
ln = nurbs("JLn", [(1, 70, 0), (1, 67, 0)], order=2)
bpy.context.view_layer.update()
r = bpy.ops.nproto.join_curves(names=json.dumps([arc.name, ln.name]), tolerance=0.001)
JA = ctx.active_object
P = _samples(JA, 400)
rad = np.linalg.norm(P - np.array([0, 70, 0]), axis=1)
on_arc = P[:, 1] >= 70 + 1e-6
check("join curves keeps a rational arc exact", r == {'FINISHED'} and np.abs(rad[on_arc] - 1).max() < 1e-6,
      f"{np.abs(rad[on_arc] - 1).max():.1e}")
# smooth single curve: nothing to unjoin; two-spline object comes apart
for o in ctx.selected_objects: o.select_set(False)
sm1 = nurbs("JSmooth", [(0, 80, 0), (1, 81, 0), (2, 80, 0), (3, 81, 0), (4, 80, 0)], order=4)
sm1.select_set(True); ctx.view_layer.objects.active = sm1
r = bpy.ops.nproto.unjoin_curves()
check("unjoin curves leaves a smooth curve alone", r == {'CANCELLED'})
sp2 = sm1.data.splines.new('NURBS'); sp2.points.add(1)
sp2.points[0].co = (0, 85, 0, 1); sp2.points[1].co = (3, 85, 0, 1); sp2.order_u = 2
r = bpy.ops.nproto.unjoin_curves()
check("unjoin curves separates splines", r == {'FINISHED'} and len(ctx.selected_objects) == 2)
# Join button routes curves to Join Curves
for o in ctx.selected_objects: o.select_set(False)
for o in parts[:2]: o.select_set(True)
ctx.view_layer.objects.active = parts[0]
with ctx.temp_override(selected_objects=parts[:2], active_object=parts[0]):
    r = bpy.ops.nproto.join('INVOKE_DEFAULT')
check("Join with curves selected joins curves", r == {'FINISHED'} and ctx.active_object.type == 'CURVE')

# --- Offset Curve: exact lines / arcs, fitted free-form, corners, folds, history
from blender_nurbs import curveoffset
def _oc_pts(o, per=200):
    return np.vstack([curvejoin.eval_span(s_, np.linspace(0, 1, per)) for sp_ in o.data.splines
                      for s_ in curvejoin.transform_spans(curvejoin.spline_spans(sp_), geom.world_matrix(o))])
def _oc_dist(P, base):
    B_ = np.vstack([_oc_pts(b_, 2000) for b_ in (base if isinstance(base, list) else [base])])
    return np.array([np.min(np.linalg.norm(B_ - x, axis=1)) for x in P])
def _oc(objs, **kw):
    r_ = bpy.ops.nproto.offset_curve(names=json.dumps([o.name for o in objs]), **kw)
    return r_, (ctx.active_object if r_ == {'FINISHED'} else None), dict(curveoffset.LAST_INFO)
_ocL = [nurbs(f"OcL{i}", [a, b], order=2) for i, (a, b) in enumerate(
    [((0, 500, 0), (4, 500, 0)), ((4, 500, 0), (4, 502, 0)), ((4, 502, 0), (0, 502, 0)), ((0, 502, 0), (0, 500, 0))])]
_r, _R, _i = _oc(_ocL, distance=0.3)
_P = _oc_pts(_R)
check("offset curve: 4 lines joined -> one sharp rectangle, exact", _r == {'FINISHED'} and len(_R.data.splines) == 1
      and np.allclose(_P.min(0)[:2], [-0.3, 499.7], atol=1e-5) and np.allclose(_P.max(0)[:2], [4.3, 502.3], atol=1e-5)
      and not _i["crossings"] and features.history(_R) is not None, str(_i))
_r, _R, _i = _oc(_ocL, distance=0.3, oc_corner="ROUND")
_d = _oc_dist(_oc_pts(_R), _ocL)
check("offset curve: round corners are arcs about the corners", _r == {'FINISHED'} and np.abs(_d - 0.3).max() < 1e-3,
      f"{np.abs(_d - 0.3).max():.1e}")
_r, _R, _i = _oc(_ocL, distance=0.3, oc_mode="INDIVIDUAL")
check("offset curve: individual -> one offset per curve", _r == {'FINISHED'} and len(ctx.selected_objects) == 4)
_ocC = bpy.data.objects.new("OcCirc", bpy.data.curves.new("OcCirc", 'CURVE')); ctx.collection.objects.link(_ocC)
primitives.write_spline_object(_ocC, primitives.exact_circle_spec(1.0)); _ocC.location = (10, 500, 0)
ctx.view_layer.update()
_r, _R, _i = _oc([_ocC], distance=-0.25)
_rad = np.linalg.norm(_oc_pts(_R) - np.array([10, 500, 0]), axis=1)
check("offset curve: a circle offsets exactly (inward)", _r == {'FINISHED'} and np.abs(_rad - 0.75).max() < 1e-6,
      f"{np.abs(_rad - 0.75).max():.1e}")
_ocRR = bpy.data.objects.new("OcRR", bpy.data.curves.new("OcRR", 'CURVE')); ctx.collection.objects.link(_ocRR)
primitives.write_spline_object(_ocRR, features.run("prim_rectangle", [], {"width": 4.0, "depth": 2.0, "corner": 0.2})[1])
_ocRR.location = (20, 500, 0); ctx.view_layer.update()
_r, _R, _i = _oc([_ocRR], distance=-0.5)
_P = _oc_pts(_R)
check("offset curve: rounded rect inward drops its corner arcs (sharp inner rectangle)", _r == {'FINISHED'}
      and _i["collapsed"] == 4 and np.allclose(_P.min(0)[:2], [18.5, 499.5], atol=1e-5)
      and np.allclose(_P.max(0)[:2], [21.5, 500.5], atol=1e-5), str(_i))
_ocTZ = nurbs("OcTZ", [(30, 500, 0), (36, 500, 0), (33.5, 506, 0), (32.5, 506, 0)], order=2)
_ocTZ.data.splines[0].use_cyclic_u = True
_r, _R, _i = _oc([_ocTZ], distance=-1.2)
check("offset curve: an edge used up inward is dropped (trapezoid -> triangle)", _r == {'FINISHED'} and
      _i["dropped"] == 1 and not _i["crossings"] and len(_R.data.splines[0].points) in (4, 7), str(_i))
_ocF = nurbs("OcF", [(0, 510, 0), (1, 511, 0), (2, 510, 0), (3, 511.5, 0), (4, 510, 0)], order=4)
_r, _R, _i = _oc([_ocF], distance=0.2)
_d = _oc_dist(_oc_pts(_R), _ocF)
check("offset curve: free-form fitted within tolerance", _r == {'FINISHED'} and _i["err"] <= 1e-4
      and np.abs(_d - 0.2).max() < 2e-4, f"{np.abs(_d - 0.2).max():.1e} {len(_R.data.splines[0].points)} CVs")
_ocF.data.splines[0].points[2].co = (2, 509, 0, 1)
features.rebuild(_R, ctx)
_d = _oc_dist(_oc_pts(_R), _ocF)
check("offset curve follows its input (history)", np.abs(_d - 0.2).max() < 2e-4, f"{np.abs(_d - 0.2).max():.1e}")
_ocB = nurbs("OcB", [(0, 520, 0), (1, 520, 0), (1.5, 521, 0), (2, 520, 0), (3, 520, 0)], order=4)
_r, _R, _i = _oc([_ocB], distance=-0.6)
_d = _oc_dist(_oc_pts(_R, 400), _ocB)
check("offset curve: a fold (distance > radius) is cut out, nothing closer than the distance",
      _r == {'FINISHED'} and _i["folds"] >= 1 and _d.min() > 0.6 - 2e-3, f"{_d.min():.4f} {_i}")
for o_ in [o_ for o_ in bpy.data.objects if o_.name.startswith("Oc")]:
    bpy.data.objects.remove(o_)

# --- Save / reload
path = os.path.join(tempfile.gettempdir(), "nurbs_test.blend")
bpy.ops.wm.save_as_mainfile(filepath=path)
bpy.ops.wm.open_mainfile(filepath=path)
store._cache.clear()
L2 = bpy.data.objects["Loft"]
b = store.get_brep(L2)
check("reload brep", b.face_count() == 1)
blend2 = bpy.data.objects[G2.name] if False else [o for o in bpy.data.objects if o.name.startswith("Loft") and features.history(o)][-1]
check("reload history intact", features.history(blend2) is not None)

# --- Class-A edge fillet: a U-bend wall (straight / degree-5 G2 bend / straight) with a flat top
_Y = -0.9144
_l1 = nurbs("CA_L1", [(0, 0, 0), (1, 0, 0)], order=2)
_bd = nurbs("CA_Bend", [(1, 0, 0), (1.6, 0, 0), (2.2, 0, 0), (2.2, _Y, 0), (1.6, _Y, 0), (1, _Y, 0)], order=6)
_l2 = nurbs("CA_L2", [(1, _Y, 0), (0, _Y, 0)], order=2)
_cl = nurbs("CA_Close", [(0, _Y, 1), (0, 0, 1)], order=2)
_walls = []
for _c in (_l1, _bd, _l2):
    bpy.ops.nproto.extrude(inputs=ins((_c, "S0")), distance=1.0, direction="Z"); _walls.append(ctx.active_object)
def _top(o):
    return max(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([p[2] for p in geom.boundary_points(store.get_surfaces(o)[0], w, 8)]))
r = bpy.ops.nproto.planar(inputs=ins(*[(w, _top(w)) for w in _walls], (_cl, "S0"))); _capU = ctx.active_object
check("class-a scene: planar top from the wall rims", r == {'FINISHED'} and store.is_nurbs(_capU))
bpy.ops.object.select_all(action='DESELECT')
for o in _walls + [_capU]:
    o.select_set(True)
ctx.view_layer.objects.active = _walls[0]
bpy.ops.nproto.join(tolerance=1e-3); _uj = ctx.active_object
_bju = store.get_brep(_uj)
from blender_nurbs.ops_trim import _edge_mid_world as _emw
_top_edges = [i for i in store.shared_edges(_uj) if abs(_emw(_uj, i)[2] - 1.0) < 1e-3]
check("class-a scene: join gives 3 top edges to fillet", len(_top_edges) == 3, str(_top_edges))
_ujk = _uj.copy(); _ujk.data = _uj.data.copy(); ctx.collection.objects.link(_ujk)   # kernel-only reference
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_ujk.name, f"E{i}"] for i in _top_edges]), radius=0.1)
def _dense(o):
    out = []
    for f in store.get_brep(o).faces():
        d = ext.surface_cvs(f.surface())
        if d["count_u"] > d["degree_u"] + 1 or d["count_v"] > d["degree_v"] + 1:
            out.append((d["count_u"], d["count_v"]))
    return out
check("kernel fillet makes a dense face along the bend", r == {'FINISHED'} and any(max(c) > 12 for c in _dense(_ujk)),
      str(_dense(_ujk)))
def _lamina(o):
    b = store.get_brep(o)
    return sum(1 for e in b.edges() if len(e.faces()) == 1), sum(1 for e in b.edges() if len(e.faces()) > 2)
_cnt = {}
for _tol in (1e-3, 1e-4):
    _o = _uj.copy(); _o.data = _uj.data.copy(); ctx.collection.objects.link(_o)
    r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_o.name, f"E{i}"] for i in _top_edges]), radius=0.1,
                                   class_a=True, ca_tolerance=_tol, ca_continuity="2")
    _rep = json.loads(_o.get("nproto_class_a", "{}"))
    _fr = [f for f in _rep.get("faces", []) if "skipped" not in f]
    _cnt[_tol] = sum(f["spans"] for f in _fr)
    check(f"class-a fillet (tol {_tol}): every fillet face single-span, within tolerance, G2 ends",
          r == {'FINISHED'} and _fr and not _dense(_o) and all(f["within"] and f["dev"] <= _tol and f["angle"] <= 0.1
                                                                  and f["ends"] == ["G2", "G2"] for f in _fr),
          f"{_rep.get('error', '')} {[(f['spans'], f['degree'], round(f['dev'], 6)) for f in _fr]}")
    check(f"class-a fillet (tol {_tol}): stitched like the kernel fillet (same open edges, no non-manifold)",
          _lamina(_o) == (_lamina(_ujk)[0], 0), f"{_lamina(_o)} vs kernel {_lamina(_ujk)}")
    if _tol == 1e-4:
        _st = features.stack(_o)[-1]
        _nf = store.get_brep(_o).face_count()
        features.rebuild(_o, ctx)
        check("class-a fillet is a history step and replays", _st["params"].get("class_a") is True
              and store.get_brep(_o).face_count() == _nf)
check("class-a: a tighter tolerance uses more patches", _cnt[1e-4] >= _cnt[1e-3] >= 1, str(_cnt))

# Class-A on a Blend fillet of the symmetric U: straights exact (degree 1 along), bend symmetric
_ob = _uj.copy(); _ob.data = _uj.data.copy(); ctx.collection.objects.link(_ob)
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_ob.name, f"E{i}"] for i in _top_edges]), shape="BLEND", radius=0.1,
                               continuity="2", class_a=True, ca_tolerance=1e-4)
_rb = [f for f in json.loads(_ob.get("nproto_class_a", "{}")).get("faces", []) if "skipped" not in f]
_bend = [f for f in _rb if f["spans"] > 1 or f["degree"] > 1]
_strt = [f for f in _rb if f["spans"] == 1 and f["degree"] == 1]
check("class-a blend: straight runs are single exact patches, the bend is symmetric and G2 to them",
      r == {'FINISHED'} and len(_strt) == 2 and len(_bend) == 1 and _bend[0]["symmetric"] and _bend[0]["within"]
      and _bend[0]["exact_sections"] and _bend[0]["ends"] == ["G2", "G2"],
      str([(f["spans"], f["degree"], f.get("symmetric"), f["ends"]) for f in _rb]))
_ar = sorted((round(f.area(), 6) for f in store.get_brep(_ob).faces() if f.area() < 0.2), reverse=True)
_pairs = [abs(a - b) for a, b in zip(_ar[0::2], _ar[1::2])]
check("class-a blend: patch areas come in mirror pairs", _pairs and max(_pairs) < 1e-4, str(_ar))
check("display toggles live on the window manager (no undo step)", hasattr(bpy.context.window_manager, "nproto_view")
      and not hasattr(ctx.scene.nproto_settings, "show_cvs"))

# Class-A unit pieces: Bezier split is exact, G2 end constraint holds
from blender_nurbs.kernel import classa as _ca
_P = np.random.RandomState(3).rand(9, 4) + 0.5
_U = _ca.clamped_knots(5, 4, [0.2, 0.5, 0.7])
_pcs = _ca.bezier_pieces(_P, _U, 5)
_tt = np.linspace(0, 1, 50)
_ref = _ca.basis(_U, 5, _tt) @ _P
_br = [0.0, 0.2, 0.5, 0.7, 1.0]
_err = 0.0
for _k, _t in enumerate(_tt):
    _j = min(int(np.searchsorted(_br, _t, side="right")) - 1, 3)
    _u = (_t - _br[_j]) / (_br[_j + 1] - _br[_j])
    _err = max(_err, float(np.abs(_ca.bernstein(5, [_u])[0] @ _pcs[_j] - _ref[_k]).max()))
check("class-a: Bezier split of a B-spline is exact", len(_pcs) == 4 and _err < 1e-12, f"{_err:.1e}")

# --- Split / Trim Curves (line across a circle, as in the user's sketch)
from blender_nurbs import curvetrim as _ct, curvejoin as _cj
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
def _line(name, a, b):
    o = nurbs(name, [a, b], order=2); return o
def _on_curve(piece_obj, ref_spans):
    """max distance of points of piece_obj to the reference curve (dense samples)."""
    sp = _cj.transform_spans(_cj.spline_spans(piece_obj.data.splines[0]), geom.world_matrix(piece_obj))
    pts, _ = _ct.sample_spans(sp, 20)
    if ref_spans == "circle":
        return float(np.abs(np.linalg.norm(pts[:, :2], axis=1) - 1.0).max() + np.abs(pts[:, 2]).max())
    return float(np.abs(pts[:, 1] - 0.35).max() + np.abs(pts[:, 2]).max())
bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0))
CIR = ctx.active_object; CIR.name = "TCircle"
LN = _line("TLine", (-2.0, 0.35, 0.0), (2.0, 0.35, 0.0))
cs = _ct.CurveSet([CIR, LN])
hits = _ct.intersect(cs.world[("TCircle", 0)], cs.world[("TLine", 0)])
xs = sorted(float(h[2][0]) for h in hits)
xe = math.sqrt(1 - 0.35 ** 2)
check("curves: line x circle -> 2 exact crossings", len(hits) == 2 and abs(xs[0] + xe) < 1e-9 and abs(xs[1] - xe) < 1e-9,
      f"{xs}")
circ_ref = cs.world[("TCircle", 0)]; line_ref = cs.world[("TLine", 0)]
bpy.ops.object.select_all(action='DESELECT'); CIR.select_set(True); LN.select_set(True)
ctx.view_layer.objects.active = CIR
r = bpy.ops.nproto.split_curves()
parts = [o for o in bpy.data.objects if o.type == 'CURVE']
lines = [o for o in parts if o.name.startswith("TLine")]; circs = [o for o in parts if o.name.startswith("TCircle")]
check("split curves: line -> 3, circle -> 2", r == {'FINISHED'} and len(lines) == 3 and len(circs) == 2,
      f"{[o.name for o in parts]}")
dev = max([_on_curve(o, "circle") for o in circs] + [_on_curve(o, "line") for o in lines])
check("split curves: pieces lie on the originals", dev < 1e-6, f"{dev:.1e}")
_len = lambda o: float(np.linalg.norm(np.diff(_ct.sample_spans(_cj.spline_spans(o.data.splines[0]), 200)[0], axis=0), axis=1).sum())
tot = sum(_len(o) for o in circs)
check("split curves: circle pieces add up to 2*pi", abs(tot - 2 * math.pi) < 1e-4, f"{tot:.6f}")
arcs = sorted(_len(o) for o in circs)
check("split curves: arc lengths match the crossing angles", abs(arcs[0] - (math.pi - 2 * math.asin(0.35))) < 1e-4, f"{arcs}")
# Trim logic: cut the line by the circle and drop the outside piece on the right
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0)); CIR = ctx.active_object; CIR.name = "TCircle"
LN = _line("TLine", (-2.0, 0.35, 0.0), (2.0, 0.35, 0.0))
cs = _ct.CurveSet([CIR, LN]); key = ("TLine", 0)
cuts = cs.cuts(key, [("TCircle", 0)])
pcs = _ct.split_spans(cs.local[key], cuts, cs.closed[key])
k = _ct.piece_at(pcs, 0.95, 1)
check("trim curves: click near the right end picks the outer right piece", len(pcs) == 3 and k == 2)
made = _ct.put_back(LN, 0, _ct.remove_piece(pcs, k, False))
xs = sorted(float(c) for o in [LN] + made for p in o.data.splines[0].points for c in [p.co[0]])
check("trim curves: line now ends on the circle", len(made) == 0 and abs(max(xs) - xe) < 1e-6, f"{max(xs):.6f}")
# closed curve: the piece across the seam stays one piece
key = ("TCircle", 0)
cuts = cs.cuts(key, [("TLine", 0)])
pcs = _ct.split_spans(cs.local[key], cuts, True)
check("trim curves: circle cut twice -> 2 pieces", len(pcs) == 2)
check("trim curves: deleting the middle of the line leaves 2 parts, of a circle 1 arc",
      len(_ct.remove_piece(_ct.split_spans(cs.local[("TLine", 0)], [0.25, 0.75], False), 1, False)) == 2
      and len(_ct.remove_piece(pcs, 0, True)) == 1)
# apparent intersection: line above the circle, seen from the top
LN2 = _line("TLineHigh", (-2.0, -0.2, 1.0), (2.0, -0.2, 1.0))
cs3 = _ct.CurveSet([CIR, LN2]); cs2 = _ct.CurveSet([CIR, LN2], (0, 0, -1))
check("split curves: apparent crossings only with a view direction",
      not cs3.cuts(("TLineHigh", 0), [("TCircle", 0)]) and len(cs2.cuts(("TLineHigh", 0), [("TCircle", 0)])) == 2)
# Bezier curves crossing twice
B1 = bezier("TB1", [(0, 0), (1, 1), (2, 0), (3, 1)]); B2 = bezier("TB2", [(0, 0.5), (1.5, 0.5), (3, 0.5)])
cs4 = _ct.CurveSet([B1, B2]); h4 = _ct.intersect(cs4.world[("TB1", 0)], cs4.world[("TB2", 0)])
err = max(abs(float(h[2][1]) - 0.5) for h in h4) if h4 else 1
check("curves: Bezier x Bezier crossings on y=0.5", len(h4) == 3 and err < 1e-9, f"{len(h4)} {err:.1e}")

# --- Rebuild / Extend Curve
from blender_nurbs import curvetools as _cv
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0)); CIR = ctx.active_object; CIR.name = "RCircle"
LN = _line("RLine", (-2.0, 0.35, 0.0), (2.0, 0.35, 0.0))
bpy.ops.object.select_all(action='DESELECT'); CIR.select_set(True); LN.select_set(True); ctx.view_layer.objects.active = CIR
bpy.ops.nproto.split_curves(cutter="ALL")
arcs = [o for o in bpy.data.objects if o.name.startswith("RCircle")]
low = max(arcs, key=lambda o: _len(o))
def _wsp(o): return _cj.transform_spans(_cj.spline_spans(o.data.splines[0]), geom.world_matrix(o))
ends0 = _cj._ends(_wsp(low)); tan0 = _cv.end_tangents(_wsp(low))
bpy.ops.object.select_all(action='DESELECT'); low.select_set(True); ctx.view_layer.objects.active = low
r = bpy.ops.nproto.rebuild_curve(rc_degree="3", rc_cvs=0, tolerance=1e-4)
sp = low.data.splines[0]
pts = _ct.sample_spans(_wsp(low), 40)[0]
dev = float(np.abs(np.linalg.norm(pts[:, :2], axis=1) - 1).max())
ends1 = _cj._ends(_wsp(low)); tan1 = _cv.end_tangents(_wsp(low))
check("rebuild curve: split arc -> one uniform non-rational cubic within tolerance",
      r == {'FINISHED'} and sp.order_u == 4 and not sp.use_bezier_u and all(abs(p.co[3] - 1) < 1e-9 for p in sp.points)
      and dev <= 1.2e-4, f"{len(sp.points)} CVs dev {dev:.1e}")
check("rebuild curve: ends and end tangents kept",
      max(np.linalg.norm(ends0[0] - ends1[0]), np.linalg.norm(ends0[1] - ends1[1])) < 1e-6
      and min(float(tan0[0] @ tan1[0]), float(tan0[1] @ tan1[1])) > 1 - 1e-9)
r = bpy.ops.nproto.rebuild_curve(rc_degree="5", rc_cvs=12)
check("rebuild curve: fixed degree / CV count", len(low.data.splines[0].points) == 12 and low.data.splines[0].order_u == 6)
bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(4, 0, 0)); FC = ctx.active_object
r = bpy.ops.nproto.rebuild_curve(rc_degree="3", rc_cvs=0, tolerance=1e-4)
pts = _ct.sample_spans(_wsp(FC), 40)[0] - np.array([4, 0, 0])
dev = float(np.abs(np.linalg.norm(pts[:, :2], axis=1) - 1).max())
check("rebuild curve: closed circle -> periodic cubic", FC.data.splines[0].use_cyclic_u and dev <= 1.2e-4,
      f"{len(FC.data.splines[0].points)} CVs dev {dev:.1e}")
# Extend: a short arc naturally / as an arc stays on its circle; a line grows by the distance
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0)); CIR = ctx.active_object; CIR.name = "ECircle"
LN = _line("ELine", (-2.0, 0.35, 0.0), (2.0, 0.35, 0.0))
bpy.ops.object.select_all(action='DESELECT'); CIR.select_set(True); LN.select_set(True); ctx.view_layer.objects.active = LN
bpy.ops.nproto.split_curves(cutter="ACTIVE")
arc = min([o for o in bpy.data.objects if o.name.startswith("ECircle")], key=_len)
L0 = _len(arc)
n_arc0 = len(arc.data.splines[0].points)
for kind in ("NATURAL", "ARC"):
    bpy.ops.object.select_all(action='DESELECT'); arc.select_set(True); ctx.view_layer.objects.active = arc
    r = bpy.ops.nproto.extend_curve(ec_type=kind, ec_to="DISTANCE", ec_end="END", distance=0.3)
    pts = _ct.sample_spans(_wsp(arc), 30)[0]
    dev = float(np.abs(np.linalg.norm(pts[:, :2], axis=1) - 1).max())
    L1 = _len(arc)
    check(f"extend curve ({kind.lower()}): arc stays on its circle, +0.3 long", r == {'FINISHED'} and dev < 1e-6
          and abs(L1 - L0 - 0.3) < 1e-3, f"dev {dev:.1e} dL {L1 - L0:.4f}")
    L0 = L1
S2 = _line("EShort", (0.0, -0.2, 0.0), (0.3, -0.2, 0.0))
bpy.ops.object.select_all(action='DESELECT'); S2.select_set(True); ctx.view_layer.objects.active = S2
r = bpy.ops.nproto.extend_curve(ec_type="LINE", ec_to="DISTANCE", ec_end="BOTH", distance=0.25)
xs = [p.co[0] for p in S2.data.splines[0].points]
check("extend curve (line, both ends): end CVs moved, still 2 CVs", r == {'FINISHED'} and len(xs) == 2
      and abs(min(xs) + 0.25) < 1e-6 and abs(max(xs) - 0.55) < 1e-6, f"{xs}")
S3 = _line("EToCircle", (0.0, -0.5, 0.0), (0.4, -0.5, 0.0))
bpy.ops.object.select_all(action='DESELECT'); S3.select_set(True); CIR2 = [o for o in bpy.data.objects if o.name.startswith("ECircle") and o != arc][0]
CIR2.select_set(True); ctx.view_layer.objects.active = CIR2
r = bpy.ops.nproto.extend_curve(ec_type="LINE", ec_to="ACTIVE", ec_end="NEAREST")
xs = [p.co[0] for p in S3.data.splines[0].points]
check("extend curve to the active curve (nearest end)", r == {'FINISHED'} and abs(max(xs) - math.sqrt(0.75)) < 1e-5
      and abs(min(xs)) < 1e-9, f"{xs}")

check("extend curve: an exact arc keeps its CV count", len(arc.data.splines[0].points) == n_arc0,
      f"{n_arc0} -> {len(arc.data.splines[0].points)}")
BZ = bezier("EBez", [(0, 2), (1, 2.5), (2, 2)])
bpy.ops.object.select_all(action='DESELECT'); BZ.select_set(True); ctx.view_layer.objects.active = BZ
r = bpy.ops.nproto.extend_curve(ec_type="NATURAL", ec_to="DISTANCE", ec_end="END", distance=0.4)
check("extend curve: a Bezier spline stays Bezier with its point count",
      r == {'FINISHED'} and BZ.data.splines[0].type == 'BEZIER' and len(BZ.data.splines[0].bezier_points) == 3)
UC = nurbs("EUni", [(0, 3, 0), (1, 3.5, 0), (2, 3, 0), (3, 3.5, 0), (4, 3, 0), (5, 3.5, 0)])
bpy.ops.object.select_all(action='DESELECT'); UC.select_set(True); ctx.view_layer.objects.active = UC
r = bpy.ops.nproto.extend_curve(ec_type="NATURAL", ec_to="DISTANCE", ec_end="END", distance=0.3, keep_cvs=True)
check("extend curve: Keep CV Count refits a uniform spline", r == {'FINISHED'} and len(UC.data.splines[0].points) == 6)
# To Next Curve: a line inside the circle reaches it
S4 = _line("ENext", (0.0, 0.1, 0.0), (0.2, 0.1, 0.0))
bpy.ops.object.select_all(action='DESELECT'); S4.select_set(True); ctx.view_layer.objects.active = S4
bpy.context.scene.cursor.location = (5, 0.1, 0)
r = bpy.ops.nproto.extend_curve(ec_type="LINE", ec_to="ANY", ec_end="END")
xs = [p.co[0] for p in S4.data.splines[0].points]
check("extend curve to the next curve", r == {'FINISHED'} and len(xs) == 2 and abs(max(xs) - math.sqrt(1 - 0.01)) < 1e-5, f"{xs}")
bpy.context.scene.cursor.location = (0, 0, 0)

# --- Edit Mode: Rebuild / Extend / Split edit the splines in place
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.curve.primitive_nurbs_circle_add(radius=1.0, location=(0, 0, 0)); CIR = ctx.active_object; CIR.name = "MCircle"
TWO = nurbs("MTwo", [(-2.0, 0.35, 0.0), (2.0, 0.35, 0.0)], order=2)
s2 = TWO.data.splines.new('NURBS'); s2.points.add(1)
s2.points[0].co = (-2.0, -0.5, 0.0, 1.0); s2.points[1].co = (-1.5, -0.5, 0.0, 1.0); s2.order_u = 2; s2.use_endpoint_u = True
bpy.ops.object.select_all(action='DESELECT'); TWO.select_set(True); ctx.view_layer.objects.active = TWO
bpy.ops.object.mode_set(mode='EDIT')
for sp in TWO.data.splines:
    for p in sp.points:
        p.select = False
TWO.data.splines[1].points[1].select = True        # the short line's end point
r = bpy.ops.nproto.extend_curve(ec_type="LINE", ec_to="ANY", ec_end="NEAREST")
check("edit mode: extend the selected end up to the circle, still in Edit Mode",
      r == {'FINISHED'} and ctx.mode == 'EDIT_CURVE')
for sp in TWO.data.splines:
    for p in sp.points:
        p.select = False
TWO.data.splines[0].points[0].select = True        # the long line
r = bpy.ops.nproto.split_curves()
nspl = len(TWO.data.splines)
bpy.ops.object.mode_set(mode='OBJECT')
short = [sp for sp in TWO.data.splines if abs(sp.points[0].co[1] + 0.5) < 1e-6]
xe2 = -math.sqrt(1 - 0.25)
check("edit mode: extended line kept after leaving Edit Mode (2 CVs, ends on the circle)",
      len(short) == 1 and len(short[0].points) == 2 and abs(short[0].points[1].co[0] - xe2) < 1e-5,
      f"{[tuple(p.co) for p in short[0].points] if short else None}")
check("edit mode: split keeps the pieces as splines of the same object", r == {'FINISHED'} and nspl == 4
      and len([o for o in bpy.data.objects if o.name.startswith("MTwo")]) == 1, f"{nspl} splines")
bpy.ops.object.mode_set(mode='EDIT')
for sp in TWO.data.splines:
    for p in sp.points:
        p.select = False
CIRsel = None
bpy.ops.object.mode_set(mode='OBJECT')
CIR.select_set(True); TWO.select_set(False); ctx.view_layer.objects.active = CIR
bpy.ops.object.mode_set(mode='EDIT')
r = bpy.ops.nproto.rebuild_curve(rc_degree="3", rc_cvs=10)
ok_edit = ctx.mode == 'EDIT_CURVE'
bpy.ops.object.mode_set(mode='OBJECT')
check("edit mode: rebuild in place (circle -> 10 periodic CVs)", r == {'FINISHED'} and ok_edit
      and len(CIR.data.splines[0].points) == 10 and CIR.data.splines[0].use_cyclic_u)

# --- Sketch tools: CV Curve (CVs / Through Points) and Line
from blender_nurbs import drawtools as _dt
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
if ctx.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
clicks = [(0, 0, 0), (1, 0.8, 0), (2.5, 0.2, 0), (3, 1.5, 0), (4.2, 1.0, 0)]
def _curve_pts(o, per=4000):
    sp = _cj.transform_spans(_cj.spline_spans(o.data.splines[0]), geom.world_matrix(o))
    return _ct.sample_spans(sp, per)[0]
def _passes(o, pts):
    from blender_nurbs import curvetools as _cvt
    C = _curve_pts(o, 400)
    return float(_cvt._seg_dist(np.array(pts, float), C).max())
data = json.dumps({"pts": clicks, "start_out": None, "end_out": None, "edit": ""})
r = bpy.ops.nproto.draw_curve(data=data, mode="THROUGH", degree="3", closed=False)
o = ctx.active_object; sp = o.data.splines[0]
d = _passes(o, clicks)
check("cv curve (through points): passes through every click, uniform cubic",
      r == {'FINISHED'} and sp.order_u == 4 and sp.use_endpoint_u and not sp.use_bezier_u and len(sp.points) == 5
      and d < 5e-5, f"{d:.1e}")
r = bpy.ops.nproto.draw_curve(data=data, mode="CV", degree="3", closed=False)
o = ctx.active_object; M = np.array(geom.world_matrix(o))
cvs = [M[:3, :3] @ np.array(p.co[:3]) + M[:3, 3] for p in o.data.splines[0].points]
check("cv curve (CVs): the clicks are the CVs", r == {'FINISHED'} and
      max(np.linalg.norm(np.array(a) - b) for a, b in zip(clicks, cvs)) < 1e-6)
r = bpy.ops.nproto.draw_curve(data=data, mode="THROUGH", degree="3", closed=True)
o = ctx.active_object
d = _passes(o, clicks)
check("cv curve (through points, closed): periodic through every click",
      r == {'FINISHED'} and o.data.splines[0].use_cyclic_u and d < 5e-5, f"{d:.1e}")
d2 = json.dumps({"pts": [(0, 0, 0), (1, 1, 0), (1.5, 0.5, 0), (2, 0, 0)], "start_out": (1, 0, 0), "end_out": (0, -1, 0), "edit": ""})
r = bpy.ops.nproto.draw_curve(data=d2, mode="CV", degree="2", closed=False, start_cont="G1", end_cont="G1")
o = ctx.active_object; M = np.array(geom.world_matrix(o))
cv = [M[:3, :3] @ np.array(p.co[:3]) + M[:3, 3] for p in o.data.splines[0].points]
check("cv curve: tangent match puts the 2nd CV on the snapped end's tangent", r == {'FINISHED'}
      and abs(cv[1][1]) < 1e-6 and cv[1][0] > 0 and abs(cv[2][0] - 2) < 1e-6 and cv[2][1] < 0, f"{cv}")
X, p_ = _dt.interpolate(np.random.RandomState(1).rand(7, 3), 5)
check("cv curve: interpolation degree 5 keeps one CV per point", len(X) == 7 and p_ == 5)
lines = [[(0, 0, 0), (1, 0, 0)], [(1, 0, 0), (1, 1, 0)], [(3, 0, 0), (4, 0, 0)]]
n0 = len(bpy.data.objects)
r = bpy.ops.nproto.draw_line(data=json.dumps({"lines": lines, "edit": ""}), polyline=False)
check("line: one 2-CV curve per line", r == {'FINISHED'} and len(bpy.data.objects) - n0 == 3
      and all(len(o.data.splines[0].points) == 2 for o in ctx.selected_objects))
n0 = len(bpy.data.objects)
r = bpy.ops.nproto.draw_line(data=json.dumps({"lines": lines, "edit": ""}), polyline=True)
sizes = sorted(len(o.data.splines[0].points) for o in ctx.selected_objects)
check("line (polyline): connected lines become one degree-1 curve", r == {'FINISHED'} and sizes == [2, 3], f"{sizes}")
host = ctx.selected_objects[0]
bpy.ops.object.select_all(action='DESELECT'); host.select_set(True); ctx.view_layer.objects.active = host
bpy.ops.object.mode_set(mode='EDIT')
ns = len(host.data.splines)
r = bpy.ops.nproto.draw_curve(data=json.dumps({"pts": clicks, "start_out": None, "end_out": None, "edit": host.name}),
                              mode="CV", degree="3", closed=False)
ok_mode = ctx.mode == 'EDIT_CURVE'
bpy.ops.object.mode_set(mode='OBJECT')
check("cv curve in Edit Mode: added as a spline of the edited object", r == {'FINISHED'} and ok_mode
      and len(host.data.splines) == ns + 1)

# line tool: typed length / angle
P = _dt.polar_point((1, 1, 0), (1, 0, 0), (0, 1, 0), 2.0, 90.0)
check("line: typed 2 at 90 deg goes straight up", np.allclose(P, (1, 3, 0)))
P = _dt.polar_point((0, 0, 0), (1, 0, 0), (0, 0, 1), 1.0, 30.0)
check("line: typed angle in a front-view plane", np.allclose(P, (math.cos(math.radians(30)), 0, 0.5)))
us = ctx.scene.unit_settings
us.system, us.length_unit = 'IMPERIAL', 'INCHES'
v1, v2, v3 = _dt.parse_length(ctx, "3"), _dt.parse_length(ctx, "2ft 6in"), _dt.parse_length(ctx, "10mm")
check("line: typed lengths use the display unit (inches) and honour unit letters",
      abs(v1 - 0.0762) < 1e-6 and abs(v2 - 0.762) < 1e-6 and abs(v3 - 0.01) < 1e-6, f"{v1} {v2} {v3}")
us.system = 'METRIC'; us.length_unit = 'METERS'

# --- Sketch snapping (CV Curve / Line): curve ends, middles, points on curves,
#     surface edges, face centres, shown surface CVs; nearest the viewer wins
from blender_nurbs import snapping as _sn, draw as _drw
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
ctx.scene.cursor.location = (0, 0, 0)
bpy.ops.nproto.add_primitive(kind='CYLINDER', radius=0.5, height=1.2, caps=True)
SCY = ctx.active_object
SLN = nurbs("SnapLine", [(1.0, -1.0, 0.0), (3.0, 1.0, 0.0)], order=2)
def _snap_view(front=False, s=2.0, W=1000):
    """Orthographic top (or front) view, camera 10 away; 1 unit = W/(2s) px."""
    VM = np.eye(4)
    if front:                                  # looking along +Y: screen x = X, y = Z
        VM[:3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    VM[2, 3] = -10.0
    PM = np.diag([1 / s, 1 / s, -0.01, 1.0]) @ VM
    return _sn.View(PM, VM, W, W)
_px = lambda x, y, s=2.0, W=1000: np.array([W / 2 + W / 2 * x / s, W / 2 + W / 2 * y / s])
top, front = _snap_view(), _snap_view(True)
T = _sn.collect([SCY, SLN])
h = T.pick(top, _px(1, -1) + (3, 2))
check("snap: curve end point (exact)", h is not None and h[1] == "end" and np.allclose(h[0], (1, -1, 0), atol=1e-12)
      and h[2] is not None, f"{h}")
h = T.pick(top, _px(2, 0) + (2, -3))
check("snap: middle of a line", h is not None and h[1] == "mid" and np.allclose(h[0], (2, 0, 0), atol=1e-9), f"{h}")
q = np.array([1.6, -0.4, 0.0])
h = T.pick(top, _px(*q[:2]) + 5 * np.array([-1, 1]) / math.sqrt(2))
dl = abs(np.cross(h[0] - (1, -1, 0), np.array([1, 1, 0]) / math.sqrt(2))[2]) if h else 1
check("snap: a point on a curve lies on it", h is not None and h[1] == "curve" and dl < 1e-12
      and np.linalg.norm(h[0] - q) < 0.02, f"{h} off {dl:.1e}")
a = math.radians(30)
c30 = 0.5 * np.array([math.cos(a), math.sin(a)])
h = T.pick(top, _px(*c30) + 4 * c30 / 0.5)
_ztop = max(f_.centroid()[2] for f_ in store.get_brep(SCY).faces())      # the cap plane (stored in float32)
_de = min(e_.closest_point(tuple(h[0]))[2] for e_ in store.get_brep(SCY).edges()) if h else 1
check("snap: a point on a surface edge is exactly on it, the top rim (nearest the viewer)",
      h is not None and h[1] == "edge" and _de < 1e-9 and abs(h[0][2] - _ztop) < 1e-7
      and np.linalg.norm(h[0][:2] - c30) < 0.02, f"{h} off {_de:.1e}")
h2 = T.pick(top, _px(*c30) + 4 * c30 / 0.5, edges=False)
check("snap: edges off (Ctrl / Shift / lock) leaves only points", h2 is None, f"{h2}")
h = T.pick(top, _px(0, 0) + (3, 3))
check("snap: face centre of a disk is its exact centre, the top cap over the bottom one",
      h is not None and h[1] == "center" and np.linalg.norm(h[0][:2]) < 1e-8 and abs(h[0][2] - _ztop) < 1e-9, f"{h}")
_cs = [_sn.face_center(SCY, i_) for i_ in range(3)]
check("snap: a cylinder side wraps its centroid: its centre is the axis point",
      sum(1 for c_ in _cs if c_ is not None and np.linalg.norm(c_) < 1e-5) == 1, f"{_cs}")
h = T.pick(front, _px(0, 0) + (2, -2))
check("snap: in front of the axis point, the seam edge's middle is nearer the viewer and wins",
      h is not None and h[1] == "mid" and np.linalg.norm(h[0] - (0, -0.5, 0)) < 1e-7, f"{h}")
h = T.pick(top, _px(0, -0.5) + (2, 2))
check("snap: surface edge end point (the seam vertex)", h is not None and h[1] == "end"
      and abs(np.linalg.norm(h[0][:2]) - 0.5) < 1e-9, f"{h}")
arcA = nurbs("ArcA", [(0, 0, 0), (1, 0, 0.4), (2, 0, 0)], order=3)
arcB = nurbs("ArcB", [(0, 2, 0), (1, 2, 0.4), (2, 2, 0)], order=3)
bpy.ops.nproto.loft(inputs=ins((arcA, "S0"), (arcB, "S0")))
SLF = ctx.active_object
T = _sn.collect([SLF])
h = T.pick(top, _px(1, 1) + (3, -2))
fc = store.get_brep(SLF).faces()[0]
dfc = fc.closest_point(tuple(h[0]))[1] if h else 1
check("snap: a curved face's centre lies on the face, in its middle", h is not None and h[1] == "center"
      and dfc < 1e-9 and np.linalg.norm(h[0][:2] - (1, 1)) < 1e-3 and 0.15 < h[0][2] < 0.25, f"{h} off {dfc:.1e}")
ncv = T.kinds.count("cv")
Ts = _sn.collect([SLF], lambda o: True)
nnet = sum(len(n_["p"]) for n_ in store.display(SLF)["nets"])
h = Ts.pick(front, _px(1, 0.4) + (2, 1))
check("snap: surface CVs snap only when the CV net is shown", ncv == 0 and Ts.kinds.count("cv") == nnet > 0
      and h is not None and h[1] == "cv" and np.allclose(h[0], (1, 0, 0.4), atol=1e-6), f"{ncv} {nnet} {h}")
v_ = ctx.window_manager.nproto_view
old_cv = v_.show_cvs
bpy.ops.object.select_all(action='DESELECT'); SLF.select_set(True)
v_.show_cvs = True
on = _drw.cv_net_shown(ctx, SLF)
SLF.select_set(False)
off = _drw.cv_net_shown(ctx, SLF)
v_.show_cvs = old_cv
check("snap: a surface's CV net counts as shown when CVs / Hulls is on and it is selected", on and not off)
import time as _time
T = _sn.collect([SCY, SLN, SLF], lambda o: True)
t0 = _time.time()
for k in range(50):
    T.pick(top, _px(0.01 * k, 0.3))
dt = (_time.time() - t0) / 50
check("snap: picking stays quick", dt < 0.02, f"{dt * 1e3:.2f} ms per mouse move")

# --- Line with Polyline on: every click adds a segment; clicking the first point closes it
_a, _b, _c = np.array((0.0, 0, 0)), np.array((1.0, 0, 0)), np.array((1.0, 1, 0))
_ls = [(_a, _b)]
_s1 = _dt.next_start(_ls, _b, True)
_ls.append((_b, _c))
_s2 = _dt.next_start(_ls, _c, True)
_ls.append((_c, _a))
_s3 = _dt.next_start(_ls, _a, True)
_s4 = _dt.next_start([(_a, _b)], _b, False)
check("polyline: each click starts the next segment, the first point closes it, off: no chaining",
      np.allclose(_s1, _b) and np.allclose(_s2, _c) and _s3 is None and _s4 is None, f"{_s1} {_s2} {_s3} {_s4}")
r = bpy.ops.nproto.draw_line(data=json.dumps({"lines": [[list(a_), list(b_)] for a_, b_ in _ls], "edit": ""}),
                             polyline=True)
_o = ctx.active_object
check("polyline: a closed chain of clicks becomes one closed degree-1 curve", r == {'FINISHED'}
      and len(ctx.selected_objects) == 1 and _o.data.splines[0].use_cyclic_u and len(_o.data.splines[0].points) == 3)

# --- Sketch axis locks (X / Y / Z, Shift: the plane of the other two)
P = _dt.axis_lock_point("X", (1, 2, 3), ((5, -3, 10), (0, 0, -1)))
check("axis lock X: slides along X through the previous point", np.allclose(P, (5, 2, 3)), f"{P}")
P = _dt.axis_lock_point("Z", (1, 2, 3), ((0, -10, 7), (0, 1, 0)))
check("axis lock Z: height from the mouse ray in a front view", np.allclose(P, (1, 2, 7)), f"{P}")
P = _dt.axis_lock_point("Z", (1, 2, 3), ((4, 4, 10), (0, 0, -1)))
check("axis lock Z seen end-on stays at the previous point", np.allclose(P, (1, 2, 3)), f"{P}")
P = _dt.axis_lock_point("-Z", (1, 2, 3), ((0.5, -10, 7), (0.1, 1, -0.2)))
check("plane lock Shift+Z: stays at the previous point's height", abs(P[2] - 3) < 1e-12, f"{P}")
P = _dt.axis_lock_point("-X", (1, 2, 3), ((4, 4, 10), (0, 0, -1)), plane_point=(4.0, 4.0, 0.0))
check("plane lock Shift+X seen edge-on: the drawing-plane point projected into it",
      np.allclose(P, (1, 4, 0)), f"{P}")
P = _dt.axis_lock_point("X", (1, 2, 3), ((0, 0, 10), (0, 0, -1)), snapped=(3.3, 7, -1))
check("axis lock X: a snapped point is projected onto the lock line", np.allclose(P, (3.3, 2, 3)), f"{P}")
P = _dt.axis_lock_point("-Y", (0, 5, 0), ((0.26, 0, 0.74), (0, 1, 0)), step=0.25)
check("plane lock with Ctrl: grid steps from the previous point", np.allclose(P, (0.25, 5, 0.75)), f"{P}")

# --- CV Curve start / end conditions (G0 / G1 / G2 to the snapped curve or
#     surface edge) and the Align history step they add
from blender_nurbs import align as _al
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
ctx.scene.nproto_settings.construction_history = True
TGT = nurbs("CondTarget", [(-3, 0, 0), (-2, 0.5, 0), (-1, 1.5, 0.3), (0, 1, 0)])
def _draw(pts, start=None, end=None, **kw):
    kw.setdefault("keep_aligned", True)
    r_ = bpy.ops.nproto.draw_curve(data=json.dumps({"pts": pts, "start": start, "end": end, "edit": ""}), **kw)
    return r_, ctx.active_object
def _cvs(o):
    M_ = np.array(geom.world_matrix(o))
    return np.array([M_[:3, :3] @ np.array(p_.co[:3]) + M_[:3, 3] for p_ in o.data.splines[0].points])
clk = [(0, 1, 0), (1, 0.4, 0.2), (2, 1.2, 0.0), (3, 0.5, -0.2)]
r, CG2 = _draw(clk, [TGT.name, "S0", "END"], None, mode="CV", degree="3", start_cont="G2", end_cont="G1")
gap, ang, dk = _al.measure(CG2, 0, TGT, "S0", "END")
h_ = features.history(CG2)
check("cv curve G2 start on a curve end: continues it (gap, angle, curvature)", r == {'FINISHED'}
      and gap < 1e-6 and ang < 1e-4 and dk < 1e-6, f"{gap:.1e} {ang:.1e}° {dk:.1e}")
check("cv curve G2: an Align history step is added at once", h_ is not None and h_["op"] == "align_curve"
      and h_["params"]["ends"]["0"]["cont"] == "G2" and "1" not in h_["params"]["ends"], f"{h_}")
X0 = _cvs(CG2)
check("cv curve G2: only the end CVs moved, the others are the clicks",
      np.allclose(X0[3], clk[3], atol=1e-6) and np.linalg.norm(X0[1] - clk[1]) > 1e-3, f"{X0}")
TGT.data.splines[0].points[2].co = (-1, 2.2, 0.6, 1.0)
features.rebuild(CG2, ctx)
gap, ang, dk = _al.measure(CG2, 0, TGT, "S0", "END")
X1 = _cvs(CG2)
check("cv curve G2: moving the target keeps the start continuous (history)", gap < 1e-6 and ang < 1e-4 and dk < 1e-3
      and np.allclose(X1[3], X0[3], atol=1e-6) and np.linalg.norm(X1[2] - X0[2]) > 1e-3,
      f"{gap:.1e} {ang:.1e}° {dk:.1e}")
r, CN = _draw(clk, [TGT.name, "S0", "END"], None, mode="CV", degree="3", start_cont="G2", end_cont="G1", keep_aligned=False)
gap, ang, dk = _al.measure(CN, 0, TGT, "S0", "END")
check("cv curve G2, Keep Aligned off: aligned once, no history", r == {'FINISHED'} and features.history(CN) is None
      and gap < 1e-6 and ang < 1e-4 and dk < 1e-3, f"{gap:.1e} {ang:.1e}° {dk:.1e}")
r, C0 = _draw(clk, [TGT.name, "S0", "END"], None, mode="CV", degree="3", start_cont="G0", end_cont="G0")
check("cv curve G0 on a curve end: no history, the clicks are the CVs", r == {'FINISHED'}
      and features.history(C0) is None and np.allclose(_cvs(C0), clk, atol=1e-6))
r, C2 = _draw([(0, 1, 0), (1.5, 2, 0.5)], [TGT.name, "S0", "END"], None, mode="CV", degree="3", start_cont="G2")
gap, ang, dk = _al.measure(C2, 0, TGT, "S0", "END")
check("cv curve G2 with only 2 clicks: more CVs (and degree 3) so it can hold", r == {'FINISHED'}
      and len(C2.data.splines[0].points) >= 3 and C2.data.splines[0].order_u == 4 and gap < 1e-6 and ang < 1e-4,
      f"{len(C2.data.splines[0].points)} CVs {gap:.1e} {ang:.1e}°")
# along a curve (NEAREST): the new curve leaves tangent to it, the way it was drawn
kc_ = features.input_curve(TGT, "S0")
t0_, t1_ = kc_.parameter_range()
Pm = np.array(kc_.evaluate(0.5 * (t0_ + t1_)), float)
Tm = np.array(_al._k()[0].evaluate_curve(kc_, 0.5 * (t0_ + t1_))["tangent"], float)
side = Pm + np.cross(Tm, (0, 0, 1)) / np.linalg.norm(Tm)
r, CN = _draw([tuple(Pm), tuple(side - 0.3 * Tm / np.linalg.norm(Tm)), tuple(side - Tm)],
              [TGT.name, "S0", "NEAREST"], None, mode="CV", degree="3", start_cont="G1")
X = _cvs(CN)
d1 = (X[1] - X[0]) / np.linalg.norm(X[1] - X[0])
tm = Tm / np.linalg.norm(Tm)
check("cv curve G1 along a curve: leaves along its tangent, in the drawn direction (flipped)",
      r == {'FINISHED'} and np.linalg.norm(np.cross(d1, tm)) < 1e-6 and d1 @ tm < 0
      and features.history(CN)["params"]["ends"]["0"]["flip"], f"{d1} {tm}")
# a curved surface edge, Through Points: tangent to the surface, its normal
# curvature matched (G2), still through every click
arcA = nurbs("ArcA", [(0, 0, 0), (1, 0, 0.4), (2, 0, 0)], order=3)
arcB = nurbs("ArcB", [(0, 2, 0), (1, 2, 0.4), (2, 2, 0)], order=3)
bpy.ops.nproto.loft(inputs=ins((arcA, "S0"), (arcB, "S0")))
ARC = ctx.active_object
dE = store.display(ARC)["edges"]
ei = min(range(len(dE)), key=lambda i_: abs(np.mean(np.array(dE[i_])[:, 1])) + 10 * (np.ptp(np.array(dE[i_])[:, 2]) < 1e-6))
clk = [(0.5, 0.0, 0.15), (1.1, -0.6, 0.35), (1.6, -1.4, 0.3), (2.0, -2.0, 0.0)]
r, CS = _draw(clk, [ARC.name, f"E{ei}", "NEAREST"], None, mode="THROUGH", degree="3", start_cont="G2",
              edge_mode="SURFACE", start_dir="FREE")      # (Free: the clicks lead it across the arc, curved)
kcs = features.input_curve(CS, "S0")
ds = _al._k()[0].evaluate_curve(kcs, kcs.parameter_range()[0])
c1, c2 = np.array(ds["tangent"], float), np.array(ds["curvature"], float)
nrm, k_n = _al.surface_frame(ARC, f"E{ei}", np.array(ds["point"], float))
tilt = math.degrees(math.asin(min(1.0, abs(c1 @ nrm) / np.linalg.norm(c1))))
kc_curve = (c2 @ nrm) / (c1 @ c1)
kc_surf = k_n(c1 / np.linalg.norm(c1))
check("cv curve through points, G2 on a surface edge: tangent to the surface, normal curvature matched",
      r == {'FINISHED'} and tilt < 1e-5 and abs(kc_surf) > 0.02 and abs(kc_curve - kc_surf) < 1e-5 * abs(kc_surf)
      and np.linalg.norm(np.array(ds["point"]) - clk[0]) < 1e-6, f"tilt {tilt:.1e}° k {kc_curve:.5f} vs {kc_surf:.5f}")
d = _passes(CS, clk)
check("cv curve through points with G2: still passes through every click", d < 5e-5, f"{d:.1e}")
Xb = _cvs(CS)
kind_, deg_, Qs, _ = _al.solve(CS, features.history(CS)["params"]["ends"],
                               [(CS, "S0"), (ARC, f"E{ei}")])
M_ = np.array(geom.world_matrix(CS))
dq = np.abs(Qs @ M_[:3, :3].T + M_[:3, 3] - Xb).max()
check("cv curve: the recorded Align step rebuilds the drawn curve unchanged", dq < 1e-6, f"{dq:.1e}")
ctx.scene.nproto_settings.construction_history = False
r, CH = _draw(clk, [ARC.name, f"E{ei}", "NEAREST"], None, mode="THROUGH", degree="3", start_cont="G1")
ctx.scene.nproto_settings.construction_history = True
check("cv curve with history off: aligned once, no history", r == {'FINISHED'} and features.history(CH) is None)

# --- The sketch tools driven through their modal code (fake top view, mouse
#     events): snap -> reference (edge / face, T cycles) -> G1 end / tangent line
import types as _types
from mathutils import Matrix as _Mx, Quaternion as _Qt
class _FArea:
    def __init__(self, region, rv3d):
        self.regions = [region]
        self.spaces = _types.SimpleNamespace(active=_types.SimpleNamespace(region_3d=rv3d))
        self.type = 'VIEW_3D'
        self.text = None
    def header_text_set(self, t):
        self.text = t
    def tag_redraw(self):
        pass
class _FCtx:
    """bpy.context with a top orthographic 3D view of 1000 px = 4 units."""
    def __init__(self):
        V = _Mx.Translation((0, 0, -10))
        Wm = _Mx(((0.5, 0, 0, 0), (0, 0.5, 0, 0), (0, 0, -0.01, 0), (0, 0, 0, 1)))
        self.region = _types.SimpleNamespace(x=0, y=0, width=1000, height=1000, type='WINDOW')
        self.region_data = _types.SimpleNamespace(view_matrix=V, perspective_matrix=Wm @ V, window_matrix=Wm,
                                                  view_rotation=_Qt(), is_perspective=False, view_distance=10.0,
                                                  view_perspective='ORTHO')
        self.area = _FArea(self.region, self.region_data)
    def __getattr__(self, k):
        return getattr(bpy.context, k)
def _ev(type_, xy=(0, 0), value='PRESS', shift=False, ctrl=False):
    return _types.SimpleNamespace(type=type_, value=value, mouse_x=float(xy[0]), mouse_y=float(xy[1]), shift=shift,
                                  ctrl=ctrl, alt=False, oskey=False, unicode="")
_wpx = lambda x, y: (500 + 250 * x, 500 + 250 * y)
def _fake(op_cls, **props):
    cls = type("Fake" + op_cls.__name__, (_dt.SketchMixin,), {})
    for k_, v_ in op_cls.__dict__.items():
        if callable(v_) or isinstance(v_, staticmethod):
            if k_ not in ("passthrough", "invoke", "__init__") and not k_.startswith("bl_"):
                setattr(cls, k_, v_)
    cls.passthrough = lambda self, c, e: False
    cls.report = lambda self, t, m: self.msgs.append(m)
    f = cls()
    f.msgs, f.data = [], ""
    for k_, v_ in getattr(op_cls, "__annotations__", {}).items():   # the operator's own defaults (Show CVs / Comb)
        if "default" in getattr(v_, "keywords", {}):
            setattr(f, k_, v_.keywords["default"])
    for k_, v_ in props.items():
        setattr(f, k_, v_)
    return f
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
ctx.scene.cursor.location = (0, 0, 0)
ctx.scene.nproto_settings.construction_history = True
bpy.ops.nproto.add_primitive(kind='CYLINDER', radius=0.5, height=1.2, caps=True)
MCY = ctx.active_object
fc = _FCtx()
rim = 0.5 * np.array([math.cos(math.radians(30)), math.sin(math.radians(30))])
# (Free: Square would run the curve down the cylinder's side, into the Line checks below)
cv = _fake(_dt.NPROTO_OT_draw_curve, mode="CV", degree="3", closed=False, start_cont="G1", end_cont="G1",
           edge_mode="SURFACE", keep_aligned=True, start_dir="FREE", end_dir="FREE")
cv._edit, cv._pts, cv._snapinfo, cv._frames, cv._start_refs = None, [], [], {}, None
cv.sketch_start(fc)
cv.modal(fc, _ev('MOUSEMOVE', _wpx(*rim), 'NOTHING'))
lab1 = _drw.sketch_overlay["snap_label"]
nref = len(_drw.sketch_overlay["refs"])
src1 = cv._refs["cands"][cv._refs["i"]]["src"] if cv._refs else None
cv.modal(fc, _ev('T', _wpx(*rim)))
lab2 = _drw.sketch_overlay["snap_label"]
src2 = cv._refs["cands"][cv._refs["i"]]["src"] if cv._refs else None
check("sketch modal: a rim point offers both faces along the edge, T switches face (highlighted)",
      lab1 == "Edge · Face 1/2 (T)" and lab2 == "Edge · Face 2/2 (T)" and nref >= 2 and src1 and src2
      and src1[3] != src2[3] and src1[:2] == src2[:2], f"{lab1} / {lab2} {src1} {src2}")
cv.modal(fc, _ev('LEFTMOUSE', _wpx(*rim)))
cv.modal(fc, _ev('LEFTMOUSE', _wpx(0.95, 0.55)))
cv.modal(fc, _ev('LEFTMOUSE', _wpx(1.4, 0.3)))
cv.modal(fc, _ev('LEFTMOUSE', _wpx(1.9, 0.5)))
r = cv.modal(fc, _ev('RET'))
MO = ctx.active_object
h_ = features.history(MO) if MO is not None and MO.type == 'CURVE' else None
e0 = h_["params"]["ends"]["0"] if h_ else {}
kcs = features.input_curve(MO, "S0") if h_ else None
if kcs is not None:
    ds = _al._k()[0].evaluate_curve(kcs, kcs.parameter_range()[0])
    nrm, _kn = _al.surface_frame(MCY, src2[1], np.array(ds["point"], float), src2[3])
    tilt = math.degrees(math.asin(min(1.0, abs(np.array(ds["tangent"]) @ nrm) / np.linalg.norm(ds["tangent"]))))
else:
    tilt = 99
check("sketch modal: clicked CV curve is G1 to the face chosen with T, with an Align history step",
      r == {'FINISHED'} and h_ is not None and e0.get("face") == src2[3] and e0.get("mode") == "SURFACE"
      and e0.get("cont") == "G1" and tilt < 1e-5, f"{r} {e0} tilt {tilt:.1e}° {cv.msgs}")
# Line: G1 start on the rim = tangent to the edge there
ln = _fake(_dt.NPROTO_OT_draw_line, polyline=False, start_mode="TANGENT", line_mode="EDGE")
ln._edit, ln._lines, ln._start, ln._lock, ln._start_refs = None, [], None, None, None
ln._reset_typing()
ln.sketch_start(fc)
ln.modal(fc, _ev('MOUSEMOVE', _wpx(*rim), 'NOTHING'))
ln.modal(fc, _ev('LEFTMOUSE', _wpx(*rim)))
lock1 = ln._lock
ln.modal(fc, _ev('MOUSEMOVE', _wpx(0.9, 1.2), 'NOTHING'))
ln.modal(fc, _ev('LEFTMOUSE', _wpx(0.9, 1.2)))
a_, b_ = ln._lines[0] if ln._lines else (np.zeros(3), np.zeros(3))
dirn = (b_ - a_) / (np.linalg.norm(b_ - a_) or 1)
check("sketch modal: a Line started on the rim with S (G1) runs tangent to the rim",
      lock1 == "T" and abs(dirn @ (a_ * (1, 1, 0)) / np.linalg.norm(a_[:2])) < 1e-6 and abs(dirn[2]) < 1e-9
      and np.linalg.norm(b_ - a_) > 0.1, f"{lock1} {dirn}")
r = ln.modal(fc, _ev('RIGHTMOUSE'))
check("sketch modal: the tangent line is made", r == {'FINISHED'} and ctx.active_object.type == 'CURVE')
# Line at the seam corner: two edges meet (the rim, the seam); T switches which one the tangent follows
ln = _fake(_dt.NPROTO_OT_draw_line, polyline=False, start_mode="TANGENT", line_mode="EDGE")
ln._edit, ln._lines, ln._start, ln._lock, ln._start_refs = None, [], None, None, None
ln._reset_typing()
ln.sketch_start(fc)
seam = None
for e_ in store.get_brep(MCY).edges():
    p0_ = np.array(e_.point_at(e_.parameter_range()[0]))
    if p0_[2] > 0.5:
        seam = p0_
        break
ln.modal(fc, _ev('MOUSEMOVE', _wpx(seam[0], seam[1]), 'NOTHING'))
labc = _drw.sketch_overlay["snap_label"]
ln.modal(fc, _ev('LEFTMOUSE', _wpx(seam[0], seam[1])))
t1 = np.array(ln._tvec) / np.linalg.norm(ln._tvec) if getattr(ln, "_tvec", None) is not None else None
ln.modal(fc, _ev('T', _wpx(seam[0], seam[1])))
t2 = np.array(ln._tvec) / np.linalg.norm(ln._tvec) if getattr(ln, "_tvec", None) is not None else None
vert = lambda t: t is not None and abs(abs(t[2]) - 1) < 1e-6
horiz = lambda t: t is not None and abs(t[2]) < 1e-6
check("sketch modal: at a corner T switches between the edges meeting there (rim tangent / seam)",
      labc is not None and labc.startswith("End · Edge 1/") and ((vert(t1) and horiz(t2)) or (horiz(t1) and vert(t2))),
      f"{labc} {t1} {t2}")
ln.modal(fc, _ev('ESC'))
# Polyline: click, click, click, click the first point -> closed, then right-click
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
pl_ = _fake(_dt.NPROTO_OT_draw_line, polyline=True, start_mode="FREE", line_mode="EDGE")
pl_._edit, pl_._lines, pl_._start, pl_._lock, pl_._start_refs = None, [], None, None, None
pl_._reset_typing()
pl_.sketch_start(fc)
for q_ in ((0, 0), (1, 0), (1, 1), (0, 0)):
    pl_.modal(fc, _ev('MOUSEMOVE', _wpx(*q_), 'NOTHING'))
    pl_.modal(fc, _ev('LEFTMOUSE', _wpx(*q_)))
n_l, st_ = len(pl_._lines), pl_._start
r = pl_.modal(fc, _ev('RIGHTMOUSE'))
o_ = ctx.active_object
check("sketch modal: polyline clicks chain, clicking the first point closes it into one curve",
      n_l == 3 and st_ is None and r == {'FINISHED'} and o_ is not None and o_.data.splines[0].use_cyclic_u
      and len(o_.data.splines[0].points) == 3, f"{n_l} {st_} {r}")

# --- At a corner shared by two edges: the edge slid along stays the reference;
#     L locks the point onto one edge
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.nproto.add_primitive(kind="PLANE", width=2, depth=2, pl_degree_u=1, pl_degree_v=1, spans_u=1, spans_v=1)
def _new_ln():
    f = _fake(_dt.NPROTO_OT_draw_line, polyline=False, start_mode="TANGENT", line_mode="EDGE")
    f._edit, f._lines, f._start, f._lock, f._start_refs = None, [], None, None, None
    f._reset_typing()
    f.sketch_start(fc)
    return f
def _corner_from(path):
    f = _new_ln()
    for q_ in path:
        f.modal(fc, _ev('MOUSEMOVE', _wpx(*q_), 'NOTHING'))
    lab_ = _drw.sketch_overlay["snap_label"]
    f.modal(fc, _ev('LEFTMOUSE', _wpx(*path[-1])))
    t_ = np.array(f._tvec) / np.linalg.norm(f._tvec) if f._lock == "T" else np.zeros(3)
    f.modal(fc, _ev('ESC'))
    return lab_, t_
lab_a, ta = _corner_from([(0.2, 1), (0.6, 1), (0.9, 1), (1, 1)])
lab_b, tb = _corner_from([(1, 0.2), (1, 0.6), (1, 0.9), (1, 1)])
check("corner: the edge slid along into a shared end stays the reference (top edge -> X, right edge -> Y)",
      abs(abs(ta[0]) - 1) < 1e-9 and abs(abs(tb[1]) - 1) < 1e-9 and lab_a == lab_b == "End · Edge 1/2 (T)",
      f"{lab_a} {ta} / {lab_b} {tb}")
f = _new_ln()
f.modal(fc, _ev('MOUSEMOVE', _wpx(0.3, 1), 'NOTHING'))
lab0 = _drw.sketch_overlay["snap_label"]
f.modal(fc, _ev('L', _wpx(0.3, 1)))
f.modal(fc, _ev('MOUSEMOVE', _wpx(0.3, 0.2), 'NOTHING'))
P_l, sn_l = f.pick_point(fc, _ev('MOUSEMOVE', _wpx(0.3, 0.2), 'NOTHING'))
check("lock onto an edge (L): the point stays on the edge wherever the mouse goes",
      lab0 == "Edge" and np.allclose(P_l, (0.3, 1, 0), atol=1e-6) and sn_l is not None
      and sn_l[3] == "Edge · locked (L)", f"{lab0} {P_l} {sn_l}")
f.modal(fc, _ev('MOUSEMOVE', _wpx(1.0, 0.97), 'NOTHING'))
lab1 = _drw.sketch_overlay["snap_label"]
f.modal(fc, _ev('LEFTMOUSE', _wpx(1.0, 0.97)))
t1 = np.array(f._tvec) / np.linalg.norm(f._tvec) if f._lock == "T" else np.zeros(3)
check("lock onto an edge: at the shared corner only that edge counts; the click releases the lock",
      lab1 == "End · locked (L)" and np.allclose(f._start, (1, 1, 0), atol=1e-9) and abs(abs(t1[0]) - 1) < 1e-9
      and f._edge_lock is None, f"{lab1} {f._start} {t1}")
f.modal(fc, _ev('ESC'))

# --- Profile: circle / ellipse / square / rectangle perpendicular to a curve or
#     surface edge, following it (history)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
PPATH = nurbs("PPath", [(-1, 0, 0), (1, 0, 0)], order=2)
def _type(f, ch, type_, xy):
    return f.modal(fc, _types.SimpleNamespace(type=type_, value='PRESS', mouse_x=float(xy[0]), mouse_y=float(xy[1]),
                                              shift=False, ctrl=False, alt=False, oskey=False, unicode=ch))
def _new_pf(**kw):
    props = dict(shape="CIRCLE", radius=0.25, radius2=0.15, width=0.5, depth=0.3, at=0.0, rotation=0.0,
                 degree="0", spans_u=8)
    props.update(kw)
    f = _fake(_dt.NPROTO_OT_draw_profile, **props)
    f._center = f._src = f._frame = None
    f._lock = None
    f._buf, f._field, f._s1, f._s2 = "", 1, None, None
    f.sketch_start(fc)
    return f
def _samples(o, n=24):
    kc_ = features.input_curve(o, "S0")
    a_, b_ = kc_.parameter_range()
    return np.array([kc_.evaluate(a_ + (b_ - a_) * k / n) for k in range(n)], float)
M_ = _pt_frame = __import__("blender_nurbs.primitives", fromlist=["x"]).profile_frame(PPATH, "S0", 0.5)
check("profile frame on a line: Z along the line, Y up, at the middle",
      abs(abs(M_[0, 2]) - 1) < 1e-9 and np.allclose(M_[:3, 1], (0, 0, 1)) and np.allclose(M_[:3, 3], (0, 0, 0)),
      f"{M_}")
pf = _new_pf()
pf.modal(fc, _ev('MOUSEMOVE', _wpx(0.3, 0), 'NOTHING'))
prev_ok = bool(_drw.sketch_overlay["curves"])
pf.modal(fc, _ev('LEFTMOUSE', _wpx(0.3, 0)))
at_ = pf.at
pf.modal(fc, _ev('MOUSEMOVE', _wpx(0.3, 0.4), 'NOTHING'))
r = pf.modal(fc, _ev('LEFTMOUSE', _wpx(0.3, 0.4)))
PCI = ctx.active_object
S_ = _samples(PCI) if r == {'FINISHED'} else np.zeros((3, 3))
h_ = features.history(PCI) if r == {'FINISHED'} else None
check("profile (circle) on a curve: centred on it, perpendicular, radius dragged out, history to the curve",
      prev_ok and r == {'FINISHED'} and abs(at_ - 0.65) < 1e-6 and np.allclose(S_[:, 0], 0.3, atol=1e-6)
      and np.allclose(np.linalg.norm(S_ - (0.3, 0, 0), axis=1), 0.4, atol=1e-5) and h_ is not None
      and h_["op"] == "perp_profile" and h_["inputs"][0][0] == PPATH, f"{r} {at_} {S_[:2]}")
PPATH.location = (0, 0.5, 0.2)
features.rebuild(PCI, ctx)
S_ = _samples(PCI)
check("profile: moving the curve moves the profile with it (still perpendicular, same size)",
      np.allclose(S_.mean(axis=0), (0.3, 0.5, 0.2), atol=1e-6) and np.allclose(S_[:, 0], 0.3, atol=1e-6)
      and np.allclose(np.linalg.norm(S_ - (0.3, 0.5, 0.2), axis=1), 0.4, atol=1e-5), f"{S_.mean(axis=0)}")
PCI.nproto_feature.radius = 0.2
S_ = _samples(PCI)
r2_ = np.linalg.norm(S_ - (0.3, 0.5, 0.2), axis=1)
PCI.nproto_feature.pp_shape = "SQUARE"
PCI.nproto_feature.width = 0.3
cv_ = np.array([(PCI.matrix_world @ Vector(p_.co[:3]))[:] for p_ in PCI.data.splines[0].points])
check("profile: Radius / Shape / Size edited in the Feature panel rebuild it in place",
      np.allclose(r2_, 0.2, atol=1e-5) and len(cv_) == 4 and np.allclose(cv_[:, 0], 0.3, atol=1e-6)
      and np.allclose(np.abs(cv_[:, 1] - 0.5), 0.15, atol=1e-6) and np.allclose(np.abs(cv_[:, 2] - 0.2), 0.15, atol=1e-6),
      f"{r2_[:2]} {cv_}")
PPATH.location = (0, 0, 0)
bpy.ops.nproto.add_primitive(kind="PLANE", width=2, depth=2, pl_degree_u=1, pl_degree_v=1, spans_u=1, spans_v=1)
pf = _new_pf()
pf.modal(fc, _ev('MOUSEMOVE', _wpx(0.2, 1), 'NOTHING'))
lab_ = _drw.sketch_overlay["snap_label"]
pf.modal(fc, _ev('S', _wpx(0.2, 1)))
pf.modal(fc, _ev('LEFTMOUSE', _wpx(0.2, 1)))
_type(pf, "0", 'ZERO', _wpx(0.2, 1.1))
_type(pf, ".", 'PERIOD', _wpx(0.2, 1.1))
_type(pf, "5", 'FIVE', _wpx(0.2, 1.1))
r = _type(pf, "", 'RET', _wpx(0.2, 1.1))
PSQ = ctx.active_object
cv_ = np.array([(PSQ.matrix_world @ Vector(p_.co[:3]))[:] for p_ in PSQ.data.splines[0].points]) if r == {'FINISHED'} \
    else np.zeros((4, 3))
check("profile (square, typed 0.5) on a surface edge: square to the surface (sides along its normal)",
      r == {'FINISHED'} and len(cv_) == 4 and np.allclose(cv_[:, 0], 0.2, atol=1e-6)
      and np.allclose(np.abs(cv_[:, 1] - 1), 0.25, atol=1e-6) and np.allclose(np.abs(cv_[:, 2]), 0.25, atol=1e-6)
      and features.history(PSQ)["params"]["pp_shape"] == "SQUARE", f"{lab_} {r} {cv_}")

# --- Draw Plane: two corners or typed X / Y; degree and spans per direction
C_, X_, Y_, sg_ = _dt.plane_rect((0, 0, 0), (1.5, -0.8, 0), (1, 0, 0), (0, 1, 0))
check("draw plane: two corners give the centre and the X / Y sizes",
      np.allclose(C_, (0.75, -0.4, 0)) and abs(X_ - 1.5) < 1e-12 and abs(Y_ - 0.8) < 1e-12 and sg_ == (1.0, -1.0))
C_, X_, Y_, _ = _dt.plane_rect((1, 1, 0), (2, 1.5, 0), (1, 0, 0), (0, 1, 0), from_center=True)
C2_, X2_, Y2_, _ = _dt.plane_rect((0, 0, 0), (1, 0.4, 0), (1, 0, 0), (0, 1, 0), square=True)
C3_, X3_, Y3_, _ = _dt.plane_rect((0, 0, 0), (-0.3, 0.2, 0), (1, 0, 0), (0, 1, 0), size=(3.0, None))
check("draw plane: from the centre, square, typed X (the mouse gives the side)",
      np.allclose(C_, (1, 1, 0)) and np.allclose((X_, Y_), (2, 1)) and np.allclose(C2_, (0.5, 0.5, 0))
      and np.allclose((X2_, Y2_), (1, 1)) and np.allclose(C3_, (-1.5, 0.1, 0)) and np.allclose((X3_, Y3_), (3, 0.2)))
Mf_ = _dt.plane_matrix((0, 0, 0), (1, 0, 0), (0, 0, 1))
check("draw plane: drawn in a front view it stands up, its normal towards the viewer (-Y)",
      np.allclose(Mf_[:3, 2], (0, -1, 0)) and abs(np.linalg.det(Mf_[:3, :3]) - 1) < 1e-12)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
def _new_dp(**kw):
    props = dict(size_x=2.0, size_y=2.0, from_center=False, degree_u=3, degree_v=2, spans_u=2, spans_v=1)
    props.update(kw)
    f = _fake(_dt.NPROTO_OT_draw_plane, **props)
    f._start, f._lock = None, None
    f._reset_typing()
    f.sketch_start(fc)
    return f
def _surf_info(o):
    d_ = ext.surface_cvs(store.get_surfaces(o)[0])
    M_ = np.array(geom.world_matrix(o))
    W_ = np.array(d_["points"], float) @ M_[:3, :3].T + M_[:3, 3]
    return d_["count_u"], d_["count_v"], d_["degree_u"], d_["degree_v"], W_.min(axis=0), W_.max(axis=0)
dp = _new_dp()
dp.modal(fc, _ev('MOUSEMOVE', _wpx(0.2, 0.1), 'NOTHING'))
dp.modal(fc, _ev('LEFTMOUSE', _wpx(0.2, 0.1)))
dp.modal(fc, _ev('MOUSEMOVE', _wpx(1.4, 0.9), 'NOTHING'))
r = dp.modal(fc, _ev('LEFTMOUSE', _wpx(1.4, 0.9)))
PL = ctx.active_object
nu_, nv_, du_, dv_, lo_, hi_ = _surf_info(PL) if r == {'FINISHED'} else (0, 0, 0, 0, np.zeros(3), np.zeros(3))
hp_ = features.history(PL)["params"] if r == {'FINISHED'} else {}
check("draw plane (clicks): corner to corner, degree 3 x 2 with 2 x 1 spans -> 5 x 3 CVs, in the drawing plane",
      r == {'FINISHED'} and (nu_, nv_, du_, dv_) == (5, 3, 3, 2) and np.allclose(lo_, (0.2, 0.1, 0), atol=1e-6)
      and np.allclose(hi_, (1.4, 0.9, 0), atol=1e-6) and abs(hp_.get("width", 0) - 1.2) < 1e-6
      and abs(hp_.get("depth", 0) - 0.8) < 1e-6, f"{r} {(nu_, nv_, du_, dv_)} {lo_} {hi_} {hp_}")
def _type(f, ch, type_, xy):
    return f.modal(fc, _types.SimpleNamespace(type=type_, value='PRESS', mouse_x=float(xy[0]), mouse_y=float(xy[1]),
                                              shift=False, ctrl=False, alt=False, oskey=False, unicode=ch))
dp = _new_dp(degree_u=3, degree_v=3, spans_u=1, spans_v=1)
q0 = _wpx(0, 0)
dp.modal(fc, _ev('LEFTMOUSE', q0))
qm = _wpx(-0.5, 0.5)
_type(dp, "2", 'TWO', qm)
_type(dp, "", 'TAB', qm)
_type(dp, "1", 'ONE', qm)
r = _type(dp, "", 'RET', qm)
PT = ctx.active_object
nu_, nv_, du_, dv_, lo_, hi_ = _surf_info(PT) if r == {'FINISHED'} else (0, 0, 0, 0, np.zeros(3), np.zeros(3))
check("draw plane (typed): X 2, Tab, Y 1, Enter -> 2 x 1 on the side of the mouse, 4 x 4 CVs",
      r == {'FINISHED'} and (nu_, nv_) == (4, 4) and np.allclose(lo_, (-2, 0, 0), atol=1e-6)
      and np.allclose(hi_, (0, 1, 0), atol=1e-6), f"{r} {lo_} {hi_}")
PL.nproto_feature.pl_degree_u = 5
nu_, nv_, du_, dv_, lo_, hi_ = _surf_info(PL)
check("plane: Degree U changed in the Feature panel rebuilds it, same size", (nu_, du_, nv_, dv_) == (7, 5, 3, 2)
      and np.allclose(lo_, (0.2, 0.1, 0), atol=1e-6) and np.allclose(hi_, (1.4, 0.9, 0), atol=1e-6), f"{nu_} {du_}")
_old = ext.surface_cvs(features.run("prim_plane", [], {"width": 1.0, "depth": 1.0, "degree": 1, "spans_u": 1,
                                                       "spans_v": 1}))
check("plane: one made before Degree U / V (one degree) still builds the same", _old["degree_u"] == 1
      and _old["degree_v"] == 1 and _old["count_u"] == 2)

# --- Loft: Closed (periodic through the sections and back to the first)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
def _ring_square(name, a_deg, R=2.0, h=0.2):
    a = math.radians(a_deg)
    c = np.array([R * math.cos(a), R * math.sin(a), 0.0])
    X = np.array([-math.cos(a), -math.sin(a), 0.0])
    Z = np.array([0.0, 0.0, 1.0])
    pts = [c - h * X - h * Z, c + h * X - h * Z, c + h * X + h * Z, c - h * X + h * Z]
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('NURBS'); sp.points.add(3)
    for p_, q_ in zip(sp.points, pts):
        p_.co = (*q_, 1.0)
    sp.order_u = 2; sp.use_cyclic_u = True
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o)
    return o, pts
sq = [_ring_square(f"RS{k}", 90 * k) for k in range(4)]
order = [0, 2, 1, 3]                       # picked across the ring: the loft must go round it
r = bpy.ops.nproto.loft(inputs=ins(*[(sq[k][0], "S0") for k in order]), closed=True)
RL = ctx.active_object
def _raw(o):
    """The loft's surface before it is split at its folds."""
    h_ = features.history(o)
    return features._build_loft(h_["inputs"], h_["params"])
def _holds(o, sections, n=12):
    """Largest distance between each section curve and the surface's iso curve
    through it (U runs along the loft, V along the sections)."""
    srf = _raw(o)
    (u0, u1), (v0, v1) = srf.u_domain(), srf.v_domain()
    worst = 0.0
    for so in sections:
        kc_ = features.input_curve(so, "S0")
        a_, b_ = kc_.parameter_range()
        C = [np.array(kc_.evaluate(a_ + (b_ - a_) * j / n), float) for j in range(n + 1)]
        f = lambda u: np.linalg.norm(np.array(srf.evaluate(u, v0), float) - C[0])
        us = np.linspace(u0, u1, 801)
        k = int(np.argmin([f(u) for u in us]))
        lo, hi = us[max(k - 1, 0)], us[min(k + 1, 800)]
        for _ in range(80):                          # golden section on the start point
            m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
            lo, hi = (lo, m2) if f(m1) < f(m2) else (m1, hi)
        u = 0.5 * (lo + hi)
        worst = max(worst, max(np.linalg.norm(np.array(srf.evaluate(u, v0 + (v1 - v0) * j / n), float) - C[j])
                               for j in range(n + 1)))
    return worst
def _seam(o):
    """(gap, tangent jump, curvature jump) where the loft direction (U) closes."""
    srf = _raw(o)
    (u0, u1), (v0, v1) = srf.u_domain(), srf.v_domain()
    hh = 1e-4 * (u1 - u0)
    gap = d1 = d2 = 0.0
    for k in range(7):
        v = v0 + (v1 - v0) * (k + 0.5) / 7
        S = lambda u: np.array(srf.evaluate(u, v), float)
        a0, a1, a2 = S(u0), S(u0 + hh), S(u0 + 2 * hh)
        b0, b1, b2 = S(u1), S(u1 - hh), S(u1 - 2 * hh)
        gap = max(gap, np.linalg.norm(a0 - b0))
        t0, t1 = (a1 - a0) / hh, (b0 - b1) / hh
        d1 = max(d1, np.linalg.norm(t0 - t1) / (np.linalg.norm(t0) or 1))
        c0, c1 = (a2 - 2 * a1 + a0) / hh ** 2, (b2 - 2 * b1 + b0) / hh ** 2
        d2 = max(d2, np.linalg.norm(c0 - c1) / (np.linalg.norm(c0) + 1e-9))
    return gap, d1, d2
dmax = _holds(RL, [o for o, _ in sq]) if r == {'FINISHED'} else 1
gap, d1, d2 = _seam(RL) if r == {'FINISHED'} else (1, 1, 1)
check("loft closed: through all four sections (given out of order) and round to the first",
      r == {'FINISHED'} and dmax < 1e-6 and gap < 1e-9, f"{r} {dmax:.1e} gap {gap:.1e}")
check("loft closed: smooth across the seam (tangent and curvature continuous)", d1 < 1e-3 and d2 < 1e-2,
      f"{d1:.1e} {d2:.1e}")
RL.nproto_feature.closed = False
gap_o, _, _ = _seam(RL)
check("loft: switching Closed off in the Feature panel gives the open loft again", gap_o > 0.5, f"{gap_o:.2f}")
circ = []
for k in range(4):
    a = math.radians(90 * k)
    bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.3, degree="0")
    o = ctx.active_object
    o.location = (2 * math.cos(a), 2 * math.sin(a), 0.0)
    o.rotation_euler = (math.pi / 2, 0.0, a)
    circ.append(o)
ctx.view_layer.update()
r = bpy.ops.nproto.loft(inputs=ins(*[(o, "S0") for o in circ]), closed=True)
RC = ctx.active_object
dmax = _holds(RC, circ) if r == {'FINISHED'} else 1
gap, d1, d2 = _seam(RC) if r == {'FINISHED'} else (1, 1, 1)
check("loft closed through exact circles: holds every circle (rational), closed and smooth at the seam",
      r == {'FINISHED'} and dmax < 1e-6 and gap < 1e-9 and d1 < 1e-3, f"{dmax:.1e} {gap:.1e} {d1:.1e}")
# a loft through rectangles has four folds (C0 lines): tessellated piece by piece
def _tilted(P, N, T):
    P, N, T = np.asarray(P, float), np.asarray(N, float), store._tri_array(T)
    fn = np.cross(P[T[:, 1]] - P[T[:, 0]], P[T[:, 2]] - P[T[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1), 1e-30)[:, None]
    Nu = N / np.maximum(np.linalg.norm(N, axis=1), 1e-30)[:, None]
    return int((np.min(np.stack([np.abs(np.sum(fn * Nu[T[:, k]], axis=1)) for k in range(3)], 1), axis=1) < 0.7).sum())
RL.nproto_feature.closed = True
srf_ = _raw(RL)
pcs_ = store._crease_pieces(srf_, 5.0) or []
dev_ = 0.0
for pc_ in pcs_:
    (a0, a1), (b0, b1) = pc_.u_domain(), pc_.v_domain()
    for uu in np.linspace(a0, a1, 5):
        for vv in np.linspace(b0, b1, 5):
            dev_ = max(dev_, np.linalg.norm(np.array(pc_.evaluate(uu, vv)) - np.array(srf_.evaluate(uu, vv))))
P1_, N1_, T1_ = store._tessellate(store.to_brep(srf_), 0.0005, 5.0, 2.0, 1 / 30, natural=True)
P0_, N0_, T0_ = store._tessellate(store.to_brep(srf_), 0.0005, 5.0, 2.0, 1 / 30, natural=False)
check("a surface with 4 folds splits exactly into smooth pieces, tessellated with sharp creases",
      len(pcs_) == 4 and dev_ < 1e-12 and _tilted(P1_, N1_, T1_) == 0 and _tilted(P0_, N0_, T0_) > 100,
      f"{len(pcs_)} pieces, dev {dev_:.1e}, smeared {_tilted(P1_, N1_, T1_)} (was {_tilted(P0_, N0_, T0_)}), "
      f"{len(P1_)} verts (was {len(P0_)})")
from blender_nurbs import contlabels as _clb
RL.nproto_feature.closed = False
_labs = _clb.labels(RL)
check("loft through rectangles: its G0 labels hold (the check finds the right side of each fold)",
      len(_labs) == 2 and all(L_["check"] and L_["check"]["ok"] for L_ in _labs),
      f"{[L_['check'] and L_['check']['text'] for L_ in _labs]}")
RL.nproto_feature.closed = True
_brl = store.get_brep(RL)
_sh = store.shared_edges(RL)
check("loft through rectangles: one face per side, joined along the folds (a closed ring: a solid)",
      _brl.face_count() == 4 and len(_sh) >= 4 and bool(_brl.is_manifold_solid()),
      f"{_brl.face_count()} faces, {len(_sh)} shared edges, solid {_brl.is_manifold_solid()}")
_n_before = _brl.face_count()
_fold_edges = [i for i in _sh if np.ptp(np.array(store.display(RL)["edges"][i])[:, 2]) < 1e-6
               and abs(np.array(store.display(RL)["edges"][i])[0, 2]) > 0.1]       # the 4 ring corners (z = +-0.2)
try:
    r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[RL.name, f"E{i}"] for i in _fold_edges]), radius=0.05)
except RuntimeError as exc:
    r = str(exc)
check("Edge Fillet rounds the folds of a loft through rectangles", r == {'FINISHED'} and len(_fold_edges) == 4
      and store.get_brep(RL).face_count() > _n_before, f"{r} {len(_fold_edges)} edges -> {store.get_brep(RL).face_count()} faces")
_labs = _clb.labels(RL)
check("after Edge Fillet the loft's G0 labels still hold (checked on the loft, not the rounded result)",
      len(_labs) == 2 and all(L_["check"] and L_["check"]["ok"] for L_ in _labs),
      f"{[L_['check'] and L_['check']['text'] for L_ in _labs]}")
_cl_old = _clb.measure(RL, *_labs[0]["src"], 0)
check("(the rounded result itself is off the sections at the corners)", not _cl_old["ok"], _cl_old["text"])
check("a smooth closed loft (exact circles: G1 at its double knots) stays one face",
      store._crease_pieces(store.get_surfaces(RC)[0], 1.0) is None and store.get_brep(RC).face_count() == 1)
try:
    r = bpy.ops.nproto.loft(inputs=ins((circ[0], "S0"), (circ[1], "S0")), closed=True)
except RuntimeError as exc:
    r = str(exc)
check("loft closed with only two sections is refused", "at least 3 sections" in str(r), f"{r}")

# --- STEP import: our own export comes back with the same shape
from blender_nurbs import step_import as _si, step_export as _se
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
if ctx.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
ctx.scene.unit_settings.scale_length = 1.0
bpy.context.scene.cursor.location = (0, 0, 0)
r = bpy.ops.nproto.add_primitive(kind='CYLINDER', radius=0.5, height=1.2, caps=True)
CYL = ctx.active_object
tmp_step = os.path.join(tempfile.mkdtemp(), "cyl.step")
_se.export_step(tmp_step, [CYL], unit="MM")
res, rep = _si.read_step(tmp_step, 1.0)
check("step import: our exported capped cylinder comes back as one closed solid",
      rep["bodies"] == 1 and rep["failed"] == 0 and rep["solids"] == 1, f"{rep}")
sm_ = _kl()
bb = np.array(sm_.brep_bounding_box(res[0][1], True))
check("step import: millimetre file -> Blender units, same size", np.allclose(bb[1] - bb[0], (1.0, 1.0, 1.2), atol=2e-3),
      f"{bb.tolist()}")
r = bpy.ops.nproto.import_step(filepath=tmp_step)
imp_objs = [o for o in ctx.selected_objects]
check("step import operator: one NURBS object in a new collection", r == {'FINISHED'} and len(imp_objs) == 1
      and store.is_nurbs(imp_objs[0]) and imp_objs[0].users_collection[0].name.startswith("cyl"))
# analytic surfaces: a hand-written STEP cylinder (CYLINDRICAL_SURFACE + 2 PLANEs + circles + seam line), in inches
step_txt = """ISO-10303-21;
HEADER;FILE_DESCRIPTION((''),'2;1');FILE_NAME('c','',(''),(''),'','','');FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));ENDSEC;
DATA;
#1=(LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.));
#2=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(25.4),#1);
#3=DIMENSIONAL_EXPONENTS(1.,0.,0.,0.,0.,0.,0.);
#4=(CONVERSION_BASED_UNIT('INCH',#2) LENGTH_UNIT() NAMED_UNIT(#3));
#5=(NAMED_UNIT(*) PLANE_ANGLE_UNIT() SI_UNIT($,.RADIAN.));
#6=(GEOMETRIC_REPRESENTATION_CONTEXT(3) GLOBAL_UNIT_ASSIGNED_CONTEXT((#4,#5)) REPRESENTATION_CONTEXT('',''));
#10=CARTESIAN_POINT('',(0.,0.,0.));
#11=DIRECTION('',(0.,0.,1.));
#12=DIRECTION('',(1.,0.,0.));
#13=AXIS2_PLACEMENT_3D('',#10,#11,#12);
#14=CARTESIAN_POINT('',(0.,0.,2.));
#15=AXIS2_PLACEMENT_3D('',#14,#11,#12);
#20=CYLINDRICAL_SURFACE('',#13,1.);
#21=PLANE('',#13);
#22=PLANE('',#15);
#30=CIRCLE('',#13,1.);
#31=CIRCLE('',#15,1.);
#32=CARTESIAN_POINT('',(1.,0.,0.));
#33=CARTESIAN_POINT('',(1.,0.,2.));
#34=VERTEX_POINT('',#32);
#35=VERTEX_POINT('',#33);
#36=VECTOR('',#11,1.);
#37=LINE('',#32,#36);
#40=EDGE_CURVE('',#34,#34,#30,.T.);
#41=EDGE_CURVE('',#35,#35,#31,.T.);
#42=EDGE_CURVE('',#34,#35,#37,.T.);
#50=ORIENTED_EDGE('',*,*,#40,.T.);
#51=ORIENTED_EDGE('',*,*,#42,.T.);
#52=ORIENTED_EDGE('',*,*,#41,.F.);
#53=ORIENTED_EDGE('',*,*,#42,.F.);
#54=EDGE_LOOP('',(#50,#51,#52,#53));
#55=FACE_OUTER_BOUND('',#54,.T.);
#56=ADVANCED_FACE('',(#55),#20,.T.);
#60=ORIENTED_EDGE('',*,*,#40,.F.);
#61=EDGE_LOOP('',(#60));
#62=FACE_OUTER_BOUND('',#61,.T.);
#63=ADVANCED_FACE('',(#62),#21,.F.);
#64=ORIENTED_EDGE('',*,*,#41,.T.);
#65=EDGE_LOOP('',(#64));
#66=FACE_OUTER_BOUND('',#65,.T.);
#67=ADVANCED_FACE('',(#66),#22,.T.);
#70=CLOSED_SHELL('',(#56,#63,#67));
#71=MANIFOLD_SOLID_BREP('Can',#70);
#72=ADVANCED_BREP_SHAPE_REPRESENTATION('',(#71,#13),#6);
ENDSEC;
END-ISO-10303-21;
"""
tmp2 = os.path.join(tempfile.mkdtemp(), "can.step")
open(tmp2, "w").write(step_txt)
res, rep = _si.read_step(tmp2, 1.0)
bb = np.array(sm_.brep_bounding_box(res[0][1], True)) if res else None
check("step import: analytic cylinder + planes (inch file) -> closed exact solid",
      rep["faces"] == 3 and rep["solids"] == 1 and bb is not None
      and np.allclose(bb[1] - bb[0], (0.0508, 0.0508, 0.0508), atol=1e-5), f"{rep} {None if bb is None else bb.tolist()}")
vol = sm_.compute_volume(res[0][1]) if res else 0
vol = vol[0] if isinstance(vol, tuple) else vol
check("step import: cylinder volume matches pi r^2 h (kernel volume is numeric)", abs(abs(vol) / (math.pi * 0.0254 ** 2 * 0.0508) - 1) < 1e-4,
      f"{vol} vs {math.pi * 0.0254 ** 2 * 0.0508}")


# --- Surface Align (edges G0 / G1 / G2 to the touching surfaces) ------------
from math import comb as _comb
from blender_nurbs import salign as _sa
ctx.scene.nproto_settings.construction_history = True
_F = lambda x, y: 0.05 * x**3 - 0.1 * x * y**2 + 0.08 * y**3 + 0.2 * x * y - 0.1 * y**2
_Bi = np.linalg.inv(np.array([[_comb(3, j) * s**j * (1 - s)**(3 - j) for j in range(4)] for s in np.linspace(0, 1, 4)]))
def _quad(name, x0, x1, y0, y1):
    X, Y = np.meshgrid(np.linspace(x0, x1, 4), np.linspace(y0, y1, 4), indexing="ij")
    C = [_Bi @ A @ _Bi.T for A in (X, Y, _F(X, Y))]
    pts = [(C[0][i, j] + 20, C[1][i, j], C[2][i, j]) for i in range(4) for j in range(4)]
    return store.new_object(ctx, name, ext.surface_from_cvs(3, 3, pts, [0]*4 + [1]*4, [0]*4 + [1]*4))
saN, saN2, saNR = _quad("SA_N", 0, 2, 0, 2), _quad("SA_N2", 2, 4, -2, 0), _quad("SA_NR", 4, 6, 0, 2)
_xs, _ys = np.linspace(2, 4, 6), np.linspace(0, 2, 6)
_pts = [(x + 20, y, _F(x, y) + 0.4 * (x - 2) * (4 - x) * y * (2 - y) + 0.01 * math.sin(3 * y)) for x in _xs for y in _ys]
saS = store.new_object(ctx, "SA_S", ext.surface_from_cvs(3, 3, _pts, geom.clamped_uniform_knots(6, 3), geom.clamped_uniform_knots(6, 3)))
def _sa_rep(o, jobs_):
    rep = []
    _sa.align_text(o, o[store.PROP_BREP], jobs_, report=rep, measure_only=True)
    return rep
def _sa_nums(rep):
    import re
    return [[float(x) for x in re.findall(r"(?:gap|angle|curvature) ([-0-9.e+]+)", r)] for r in rep]
_before = _sa_nums(_sa_rep(saS, [("U0", (saN, "U1"), "G2")]))[0]
r = bpy.ops.nproto.surface_align(inputs=ins((saS, "U0"), (saS, "V0"), (saS, "U1")), cont_U0="G2", cont_V0="G2", cont_U1="G2")
_jobs = [("U0", (saN, "U1"), "G2"), ("V0", (saN2, "V1"), "G2"), ("U1", (saNR, "U0"), "G2")]
_after = _sa_nums(_sa_rep(saS, _jobs))
check("surface align: 3 edges G2 at once, neighbours found automatically (gap, angle, curvature %)",
      r == {'FINISHED'} and _before[1] > 5 and all(g < 1e-4 and a < 0.02 and k < 5 for g, a, k in _after)
      and [e["op"] for e in features.stack(saS)] == ["surface_align"] and store.is_natural(saS), f"{_before} -> {_after}")
features.remove_step(saS, 0); features.rebuild(saS, ctx, force=True)
r = bpy.ops.nproto.surface_align(inputs=ins((saS, "U0")), cont_U0="G1")
saN.location.z += 0.2; ctx.view_layer.update(); features.rebuild(saS, ctx)
_f = _sa_nums(_sa_rep(saS, [("U0", (saN, "U1"), "G1")]))[0]
check("surface align: history -- moving the neighbour realigns the edge (G1)", r == {'FINISHED'} and _f[0] < 1e-4 and _f[1] < 0.02, f"{_f}")
saN.location.z -= 0.2; ctx.view_layer.update()
from blender_nurbs import contlabels as _cl
_cl._cache.clear()
_L = _cl.labels(saS)
_r = bpy.ops.nproto.cycle_continuity(obj_name=saS.name, index=0) if _L else None
_cl._cache.clear()
_L2 = _cl.labels(saS)
_f = _sa_nums(_sa_rep(saS, [("U0", (saN, "U1"), "G2")]))[0]
check("continuity labels: Surface Align edge shows a label, click steps G1 -> G2 and realigns",
      len(_L) == 1 and _L[0]["value"] == "G1" and _r == {'FINISHED'} and _L2[0]["value"] == "G2"
      and features.stack(saS)[0]["params"]["cont"] == ["G2"] and len(_f) == 3 and _f[2] < 5, f"{_L} {_L2} {_f}")
# too few rows: a 4-row cubic aligned G2 at both ends gets knots inserted
_pts4 = [(x + 20, y, 0.0) for x in np.linspace(2, 4, 4) for y in _ys]
saS4 = store.new_object(ctx, "SA_S4", ext.surface_from_cvs(3, 3, _pts4, geom.clamped_uniform_knots(4, 3), geom.clamped_uniform_knots(6, 3)))
r = bpy.ops.nproto.surface_align(inputs=ins((saS4, "U0"), (saN, "U1"), (saS4, "U1"), (saNR, "U0")), cont_U0="G2", cont_U1="G2")
_f = _sa_nums(_sa_rep(saS4, [("U0", (saN, "U1"), "G2"), ("U1", (saNR, "U0"), "G2")]))
check("surface align: picked targets, rows added for G2 at both ends",
      r == {'FINISHED'} and ext.surface_cvs(store.get_surfaces(saS4)[0])["count_u"] >= 6
      and all(g < 1e-4 and a < 0.02 and k < 5 for g, a, k in _f), f"{_f}")



# --- continuity check (red ring) and overlapping neighbours -----------------
_m_bad = _cl.measure(saS4, saN, "U1", 2) if False else None
_ov = _quad("SA_OV", 2, 3, 0, 2)        # a second fill on S's own side, sharing S's U0 edge
_fresh = store.new_object(ctx, "SA_S5", ext.surface_from_cvs(3, 3, _pts, geom.clamped_uniform_knots(6, 3), geom.clamped_uniform_knots(6, 3)))
_c = [c for c in __import__("blender_nurbs.ops", fromlist=["x"])._pick_candidates(ctx) if c[0] != _fresh]
_nb = _sa.find_neighbour(_fresh, "U0", _c)
check("surface align: an overlapping fill of the same opening is not taken as the neighbour",
      _nb is not None and _nb[0] == saN and _sa.overlaps(_fresh, "U0", _ov, "U0") and not _sa.overlaps(_fresh, "U0", saN, "U1"),
      f"{_nb and (_nb[0].name, _nb[1])}")
_before_chk = _cl.measure(_fresh, saN, "U1", 2)
r = bpy.ops.nproto.surface_align(inputs=ins((_fresh, "U0")), cont_U0="G2")
_cl._cache.clear()
_lab = _cl.labels(_fresh)
check("continuity check: unaligned edge FAILS G2, aligned label holds (no red ring)",
      not _before_chk["ok"] and r == {'FINISHED'} and _lab and _lab[0]["check"] and _lab[0]["check"]["ok"],
      f"{_before_chk['text']} | {_lab and _lab[0]['check']}")
_lab_l = _cl.labels(bpy.data.objects["SA_S"]) if "SA_S" in bpy.data.objects else []

# --- Match Surface (edge rows set from the target's CV rows: exact G0 / G1 / G2) ---
from blender_nurbs import match as _ma
ctx.scene.nproto_settings.construction_history = True
_mrng = np.random.default_rng(5)
def _mk(name, G, Uu, Uv, du=3, dv=3):
    pts = [tuple(map(float, G[i, j])) for i in range(G.shape[0]) for j in range(G.shape[1])]
    return store.new_object(ctx, name, ext.surface_from_cvs(du, dv, pts, list(Uu), list(Uv)))
def _mgrid(o):
    d = ext.surface_cvs(store.get_surfaces(o)[0])
    return np.array(d["points"], float).reshape(d["count_u"], d["count_v"], 3), d
def _msplit(G, U, t, axis, p=3):
    """Cut a CV grid at knot t of one axis: two exact pieces (knots 0..1 each)."""
    Gt = G if axis == 0 else G.transpose(1, 0, 2)
    U = list(U)
    while U.count(t) < p:
        Gt, U = _sa.insert_knot(Gt, U, p, t)
    k = max(i for i, x in enumerate(U) if x == t) - p
    a, b = Gt[:k + 1], Gt[k:]
    Ua, Ub = [x / t for x in U[:k + p + 1]] + [1.0], [0.0] + [(x - t) / (1 - t) for x in U[k + 1:]]
    if axis == 1:
        a, b = a.transpose(1, 0, 2), b.transpose(1, 0, 2)
    return (np.ascontiguousarray(a), Ua), (np.ascontiguousarray(b), Ub)
def _exact(m, lv=2):
    """Exact as far as Continuity Check can tell (its own floor: ~1e-11 gaps, ~1e-6 deg)."""
    return m is not None and m["gap"] < 1e-9 and (lv < 1 or m["angle"] < 1e-5) and (lv < 2 or m["curv"] < 1e-6)
def _mmeas(*a):
    try:
        return _ma.measure(*a)
    except ValueError:              # Continuity Check: "the edges don't meet"
        return None
# one surface cut in 2 x 2 at u = v = 0.5: the top-right piece S, perturbed, against its neighbours
_mn = 7
_mX, _mY = np.meshgrid(np.linspace(300, 304, _mn), np.linspace(0, 4, _mn), indexing="ij")
_mG = np.stack([_mX, _mY, 0.4 * np.sin(0.8 * (_mX - 300)) * np.cos(0.6 * _mY) + 0.05 * _mrng.standard_normal(_mX.shape)], 2)
_mU = geom.clamped_uniform_knots(_mn, 3)
(_mL, _mUL), (_mR, _mUR) = _msplit(_mG, _mU, 0.5, 0)
(_mL0, _), (_mL1, _mVL1) = _msplit(_mL, _mU, 0.5, 1)
(_mR0, _mVR0), (_mR1, _mVR1) = _msplit(_mR, _mU, 0.5, 1)
maL = _mk("MA_L", _mL1, _mUL, _mVL1)          # left of S, across its U0
maB = _mk("MA_B", _mR0, _mUR, _mVR0)          # below S, across its V0
_mP = _mR1.copy(); _mP[0] += 0.02 * _mrng.standard_normal(_mP[0].shape); _mP[1:3] += 0.12 * _mrng.standard_normal(_mP[1:3].shape)
maS = _mk("MA_S", _mP, _mUR, _mVR1)
_m0 = _mmeas(maS, "U0", maL, "U1")
r = bpy.ops.nproto.match_surface(inputs=ins((maS, "U0")), cont_U0="G2")      # the target is found by itself
_m1 = _ma.measure(maS, "U0", maL, "U1")
_mg, _md = _mgrid(maS)
_hull = max(np.linalg.norm(np.cross(_mg[1, j] - _mg[0, j], _mL1[-1, j] - _mL1[-2, j])) for j in range(_mg.shape[1]))
check("match surface: same layout, G2 to the touching edge -- exact (Continuity Check), row 0 IS the target's row, "
      "only rows 0-2 move, CV hull lines straight across",
      r == {'FINISHED'} and not _exact(_m0) and _exact(_m1) and (_md["count_u"], _md["count_v"]) == _mP.shape[:2]
      and np.abs(_mg[0] - _mL1[-1]).max() < 1e-12 and np.abs(_mg[3:] - _mP[3:]).max() < 1e-12 and _hull < 1e-12
      and [e["op"] for e in features.stack(maS)] == ["match_surface"] and store.is_natural(maS), f"{_m0} -> {_m1}")
# history: the target moves, the edge follows; its label steps G2 -> G0 and back to G2
maL.location.z += 0.3; ctx.view_layer.update(); features.rebuild(maS, ctx)
_m2 = _ma.measure(maS, "U0", maL, "U1")
maL.location.z = 0.0; ctx.view_layer.update(); features.rebuild(maS, ctx)      # (exactly: -= 0.3 leaves float32 dust)
_cl._cache.clear()
_mlab = _cl.labels(maS)
_r1 = bpy.ops.nproto.cycle_continuity(obj_name=maS.name, index=0) if _mlab else None
_cl._cache.clear()
_mlab2 = _cl.labels(maS)
_r2 = bpy.ops.nproto.cycle_continuity(obj_name=maS.name, index=0, step=-1) if _mlab2 else None   # back from G3 to G2
_cl._cache.clear()
_mlab3 = _cl.labels(maS)
check("match surface: history -- moving the target matches again; its edge label holds and steps G2 -> G3 -> G2",
      _exact(_m2) and len(_mlab) == 1 and _mlab[0]["value"] == "G2" and _mlab[0]["check"]["ok"]
      and _r1 == {'FINISHED'} and _mlab2[0]["value"] == "G3" and _r2 == {'FINISHED'} and _mlab3[0]["value"] == "G2"
      and _exact(_ma.measure(maS, "U0", maL, "U1")), f"{_m2} {[(L['value'], L['check']) for L in _mlab + _mlab2 + _mlab3]}")
# two edges meeting at a corner: both exact, and the cut-out piece comes back CV for CV
features.remove_step(maS, 0); features.rebuild(maS, ctx, force=True)
_mP2 = _mR1.copy(); _mP2[:3] += 0.1 * _mrng.standard_normal(_mP2[:3].shape); _mP2[:, :3] += 0.1 * _mrng.standard_normal(_mP2[:, :3].shape)
maS2 = _mk("MA_S2", _mP2, _mUR, _mVR1)
r = bpy.ops.nproto.match_surface(inputs=ins((maS2, "U0"), (maL, "U1"), (maS2, "V0"), (maB, "V1")), cont_U0="G2", cont_V0="G2")
_mg2, _ = _mgrid(maS2)
check("match surface: two edges at a corner (a 2 x 2 split), G2 both -- exact, the corner block is the original's CVs",
      r == {'FINISHED'} and _exact(_ma.measure(maS2, "U0", maL, "U1")) and _exact(_ma.measure(maS2, "V0", maB, "V1"))
      and np.abs(_mg2[:3, :3] - _mR1[:3, :3]).max() < 1e-10, f"{np.abs(_mg2[:3, :3] - _mR1[:3, :3]).max():.2e}")
# the target's parameter runs the other way along the edge
maR = _mk("MA_Rev", _mL1[:, ::-1].copy(), _mUL, [1 - x for x in reversed(_mVL1)])
maR.location.y = 10; ctx.view_layer.update()
maS3 = _mk("MA_S3", _mP, _mUR, _mVR1); maS3.location.y = 10; ctx.view_layer.update()
r = bpy.ops.nproto.match_surface(inputs=ins((maS3, "U0"), (maR, "U1")), cont_U0="G2")
check("match surface: a target running the other way (reversed parameter) -- turned round, exact",
      r == {'FINISHED'} and _exact(_ma.measure(maS3, "U0", maR, "U1")), f"{_ma.measure(maS3, 'U0', maR, 'U1')}")
# different layouts along the edge: target degree 5 with its own knots, surface cubic with 4 CVs
_tX, _tY = np.meshgrid(np.linspace(318, 320, 6), np.linspace(0, 2, 8), indexing="ij")
maT5 = _mk("MA_T5", np.stack([_tX, _tY, 0.2 * np.sin(1.3 * _tY) + 0.05 * (_tX - 320) * _tY], 2), geom.clamped_uniform_knots(6, 3),
           [0.0] * 6 + [0.3, 0.55] + [1.0] * 6, 3, 5)
_sX, _sY = np.meshgrid(np.linspace(320, 322, 5), np.linspace(0, 2, 4), indexing="ij")
maS5 = _mk("MA_S5", np.stack([_sX, _sY, 0.1 * _mrng.standard_normal(_sX.shape)], 2), geom.clamped_uniform_knots(5, 3),
           geom.clamped_uniform_knots(4, 3))
_before5 = _mgrid(maS5)[0].copy()
r = bpy.ops.nproto.match_surface(inputs=ins((maS5, "U0"), (maT5, "U1")), cont_U0="G2")
_g5, _d5 = _mgrid(maS5)
check("match surface: different degree and knots along the edge -- made one exactly (degree 3 -> 5, the target's knots), "
      "exact G2", r == {'FINISHED'} and _d5["degree_v"] == 5 and _d5["count_v"] == 8 and _d5["count_u"] == 5
      and np.allclose(_d5["knots_v"], [0.0] * 6 + [0.3, 0.55] + [1.0] * 6) and _exact(_ma.measure(maS5, "U0", maT5, "U1")),
      f"{_d5['degree_v']} {_d5['count_v']} {_d5['knots_v']} {_ma.measure(maS5, 'U0', maT5, 'U1')}")
# a curve as the target: G0, row 0 = the curve's CVs
maC = nurbs("MA_C", [(330 + 0.8 * i, -0.2, 0.3 * math.sin(i)) for i in range(6)])
_cX, _cY = np.meshgrid(np.linspace(330, 334, 4), np.linspace(0, 2, 5), indexing="ij")
maSC = _mk("MA_SC", np.stack([_cX, _cY, np.zeros_like(_cX)], 2), geom.clamped_uniform_knots(4, 3), geom.clamped_uniform_knots(5, 3))
r = bpy.ops.nproto.match_surface(inputs=ins((maSC, "V0"), (maC, "S0")), cont_V0="G2")
_gc, _ = _mgrid(maSC)
_crow = np.array(ext.curve_cvs(features.input_curve(maC, "S0"))["points"], float)
check("match surface: a curve as the target -- G0, the edge row is the curve's CVs",
      r == {'FINISHED'} and _ma.measure(maSC, "V0", maC, "S0")["gap"] < 1e-9 and np.abs(_gc[:, 0] - _crow).max() < 1e-12,
      f"{_ma.measure(maSC, 'V0', maC, 'S0')}")
# Closest Points: the edge runs along the middle part of a longer target edge
_lX, _lY = np.meshgrid(np.linspace(340, 346, 6), np.linspace(-2, 0, 4), indexing="ij")
maTL = _mk("MA_TL", np.stack([_lX, _lY, 0.3 * np.sin(0.5 * (_lX - 340)) + 0.1 * _lY ** 2], 2), geom.clamped_uniform_knots(6, 3),
           geom.clamped_uniform_knots(4, 3))
_mX2, _mY2 = np.meshgrid(np.linspace(342, 344.5, 5), np.linspace(0.05, 2, 4), indexing="ij")
maSM = _mk("MA_SM", np.stack([_mX2, _mY2, 0.1 + np.zeros_like(_mX2)], 2), geom.clamped_uniform_knots(5, 3), geom.clamped_uniform_knots(4, 3))
r = bpy.ops.nproto.match_surface(inputs=ins((maSM, "V0"), (maTL, "V1")), cont_V0="G2", ends="CLOSEST")
_e = np.array(geom.boundary_points(store.get_surfaces(maSM)[0], "V0", 3), float)
check("match surface: Closest Points -- part of a longer target edge, cut out exactly; G2 exact",
      r == {'FINISHED'} and _exact(_ma.measure(maSM, "V0", maTL, "V1")) and abs(_e[0][0] - 342) < 0.05 and abs(_e[-1][0] - 344.5) < 0.05,
      f"{_ma.measure(maSM, 'V0', maTL, 'V1')} {_e[[0, -1], 0]}")
# Keep Other End: a 4-row patch matched G2 keeps its far edge's position and tangent (rows inserted)
_kX, _kY = np.meshgrid(np.linspace(352, 354, 4), np.linspace(0, 2, 4), indexing="ij")
maT4 = _mk("MA_T4", np.stack([_kX - 2, _kY, 0.1 * np.cos(_kY)], 2), geom.clamped_uniform_knots(4, 3), geom.clamped_uniform_knots(4, 3))
maS4 = _mk("MA_S4", np.stack([_kX, _kY, 0.2 * _mrng.standard_normal(_kX.shape)], 2), geom.clamped_uniform_knots(4, 3),
           geom.clamped_uniform_knots(4, 3))
_far = lambda o: np.array([np.r_[store.get_surfaces(o)[0].evaluate(1.0, v), store.get_surfaces(o)[0].derivatives(1.0, v)[1]]
                           for v in np.linspace(0, 1, 7)], float)
_f0 = _far(maS4)
r = bpy.ops.nproto.match_surface(inputs=ins((maS4, "U0"), (maT4, "U1")), cont_U0="G2", keep="G1")
check("match surface: Keep Other End = Tangent -- rows inserted, the far edge keeps position and tangent, G2 exact",
      r == {'FINISHED'} and _mgrid(maS4)[1]["count_u"] == 5 and np.abs(_far(maS4) - _f0).max() < 1e-10
      and _exact(_ma.measure(maS4, "U0", maT4, "U1")), f"{np.abs(_far(maS4) - _f0).max():.2e}")
# degree 1 across: raised for G2
maP1 = _mk("MA_P1", np.stack([_kX[[0, -1]] - 4, _kY[[0, -1]], 0.5 + np.zeros((2, 4))], 2), [0, 0, 1, 1], geom.clamped_uniform_knots(4, 3), 1, 3)
r = bpy.ops.nproto.match_surface(inputs=ins((maP1, "U1"), (maT4, "U0")), cont_U1="G2")
check("match surface: degree 1 across -- raised to 2 for G2, exact", r == {'FINISHED'} and _mgrid(maP1)[1]["degree_u"] == 2
      and _exact(_ma.measure(maP1, "U1", maT4, "U0")), f"{_mgrid(maP1)[1]['degree_u']} {_ma.measure(maP1, 'U1', maT4, 'U0')}")
# trimmed target: a side along its surface's CV net is exact; a trim curve is refused with the reason
_tXX, _tYY = np.meshgrid(np.linspace(360, 364, 6), np.linspace(0, 2, 5), indexing="ij")
_tG = np.stack([_tXX, _tYY, 0.2 * np.sin(_tXX - 360) + 0.1 * _tYY ** 2], 2)
_tU, _tV = geom.clamped_uniform_knots(6, 3), geom.clamped_uniform_knots(5, 3)
_tbase = ext.surface_from_cvs(3, 3, [tuple(map(float, _tG[i, j])) for i in range(6) for j in range(5)], _tU, _tV)
_tcrv = lambda s_: ext.curve_from_cvs(3, [tuple(map(float, q)) for q in s_["P"]], [float(k) for k in s_["knots"]], None)
_trow = lambda P_, U_: {"deg": 3, "knots": U_, "P": P_, "D": None, "DD": None}
_tloop = [_tcrv(geom.sub_side(_trow(_tG[:, 0], _tU), 0.0, 0.6)),
          _kl().create_curve_interp_points([tuple(_tbase.evaluate(0.6 - 0.2 * t, t)) for t in np.linspace(0, 1, 12)]),
          _tcrv(geom.reverse_side(geom.sub_side(_trow(_tG[:, -1], _tU), 0.0, 0.4))), _tcrv(geom.reverse_side(_trow(_tG[0], _tV)))]
maTT = store.new_object(ctx, "MA_TT", _kl().trim_surface_with_3d_curves(_tbase, [4], _tloop))
_tedges = store.display(maTT)["edges"]
_tnear = lambda q_: "E%d" % min(range(len(_tedges)), key=lambda i: np.linalg.norm(np.array(_tedges[i][len(_tedges[i]) // 2]) - np.array(q_)))
_tside, _ttrim = _tnear(_tbase.evaluate(0.3, 0.0)), _tnear(_tbase.evaluate(0.5, 0.5))
_xe = float(_tbase.evaluate(0.6, 0.0)[0])
_sXX, _sYY = np.meshgrid(np.linspace(360, _xe, 5), np.linspace(-1.5, 0, 4), indexing="ij")
_sG = np.stack([_sXX, _sYY, 0.1 * _mrng.standard_normal(_sXX.shape)], 2)
maST = _mk("MA_ST", _sG, geom.clamped_uniform_knots(5, 3), geom.clamped_uniform_knots(4, 3))
r = bpy.ops.nproto.match_surface(inputs=ins((maST, "V1"), (maTT, _tside)), cont_V1="G2")
maST2 = _mk("MA_ST2", _sG, geom.clamped_uniform_knots(5, 3), geom.clamped_uniform_knots(4, 3))
r2 = bpy.ops.nproto.match_surface(inputs=ins((maST2, "V1"), (maTT, _ttrim)), cont_V1="G2")
try:
    _ma.match_text(maST2, maST2[store.PROP_BREP], [("V1", (maTT, _ttrim), "G2")]); _why = ""
except _ma.MatchError as exc:
    _why = str(exc)
check("match surface: a trimmed face's side along its CV net -- exact; its trim curve is refused (says why, points to Surface Align)",
      not store.is_natural(maTT) and r == {'FINISHED'} and _exact(_ma.measure(maST, "V1", maTT, _tside))
      and r2 == {'CANCELLED'} and "trim curve" in _why and "Surface Align" in _why and _mgrid(maST2)[0].tolist() == _sG.tolist(),
      f"{_tside} {_ttrim} {_ma.measure(maST, 'V1', maTT, _tside)} {_why}")
# Average Both: the two move to meet halfway (different layouts), history of both removed
_aX, _aY = np.meshgrid(np.linspace(370, 372, 5), np.linspace(0, 3, 6), indexing="ij")
maA = _mk("MA_A", np.stack([_aX, _aY, 0.3 * (_aX - 370) + 0.1 * np.sin(_aY)], 2), geom.clamped_uniform_knots(5, 3), geom.clamped_uniform_knots(6, 3))
_bX, _bY = np.meshgrid(np.linspace(372.2, 374, 4), np.linspace(0, 3, 4), indexing="ij")
maBB = _mk("MA_BB", np.stack([_bX, _bY, -0.2 * (_bX - 372.2) + 0.8 + 0.05 * _bY], 2), geom.clamped_uniform_knots(4, 3), geom.clamped_uniform_knots(4, 3))
features.push_step(maA, "flip_normal", [], {}); features.rebuild(maA, ctx, force=True)          # some history to remove
_ea0 = np.array(geom.boundary_points(store.get_surfaces(maA)[0], "U1", 7)); _eb0 = np.array(geom.boundary_points(store.get_surfaces(maBB)[0], "U0", 7))
r = bpy.ops.nproto.match_surface(inputs=ins((maBB, "U0"), (maA, "U1")), cont_U0="G2", average=True)
_ea1 = np.array(geom.boundary_points(store.get_surfaces(maA)[0], "U1", 7))
check("match surface: Average Both -- both surfaces move to meet halfway, exact G2, no history left",
      r == {'FINISHED'} and _exact(_ma.measure(maBB, "U0", maA, "U1")) and np.abs(_ea1 - 0.5 * (_ea0[:, :] + _eb0)).max() < 1e-9
      and np.abs(_ea1 - _ea0).max() > 0.1 and not features.has_history(maA) and not features.has_history(maBB),
      f"{_ma.measure(maBB, 'U0', maA, 'U1')} {np.abs(_ea1 - 0.5 * (_ea0 + _eb0)).max():.2e}")
# a target built from the surface (history) would loop: refused; the operator is undoable
r = bpy.ops.nproto.match_surface(inputs=ins((maS3, "U0"), (maR, "U1")), cont_U0="G1")      # (MA_S3 is built from MA_Rev)
r2 = bpy.ops.nproto.match_surface(inputs=ins((maR, "U1"), (maS3, "U0")), cont_U1="G1")
check("match surface: matching a target that is built from the surface is refused (history loop); the command is undoable",
      r == {'FINISHED'} and r2 == {'CANCELLED'} and 'UNDO' in bpy.types.NPROTO_OT_match_surface.bl_options
      and features.depends_on(maS3, maR), f"{r} {r2}")

# --- G3: the curvature rate (Continuity Check, Match Surface, Surface Align) ---
from blender_nurbs import contcheck as _cc, surfeval as _se
# the third-order form against an analytic cubic graph z = f(x, y) at its flat point
_g3f = lambda x, y: 0.3 * x * x + 0.1 * x * y - 0.2 * y * y + 0.15 * x ** 3 - 0.05 * x * x * y + 0.07 * y ** 3
_X, _Y = np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4), indexing="ij")
_g3G = np.stack([_Bi @ A_ @ _Bi.T for A_ in (_X + 500, _Y, _g3f(_X, _Y))], -1)
_g3o = _mk("G3_Graph", _g3G, [0] * 4 + [1] * 4, [0] * 4 + [1] * 4)
_, _, _g3C = _se.forms(_se.Surf(store.get_surfaces(_g3o)[0]).at([0.5], [0.5], 3))
_dd = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
_want = [0.9, 0.42, 0.9 * _dd[0] ** 3 - 0.3 * _dd[0] ** 2 * _dd[1] + 0.42 * _dd[1] ** 3]
_got = [float(_se.cubic(_g3C, d_[None])[0]) for d_ in (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), _dd)]
check("G3: third-order form = the third derivatives of a cubic graph (C(x,x,x), C(y,y,y), C(d,d,d))",
      np.allclose(_got, _want, atol=1e-9), f"{_got} vs {_want}")
# one smooth surface cut between knots (C-inf there: G3 holds) and at a simple cubic knot (C2 only: G3 fails)
_g3n = 8
_X, _Y = np.meshgrid(np.linspace(0, 4, _g3n), np.linspace(0, 4, _g3n), indexing="ij")
_g3S = np.stack([_X + 510, _Y, 0.5 * np.sin(0.9 * _X) * np.cos(0.7 * _Y) + 0.05 * _mrng.standard_normal(_X.shape)], -1)
_g3U = geom.clamped_uniform_knots(_g3n, 3)
_g3k = sorted(set(_g3U[4:-4]))
def _g3cut(t, name):
    (a_, ua_), (b_, ub_) = _msplit(_g3S, _g3U, t, 0)
    A_, B_ = _mk(name + "A", a_, ua_, _g3U), _mk(name + "B", b_, ub_, _g3U)
    return _cc.analyze({"a": _cc.side_key(A_, "U1"), "b": _cc.side_key(B_, "U0")}, 80)
_rs, _rk = _g3cut(0.5 * (_g3k[1] + _g3k[2]), "G3_Cut"), _g3cut(_g3k[2], "G3_Knot")
check("G3 check: a cut between knots holds G3 (~1e-11); at a simple cubic knot G2 holds but G3 does not",
      _rs["g3rel"].max() < 1e-8 and _rk["crel"].max() < 1e-7 and _rk["g3rel"].max() > 0.2,
      f"{_rs['g3rel'].max():.2e} / {_rk['crel'].max():.2e} {_rk['g3rel'].max():.2e}")
def _exact3(m):
    return _exact(m) and m["rate"] is not None and m["rate"] < 1e-6
# Match G3: one edge of the cut-out piece -- exact; only rows 0-3 move
_g3P = _mR1.copy(); _g3P[0] += 0.02 * _mrng.standard_normal(_g3P[0].shape); _g3P[1:4] += 0.1 * _mrng.standard_normal(_g3P[1:4].shape)
g3S = _mk("G3_S", _g3P, _mUR, _mVR1)
_m0 = _mmeas(g3S, "U0", maL, "U1")
r = bpy.ops.nproto.match_surface(inputs=ins((g3S, "U0"), (maL, "U1")), cont_U0="G3")
_m1 = _ma.measure(g3S, "U0", maL, "U1")
_mg, _ = _mgrid(g3S)
check("match surface G3: one edge -- exact G0-G3 (Continuity Check), only rows 0-3 move",
      r == {'FINISHED'} and not _exact3(_m0) and _exact3(_m1) and np.abs(_mg[4:] - _g3P[4:]).max() < 1e-12, f"{_m0} -> {_m1}")
# two edges at a corner, both G3: the cut-out piece comes back CV for CV (cut at 0.45, not a knot:
# at u = v = 0.5 -- a knot of this cubic -- the pieces are only C2 to each other)
(_c45L, _c45UL), (_c45R, _c45UR) = _msplit(_mG + np.array([0.0, 20.0, 0.0]), _mU, 0.45, 0)
(_, _), (_c45L1, _c45VL1) = _msplit(_c45L, _mU, 0.45, 1)
(_c45R0, _c45VR0), (_c45R1, _c45VR1) = _msplit(_c45R, _mU, 0.45, 1)
g3L, g3Bt = _mk("G3_L", _c45L1, _c45UL, _c45VL1), _mk("G3_Bt", _c45R0, _c45UR, _c45VR0)
_g3P2 = _c45R1.copy(); _g3P2[:4] += 0.1 * _mrng.standard_normal(_g3P2[:4].shape); _g3P2[:, :4] += 0.1 * _mrng.standard_normal(_g3P2[:, :4].shape)
g3S2 = _mk("G3_S2", _g3P2, _c45UR, _c45VR1)
r = bpy.ops.nproto.match_surface(inputs=ins((g3S2, "U0"), (g3L, "U1"), (g3S2, "V0"), (g3Bt, "V1")), cont_U0="G3", cont_V0="G3")
_mg, _ = _mgrid(g3S2)
check("match surface G3: two edges at a corner -- both exact, the cut-out piece comes back CV for CV",
      r == {'FINISHED'} and _exact3(_ma.measure(g3S2, "U0", g3L, "U1")) and _exact3(_ma.measure(g3S2, "V0", g3Bt, "V1"))
      and _mg.shape == _c45R1.shape and np.abs(_mg - _c45R1).max() < 1e-9, f"{_mg.shape} {_c45R1.shape}")
# G2 leaves row 3 (and G3) alone; a 4-row cubic gets rows for G3 + Keep G3; a degree-5 target; Average
g3S3 = _mk("G3_S3", _g3P, _mUR, _mVR1)
bpy.ops.nproto.match_surface(inputs=ins((g3S3, "U0"), (maL, "U1")), cont_U0="G2")
_m2 = _ma.measure(g3S3, "U0", maL, "U1")
_X, _Y = np.meshgrid(np.linspace(520, 522, 4), np.linspace(0, 2, 5), indexing="ij")
g3T4 = _mk("G3_T4", np.stack([_X, _Y, 0.1 * np.sin(_X) + 0.05 * _Y ** 2], 2), geom.clamped_uniform_knots(4, 3), geom.clamped_uniform_knots(5, 3))
_X, _Y = np.meshgrid(np.linspace(522, 524, 4), np.linspace(0, 2, 5), indexing="ij")
g3S4 = _mk("G3_S4", np.stack([_X, _Y, 0.2 * np.cos(_X) - 0.1 * _Y], 2), geom.clamped_uniform_knots(4, 3), geom.clamped_uniform_knots(5, 3))
_far0 = np.array(geom.boundary_points(store.get_surfaces(g3S4)[0], "U1", 9))
r = bpy.ops.nproto.match_surface(inputs=ins((g3S4, "U0"), (g3T4, "U1")), cont_U0="G3", keep="G3")
_far1 = np.array(geom.boundary_points(store.get_surfaces(g3S4)[0], "U1", 9))
_X, _Y = np.meshgrid(np.linspace(530, 532, 6), np.linspace(0, 2, 5), indexing="ij")
g3T5 = _mk("G3_T5", np.stack([_X, _Y, 0.3 * np.sin(1.3 * _X) * (1 + 0.2 * _Y)], 2), geom.clamped_uniform_knots(6, 5),
           geom.clamped_uniform_knots(5, 3), du=5)
_X, _Y = np.meshgrid(np.linspace(532, 534, 6), np.linspace(0, 2, 5), indexing="ij")
g3S5 = _mk("G3_S5", np.stack([_X, _Y, 0.1 * _X - 53.0 + 0.05 * _Y], 2), geom.clamped_uniform_knots(6, 3), geom.clamped_uniform_knots(5, 3))
r5 = bpy.ops.nproto.match_surface(inputs=ins((g3S5, "U0"), (g3T5, "U1")), cont_U0="G3")
_X, _Y = np.meshgrid(np.linspace(540, 542, 6), np.linspace(0, 2, 5), indexing="ij")
g3A = _mk("G3_A", np.stack([_X, _Y, 0.2 * np.sin(2 * _X) + 0.1 * _Y], 2), geom.clamped_uniform_knots(6, 3), geom.clamped_uniform_knots(5, 3))
_X, _Y = np.meshgrid(np.linspace(542, 544, 6), np.linspace(0, 2, 5), indexing="ij")
g3B = _mk("G3_B", np.stack([_X, _Y, 0.25 * np.cos(1.5 * _X) - 0.05 * _Y], 2), geom.clamped_uniform_knots(6, 3), geom.clamped_uniform_knots(5, 3))
ra = bpy.ops.nproto.match_surface(inputs=ins((g3B, "U0"), (g3A, "U1")), cont_U0="G3", average=True)
check("match surface G3: G2 leaves G3 open; rows added for G3 + Keep G3 (far edge stays); degree-5 target; Average Both",
      _m2["curv"] < 1e-6 and _m2["rate"] > 1e-3
      and r == {'FINISHED'} and _exact3(_ma.measure(g3S4, "U0", g3T4, "U1")) and _mgrid(g3S4)[1]["count_u"] == 8
      and np.abs(_far1 - _far0).max() < 1e-9
      and r5 == {'FINISHED'} and _exact3(_ma.measure(g3S5, "U0", g3T5, "U1"))
      and ra == {'FINISHED'} and _exact3(_ma.measure(g3B, "U0", g3A, "U1")), f"{_m2}")
# Surface Align G3: one edge on the neighbour's layout -- G3 to the check's tolerance; its label shows G3 and holds
_pts1 = [(x + 20, y, _F(x, y) + 0.3 * (x - 2) * (4 - x) * (y + 0.3) + 0.02 * math.sin(3 * y))
         for x in np.linspace(2, 4, 7) for y in np.linspace(0, 2, 4)]
g3SA = store.new_object(ctx, "G3_SA", ext.surface_from_cvs(3, 3, _pts1, geom.clamped_uniform_knots(7, 3), [0] * 4 + [1] * 4))
_b = _ma.measure(g3SA, "U0", saN, "U1")
r = bpy.ops.nproto.surface_align(inputs=ins((g3SA, "U0"), (saN, "U1")), cont_U0="G3")
_a = _ma.measure(g3SA, "U0", saN, "U1")
_cl._cache.clear()
_lab = _cl.labels(g3SA)
check("surface align G3: one edge -- curvature rate within 1% (it was 40%); its label shows G3 and holds",
      r == {'FINISHED'} and _b["rate"] > 0.1 and _a["gap"] < 1e-4 and _a["curv"] < 0.01 and _a["rate"] < 0.01
      and len(_lab) == 1 and _lab[0]["value"] == "G3" and _lab[0]["check"]["ok"], f"{_b} -> {_a} {_lab and _lab[0]['check']}")

# Loft G3: between natural edges exact on the inputs' layout (the same, or two raised to one), degree 7 across;
# G3 / G0 and G2 / G3 mixes hold what they ask for, and no more
from blender_nurbs import contcheck as _cc3
def _lwavy(name, x0, nv, seed):
    r_ = np.random.default_rng(seed)
    X_, Y_ = np.meshgrid(np.linspace(x0, x0 + 4, 6), np.linspace(0, 5, nv), indexing="ij")
    Z_ = 0.6 * np.sin(0.8 * X_ + 0.3) * np.cos(0.6 * Y_) + 0.15 * r_.standard_normal(X_.shape)
    return _mk(name, np.stack([X_, Y_ + 600, Z_], -1), geom.clamped_uniform_knots(6, 3), geom.clamped_uniform_knots(nv, 3))
def _lg3(lo, src, e_src, lo_e):
    r_ = _cc3.analyze({"a": _cc3.side_key(lo, lo_e), "b": _cc3.side_key(src, e_src)}, 60)
    return float(r_["gap"].max()), float(r_["crel"].max()), float(np.nanmax(r_["g3rel"]))
lgA, lgB, lgB9 = _lwavy("LG3_A", 0, 7, 1), _lwavy("LG3_B", 7, 7, 2), _lwavy("LG3_B9", 7, 9, 3)
_lres = {}
for _nm, _bb in (("same", lgB), ("raised", lgB9)):
    for _ca, _cb in (("G3", "G3"), ("G3", "G0"), ("G2", "G3")):
        r = bpy.ops.nproto.loft(inputs=ins((lgA, "U1"), (_bb, "U0")), cont_a=_ca, cont_b=_cb)
        _lo = ctx.active_object
        _lres[_nm, _ca, _cb] = (r, ext.surface_cvs(store.get_surfaces(_lo)[0])["degree_u"],
                                _lg3(_lo, lgA, "U1", "U0"), _lg3(_lo, _bb, "U0", "U1"))
_okl = lambda g, lvl: g[0] < 1e-8 and (lvl < 2 or g[1] < 1e-7) and (lvl < 3 or g[2] < 1e-7)
check("loft G3 between surface edges: exact (Continuity Check ~1e-10) on one layout or two raised to one, degree 7 "
      "across; G3 / G0 and G2 / G3 hold G3 only where asked",
      all(v[0] == {'FINISHED'} for v in _lres.values())
      and all(_lres[k, "G3", "G3"][1] == 7 and _okl(_lres[k, "G3", "G3"][2], 3) and _okl(_lres[k, "G3", "G3"][3], 3)
              and _lres[k, "G3", "G0"][1] == 4 and _okl(_lres[k, "G3", "G0"][2], 3)
              and _lres[k, "G2", "G3"][1] == 6 and _okl(_lres[k, "G2", "G3"][2], 2) and _lres[k, "G2", "G3"][2][2] > 0.05
              and _okl(_lres[k, "G2", "G3"][3], 3) for k in ("same", "raised")),
      f"{_lres}")

# Loft G3 to a trimmed edge (sampled: the face's third-order form) -- Free / Align Edges / Align to Surface,
# Bezier Spans and Fitted: both labels G3 and hold
_X, _Y = np.meshgrid(np.linspace(0, 4, 6), np.linspace(-2, 2, 6), indexing="ij")
lgBowl = _mk("LG3_Bowl", np.stack([_X, _Y + 620, 0.4 * np.sin(0.9 * _X) * np.cos(0.7 * _Y) + 0.05 * _X * _Y], -1),
             geom.clamped_uniform_knots(6, 3), geom.clamped_uniform_knots(6, 3))
_lgcut = nurbs("LG3_Cut", [(3.0, 617, 5), (2.7, 619, 5), (3.2, 621, 5), (3.0, 623, 5)])
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.trim(action="KEEP", target=lgBowl.name, cutters=ins((_lgcut, "S0")), ref_point=(1, 620, 0.3),
                    normal=(0, 0, 1), view_dir=(0, 0, -1), projection="Z")
_lge = _trim_edge(lgBowl, lambda m: 2.6 < m.x < 3.3 and abs(m.y - 620) < 1.5)
_X, _Y = np.meshgrid(np.linspace(5, 8, 5), np.linspace(-1.5, 1.5, 5), indexing="ij")
lgPl = _mk("LG3_Plate", np.stack([_X, _Y + 620, 0.3 + 0.2 * np.sin(1.1 * _Y) + 0.1 * (_X - 5) ** 2], -1),
           geom.clamped_uniform_knots(5, 3), geom.clamped_uniform_knots(5, 3))
_lbad = []
for _fit in ("SPANS", "DENSE"):
    for _mode in ("FREE", "EDGES", "SURFACE"):
        _cl._cache.clear()
        r = bpy.ops.nproto.loft(inputs=json.dumps([[lgBowl.name, _lge], [lgPl.name, "U0"]]), cont_a="G3", cont_b="G3",
                                fit=_fit, edges_a=_mode, edges_b=_mode)
        _lo = ctx.active_object
        _lb3 = _cl.labels(_lo)
        if r != {'FINISHED'} or len(_lb3) != 2 or not all(L_["value"] == "G3" and L_["check"]["ok"]
                                                            and L_["check"]["rate"] < 0.01 for L_ in _lb3):
            _lbad.append((_fit, _mode, [L_["check"] and L_["check"]["text"] for L_ in _lb3]))
check("loft G3 to a trimmed edge: Free / Align Edges / Align to Surface, Spans and Fitted -- both labels G3, "
      "curvature rate within 1%", _lge and not _lbad, f"{_lge} {_lbad}")

# Loft's labels and Feature panel go to G3; a Blend Curve's stay G0 - G2
r = bpy.ops.nproto.loft(inputs=ins((lgA, "U1"), (lgB, "U0")), cont_a="G2", cont_b="G2")
_lo = ctx.active_object
_cl._cache.clear()
_vals = [L_["values"] for L_ in _cl.labels(_lo)]
r1 = bpy.ops.nproto.cycle_continuity(obj_name=_lo.name, index=0)
_c1 = features.history(_lo)["params"]["cont_a"]
_cl._cache.clear()
_g3after = _lg3(_lo, lgA, "U1", "U0")
from blender_nurbs import feature_ui as _fui
_fui.sync_from_history(_lo)
_lbc = [nurbs(f"LG3_BC{k}", [(k * 3.0, 640, 0), (k * 3.0 + 1, 640.5, 0.2), (k * 3.0 + 2, 640, 0)]) for k in (0, 1)]
bpy.ops.nproto.blend_curve(inputs=ins((_lbc[0], "S0"), (_lbc[1], "S0")), end_a=1, end_b=0, cont_a="G2", cont_b="G2")
bcf = ctx.active_object
_li = [it[0] for it in _fui._side_cont_items(_lo.nproto_feature, ctx)]
_bi = [it[0] for it in _fui._side_cont_items(bcf.nproto_feature, ctx)]
check("loft labels offer G0 - G3 and a click steps G2 -> G3 (rebuilt G3); the Feature panel offers G3 for a loft "
      "and a blend curve",
      _vals == [["G0", "G1", "G2", "G3"]] * 2 and r1 == {'FINISHED'} and _c1 == "G3" and _g3after[2] < 1e-7
      and _li == ["G0", "G1", "G2", "G3"] and _bi == ["G0", "G1", "G2", "G3"] and _lo.nproto_feature.cont_a == "G3",
      f"{_vals} {_c1} {_g3after} {_li} {_bi} {_lo.nproto_feature.cont_a}")

# --- Boundary: exact Coons edges, degree / spans ------------------------------
_bx = 40.0
bdB = nurbs("BD_B", [(_bx + 0, 0, 0), (_bx + 1, -0.2, 0.3), (_bx + 2, 0.3, -0.2), (_bx + 2.5, -0.1, 0.4), (_bx + 3, 0, 0)])
bdR = nurbs("BD_R", [(_bx + 3, 0, 0), (_bx + 3.4, 1, 0.5), (_bx + 3, 2, 0)], order=3)
bdT = nurbs("BD_T", [(_bx + 0, 2, 0.1), (_bx + 1.5, 2.3, 0.6), (_bx + 3, 2, 0)], order=3)      # runs backwards
bdL = nurbs("BD_L", [(_bx + 0, 2, 0.1), (_bx - 0.3, 1.2, -0.3), (_bx + 0.2, 0.6, 0.2), (_bx + 0, 0, 0)])
_bd_in = ins((bdB, "S0"), (bdR, "S0"), (bdT, "S0"), (bdL, "S0"))
def _bd_err(o):
    face = store.get_brep(o).faces()[0]
    err = 0.0
    for c in (bdB, bdR, bdT, bdL):
        kc = features.input_curve(c, "S0"); t0, t1 = kc.parameter_range()
        err = max(err, max(face.closest_point(tuple(kc.evaluate(t0 + (t1 - t0) * k / 30)))[1] for k in range(31)))
    return err
from blender_nurbs import sweeps as _sw
def _bd(**kw):
    r = bpy.ops.nproto.boundary(inputs=_bd_in, **kw)
    o = ctx.active_object
    d = ext.surface_cvs(store.get_surfaces(o)[0])
    return r, o, (d["degree_u"], d["degree_v"]), (d["count_u"], d["count_v"]), _bd_err(o), dict(_sw.boundary_info)
r, bdE, deg, cnt, err, inf = _bd()
check("boundary: exact Coons -- edges are the input curves, their own degree / CVs",
      r == {'FINISHED'} and inf["exact"] and err < 1e-7 and deg == (3, 3) and cnt == (5, 4), f"{deg} {cnt} {err:.2e} {inf}")
r, bdF, deg, cnt, errF, inf = _bd(bd_method="FIT", samples=12)
check("boundary: Exact Edges off -> sampled fit, edges only approximate", r == {'FINISHED'} and not inf["exact"] and errF > 1e-6,
      f"{deg} {cnt} {errF:.2e}")
r, bd5, deg, cnt, err, inf = _bd(sq_degree_u="5", sq_degree_v="4")
check("boundary: higher degree asked -> still exact (degree raised)", r == {'FINISHED'} and inf["exact"] and deg == (5, 4) and err < 1e-7,
      f"{deg} {cnt} {err:.2e}")
r, bdS, deg, cnt, err, inf = _bd(sq_spans_u=4, sq_spans_v=3)
check("boundary: spans that the curves don't have -> fitted with exactly that layout",
      r == {'FINISHED'} and not inf["exact"] and cnt == (4 + 3, 3 + 3) and err < 2e-2, f"{deg} {cnt} {err:.2e}")

r, bdW, deg, cnt, err, inf = _bd(sq_degree_u="2")
check("boundary: exact not possible -> fitted, and the panel says why", r == {'FINISHED'} and not inf["exact"]
      and "below the curves' degree 3" in (inf.get("why") or "") and deg[0] == 2, f"{inf}")


# --- Patch: N-sided trimmed fill, G0 / G1 per edge ---------------------------
from blender_nurbs import patch as _pt, square as _sqm
from blender_nurbs.drawtools import interpolate as _interp
_smk = _kl()
_PF = lambda x, y: 0.25 * math.sin(0.9 * x) * math.cos(0.7 * y) + 0.05 * x * y
_pxs = np.linspace(-3, 3, 10)
_pbase = ext.surface_from_cvs(3, 3, [(x + 60, y, _PF(x, y)) for x in _pxs for y in _pxs],
                              geom.clamped_uniform_knots(10, 3), geom.clamped_uniform_knots(10, 3))
def _pcurve(srf, a, b, z, n=12):
    Q = []
    for t in np.linspace(0, 1, n):
        x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        r_ = _smk.surface_closest_point(srf, (x, y, z(x, y)))
        r_ = r_[0] if isinstance(r_, tuple) else r_
        Q.append((r_.x, r_.y, r_.z))
    X, d_ = _interp(np.array(Q), 3)
    return ext.curve_from_cvs(d_, [tuple(map(float, x)) for x in X], geom.clamped_uniform_knots(len(X), d_), None)
_pent = [(60 + 1.3 * math.cos(a), 1.1 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 6)[:-1] + 0.3]
_zf = lambda x, y: _PF(x - 60, y)
_hole = [_pcurve(_pbase, _pent[i], _pent[(i + 1) % 5], _zf) for i in range(5)]
_cor = [(57.1, -2.9), (62.9, -2.9), (62.9, 2.9), (57.1, 2.9)]
_outer = [_pcurve(_pbase, _cor[i], _cor[(i + 1) % 4], _zf) for i in range(4)]
ptH = store.new_object(ctx, "PT_Holed", _smk.trim_surface_with_3d_curves(_pbase, [4, 5], _outer + _hole))
_hrefs = [f"E{i}" for i, pl in enumerate(store.display(ptH)["edges"])
          if np.linalg.norm(np.array(pl[len(pl) // 2])[:2] - (60, 0)) < 1.6]
r = bpy.ops.nproto.patch(inputs=ins(*[(ptH, x) for x in _hrefs]), default_cont="G1")
ptP = ctx.active_object
_pm = [_cl.measure(ptP, ptH, x, 1) for x in _hrefs]
check("patch: 5-sided hole in a curved surface -> one trimmed face, G1 on every edge",
      r == {'FINISHED'} and len(_hrefs) == 5 and store.get_brep(ptP).face_count() == 1 and not store.is_natural(ptP)
      and all(m_["ok"] and m_["angle"] < 0.05 for m_ in _pm), str([m_["text"] for m_ in _pm]))
_cl._cache.clear()
_pl = _cl.labels(ptP)
_r2 = bpy.ops.nproto.cycle_continuity(obj_name=ptP.name, index=0, step=-1)
_cl._cache.clear()
_pl2 = _cl.labels(ptP)
check("patch: one label per edge; clicking steps that edge G1 -> G0 and rebuilds",
      len(_pl) == 5 and all(L["value"] == "G1" for L in _pl) and _r2 == {'FINISHED'}
      and _pl2[0]["value"] == "G0" and features.history(ptP)["params"]["cont"][0] == "G0"
      and all(L["check"] and L["check"]["ok"] for L in _pl2), f"{[(L['value'], L['check'] and L['check']['text']) for L in _pl2]}")
# a window in a cylinder wall (about 100 degrees around)
_cr = 1.0
_cyl = ext.surface_from_cvs(3, 3, [(80 + _cr * math.cos(a), _cr * math.sin(a), z) for a in np.linspace(-1.6, 1.6, 9)
                                  for z in np.linspace(-2, 2, 6)], geom.clamped_uniform_knots(9, 3), geom.clamped_uniform_knots(6, 3))
def _ccurve(a0, z0, a1, z1, n=14):
    Q = []
    for t in np.linspace(0, 1, n):
        a, z = a0 + t * (a1 - a0), z0 + t * (z1 - z0)
        r_ = _smk.surface_closest_point(_cyl, (80 + _cr * math.cos(a), _cr * math.sin(a), z))
        r_ = r_[0] if isinstance(r_, tuple) else r_
        Q.append((r_.x, r_.y, r_.z))
    X, d_ = _interp(np.array(Q), 3)
    return ext.curve_from_cvs(d_, [tuple(map(float, x)) for x in X], geom.clamped_uniform_knots(len(X), d_), None)
_win = [(-0.85, -1), (0.85, -1.2), (0.95, 0.4), (0.0, 1.2), (-0.9, 0.6)]
_wc = [_ccurve(*_win[i], *_win[(i + 1) % 5]) for i in range(5)]
_bnd = [(-1.55, -1.9), (1.55, -1.9), (1.55, 1.9), (-1.55, 1.9)]
_bc = [_ccurve(*_bnd[i], *_bnd[(i + 1) % 4]) for i in range(4)]
ptC = store.new_object(ctx, "PT_Cyl", _smk.trim_surface_with_3d_curves(_cyl, [4, 5], _bc + _wc))
_crefs = [f"E{i}" for i, pl in enumerate(store.display(ptC)["edges"]) if abs(np.array(pl[len(pl) // 2])[2]) < 1.3
          and abs(math.atan2(pl[len(pl) // 2][1], pl[len(pl) // 2][0] - 80)) < 1.2]
r = bpy.ops.nproto.patch(inputs=ins(*[(ptC, x) for x in _crefs]), default_cont="G1")
ptPC = ctx.active_object
_pc = [_cl.measure(ptPC, ptC, x, 1) for x in _crefs]
_s = store.get_brep(ptPC).faces()[0].surface()
_dev = []
for a in (-0.4, 0.0, 0.4):
    for z in (-0.5, 0.0, 0.5):
        q_ = _smk.surface_closest_point(_cyl, (80 + math.cos(a), math.sin(a), z))
        q_ = q_[0] if isinstance(q_, tuple) else q_
        q_ = np.array([q_.x, q_.y, q_.z])
        _dev.append(float(np.linalg.norm(np.array(_s.evaluate(*_sqm._foot(_s, q_))) - q_)))
check("patch: window in a cylinder wall -> G1 all round, close to the wall inside",
      r == {'FINISHED'} and len(_crefs) == 5 and all(m_["ok"] for m_ in _pc) and max(_dev) < 0.06,
      f"{[m_['text'] for m_ in _pc]} dev {max(_dev):.4f}")

_cvP = np.array(ext.surface_cvs(_s)["points"]) if hasattr(ext, "surface_cvs") else None
_loopP = np.vstack([np.array([e.evaluate(t) for t in np.linspace(*e.parameter_range(), 20)]) for e in _wc])
_grow = (np.ptp(_cvP, axis=0).max() / np.ptp(_loopP, axis=0).max()) if _cvP is not None else 1.0
check("patch: the net stays close around the opening (four-sided base, no loose CVs)",
      _grow < 1.6 and _pt.info.get("method") == "four-sided", f"net / opening size {_grow:.2f}, {_pt.info}")

# a corner inside one of the four chains (the pentagon has 5) is laid out off
# the straight border, so the surface doesn't crease across from it
_raw = [_sqm.sample_side(ptH, x, 24) for x in _hrefs]
_ord, _ = _pt.chain(_raw)
_sds = [_raw[k].reversed() if rv else _raw[k] for k, rv in _ord]
_idx, _P5, _ = _pt.corners(_sds)
_lay = _pt.layout_quad(_sds, _idx, 0.5)
_starts = np.cumsum([0] + [len(sd.P) - 1 for sd in _sds[:-1]])
_free = [k for k, st in enumerate(_starts) if min(abs(int(st) - i) for i in _idx) > 2]
_off = [float(min(abs(_lay[k][0][0]), abs(1 - _lay[k][0][0]), abs(_lay[k][0][1]), abs(1 - _lay[k][0][1]))) for k in _free]
check("patch: a corner inside a side bulges off the border in (u, v) (no crease across the patch)",
      len(_free) == 1 and _off[0] > 0.03, f"corners {_idx}, free {_free}, off {_off}")

# a trimmed patch's display mesh keeps to Max Edge (the kernel leaves long
# fans where a trim crosses its cells; they shaded as facets)
_bb = store.get_brep(ptPC).bounding_box()
_diag = float(np.linalg.norm(np.array(_bb[1]) - np.array(_bb[0])))
_V = np.array([v.co for v in ptPC.data.vertices])
_me = max(max(np.linalg.norm(_V[pl.vertices[i]] - _V[pl.vertices[(i + 1) % len(pl.vertices)]]) for i in range(len(pl.vertices)))
          for pl in ptPC.data.polygons)
check("patch: display mesh refined to Max Edge on the trimmed face", _me <= 1.01 * store.max_edge_setting(ctx) * _diag,
      f"longest edge {_me:.4f}, limit {store.max_edge_setting(ctx) * _diag:.4f}, {len(ptPC.data.polygons)} tris")

# G2: the 5-sided hole in the freeform surface, curvature-continuous all round
r = bpy.ops.nproto.patch(inputs=ins(*[(ptH, x) for x in _hrefs]), default_cont="G2")
ptG2 = ctx.active_object
_g2m = [_cl.measure(ptG2, ptH, x, 2) for x in _hrefs]
check("patch: G2 on every edge of the 5-sided hole holds (gap, angle, curvature)",
      r == {'FINISHED'} and all(m_["ok"] for m_ in _g2m), str([m_["text"] for m_ in _g2m]))
_cl._cache.clear()
_l2 = _cl.labels(ptG2)
check("patch: labels offer G2", all(L["values"] == ["G0", "G1", "G2"] and L["value"] == "G2" for L in _l2),
      str([(L["value"], L["values"]) for L in _l2]))

# G2 on a strongly curved neighbour (the cylinder window): the cross
# direction stays pinned in the Gauss-Newton model, so curvature follows
r = bpy.ops.nproto.patch(inputs=ins(*[(ptC, x) for x in _crefs]), default_cont="G2")
ptCG2 = ctx.active_object
_cg2 = [_cl.measure(ptCG2, ptC, x, 2) for x in _crefs]
check("patch: G2 on the cylinder window: every edge within 10% curvature and 0.15 deg",
      r == {'FINISHED'} and all(m_["curv"] < 0.10 and m_["angle"] < 0.15 for m_ in _cg2)
      and sum(m_["ok"] for m_ in _cg2) >= 4, str([m_["text"] for m_ in _cg2]))

# the CV net itself runs evenly (net fairness on the Greville differences)
def _net_rough(obj):
    d_ = ext.surface_cvs(store.get_brep(obj).faces()[0].surface())
    G_ = np.array(d_["points"]).reshape(d_["count_u"], d_["count_v"], 3)
    gr = lambda U_, p_, n_: np.array([sum(U_[i + 1:i + p_ + 1]) / p_ for i in range(n_)])
    out_ = []
    for ax, g_ in ((0, gr(d_["knots_u"], d_["degree_u"], d_["count_u"])), (1, gr(d_["knots_v"], d_["degree_v"], d_["count_v"]))):
        X_ = np.moveaxis(G_, ax, 0)
        h_ = np.diff(g_)
        s_ = np.diff(X_, axis=0) / h_[:, None, None]
        dd_ = np.diff(s_, axis=0) / ((h_[1:] + h_[:-1]) / 2)[:, None, None]
        out_.append(np.linalg.norm(np.diff(dd_, axis=0), axis=2).ravel())
    return float(np.median(np.concatenate(out_))) / float(np.linalg.norm(np.ptp(G_.reshape(-1, 3), axis=0)))
_orig_sq = _pt.solve_quad
_rough = []
for _kw in ({"cw": 0.0, "geo": 0}, {"geo": 0}):
    _pt.solve_quad = lambda *a, _k=_kw, **k: _orig_sq(*a, **dict(k, **_k))
    r = bpy.ops.nproto.patch(inputs=ins(*[(ptC, x) for x in _crefs]), default_cont="G1")
    _rough.append(_net_rough(ctx.active_object))
_pt.solve_quad = _orig_sq
r = bpy.ops.nproto.patch(inputs=ins(*[(ptC, x) for x in _crefs]), default_cont="G1")
_rough.append(_net_rough(ctx.active_object))
_pcn = [_cl.measure(ctx.active_object, ptC, x, 1) for x in _crefs]
check("patch: the CV net runs evenly: net fairness evens the (u, v)-faired net, the surface-faired one "
      "is as even again (edges hold)",
      _rough[1] < 0.8 * _rough[0] and _rough[2] <= _rough[1] and all(m_["ok"] for m_ in _pcn),
      "rough plain / + net fairness / + surface passes " + " / ".join(f"{x:.2f}" for x in _rough))

# zebra mesh of a trimmed patch: no folded triangles, and normals bend at most
# a little more than the zebra's angle tolerance across any edge
from blender_nurbs import diagnostics as _dg
_zP, _zN, _zT = _dg.zebra_mesh(ptPC, 2.0, 0.0005)
_zP = np.asarray(_zP, float); _zN = np.asarray(_zN, float)
_zN /= np.linalg.norm(_zN, axis=1)[:, None]
_fn = np.cross(_zP[_zT[:, 1]] - _zP[_zT[:, 0]], _zP[_zT[:, 2]] - _zP[_zT[:, 0]])
_ar = np.linalg.norm(_fn, axis=1)
_nv = _zN[_zT[:, 0]] + _zN[_zT[:, 1]] + _zN[_zT[:, 2]]
_ok = _ar > 1e-2 * np.median(_ar)          # slivers along the trim edge are invisible
_cos = np.sum(_fn * _nv, axis=1)[_ok] / (_ar[_ok] * np.linalg.norm(_nv, axis=1)[_ok])
_eang = np.degrees(np.arccos(np.clip(np.concatenate([np.sum(_zN[_zT[:, i]] * _zN[_zT[:, j]], axis=1)
                                                     for i, j in ((0, 1), (1, 2), (2, 0))]), -1, 1)))
check("zebra mesh: no folded triangles, normals within the angle tolerance across edges",
      int(np.sum(_cos < 0.3)) == 0 and float(np.percentile(_eang, 99.9)) < 2.5,
      f"folded {int(np.sum(_cos < 0.3))}, edge normal angle p99.9 {np.percentile(_eang, 99.9):.2f} deg, {len(_zT)} tris")

# crack repair: open edges of the fine zebra mesh lie on the B-rep's edges
# only, and a real trimmed hole stays open
def _open_edges(T_):
    c_ = {}
    for t in T_:
        t = [int(x) for x in t]
        for k in range(len(t)):
            a_, b_ = t[k], t[(k + 1) % len(t)]
            key = (min(a_, b_), max(a_, b_))
            c_[key] = c_.get(key, 0) + 1
    return [k for k, v in c_.items() if v == 1]
_zP, _zN, _zT = _dg.zebra_mesh(ptCG2, 1.0, 0.0005)
_zP = np.asarray(_zP, float)
_E = store._edge_samples(store.get_brep(ptCG2), 400)
_op = _open_edges(_zT)
_far = [e for e in _op if np.min(np.linalg.norm(_E - (_zP[e[0]] + _zP[e[1]]) / 2, axis=1)) > 0.01]
_hl = _open_edges(store._tessellate(store.get_brep(ptH), *store.tess_settings(), store.edge_angle_setting(), store.max_edge_setting())[2])
_hP = np.asarray(store._tessellate(store.get_brep(ptH), *store.tess_settings(), store.edge_angle_setting(), store.max_edge_setting())[0], float)
_near_hole = [e for e in _hl if np.linalg.norm(((_hP[e[0]] + _hP[e[1]]) / 2)[:2] - (60, 0)) < 1.6]
check("tessellation: no cracks inside a face, a real trimmed hole stays open",
      len(_far) <= 0.01 * len(_op) and len(_near_hole) >= 10,
      f"open edges off the B-rep edges {len(_far)} of {len(_op)}; hole rim edges {len(_near_hole)}")

# corner conflict: a square opening between four flat strips, one strip
# tilted 3 degrees about its own edge -- its tangent plane can't contain the
# neighbouring edges, so G1 there is eased toward both its corners (orange
# label) instead of squashing the patch flat at the corner (a fold)
def _strip(name, a, b, out, k=1):
    a, b = np.array(a, float), np.array(b, float)
    o_ = np.array(out, float)
    pts = [tuple(a), tuple(a + o_), tuple(b), tuple(b + o_)]
    return store.new_object(ctx, name, ext.surface_from_cvs(1, 1, pts, [0, 0, 1, 1], [0, 0, 1, 1]))
_th = math.radians(3.0)
_X0 = 200.0
_sq = [(_X0, 0, 0), (_X0 + 4, 0, 0), (_X0 + 4, 4, 0), (_X0, 4, 0)]
_strips = [_strip("CC_A", _sq[0], _sq[1], (0, -2 * math.cos(_th), 2 * math.sin(_th))),
           _strip("CC_B", _sq[1], _sq[2], (2, 0, 0)),
           _strip("CC_C", _sq[2], _sq[3], (0, 2, 0)),
           _strip("CC_D", _sq[3], _sq[0], (-2, 0, 0))]
def _inner_edge(o_, a, b):
    m_ = (np.array(a) + np.array(b)) / 2
    eds = store.display(o_)["edges"]
    return f"E{min(range(len(eds)), key=lambda i: np.linalg.norm(np.array(eds[i][len(eds[i]) // 2]) - m_))}"
_cin = [(o_, _inner_edge(o_, _sq[i], _sq[(i + 1) % 4])) for i, o_ in enumerate(_strips)]
_cc_ref = {}
for o_, r_ in _cin:
    _cc_ref[o_.name] = r_
_ccn = [_strips[i].name for i in range(4)]
r = bpy.ops.nproto.patch(inputs=ins(*_cin), default_cont="G1", pt_quad=False)    # the trimmed fill
ptCC = ctx.active_object
_cl._cache.clear()
_ccl = _cl.labels(ptCC)
_s = store.get_brep(ptCC).faces()[0].surface()
(_u0, _u1), (_v0, _v1) = _s.u_domain(), _s.v_domain()
_J = np.array([np.linalg.norm(np.cross(*_s.derivatives(u_, v_)[1:])) for u_ in np.linspace(_u0, _u1, 40)
               for v_ in np.linspace(_v0, _v1, 40)])
_byname = {L["name"].split()[0]: L for L in _ccl}
check("patch: a corner no surface can be G1 to is eased (orange), the other edges hold, no fold",
      r == {'FINISHED'} and _byname["CC_A"]["check"].get("eased") and not _byname["CC_A"]["check"]["ok"]
      and all(_byname[n_]["check"]["ok"] for n_ in ("CC_B", "CC_C", "CC_D")) and _J.min() / np.median(_J) > 0.1,
      f"{[(n_, _byname[n_]['check']['text'][:70]) for n_ in _ccn]} jmin {_J.min() / np.median(_J):.3f}")
_mk = _cl.corner_marks(ptCC)
_mkp = sorted([tuple(np.round(np.array(m_["pos"]), 3)) for m_ in _mk])
check("patch: a square marker on each eased corner, with the angle",
      len(_mk) == 2 and _mkp == sorted([tuple(map(float, _sq[0])), tuple(map(float, _sq[1]))])
      and all(abs(m_["deg"] - 3.0) < 0.1 for m_ in _mk), str([(p_, round(m_["deg"], 2)) for p_, m_ in zip(_mkp, _mk)]))

# --- Patch robustness: openings of many kinds ------------------------------
from mathutils import Matrix as _Mx
_RF = {}
def _rb_base(kind, X0):
    xs = np.linspace(-3, 3, 10)
    f = {"free": lambda x, y: 0.25 * math.sin(0.9 * x) * math.cos(0.7 * y) + 0.05 * x * y,
         "flat": lambda x, y: 0.0,
         "dome": lambda x, y: 0.6 - 0.07 * (x * x + y * y)}[kind]
    U_ = geom.clamped_uniform_knots(10, 3)
    srf = ext.surface_from_cvs(3, 3, [(x + X0, y, f(x, y)) for x in xs for y in xs], U_, U_)
    _RF[id(srf)] = (f, X0)
    return srf
def _rb_on(srf, xy):
    f, X0 = _RF[id(srf)]
    Q = []
    for x, y in xy:
        r_ = _smk.surface_closest_point(srf, (x, y, f(x - X0, y)))
        r_ = r_[0] if isinstance(r_, tuple) else r_
        Q.append((r_.x, r_.y, r_.z))
    X_, d_ = _interp(np.array(Q), 3)
    return ext.curve_from_cvs(d_, [tuple(map(float, x)) for x in X_], geom.clamped_uniform_knots(len(X_), d_), None)
_rb_seg = lambda a, b, n=14: [(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])) for t in np.linspace(0, 1, n)]
_rb_arc = lambda c, r, a0, a1, n=16: [(c[0] + r * math.cos(a), c[1] + r * math.sin(a)) for a in np.linspace(a0, a1, n)]
_rb_poly = lambda pts: [_rb_seg(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
def _rb_holed(name, kind, X0, pieces, M=None):
    srf = _rb_base(kind, X0)
    cor = [(X0 - 2.9, -2.9), (X0 + 2.9, -2.9), (X0 + 2.9, 2.9), (X0 - 2.9, 2.9)]
    outer = [_rb_on(srf, _rb_seg(cor[i], cor[(i + 1) % 4])) for i in range(4)]
    hole = [_rb_on(srf, [(x + X0, y) for x, y in pc]) for pc in pieces]
    o_ = store.new_object(ctx, name, _smk.trim_surface_with_3d_curves(srf, [4, len(hole)], outer + hole))
    eds = store.display(o_)["edges"]
    refs = [f"E{i}" for i, pl in enumerate(eds) if pl and max(abs(pl[len(pl) // 2][0] - X0), abs(pl[len(pl) // 2][1])) < 2.4]
    if M is not None:
        o_.matrix_world = M
    return o_, refs, srf
def _rb_jmin(o_):
    s_ = store.get_brep(o_).faces()[0].surface()
    (a0, a1), (b0, b1) = s_.u_domain(), s_.v_domain()
    J_ = np.array([np.linalg.norm(np.cross(*s_.derivatives(u_, v_)[1:])) for u_ in np.linspace(a0, a1, 30)
                   for v_ in np.linspace(b0, b1, 30)])
    return J_.min() / np.median(J_)
def _rb_dev(o_, srf, X0, pieces):
    pts = np.vstack([np.array(pc) for pc in pieces]); c_ = pts.mean(axis=0)
    s_ = store.get_brep(o_).faces()[0].surface(); f, _ = _RF[id(srf)]; d_ = 0.0
    for fr in (0.0, 0.3, 0.6):
        for p_ in pts[::3]:
            q_ = c_ + fr * (p_ - c_)
            r_ = _smk.surface_closest_point(srf, (q_[0] + X0, q_[1], f(q_[0], q_[1])))
            r_ = r_[0] if isinstance(r_, tuple) else r_
            P_ = np.array([r_.x, r_.y, r_.z])
            d_ = max(d_, float(np.linalg.norm(np.array(s_.evaluate(*_sqm._foot(s_, P_))) - P_)))
    return d_
_Lsh = [(-1.2, -1.2), (1.2, -1.2), (1.2, 0.0), (0.0, 0.0), (0.0, 1.2), (-1.2, 1.2)]
_star = [((1.4 if i % 2 == 0 else 0.7) * math.cos(a), (1.4 if i % 2 == 0 else 0.7) * math.sin(a))
         for i, a in enumerate(np.linspace(0, 2 * math.pi, 11)[:-1])]
_RB = [  # name, kind, pieces, continuity, max deviation from the surface the hole was cut from, matrix
    ("triangle", "free", _rb_poly([(-1.3, -1.0), (1.4, -0.8), (0.1, 1.3)]), "G2", 0.02, None),
    ("L_flat", "flat", _rb_poly(_Lsh), "G1", 1e-4, None),
    ("L_free", "free", _rb_poly(_Lsh), "G2", 0.02, None),
    ("star", "dome", _rb_poly(_star), "G1", 0.02, None),
    ("circle2", "free", [_rb_arc((0, 0), 1.2, 0, math.pi), _rb_arc((0, 0), 1.2, math.pi, 2 * math.pi)], "G2", 0.02, None),
    ("L_xform", "free", _rb_poly(_Lsh), "G2", None,
     _Mx.Translation((5, -3, 2)) @ _Mx.Rotation(0.5, 4, 'X') @ _Mx.Rotation(0.4, 4, 'Z') @ _Mx.Scale(2.0, 4)),
]
_rb_out = []
for _i, (_nm, _kd, _pc, _cn, _dmax, _M) in enumerate(_RB):
    _X0 = 300.0 + 10.0 * _i
    _o, _rf, _sf = _rb_holed("RB_" + _nm, _kd, _X0, _pc, _M)
    try:
        r = bpy.ops.nproto.patch(inputs=ins(*[(_o, x) for x in _rf]), default_cont=_cn)
    except Exception as exc:
        _rb_out.append((_nm, "EXC " + str(exc)[:80])); continue
    _P = ctx.active_object
    _cl._cache.clear()
    _bad = [L["check"]["text"][:40] for L in _cl.labels(_P) if L["check"] and not L["check"]["ok"]]
    _jm = _rb_jmin(_P)
    _dv = _rb_dev(_P, _sf, _X0, _pc) if _dmax is not None else 0.0
    _rb_out.append((_nm, r == {'FINISHED'} and not _bad and _jm > 0.1 and (_dmax is None or _dv < _dmax),
                    len(_rf), _bad[:2], round(_jm, 3), round(_dv, 4)))
check("patch robustness: triangle, concave L (flat / curved / transformed), 5-point star, two-arc circle "
      "-- every edge holds, no fold, close to the surface the hole was cut from",
      all(x[1] is True for x in _rb_out), str(_rb_out))
_cv5 = []
for _i, (_a, _b) in enumerate(zip([(400 + 1.3 * math.cos(t), 1.3 * math.sin(t), 0.3 * math.sin(2 * t))
                                   for t in np.linspace(0, 2 * math.pi, 6)[:-1]], [None] * 5)):
    _cv5.append(_a)
_cvo = [nurbs(f"RB_cv{i}", [_cv5[i], _cv5[(i + 1) % 5]], order=2) for i in range(5)]
r = bpy.ops.nproto.patch(inputs=ins(*[(o_, "S0") for o_ in _cvo]))
check("patch robustness: five plain curves in space (G0)", r == {'FINISHED'} and _rb_jmin(ctx.active_object) > 0.1,
      f"jmin {_rb_jmin(ctx.active_object):.3f}")

# fairness measured on the surface (not in (u, v)): where the layout bends
# the parameter lines, the (u, v) thin plate left ridges -- curvature varies
# far less inside with the surface passes
def _pt_hvar(G, U, p, loop, m=160):
    """int |grad H|^2 dA (times the loop's size^2) inside the loop, a little
    off it."""
    nu, nv = G.shape[:2]
    X = G.reshape(-1, 3)
    uv = _pt._project(X, _pt._grid_uv(X, (nu, nv), p, U, loop), loop, (nu, nv), p, U, it=4)
    t = np.linspace(0.0, 1.0, m)
    def B(n_, U_):
        out = np.zeros((3, m, n_))
        for a, s in enumerate(t):
            sp = geom.find_span(n_ - 1, p, float(s), U_)
            out[:, a, sp - p:sp + 1] = _pt.ders_basis(sp, float(s), p, U_, 2)
        return out
    Bu, Bv = B(nu, U[0]), B(nv, U[1])
    S = lambda a, b: np.einsum("ai,ijc,bj->abc", Bu[a], G, Bv[b])
    Su, Sv, Suu, Suv, Svv = S(1, 0), S(0, 1), S(2, 0), S(1, 1), S(0, 2)
    N = np.cross(Su, Sv)
    N /= np.linalg.norm(N, axis=2, keepdims=True)
    E, F, Gg = (Su * Su).sum(2), (Su * Sv).sum(2), (Sv * Sv).sum(2)
    det = E * Gg - F * F
    H = (E * (Svv * N).sum(2) - 2 * F * (Suv * N).sum(2) + Gg * (Suu * N).sum(2)) / (2 * det)
    Hu, Hv = np.gradient(H, t, t)
    g2 = (Gg * Hu * Hu - 2 * F * Hu * Hv + E * Hv * Hv) / det
    UU, VV = np.meshgrid(t, t, indexing="ij")
    x, y = UU.ravel(), VV.ravel()
    msk = np.zeros(len(x), bool)
    px, py = uv[:, 0], uv[:, 1]
    for a, b, c, d in zip(px, py, np.roll(px, -1), np.roll(py, -1)):
        msk ^= ((b > y) != (d > y)) & (x < a + (y - b) * (c - a) / ((d - b) if d != b else 1e-30))
    msk = msk.reshape(m, m)
    for _ in range(int(m * 0.06)):
        msk[1:-1, 1:-1] &= msk[:-2, 1:-1] & msk[2:, 1:-1] & msk[1:-1, :-2] & msk[1:-1, 2:]
    return float((g2 * np.sqrt(det) * (t[1] - t[0]) ** 2)[msk].sum()) * float(np.linalg.norm(np.ptp(loop, axis=0))) ** 2
_hv = {}
_hv_objs = [("pentagon G2", ptG2), ("cylinder window G2", ptCG2)] + \
    [(nm, ob) for nm in ("L_free", "star") for ob in bpy.data.objects
     if (features.history(ob) or {}).get("params", {}).get("pt_method") == "QUAD"
     and features.history(ob)["inputs"] and features.history(ob)["inputs"][0][0].name == "RB_" + nm]
for _nm, _ob in _hv_objs:
    _h = features.history(_ob)
    _sds, _ord, _, _, _ = _pt.loop_sides(_h["inputs"], _h["params"])
    _sc = [_pt.CONT.get(_h["params"]["cont"][k], 1) for k, _ in _ord]
    _ix = _pt.corners(_sds, fixed=_pt.corner_loop_indices(_sds, _ord, []))[0]
    _lp = np.vstack([sd.P for sd in _sds])
    _hv[_nm] = []
    for _g in (0, _pt.GEO_PASSES):
        _rp = {"check": True}
        _G, _U, _ = _pt.solve_quad(_sds, _sc, n=(24, 24), p=5, fair=1.0, idx=_ix, geo=_g, report=_rp)
        _hv[_nm].append((_pt_hvar(_G, _U, 5, _lp), max(e[0] for e in _rp["errors"])))
check("patch: fairness on the surface -- curvature varies far less inside than with the (u, v) thin plate "
      "(pentagon, cylinder window, L, star; under 0.6x), edges as close",
      len(_hv) == 4 and all(v[1][0] < 0.6 * v[0][0] and v[1][1] < max(2 * v[0][1], 5e-5) for v in _hv.values()),
      str({k: [(round(a, 3), f"{b:.1e}") for a, b in v] for k, v in _hv.items()}))

# --- Patch solver speed-ups: the vectorised basis is the scalar one to the
# last bit, and the block solve's premise -- nothing couples CV rows more
# than the degree apart -- holds for every system a Patch solves
_rng = np.random.default_rng(7)
_bd = 0.0
for _p in (1, 2, 3, 5):
    for _n in (_p + 1, _p + 5, 24):
        _Uk = geom.clamped_uniform_knots(_n, _p)
        _ts = np.concatenate([_rng.random(60), np.unique(_Uk)])
        _sp = _pt.find_spans(_n - 1, _p, _ts, _Uk)
        _bd = max(_bd, float(np.abs(_sp - [geom.find_span(_n - 1, _p, float(t), _Uk) for t in _ts]).max()))
        _D = _pt.ders_basis_many(_sp, _ts, _p, _Uk, min(3, _p))
        _bd = max(_bd, max(float(np.abs(_D[k] - _pt.ders_basis(int(s), float(t), _p, _Uk, min(3, _p))).max())
                           for k, (s, t) in enumerate(zip(_sp, _ts))))
_U2 = (geom.clamped_uniform_knots(9, 3), geom.clamped_uniform_knots(12, 3))
_uvs = _rng.random((30, 2))
_bd = max(_bd, float(np.abs(_pt.surface_rows_many(_uvs, (9, 12), 3, _U2) -
                            np.array([_pt.surface_rows(float(a), float(b), (9, 12), 3, _U2) for a, b in _uvs])).max()))
_Su, _Sv, _r = _rng.normal(size=(20, 3)), _rng.normal(size=(20, 3)), _rng.normal(size=(20, 3))
_Sv[0] = 2 * _Su[0]                                   # a degenerate pair: lstsq's minimum-norm answer
_l2 = float(np.abs(_pt.lstsq2(_Su, _Sv, _r) - np.array([np.linalg.lstsq(np.c_[a, b], c, rcond=None)[0]
                                                        for a, b, c in zip(_Su, _Sv, _r)])).max())
check("patch solver: vectorised spans / basis / rows equal the scalar ones exactly, lstsq2 = lstsq",
      _bd == 0.0 and _l2 < 1e-12, f"basis {_bd}, lstsq2 {_l2:.1e}")
_h = features.history(ptG2)
_sds, _ord, _, _, _ = _pt.loop_sides(_h["inputs"], _h["params"])
_sc = [_pt.CONT.get(_h["params"]["cont"][k], 1) for k, _ in _ord]
_ix = _pt.corners(_sds, fixed=_pt.corner_loop_indices(_sds, _ord, []))[0]
_real_ns = _pt.net_solve
_seen = []
def _ns_spy(Q, b, nu, p):
    nv = len(Q) // (3 * nu)
    R = (Q.reshape(nu, 3 * nv, nu, 3 * nv) != 0).any(axis=(1, 3))
    _i, _j = np.nonzero(R)
    x = _real_ns(Q, b, nu, p)
    _seen.append((int(np.abs(_i - _j).max()), max(p, 2),
                  float(np.abs(x - np.linalg.solve(Q, b)).max() / np.abs(np.linalg.solve(Q, b)).max())))
    return x
_pt.net_solve = _ns_spy
try:
    for _n2, _pp, _kw in (((24, 24), 5, {}), ((20, 14), 3, {"tension": [1.8] + [1.0] * (len(_sds) - 1)})):
        _pt.solve_quad(_sds, _sc, n=_n2, p=_pp, fair=1.0, idx=_ix, **_kw)
finally:
    _pt.net_solve = _real_ns
check("patch solver: every system is zero between CV rows more than the degree apart (degree 5 and 3, "
      "tension), and the block solve gives the dense solve's answer",
      len(_seen) >= 8 and all(bw <= lim and d < 1e-7 for bw, lim, d in _seen),
      str([(bw, lim, f"{d:.0e}") for bw, lim, d in _seen]))

# --- Patch: separate U / V counts, lightest net, tension, corners by hand, guide curves
from blender_nurbs import feature_ui as _fui
def _pt_net(o_):
    d_ = _smk.blender_ext.surface_cvs(store.get_brep(o_).faces()[0].surface())
    return d_["count_u"], d_["count_v"], np.array(d_["points"], float)
def _pt_bad(o_):
    _cl._cache.clear()
    return [L["name"] + " " + L["check"]["text"][:40] for L in _cl.labels(o_)
            if L["check"] and not L["check"]["ok"] and not L["check"].get("eased")]
_o, _rf, _sf = _rb_holed("PF_L", "free", 400.0, _rb_poly(_Lsh))
_loop = [(_o, x) for x in _rf]
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G1", pt_cvs=16, pt_cvs_v=10)
_P = ctx.active_object
check("patch: separate CV counts U / V (16 x 10), every edge holds",
      r == {'FINISHED'} and _pt_net(_P)[:2] == (16, 10) and not _pt_bad(_P), str((_pt_net(_P)[:2], _pt_bad(_P))))
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G2", pt_auto_cvs=True)
_P = ctx.active_object
_nu, _nv, _ = _pt_net(_P)
check("patch: Lightest -- far fewer CVs than 24 x 24, every edge still holds G2, no fold",
      r == {'FINISHED'} and _nu * _nv <= 16 * 16 and not _pt_bad(_P) and _rb_jmin(_P) > 0.1,
      str((_nu, _nv, _pt_bad(_P), round(_rb_jmin(_P), 3))))
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G1")
_P = ctx.active_object
_, _, _G0 = _pt_net(_P)
_size = float(np.linalg.norm(np.ptp(_G0, axis=0)))
_fui.sync_from_history(_P)
_pg = _P.nproto_feature
_moves = []
for _t in (1.8, 0.6):
    _pg.pt_edges[0].tension = _t
    _moves.append((_t, round(float(np.abs(_pt_net(_P)[2] - _G0).max()) / _size, 4), _pt_bad(_P)))
_pg.pt_edges[0].tension = 1.0
_back = float(np.abs(_pt_net(_P)[2] - _G0).max())
check("patch: tension per edge (panel) reshapes the patch, every edge keeps G1; back to 1 = the plain patch",
      all(m_ > 0.005 and not b_ for _, m_, b_ in _moves) and _back < 1e-9
      and "pt_tension" not in features.history(_P)["params"], str((_moves, _back)))
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G2", pt_tension_all=1.6)
check("patch: tension on every edge (operator) with G2, every edge holds",
      r == {'FINISHED'} and not _pt_bad(ctx.active_object), str(_pt_bad(ctx.active_object)))
# fairness on the surface: on by default (not stored); off in the operator =
# the (u, v) fairness (stored); the Feature panel switches it back on
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G2", pt_surface_fair=False)
_Ps = ctx.active_object
_, _, _Goff = _pt_net(_Ps)
_fui.sync_from_history(_Ps)
_pgs = _Ps.nproto_feature
_off_ok = (r == {'FINISHED'} and features.history(_Ps)["params"].get("pt_surface_fair") is False
           and _pgs.pt_surface_fair is False and not _pt_bad(_Ps))
_pgs.pt_surface_fair = True
_, _, _Gon = _pt_net(_Ps)
_on_ok = features.history(_Ps)["params"].get("pt_surface_fair") is True and not _pt_bad(_Ps)
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G2")
_Pd = ctx.active_object
_, _, _Gdef = _pt_net(_Pd)
_fui.sync_from_history(_Pd)
_sw = (float(np.abs(_Gon - _Goff).max()) / _size, float(np.abs(_Gon - _Gdef).max()))
check("patch: 'On Surface' fairness -- on by default (not stored), off in the operator = (u, v) fairness (stored), "
      "the Feature panel switches it back on (= the default patch), edges hold",
      _off_ok and _on_ok and "pt_surface_fair" not in features.history(_Pd)["params"] and _Pd.nproto_feature.pt_surface_fair
      and _sw[0] > 1e-3 and _sw[1] < 1e-9, str((_off_ok, _on_ok, _sw)))
# corners by hand: click a junction that isn't a corner -> the nearest corner moves there
_cps = _pt.corner_points(_P)
_j = next(i for i, c_ in enumerate(_cps) if not c_["corner"])
_jk = (_cps[_j]["k"], _cps[_j]["f"])
r = bpy.ops.nproto.patch_corner(obj_name=_P.name, index=_j)
_pc_ = features.history(_P)["params"].get("pt_corners")
_now = [(c_["k"], c_["f"]) for c_ in _pt.corner_points(_P) if c_["corner"]]
_moved = float(np.abs(_pt_net(_P)[2] - _G0).max())
r2 = bpy.ops.nproto.patch_corner(obj_name=_P.name, index=0, reset=True)
check("patch: a click on a junction makes it a corner (stored, edges hold); Alt+click back to automatic",
      r == {'FINISHED'} and _pc_ and len(_pc_) == 4 and _jk in _now and _moved > 1e-4 and not _pt_bad(_P)
      and r2 == {'FINISHED'} and "pt_corners" not in features.history(_P)["params"]
      and float(np.abs(_pt_net(_P)[2] - _G0).max()) < 1e-9, str((_pc_, _now, _jk, _moved)))
# guide curves: one inside the opening, lifted 0.12 above the surface the hole was cut from
_f, _ = _RF[id(_sf)]
_gp = []
for _k in range(13):
    _x, _y = -0.9 + 0.6 * _k / 12, -0.6 + 1.2 * _k / 12
    _q = _smk.surface_closest_point(_sf, (_x + 400.0, _y, _f(_x, _y)))
    _q = _q[0] if isinstance(_q, tuple) else _q
    _gp.append((_q.x, _q.y, _q.z + 0.12 * math.sin(math.pi * _k / 12) ** 2))
_go = nurbs("PF_guide", _gp)
ctx.view_layer.update()
r = bpy.ops.nproto.patch(inputs=ins(*([(_go, "S0")] + _loop)), default_cont="G1")   # guide picked first
_P = ctx.active_object
_h = features.history(_P)
_gl = [L for L in _cl.labels(_P) if L.get("guide")]
check("patch: a curve inside the opening is a guide -- the patch passes through it, every edge holds",
      r == {'FINISHED'} and _h["params"].get("pt_guides") == [0] and _gl and _gl[0]["check"]["ok"]
      and not _pt_bad(_P) and _pt.patch_info(_P).get("guide_dev", 1.0) < 1e-3,
      str((_h["params"].get("pt_guides"), _pt.patch_info(_P).get("guide_dev"), _pt_bad(_P))))
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G1")
_P = ctx.active_object
ctx.view_layer.objects.active = _P
r = bpy.ops.nproto.patch_add_guides(target=_P.name, inputs=ins((_go, "S0")))
_ok1 = r == {'FINISHED'} and features.history(_P)["params"].get("pt_guides") == [6] and not _pt_bad(_P)
r = bpy.ops.nproto.patch_remove_guide(index=6)
check("patch: guides added afterwards and removed again (back to the plain patch)",
      _ok1 and r == {'FINISHED'} and not features.history(_P)["params"].get("pt_guides")
      and len(features.history(_P)["inputs"]) == 6 and float(np.abs(_pt_net(_P)[2] - _G0).max()) < 1e-9,
      str(features.history(_P)["params"]))
# which picks are guides: dangling ones; a guide from corner to corner by pick order
_diag = nurbs("PF_diag", [_cps[0]["pos"][:], _cps[2]["pos"][:]], order=2)
check("patch: guides told from the loop (dangling, or picked after the loop closed)",
      _pt.split_loop([(_go, "S0")] + _loop) == [0] and _pt.split_loop(_loop) == []
      and _pt.split_loop(_loop + [(_diag, "S0")]) == [6],
      str((_pt.split_loop([(_go, "S0")] + _loop), _pt.split_loop(_loop + [(_diag, "S0")]))))

# --- Pause Rebuild: settings wait, Rebuild builds them; per-edge rows; "All edges" row; operator's edge rows
from blender_nurbs import history as _hist
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G1")
_P = ctx.active_object
_G1n = _pt_net(_P)[2]
ctx.scene.nproto_settings.pause_rebuild = True
_cl._cache.clear()
_li = next(i for i, L in enumerate(_cl.labels(_P)) if L["param"] == "pc:0")
r1 = bpy.ops.nproto.cycle_continuity(obj_name=_P.name, index=_li)          # G1 -> G2, held back
_pg = _P.nproto_feature
_pg.pt_edges[1].cont = "G0"                                                 # panel row, held back
_pg.pt_edges[2].tension = 1.5
_held_same = float(np.abs(_pt_net(_P)[2] - _G1n).max()) < 1e-12
_cl._cache.clear()
_unchk = all(L["check"] is None and L["pending"] for L in _cl.labels(_P))
_wt = sorted(_hist.waiting(ctx.scene))
_hp = features.history(_P)["params"]
r2 = bpy.ops.nproto.rebuild_waiting()
_moved = float(np.abs(_pt_net(_P)[2] - _G1n).max())
check("pause rebuild: label click and panel edits only change the settings (geometry unchanged, labels "
      "unchecked); Rebuild builds them and every edge holds",
      r1 == {'FINISHED'} and _held_same and _unchk and _wt == [_P.name]
      and _hp["cont"][:3] == ["G2", "G0", "G1"] and abs(_hp["pt_tension"][2] - 1.5) < 1e-9
      and r2 == {'FINISHED'} and not features.pending(_P) and _moved > 1e-4 and not _pt_bad(_P)
      and not _hist.waiting(ctx.scene), str((_held_same, _unchk, _wt, _hp.get("cont"), _moved, _pt_bad(_P))))
_pg.pt_edges[1].cont = "G1"
_pg.pt_edges[2].tension = 1.0
_cl._cache.clear()
_li = next(i for i, L in enumerate(_cl.labels(_P)) if L["param"] == "pc:0")
bpy.ops.nproto.cycle_continuity(obj_name=_P.name, index=_li, step=-1)      # G2 -> G1 (Ctrl+click)
_w2 = features.pending(_P)
ctx.scene.nproto_settings.pause_rebuild = False                              # switching off builds
check("pause rebuild: Ctrl+click steps back; switching the pause off builds what waited (= the plain G1 patch)",
      _w2 and not features.pending(_P) and float(np.abs(_pt_net(_P)[2] - _G1n).max()) < 1e-9
      and features.history(_P)["params"]["cont"][:3] == ["G1", "G1", "G1"], str(features.history(_P)["params"]))
# "All edges" row in the Feature panel
_fui.sync_from_history(_P)
_all0 = _pg.pt_cont_all
_pg.pt_cont_all = "G2"
_hp = features.history(_P)["params"]
_ok_all = _hp.get("pt_all") == "G2" and set(_hp["cont"]) == {"G2"} and all(e.cont == "G2" for e in _pg.pt_edges) \
    and not _pt_bad(_P)
_pg.pt_edges[3].cont = "G1"
_mixed = _pg.pt_cont_all == "MIXED" and "pt_all" not in features.history(_P)["params"]
_pg.pt_tension_all = 1.3
_tall = np.allclose(features.history(_P)["params"].get("pt_tension", [0]), [1.3] * 6, atol=1e-6) and all(abs(e.tension - 1.3) < 1e-6 for e in _pg.pt_edges)
_pg.pt_tension_all = 1.0
check("patch panel: 'All edges' sets every edge (stored), an edge set by itself makes it Mixed; all tensions at once",
      _all0 == "G1" and _ok_all and _mixed and _tall and "pt_tension" not in features.history(_P)["params"],
      str((_all0, _ok_all, _mixed, _tall, features.history(_P)["params"])))
# the operator's redo panel: edge rows (a redo keeps them; an edge set by itself -> Mixed)
_rows = [{"name": "", "index": k, "label": "e", "curve": False, "cont": "G2" if k == 0 else "G1", "tension": 1.4 if k == 1 else 1.0}
         for k in range(6)]
r = bpy.ops.nproto.patch(inputs=ins(*_loop), default_cont="G1", last_all="G1", last_conts=",".join(["G1"] * 6),
                         pt_tension_all=1.0, last_tension=1.0, edges=_rows)
_hp = features.history(ctx.active_object)["params"]
check("patch operator: per-edge continuity and tension rows (redo panel) are built as set",
      r == {'FINISHED'} and _hp["cont"] == ["G2"] + ["G1"] * 5 and abs(_hp.get("pt_tension", [0, 0])[1] - 1.4) < 1e-6
      and "pt_all" not in _hp and not _pt_bad(ctx.active_object), str(_hp))

# --- tessellation: a periodic surface's seam meshes without folds (u wraps instead of clamping at the seam)
from blender_nurbs import store as _st
_fold_counts = []
for _op, _p in (("prim_cylinder", {"radius": 1.0, "height": 2.0, "degree": 3, "spans_u": 8, "spans_v": 1, "caps": False}),
                ("prim_torus", {"radius": 1.0, "radius2": 0.3, "degree": 0, "spans_u": 8, "spans_v": 8})):
    _b = _st.to_brep(features.run(_op, [], _p))
    _P, _N, _T = _st._tessellate(_b, 0.002, 15.0, 2.0, 1.0 / 30.0)
    _fold_counts.append((_op, _st._periodic(_b.faces()[0].surface()), len(_st._folded(np.asarray(_P, float), np.asarray(_N, float), _st._tri_array(_T)))))
check("tessellation: tube and torus (periodic) display meshes have no folded triangles at the seam",
      all(per_[0] and n_ == 0 for _, per_, n_ in _fold_counts), str(_fold_counts))

# --- UI: the four tool groups list every tool once; panels and menus draw without errors
from types import SimpleNamespace as _NS
from blender_nurbs import toolgroups as _tg, ui as _ui, primitives as _prims


class _FakeProps:
    def __init__(self, d): object.__setattr__(self, "_d", d)
    def __setattr__(self, k, v): self._d[k] = v
    def __getattr__(self, k): return self._d.get(k)


class _FakeLay:
    def __init__(self, rec): object.__setattr__(self, "_rec", rec)
    def __setattr__(self, k, v): pass
    def __getattr__(self, name):
        rec = self._rec
        def f(*a, **k):
            if name == "operator":
                e = {"op": a[0], "props": {}}
                rec.append(e)
                return _FakeProps(e["props"])
            if name == "menu":
                rec.append({"menu": a[0]})
            if name == "prop":
                rec.append({"prop": a[1]})
            return _FakeLay(rec)
        return f


def _draw(cls, **extra):
    rec = []
    cls.draw(_NS(layout=_FakeLay(rec), **extra), ctx)
    return rec


_ui_err, _drawn = [], {}
for _cls in _tg.classes:
    try:
        if issubclass(_cls, bpy.types.Panel):
            _drawn[_cls.__name__] = _draw(_cls, group=_cls.group, hint=_cls.hint)
        else:
            _drawn[_cls.__name__] = _draw(_cls, group=getattr(_cls, "group", ""))
    except Exception as _e:
        _ui_err.append(f"{_cls.__name__}: {_e!r}")
_DROPS = {d.menu for d in _tg.dropdowns()}        # a panel's dropdown menus: their tools are the panel's
_in_panels = [e["op"] for n, r in _drawn.items() if n.startswith("VIEW3D_PT") or n in _DROPS for e in r if "op" in e]
_missing = []
for _op in set(_in_panels):
    try:
        getattr(bpy.ops.nproto, _op.split(".")[1]).get_rna_type()
    except Exception:
        _missing.append(_op)
_not_tools = {"add_primitive", "cvs_cancel", "cvs_done", "click_continuity", "cycle_continuity", "loft_slide", "delete_history", "collect_inputs", "export_step",
              "import_step", "open_step", "history_focus", "history_toggle_hide", "tree_step", "step_unfocus", "mute_step", "patch_add_guides",
              "patch_auto_corners", "patch_corner", "patch_remove_guide", "rebuild", "rebuild_waiting", "remove_step",
              "pipe_point_add", "pipe_point_remove",     # Pipe / Sweep points: Feature panel rows' buttons
              "restore_inputs", "sync_feature", "tab_edit", "continuity", "continuity_clear", "section_add", "section_orient",
              "section_targets", "section_curves", "section_remove", "subsel_picker", "bake_feature", "delete_feature",
              "curve_degree", "fix_clip_start", "delete_faces", "move_pivot", "cv_nudge", "pattern_select",
              "measure",    # Measure: in the Analyze panel / header popover, with Continuity Check
              "remove_symmetry",    # Mirror's Make Symmetric: the Active Object panel's button
              "face_cvs_refine",    # Face CVs' Add CVs / Rebuild: the open face cage's panel buttons
              "move", "rotate", "scale", "transform_hold",   # by Points: hold G / R / S (movetool keymap), no button
              "align_parts",    # Align Parts: Object > Transform and the NURBS Proto menu
              "square"}     # Square: merged into Patch (Quad Sided); kept for older files, off the menus
_all_tools = {"nproto." + n for n in dir(bpy.ops.nproto)
              if n not in _not_tools and not n.startswith("workplane_")}   # the Work Planes panel's own
_menu_only = {t[0] for g in _tg.GROUPS for sec in g[2] for row in sec[1] if isinstance(row, _tg.MenuOnly)
              for t in row if t[0] != "nproto.add_primitive"}     # Edit CVs: Tab does it, the menus list it
_pbtn = [(e["op"], repr(sorted(e.get("props", {}).items()))) for n, r in _drawn.items()
         if n.startswith("VIEW3D_PT") or n in _DROPS for e in r if "op" in e]
_variant_ops = {o_ for o_ in set(_in_panels) if _in_panels.count(o_) > 1 and o_ != "nproto.add_primitive"
                and len({p_ for q_, p_ in _pbtn if q_ == o_}) == _in_panels.count(o_)}
_tool_ops = [o_ for o_ in _in_panels if o_ != "nproto.add_primitive" and o_ not in _variant_ops] + \
    sorted(_variant_ops) + sorted(_menu_only)        # (Boolean, Twist / Bend / Taper, Remove (All) Fillets:
                                                     # one button per variant, each with its own settings)
_bool_ops = sorted(e["props"].get("operation") for n, r in _drawn.items() if n.startswith("VIEW3D_PT") for e in r
                   if e.get("op") == "nproto.boolean")
_kinds = [e["props"].get("kind") for n, r in _drawn.items() if n.startswith("VIEW3D_PT") or n in _DROPS for e in r
          if e.get("op") == "nproto.add_primitive"]
check("UI: every tool is in exactly one group panel and all operators exist",
      not _ui_err and not _missing and sorted(_tool_ops) == sorted(_all_tools)
      and sorted(_kinds) == sorted(_prims.SURFACE_KINDS)
      and _bool_ops == ["DIFFERENCE", "INTERSECTION", "UNION"],
      f"bool={_bool_ops} err={_ui_err} missing={_missing} extra={sorted(set(_tool_ops) - _all_tools)} "
      f"absent={sorted(_all_tools - set(_tool_ops))} dup={sorted({o for o in _tool_ops if _tool_ops.count(o) > 1})}")
_menu_ops = sorted(e["op"] for n, r in _drawn.items() if n.startswith("VIEW3D_MT") and n != "VIEW3D_MT_nproto_tools"
                   and n not in _DROPS for e in r if "op" in e)
_subs = [e["menu"] for e in _drawn["VIEW3D_MT_nproto_tools"] if "menu" in e]
_menu_kinds = sorted(e["props"].get("kind") for n, r in _drawn.items() if n.startswith("VIEW3D_MT")
                     and n != "VIEW3D_MT_nproto_tools" and n not in _DROPS for e in r
                     if e.get("op") == "nproto.add_primitive")
check("UI: header menu holds the seven groups, the same tools as the panels, and every shape (the curve shapes "
      "only there: the panel draws them)",
      sorted(set(_menu_ops)) == sorted(set(_in_panels) | _menu_only) and _menu_only == {"nproto.edit_cvs"}
      and _menu_kinds == sorted(list(_prims.CURVE_KINDS) + list(_prims.SURFACE_KINDS))
      and _subs == ["VIEW3D_MT_nproto_curves", "VIEW3D_MT_nproto_surfaces",
                    "VIEW3D_MT_nproto_curve_edit", "VIEW3D_MT_nproto_surface_edit",
                    "VIEW3D_MT_nproto_texture", "VIEW3D_MT_nproto_convert", "VIEW3D_MT_nproto_scan"],
      f"{_subs} {_menu_kinds}")
_prim_btn = [e["menu"] for e in _drawn["VIEW3D_PT_nproto_surfaces"] if "menu" in e]
check("UI: the surface primitives are one Primitive dropdown in Create Surfaces (not seven buttons)",
      _prim_btn == ["VIEW3D_MT_nproto_primitives"]
      and not any(e.get("op") == "nproto.add_primitive" for e in _drawn["VIEW3D_PT_nproto_surfaces"]), str(_prim_btn))
_scan_ops = [e["op"] for e in _drawn["VIEW3D_PT_nproto_scan"] if "op" in e]
check("UI: the Scan panel holds the three scan tools; Convert none of them",
      _scan_ops == ["nproto.scan_tube_cage", "nproto.scan_fit_cage", "nproto.scan_drape_patch"]
      and not any("scan_" in e.get("op", "") for e in _drawn["VIEW3D_PT_nproto_convert"]), str(_scan_ops))
_sel0 = [o_ for o_ in ctx.view_layer.objects if o_.select_get()]
_act0 = ctx.view_layer.objects.active
_cd = bpy.data.curves.new("DimCurve", 'CURVE')
_cv = bpy.data.objects.new("DimCurve", _cd)
ctx.collection.objects.link(_cv)
for o_ in _sel0:
    o_.select_set(False)
ctx.view_layer.objects.active = None
_none_sel = _tg.selected_kinds(ctx)
_cv.select_set(True)
ctx.view_layer.objects.active = _cv
_have = _tg.selected_kinds(ctx)


class _ActLay:
    """Records each row's `active` and the operators drawn in it."""
    def __init__(self, rows, cur=None):
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "cur", cur)
    def __setattr__(self, k, v):
        if k == "active" and self.cur is not None:
            self.cur["active"] = v
    def row(self, **k):
        r = {"active": True, "ops": []}
        self.rows.append(r)
        return _ActLay(self.rows, r)
    def column(self, **k):
        return self
    def separator(self, **k):
        pass
    def operator(self, op, **k):
        self.cur["ops"].append(op)
        return _FakeProps({})
    def menu(self, m, **k):
        self.cur["ops"].append(m)


_rows = []
_tg.draw_panel(_ActLay(_rows), "SURFACE_EDIT", ctx)
_rows_c = []
_tg.draw_panel(_ActLay(_rows_c), "CURVE_EDIT", ctx)
_cut = {"nproto.trim", "nproto.split", "nproto.emboss",   # they take the curve (a cutter, a profile)
        "nproto.deform"}                                   # Twist / Bend / Taper: a curve is deformed too
check("UI: a curve selected dims Surface Edit (still clickable) but Trim / Split / Emboss / Twist and not Curve Edit; "
      "nothing selected dims nothing",
      _none_sel is None and _have == {"CURVE"} and _tg.fits({"NURBS"}, None)
      and _rows and all(r_["active"] == bool(set(r_["ops"]) & _cut) for r_ in _rows if r_["ops"])
      and all(r_["active"] for r_ in _rows_c),
      f"{_have} {[(r_['ops'], r_['active']) for r_ in _rows]}")
bpy.data.objects.remove(_cv)
bpy.data.curves.remove(_cd)
for o_ in _sel0:
    o_.select_set(True)
ctx.view_layer.objects.active = _act0
_act = ctx.view_layer.objects.active
try:
    _shown = _ui.VIEW3D_PT_nproto_active.poll(ctx)
    _top = _draw(_ui.VIEW3D_PT_nproto_active)
    _rest = [_draw(_ui.VIEW3D_PT_nproto_analyze_side), _draw(_ui.VIEW3D_PT_nproto_display),
             _draw(_ui.VIEW3D_PT_nproto_history)]
    _hrec = []
    _ui.header_analyze(_NS(layout=_FakeLay(_hrec)), ctx)
    ctx.view_layer.objects.active = None
    _hidden = not _ui.VIEW3D_PT_nproto_active.poll(ctx)
    _side_err = None
except Exception as _e:
    _shown = _hidden = False
    _side_err = repr(_e)
finally:
    ctx.view_layer.objects.active = _act
check("UI: Active Object panel shows for a NURBS object, hides with nothing active; Analyze and Settings draw",
      _side_err is None and _shown and _hidden and store.is_nurbs(_act), f"{_side_err} active={_act and _act.name}")
_tog = ["show_edges", "show_cvs", "show_spans", "show_cont_labels", "show_fine_grid"]
check("UI: Settings split -- Display (the five toggles, then the quality preset and the four tessellation "
      "values right under them, no sub-panel), History; the toggles also sit as icons in the 3D view header",
      _side_err is None and [e["prop"] for e in _rest[1] if "prop" in e] == _tog + [
          "display_quality", "chord_tolerance", "angle_tolerance", "edge_angle_tolerance", "max_edge_rel"]
      and "construction_history" in [e.get("prop") for e in _rest[2]]
      and [e["prop"] for e in _hrec if e.get("prop") in _tog] == _tog
      and not hasattr(_ui, "VIEW3D_PT_nproto_tessellation"), str(_hrec))

# --- Edit CVs on a Patch: trims fitted tightly (re-lifting the edges keeps them in place),
#     history and the exact B-rep kept unless a CV moves
from blender_nurbs import jobs as _jobs, patch as _pmod
_sm_ext = _smk.blender_ext


def _edge_pts(b_, n_=40):
    out_ = []
    for e_ in b_.edges():
        t0_, t1_ = e_.parameter_range()
        out_.append(np.array([list(e_.point_at(t_)) for t_ in np.linspace(t0_, t1_, n_)]))
    return out_


def _relift_move(o_):
    """Worst edge move, relative to the edge's length, when the edges are
    re-lifted from their UV trim curves onto the unchanged surface (what a CV
    edit does to edges it doesn't touch)."""
    cg_ = ncvs.make_cage(ctx, o_)
    meta_, pts_ = json.loads(cg_[ncvs.CAGE_META]), json.loads(cg_[ncvs.CAGE_CVS0])
    me_ = cg_.data
    bpy.data.objects.remove(cg_)
    bpy.data.meshes.remove(me_)
    _, br_ = _jobs.run("retarget", {"target": o_}, ["result", "untrimmed"], meta=meta_, cvs=pts_)
    a_, b_ = _edge_pts(store.get_brep(o_)), _edge_pts(store.brep_from_text(br_["result"]))
    return max(float(np.linalg.norm(x - y, axis=1).max() / (np.linalg.norm(np.diff(x, axis=0), axis=1).sum() or 1.0))
               for x, y in zip(a_, b_))


_PP = ctx.active_object
_tight = getattr(_sm_ext, "trim_surface_with_3d_curves_tol", None)
_mv_old = float("nan")
if _tight is not None:
    delattr(_sm_ext, "trim_surface_with_3d_curves_tol")      # the kernel's own trim (0.01 fit) for comparison
    features.rebuild(_PP, ctx, force=True)
    _mv_old = _relift_move(_PP)
    setattr(_sm_ext, "trim_surface_with_3d_curves_tol", _tight)
    features.rebuild(_PP, ctx, force=True)
_mv_new = _relift_move(_PP)
check("patch trims: re-lifting the edges from their UV curves keeps them within the labels' gap tolerance",
      _tight is not None and _mv_new < 0.5 * _cl.TOL_GAP_REL, f"old trim {_mv_old:.1e}, new {_mv_new:.1e} (x edge length)")

_txt0 = _PP[store.PROP_BREP]
bpy.ops.object.select_all(action='DESELECT'); _PP.select_set(True); ctx.view_layer.objects.active = _PP
r1 = bpy.ops.nproto.edit_cvs(); _cg = ctx.active_object
_hist_open = features.has_history(_PP)
ctx.view_layer.update(); ncvs._dirty.add(_cg.name); ncvs._process()      # cage update without a move
_same_open = _PP[store.PROP_BREP] == _txt0
r2 = bpy.ops.nproto.cvs_done()
check("Edit CVs on a patch, Done without moving a CV: history and the exact B-rep are kept",
      r1 == {'FINISHED'} and r2 == {'FINISHED'} and _hist_open and _same_open and features.has_history(_PP)
      and _PP[store.PROP_BREP] == _txt0 and ncvs.cage_for(_PP) is None)
bpy.ops.nproto.edit_cvs(); _cg = ctx.active_object; _cgn = _cg.name
bpy.ops.object.mode_set(mode='OBJECT'); ctx.view_layer.update(); ncvs._finish_cages()       # Tab out
check("Edit CVs on a patch, Tab out without a change: history and B-rep kept",
      _cgn not in bpy.data.objects and features.has_history(_PP) and _PP[store.PROP_BREP] == _txt0)
bpy.ops.nproto.edit_cvs(); _cg = ctx.active_object
_bm = bmesh.from_edit_mesh(_cg.data); _bm.verts.ensure_lookup_table()
_mid = len(_bm.verts) // 2 + int(math.sqrt(len(_bm.verts))) // 2
_bm.verts[_mid].co.z += 0.2; bmesh.update_edit_mesh(_cg.data)
ctx.view_layer.update(); ncvs._process()
_changed = _PP[store.PROP_BREP] != _txt0
r3 = bpy.ops.nproto.cvs_cancel()
check("Edit CVs on a patch, move a CV then Cancel: history and B-rep kept",
      r3 == {'FINISHED'} and _changed and features.has_history(_PP) and _PP[store.PROP_BREP] == _txt0)
bpy.ops.nproto.edit_cvs(); _cg = ctx.active_object
_bm = bmesh.from_edit_mesh(_cg.data); _bm.verts.ensure_lookup_table()
_bm.verts[_mid].co.z += 0.2; bmesh.update_edit_mesh(_cg.data)
ctx.view_layer.update(); ncvs._process()
r4 = bpy.ops.nproto.cvs_done()
_pp_cvs = ext.surface_cvs(store.get_surfaces(_PP)[0])["points"]
_pp_st = [e["op"] for e in features.stack(_PP)]
features.rebuild(_PP, ctx, force=True)          # the patch again, then the tweak replayed on it (trimmed: worker)
_pp_re = np.abs(np.array(ext.surface_cvs(store.get_surfaces(_PP)[0])["points"]) - np.array(_pp_cvs)).max()
check("Edit CVs on a patch, move a CV then Done: the edit stays as a CV Tweak step that replays it (trimmed)",
      r4 == {'FINISHED'} and features.history(_PP) is not None and _pp_st[-1:] == ["cv_tweak"]
      and _PP[store.PROP_BREP] != _txt0 and _pp_re < 1e-9 and not store.is_natural(_PP), (_pp_st, _pp_re))

# --- Edit CVs on a trimmed face whose edges have no UV trim curves (some STEP faces):
#     they are dropped onto the surface first, then the edit works as usual
_nuv_pts = [(x, y, 0.3 * math.sin(x) * math.cos(y)) for x in np.linspace(500, 503, 4) for y in np.linspace(0, 3, 4)]
_nuv_srf = _sm_ext.surface_from_cvs(3, 3, _nuv_pts, [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1], None)
_NU = store.new_object(ctx, "NoUVFace", _sm_ext.brep_from_surface(_nuv_srf))      # a Brep: handled as trimmed
_nuv_missing = _sm_ext.count_missing_uv_trims(store.get_brep(_NU))
_nuv_a0 = store.get_brep(_NU).area(1e-9)
bpy.ops.object.select_all(action='DESELECT'); _NU.select_set(True); ctx.view_layer.objects.active = _NU
r1 = bpy.ops.nproto.edit_cvs(); _cg = ctx.active_object
_bm = bmesh.from_edit_mesh(_cg.data); _bm.verts.ensure_lookup_table()
_bm.verts[5].co.z += 0.5; _bm.verts[10].co.z -= 0.4; bmesh.update_edit_mesh(_cg.data)
ctx.view_layer.update(); ncvs._process()
r2 = bpy.ops.nproto.cvs_done()
_nb = store.get_brep(_NU)
check("Edit CVs on a trimmed face without UV trim curves: the edit works, edges stay on the surface",
      _nuv_missing == 4 and not store.is_natural(_NU) and r1 == {'FINISHED'} and r2 == {'FINISHED'}
      and _nb.face_count() == 1 and abs(_nb.area(1e-9) - _nuv_a0) > 1e-3 and edge_gap(_NU) < 1e-6
      and _sm_ext.count_missing_uv_trims(_nb) == 0,
      f"missing before {_nuv_missing}, area {_nuv_a0:.4f} -> {_nb.area(1e-9):.4f}, gap {edge_gap(_NU):.1e}")

# --- display mesh: flat faces stay a few triangles, curved faces follow the angle tolerance
_ch, _an = store.tess_settings(ctx)
_flatb = store.to_brep(features.run("prim_plane", [], {"size": 2.0, "degree": 1, "spans_u": 1, "spans_v": 1}))
_tubeb = store.to_brep(features.run("prim_cylinder", [], {"radius": 1.0, "height": 2.0, "degree": 3,
                                                                        "spans_u": 8, "spans_v": 1, "caps": False}))
_fP, _fN, _fT = store._tessellate(_flatb, _ch, _an, 2.0, 1.0 / 30.0)
_cP, _cN, _cT = store._tessellate(_tubeb, _ch, _an, 2.0, 1.0 / 30.0)
_cT = store._tri_array(_cT)
_cn = np.asarray(_cN, float); _cn /= np.linalg.norm(_cn, axis=1, keepdims=True)
_spread = np.degrees(np.arccos(np.clip(np.min(np.vstack([np.einsum("ij,ij->i", _cn[_cT[:, a]], _cn[_cT[:, b]])
                                                         for a, b in ((0, 1), (1, 2), (2, 0))]), axis=0), -1, 1)))
check("display mesh: a flat face is a few triangles, a curved one stays within the angle tolerance",
      abs(_ch - 0.0005) < 1e-7 and abs(_an - 5.0) < 1e-6 and len(store._tri_array(_fT)) <= 8 and float(np.percentile(_spread, 99)) <= _an + 0.5,
      f"flat {len(store._tri_array(_fT))} tris, tube {len(_cT)} tris (at 0.002 / 15 deg: "
      f"{len(store._tri_array(store._tessellate(_tubeb, 0.002, 15.0, 2.0, 1.0 / 30.0)[2]))}), "
      f"99th pct normal spread {np.percentile(_spread, 99):.2f} deg")

# --- Environment matcap: Red is gone, Environment reflects a tunnel, a horizon or Blender's HDRIs
from blender_nurbs import environments as _env
_vw = ctx.window_manager.nproto_view
_styles = {i.identifier: i.value for i in _vw.bl_rna.properties["matcap_style"].enum_items}
_srcs = [i[0] for i in _ui._env_source_items(_vw, ctx)]      # a dynamic enum: bl_rna lists no items
_hn = _env.hdri_names()
check("matcap: Highlight / Gray Lines / Environment (Red removed, stored numbers kept); sources are "
      "Light Tunnel, Horizon and Blender's HDRIs",
      _styles == {"HIGHLIGHT": 0, "LINES": 2, "ENVIRONMENT": 3} and _srcs[:2] == ["TUNNEL", "HORIZON"]
      and _srcs[2:] == _hn and {"city", "studio", "sunset"} <= set(_hn), f"{_styles} {_srcs}")
_D, _dA = _env.equirect_dirs(64, 32)
_E1 = _env.irradiance(np.ones((32, 64, 3)))
_uu, _vv = _env.equirect_uv(_D)
_ui_x = ((np.arange(64) + 0.5) / 64)[None, :].repeat(32, 0).ravel()
_vi_x = ((np.arange(32) + 0.5) / 32)[:, None].repeat(64, 1).ravel()
check("environment: equirect directions and (u, v) agree, solid angles sum to 4 pi, "
      "a uniform sky gives irradiance 1 everywhere",
      np.allclose(np.mod(_uu, 1.0), _ui_x, atol=1e-9) and np.allclose(_vv, _vi_x, atol=1e-9)
      and abs(_dA.sum() - 4 * np.pi) < 0.01 and abs(_E1.min() - 1) < 0.02 and abs(_E1.max() - 1) < 0.02,
      f"sum dA {_dA.sum():.4f}, E {_E1.min():.4f}..{_E1.max():.4f}")
_lum = np.array([0.2126, 0.7152, 0.0722])
_hd = {n: _env.hdri(n) for n in ("city", "forest", "sunset", "studio")}
_up = {n: (float(d["irr"][-1].mean(axis=0) @ _lum), float(d["irr"][0].mean(axis=0) @ _lum)) for n, d in _hd.items()}
check("environment: Blender's HDRIs load at 1024 / 256 / 64 wide, sky up (outdoor ones light from above), "
      "brightest irradiance x gain = 1",
      all(d is not None and [l.shape[1] for l in d["levels"]] == [1024, 256, 64]
          and abs(float((d["irr"] @ _lum).max()) * d["gain"] - 1) < 1e-6 for d in _hd.values())
      and all(_up[n][0] > 2 * _up[n][1] for n in ("city", "forest", "sunset")), str(_up))
_ring = np.stack([np.zeros(720), np.cos(np.linspace(0, 2 * np.pi, 720, endpoint=False)),
                  np.sin(np.linspace(0, 2 * np.pi, 720, endpoint=False))], axis=1)
_on = _env.tunnel(_ring, 8)[:, 0] > 0.3
_hz = _env.horizon(np.array([[1, 0, 0.3], [1, 0, -0.3], [1, 0, 0.1]]), 0.2)[:, 0]
_Et, _gt = _env.procedural("TUNNEL", 8)
check("environment: the tunnel has 8 strips around its axis (one straight up), the horizon splits at its height",
      int(np.sum(_on & ~np.roll(_on, 1))) == 8 and _on[180] and _hz[0] > 0.5 and _hz[1] < 0.1 and _hz[2] < 0.1
      and abs(float((_Et @ _lum).max()) * _gt - 1) < 1e-6, f"strips {int(np.sum(_on & ~np.roll(_on, 1)))}, horizon {_hz}")
_vw.show_matcap, _vw.matcap_style = True, 'ENVIRONMENT'
try:
    _pa = []
    for _sv in ("TUNNEL", "HORIZON", "studio"):
        _vw.env_source = _sv
        _pa.append([e["prop"] for e in _draw(_ui.VIEW3D_PT_nproto_analyze_side) if "prop" in e])
    _env_err = None
except Exception as _e:
    _env_err = repr(_e)
finally:
    _vw.show_matcap = False
    _vw.property_unset("env_source")
check("UI: Environment shows its source, Rotation + Lights (tunnel), Height (horizon), Rotation (HDRI), "
      "paint, gloss, roughness and metallic, and no Clear Coat toggle (it is the coat)",
      _env_err is None and all({"env_source", "env_paint", "env_gloss", "env_roughness", "env_metallic"} <= set(p_)
                               and "env_coat" not in p_ for p_ in _pa)
      and "env_lights" in _pa[0] and "env_height" in _pa[1] and "env_rotation" in _pa[2]
      and "env_lights" not in _pa[2] and "zebra_direction" not in _pa[0], f"{_env_err} {_pa}")

# --- Clear Coat: offered over Draft / Curvature / Highlight / Gray Lines, not over Zebra; it has the
#     environment's controls but no paint (the overlay is the base colour)
_coat = {}
try:
    for _nm, _set in (("draft", {"show_draft": True}), ("curv", {"show_curvature": True}),
                      ("hl", {"show_matcap": True, "matcap_style": 'HIGHLIGHT'}),
                      ("lines", {"show_matcap": True, "matcap_style": 'LINES'}), ("zebra", {"show_zebra": True})):
        for _k, _val in _set.items():
            setattr(_vw, _k, _val)
        _vw.env_coat = False
        _off = [e["prop"] for e in _draw(_ui.VIEW3D_PT_nproto_analyze_side) if "prop" in e]
        _vw.env_coat = True
        _on = [e["prop"] for e in _draw(_ui.VIEW3D_PT_nproto_analyze_side) if "prop" in e]
        _hdr = []
        _ui.header_analyze(_NS(layout=_NS(popover=lambda *a, **k: _hdr.append(k.get("text")),
                                         row=lambda *a, **k: _NS(prop=lambda *a, **k: None))), ctx)
        _coat[_nm] = ("env_coat" in _off, "env_source" in _off, "env_source" in _on, "env_roughness" in _on,
                      "env_paint" in _on or "env_metallic" in _on, _hdr[0])
        for _k in ("show_draft", "show_curvature", "show_matcap", "show_zebra"):
            setattr(_vw, _k, False)
    _coat_err = None
except Exception as _e:
    _coat_err = repr(_e)
finally:
    for _k in ("show_draft", "show_curvature", "show_matcap", "show_zebra", "env_coat", "matcap_style"):
        _vw.property_unset(_k)
check("UI: Clear Coat toggle over Draft / Curvature / Highlight / Gray Lines (not Zebra); on, it shows the "
      "environment and roughness, no paint; the header names it",
      _coat_err is None
      and all(_coat[n_][:5] == (True, False, True, True, False) and _coat[n_][5].endswith("Coat")
              for n_ in ("draft", "curv", "hl", "lines"))
      and _coat["zebra"][:3] == (False, False, False) and "Coat" not in _coat["zebra"][5], f"{_coat_err} {_coat}")

# a radius left over from the last use (or the default) can be far bigger than the part:
# from the UI (fit_pending, set by invoke) it is scaled to the part; a script's radius is kept
from blender_nurbs import draw as _dr, ops_trim as _otr
_sf = poly("SmallFloor_ln", [(29.98, 0, 0), (30.02, 0, 0)])
bpy.ops.nproto.extrude(inputs=ins((_sf, "S0")), distance=0.05, direction="Y"); _sfl = ctx.active_object; _sfl.name = "SmallFloor"
_sw = poly("SmallWall_ln", [(30, 0, -0.02), (30, 0.05, -0.02)])
bpy.ops.nproto.extrude(inputs=ins((_sw, "S0")), distance=0.04, direction="Z"); _swl = ctx.active_object; _swl.name = "SmallWall"
try:
    _r_script = bpy.ops.nproto.fillet(surf_a="SmallFloor", surf_b="SmallWall", radius=0.25, trim_bases=False)
except RuntimeError as _e:
    _r_script = {'CANCELLED'} if "may be too big" in str(_e) else str(_e)
_r_ui = bpy.ops.nproto.fillet(surf_a="SmallFloor", surf_b="SmallWall", radius=0.25, trim_bases=False, fit_pending=True)
_sfil = ctx.active_object
check("fillet: a radius too big for the part is scaled to it from the UI (script radius kept: no fillet), "
      "nice_length rounds to 1/2/5",
      _r_script == {'CANCELLED'} and _r_ui == {'FINISHED'} and _sfil.name.startswith("Fillet")
      and max(_sfil.dimensions[0], _sfil.dimensions[2]) < 0.004
      and [_otr.nice_length(x) for x in (0.0027, 0.0099, 0.6, 1.0)] == [0.002, 0.005, 0.5, 1.0],
      f"{_r_script} {_r_ui} {_sfil.name} {tuple(round(d, 4) for d in _sfil.dimensions)}")

# the edges of a hidden active object are drawn as a ghost only when it was picked
# in the Construction Tree, not after hiding it with H
_swl.hide_set(False); ctx.view_layer.objects.active = _swl; _swl.hide_set(True)
_g_h = _dr.ghost["name"] == _swl.name
bpy.ops.nproto.history_focus(name=_swl.name)
_g_tree = _dr.ghost["name"] == _swl.name
_swl.hide_set(False); bpy.ops.nproto.history_focus(name=_swl.name)
check("ghost edges: only for a hidden object picked in the Construction Tree",
      not _g_h and _g_tree and _dr.ghost["name"] is None, f"{_g_h} {_g_tree} {_dr.ghost}")

# a fillet ring around a cap (closed in u, clamped knots): its mesh must not be pulled across the
# seam (mesh area = surface area), and it faces the way the cap and the wall face
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.5, degree="0", caps=False); _rw = ctx.active_object
_rw.location = (60, 0, 0)
bpy.ops.nproto.add_primitive(kind="DISK", radius=1.3, degree="0"); _rc = ctx.active_object
_rc.location = (60, 0, 1.0)
ctx.view_layer.update()
_rr = bpy.ops.nproto.fillet(surf_a=_rw.name, surf_b=_rc.name, radius=0.2, trim_bases=True)
_rf = ctx.active_object
_rme = _rf.data
_rar = np.zeros(len(_rme.polygons)); _rme.polygons.foreach_get("area", _rar)
_rfn = np.zeros(len(_rme.polygons) * 3); _rme.polygons.foreach_get("normal", _rfn); _rfn = _rfn.reshape(-1, 3)
_rce = np.zeros(len(_rme.polygons) * 3); _rme.polygons.foreach_get("center", _rce); _rce = _rce.reshape(-1, 3)
_rrad = _rce[:, :2] - (np.array(_rc.location[:2]) - np.array(_rf.location[:2])); _rrad = _rrad / np.maximum(np.linalg.norm(_rrad, axis=1), 1e-12)[:, None]
# at the rails the fillet is tangent to the bases: there it must face the way they face
def _mesh_nc(o):
    m_ = o.data; n_ = np.zeros(len(m_.polygons) * 3); c_ = np.zeros(len(m_.polygons) * 3)
    m_.polygons.foreach_get("normal", n_); m_.polygons.foreach_get("center", c_)
    W_ = np.array(o.matrix_world); c_ = c_.reshape(-1, 3) @ W_[:3, :3].T + W_[:3, 3]
    return n_.reshape(-1, 3) @ np.linalg.inv(W_[:3, :3]), c_      # (normals: rotation / translation only here)
_axis = np.array(_rw.location[:2])
def _radial(c_):
    d_ = c_[:, :2] - _axis
    return d_ / np.maximum(np.linalg.norm(d_, axis=1), 1e-12)[:, None]
_wn, _wc = _mesh_nc(_rw)
_cn, _ = _mesh_nc(_rc)
_wall_sign = np.sign(np.mean(np.sum(_wn[:, :2] * _radial(_wc), axis=1)))
_cap_sign = np.sign(np.mean(_cn[:, 2]))
_fn, _fc = _mesh_nc(_rf)
_on_cap = np.abs(_fc[:, 2] - _rc.location.z) < 0.01
_on_wall = np.abs(np.linalg.norm(_fc[:, :2] - _axis, axis=1) - 1.0) < 0.01
_rbrep = store.get_brep(_rf).area()
_rdot = min(float(np.mean(_fn[_on_cap, 2] * _cap_sign > 0)) if _on_cap.any() else 0.0,
            float(np.mean(np.sum(_fn[_on_wall, :2] * _radial(_fc[_on_wall]), axis=1) * _wall_sign > 0))
            if _on_wall.any() else 0.0)
_rar = np.zeros(len(_rf.data.polygons)); _rf.data.polygons.foreach_get("area", _rar)
check("fillet ring round a cap: mesh area = surface area (no faces across the seam), faces the way the cap and wall face at the rails",
      _rr == {'FINISHED'} and abs(float(_rar.sum()) - _rbrep) < 0.01 * _rbrep and _rdot > 0.99
      and store._periodic(store.get_brep(_rf).faces()[0].surface()) in ((True, False), (False, True)),
      f"{_rr} mesh {float(_rar.sum()):.4f} brep {_rbrep:.4f} outward {_rdot:.3f}")

# Boolean of closed solids: union / difference / intersection, with history (inputs hidden, the
# operation editable) and without (the active object receives the result)
def _box_at(x):
    bpy.ops.nproto.add_primitive(kind="BOX", width=2, depth=2, height=2, degree="0")
    o = ctx.active_object; o.location = (x, 0, 0); ctx.view_layer.update(); return o
_bv = {}
for _op in ("UNION", "DIFFERENCE", "INTERSECTION"):
    _ba, _bb = _box_at(70), _box_at(71)
    _r = bpy.ops.nproto.boolean(operation=_op, names=json.dumps([_ba.name, _bb.name]))
    _res = ctx.active_object
    _bv[_op] = (_r, round(abs(store.get_brep(_res).volume(1e-6)), 4), bool(store.get_brep(_res).is_manifold_solid()),
                features.history(_res)["op"], _ba.hide_get() and _bb.hide_get())
_hres = _res
_hres.nproto_feature.bool_op = "UNION"        # the Feature panel: edit the operation
_hv = round(abs(store.get_brep(_hres).volume(1e-6)), 4)
bpy.ops.object.select_all(action='DESELECT'); _hres.select_set(True); ctx.view_layer.objects.active = _hres
_hname = _hres.name
_rst = bpy.ops.nproto.restore_inputs()
_rst_ok = _rst == {'FINISHED'} and _hname not in [o.name for o in bpy.data.objects] and not _ba.hide_get()
_prev = ctx.scene.nproto_settings.construction_history
ctx.scene.nproto_settings.construction_history = False
_ba2, _bb2 = _box_at(80), _box_at(81)
_bb2n = _bb2.name
try:
    _r2 = bpy.ops.nproto.boolean(operation="DIFFERENCE", names=json.dumps([_ba2.name, _bb2.name]))
finally:
    ctx.scene.nproto_settings.construction_history = _prev
_nohist = (_r2, ctx.active_object.name == _ba2.name, _bb2n not in bpy.data.objects,
           round(abs(store.get_brep(_ba2).volume(1e-6)), 4))
try:
    _bad = bpy.ops.nproto.boolean(operation="UNION", names=json.dumps([_ba.name, _rw.name]))
except RuntimeError as _e:
    _bad = str(_e)
check("boolean: union 12 / difference 4 / intersection 4 of two overlapping 2-cubes, closed solids, "
      "history (inputs hidden, operation editable, Restore Inputs), no-history replaces the target, open sheet refused",
      [_bv[o][:3] for o in ("UNION", "DIFFERENCE", "INTERSECTION")]
      == [({'FINISHED'}, 12.0, True), ({'FINISHED'}, 4.0, True), ({'FINISHED'}, 4.0, True)]
      and all(v[3] == "boolean" and v[4] for v in _bv.values()) and _hv == 12.0 and _rst_ok
      and _nohist == ({'FINISHED'}, True, True, 4.0)
      and "not a closed solid" in str(_bad),
      f"{_bv} edit={_hv} restore={_rst_ok} nohist={_nohist} bad={str(_bad)[:120]}")

# --- surfaces built from other surfaces' edges face the same way; Flip Normal ---
from blender_nurbs.square import sample_side as _ss, grid_starts as _gs, _foot as _ft
def _seam_dot(obj, src, ref):
    sd = _ss(src, ref, 10)
    s_ = store.get_surfaces(obj)[0]
    M_ = np.array(geom.world_matrix(obj), float); Mi_ = np.linalg.inv(M_)
    d_ = []
    for k in range(1, 9):
        q = (Mi_ @ np.append(sd.P[k], 1))[:3]
        _, su, sv = s_.derivatives(*_ft(s_, q, _gs(s_, [q])[0]))
        n_ = M_[:3, :3] @ np.cross(su, sv)
        d_.append(float(n_ @ sd.Nout[k]) / np.linalg.norm(n_))
    return round(float(np.mean(d_)), 3)
_o1 = nurbs("OR_c1", [(90, 0, 0), (90, 0.7, 0), (90, 1.4, 0), (90, 2, 0)])
_o2 = nurbs("OR_c2", [(92, 0, 0), (92, 0.7, 0), (92, 1.4, 0), (92, 2, 0)])
_o3 = nurbs("OR_c3", [(94, 2, 0.5), (94, 1.4, 0.5), (94, 0.7, 0.5), (94, 0, 0.5)])
bpy.ops.nproto.loft(inputs=ins((_o1, "S0"), (_o2, "S0"))); _oA = ctx.active_object
_sA = store.get_surfaces(_oA)[0]
_eA = min(("U0", "U1", "V0", "V1"), key=lambda r: abs(np.mean([p[0] for p in geom.boundary_points(_sA, r, 5)]) - 92))
_votes = {}
for _flipB in (False, True):        # whichever way the curve runs, B comes out facing like A
    bpy.ops.nproto.loft(inputs=ins((_oA, _eA), (_o3, "S0")), flip_b=_flipB); _oB = ctx.active_object
    _votes[_flipB] = _seam_dot(_oB, _oA, _eA)
_fl_before = (_oA.get(features.PROP_FLIPPED), json.loads(_oB[features.PROP_HIST])["refs"][0])
_oA.nproto_feature.flip_normal = True           # Feature panel: Flip Normal on A
features.rebuild(_oA, ctx, force=True)
_ref_after = json.loads(_oB[features.PROP_HIST])["refs"][0]
features.rebuild(_oB, ctx)
_sB = store.get_surfaces(_oB)[0]
_gapB = max(float(np.linalg.norm(np.array(_sB.evaluate(*_ft(_sB, np.array(p), _gs(_sB, [p])[0]))) - np.array(p)))
            for p in geom.boundary_points(store.get_surfaces(_oA)[0], _ref_after, 9))
check("orientation: a loft from a surface edge faces like that surface (either curve direction); Flip Normal on the "
      "source turns it over, renames the edge the loft uses, and the loft follows",
      min(_votes.values()) > 0.9 and _fl_before[0] == 0 and _oA.get(features.PROP_FLIPPED) == 1
      and _ref_after == {"U0": "U1", "U1": "U0"}.get(_fl_before[1], _fl_before[1])
      and _seam_dot(_oB, _oA, _ref_after) > 0.9 and _gapB < 1e-6,
      f"votes={_votes} before={_fl_before} after={_ref_after} flipped={_oA.get(features.PROP_FLIPPED)} "
      f"dot={_seam_dot(_oB, _oA, _ref_after)} gap={_gapB:.1e}")

# --- feature batch: CV selection, Flip Normal, G2 check, Offset / Thicken, chord + blend fillets ---
from blender_nurbs import cvs as _cvs
from blender_nurbs.ops_sweep import continuity_report as _cr
def _floor(name, y, x0=0.0):
    ln = poly(name + "_ln", [(x0, y - 2, 0), (x0, y + 2, 0)])
    bpy.ops.nproto.extrude(inputs=ins((ln, "S0")), distance=4.0, direction="X")
    o = ctx.active_object; o.name = name; return o
def _wall(name, y, x=2.0):
    ln = poly(name + "_ln", [(x, y - 2, -1), (x, y + 2, -1)])
    bpy.ops.nproto.extrude(inputs=ins((ln, "S0")), distance=3.0, direction="Z")
    o = ctx.active_object; o.name = name; return o
def _sel(*objs):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs: o.select_set(True)
    ctx.view_layer.objects.active = objs[0]
def _face_normals(o):
    b = store.get_brep(o); R = np.array(geom.world_matrix(o).to_3x3())
    out = []
    for f in b.faces():
        c = f.centroid()
        _, uvs, _ = _kl().surface_closest_point(f.surface(), tuple(c))
        uv = uvs[0]; n = R @ np.array(f.outward_normal_at_uv(uv.x, uv.y)); out.append(n / np.linalg.norm(n))
    return out
def _bbz(o):
    b = store.get_brep(o); M = geom.world_matrix(o)
    lo, hi = b.bounding_box(); return round((M @ Vector(lo)).z, 4), round((M @ Vector(hi)).z, 4)

# 1. a new CV cage starts with nothing selected
_fc = _floor("FB_cage", 900)
_cage = _cvs.make_cage(ctx, _fc)
_nsel = sum(1 for v in _cage.data.vertices if v.select)
_me = _cage.data; bpy.data.objects.remove(_cage); bpy.data.meshes.remove(_me)
check("CV cage: no CV selected when it opens", _nsel == 0, f"{_nsel} selected")

# 2. Flip Normal: plain surface with history -> step, flip again -> step gone
_f1 = _floor("FB_flip", 910)
_n0 = _face_normals(_f1)[0]
_sel(_f1); _r1 = bpy.ops.nproto.flip_normal()
_n1 = _face_normals(_f1)[0]; _st1 = [e["op"] for e in features.stack(_f1)]
_sel(_f1); _r2 = bpy.ops.nproto.flip_normal()
_n2 = _face_normals(_f1)[0]; _st2 = [e["op"] for e in features.stack(_f1)]
check("flip normal: a plain surface turns over (history step), flipping again removes the step",
      _r1 == _r2 == {'FINISHED'} and _n0 @ _n1 < -0.999 and _n0 @ _n2 > 0.999 and _st1 == ["flip_normal"]
      and _st2 == [] and store.is_natural(_f1), f"{_n0} {_n1} {_n2} {_st1} {_st2}")
# trimmed: trims kept, still trimmed, history edit (input moved) keeps it flipped
_f2 = _floor("FB_flipT", 920)
bpy.ops.curve.primitive_bezier_circle_add(radius=0.5, location=(1, 920, 1)); _cc = ctx.active_object
bpy.ops.nproto.trim(action="KEEP", target=_f2.name, cutters=ins((_cc, "S0")), ref_point=(1, 920, 0), normal=(0, 0, 1),
                    view_dir=(0, 0, -1))
_a0, _m0 = area(_f2), _face_normals(_f2)[0]
_sel(_f2); _r3 = bpy.ops.nproto.flip_normal()
_a1, _m1 = area(_f2), _face_normals(_f2)[0]
check("flip normal: a trimmed surface turns over, trim kept", _r3 == {'FINISHED'} and abs(_a1 - _a0) < 1e-4 * _a0
      and _m0 @ _m1 < -0.999 and not store.is_natural(_f2), f"area {_a0:.5f} -> {_a1:.5f} n {_m0} -> {_m1}")
# joined L: both faces turn over, still joined
_L = make_L("FB_L", loc=(0, 790, 0))
_ln0 = _face_normals(_L); _sh0 = len(store.shared_edges(_L))
_sel(_L); _r4 = bpy.ops.nproto.flip_normal()
_ln1 = _face_normals(_L)
check("flip normal: a joined sheet turns over, still joined",
      _r4 == {'FINISHED'} and len(_ln1) == 2 and all(a @ b < -0.999 for a, b in zip(_ln0, _ln1))
      and len(store.shared_edges(_L)) == _sh0 == 1, f"{_ln0} -> {_ln1}")
# without construction history: geometry replaced, no stack
_prev = ctx.scene.nproto_settings.construction_history
ctx.scene.nproto_settings.construction_history = False
try:
    _f3 = _floor("FB_flipN", 930)
    _k0 = _face_normals(_f3)[0]
    _sel(_f3); _r5 = bpy.ops.nproto.flip_normal()
    _k1 = _face_normals(_f3)[0]
finally:
    ctx.scene.nproto_settings.construction_history = _prev
check("flip normal without history", _r5 == {'FINISHED'} and _k0 @ _k1 < -0.999 and not features.stack(_f3))

# 3. Continuity Check reports G2: a G2 loft to a flat floor matches its curvature, a G1 loft does not
_gf = _floor("G2_floor", 980)
_gwl = poly("G2_wall_ln", [(6, 978, 1), (6, 982, 1)])
bpy.ops.nproto.extrude(inputs=ins((_gwl, "S0")), distance=2.0, direction="Z"); _gw = ctx.active_object
def _iso_at(o, key, val):
    s_ = store.get_surfaces(o)[0]
    return min(("U0", "U1", "V0", "V1"), key=lambda r: abs(np.mean([p[key] for p in geom.boundary_points(s_, r, 5)]) - val))
_ge_f, _ge_w = _iso_at(_gf, 0, 4.0), _iso_at(_gw, 2, 1.0)
_rep = {}
for _cont in ("G1", "G2"):
    bpy.ops.nproto.loft(inputs=json.dumps([[_gf.name, _ge_f], [_gw.name, _ge_w]]), cont_a=_cont, cont_b=_cont)
    _lo = ctx.active_object
    _rep[_cont] = _cr(_lo, _gf) or _cr(_gf, _lo)
check("continuity check: G2 measured (curvature across the edge), G2 loft passes, G1 loft does not",
      _rep["G2"] and _rep["G1"] and _rep["G2"]["curv_diff"] < 0.02 and _rep["G1"]["curv_diff"] > 0.1,
      f"G1 {_rep['G1'] and round(_rep['G1']['curv_diff'], 4)} G2 {_rep['G2'] and round(_rep['G2']['curv_diff'], 5)}")

# 4. Offset of a joined sheet and of a trimmed surface; Thicken single / joined / symmetric
_L2 = make_L("OF_L", loc=(0, 810, 0))
_sel(_L2); _r6 = bpy.ops.nproto.offset(distance=0.1); _of = ctx.active_object
_sel(_f2); _r7 = bpy.ops.nproto.offset(distance=0.1); _ot = ctx.active_object
check("offset: joined sheet (2 faces, corner closed) and trimmed surface (trim kept, moved 0.1)",
      _r6 == _r7 == {'FINISHED'} and store.get_brep(_of).face_count() == 2 and len(store.shared_edges(_of)) == 1
      and abs(area(_ot) - area(_f2)) < 1e-3 * area(_f2) and abs(abs(_bbz(_ot)[0]) - 0.1) < 1e-6
      and features.history(_of)["op"] == "offset",
      f"faces {store.get_brep(_of).face_count()} shared {len(store.shared_edges(_of))} area {area(_ot):.5f}/{area(_f2):.5f} z {_bbz(_ot)}")
_t1 = _floor("TK_single", 940)
_sel(_t1); _r8 = bpy.ops.nproto.thicken(distance=0.1); _tk = ctx.active_object
_v1 = Vol(_tk)
_tk.nproto_feature.distance = 0.2; features.rebuild(_tk, ctx)
_v2 = Vol(_tk)
_t2 = _floor("TK_sym", 950)
_sel(_t2); bpy.ops.nproto.thicken(distance=0.1, symmetric=True); _ts = ctx.active_object
_L3 = make_L("TK_L", loc=(0, 830, 0))
_sel(_L3); bpy.ops.nproto.thicken(distance=0.1); _tl = ctx.active_object
check("thicken: 4x4 sheet -> closed solid 1.6, feature edit 0.2 -> 3.2, symmetric centred, sheet hidden, joined L -> solid",
      _r8 == {'FINISHED'} and abs(_v1 - 1.6) < 1e-3 and abs(_v2 - 3.2) < 1e-3 and _t1.hide_get()
      and _bbz(_ts) == (-0.05, 0.05) and abs(Vol(_ts) - 1.6) < 1e-3 and Vol(_tl) > 0,
      f"v {_v1:.4f} {_v2:.4f} sym {_bbz(_ts)} {Vol(_ts):.4f} L {Vol(_tl):.4f}")

# 5. Fillets: chord (distance between the rails) and the G2 blend between two surfaces
def _chord_of(o):
    s_ = store.get_surfaces(o)[0]; (u0, u1), (v0, v1) = s_.u_domain(), s_.v_domain()
    du = np.mean([np.linalg.norm(np.array(s_.evaluate(u0, v)) - np.array(s_.evaluate(u1, v))) for v in np.linspace(v0, v1, 9)])
    dv = np.mean([np.linalg.norm(np.array(s_.evaluate(u, v0)) - np.array(s_.evaluate(u, v1))) for u in np.linspace(u0, u1, 9)])
    return float(min(du, dv))
_cf, _cw = _floor("CH_floor", 960), _wall("CH_wall", 960)
_r9 = bpy.ops.nproto.fillet(surf_a=_cf.name, surf_b=_cw.name, radius=0.4, rail_type="CHORD", shape="CIRCULAR",
                            trim_bases=True)
_cfil = ctx.active_object
_rr = 0.4 / math.sqrt(2.0)
check("surface fillet by chord: rails 0.4 apart, ball radius 0.4 / sqrt 2 at 90 deg",
      _r9 == {'FINISHED'} and abs(_chord_of(_cfil) - 0.4) < 1e-4 and abs(area(_cfil) - 0.5 * math.pi * _rr * 4) < 2e-3,
      f"chord {_chord_of(_cfil):.6f} area {area(_cfil):.4f} vs {0.5 * math.pi * _rr * 4:.4f}")
_bf, _bw = _floor("BL_floor", 970), _wall("BL_wall", 970)
_r10 = bpy.ops.nproto.fillet(surf_a=_bf.name, surf_b=_bw.name, radius=0.3, shape="BLEND", continuity="2",
                             trim_bases=True)
_bfil = ctx.active_object
_c1, _c2 = _cr(_bfil, _bf) or _cr(_bf, _bfil), _cr(_bfil, _bw) or _cr(_bw, _bfil)
_rb = features.rebuild(_bfil, ctx, force=True)
_c3 = _cr(_bfil, _bf) or _cr(_bf, _bfil)
check("surface fillet G2 blend: G2 to both surfaces, history rebuild keeps it",
      _r10 == {'FINISHED'} and _c1 and _c2 and _c1["gap"] < 1e-5 and _c2["gap"] < 1e-5 and _c1["angle"] < 0.05
      and _c2["angle"] < 0.05 and _c1["curv_diff"] < 0.02 and _c2["curv_diff"] < 0.02 and _c3 and _c3["curv_diff"] < 0.02
      and features.history(_bfil)["params"].get("blend") == 2,
      f"{_c1} {_c2} after rebuild {_c3}")
_L4 = make_L("CH_L", loc=(0, 850, 0))
_ea0 = A(_L4)
_r11 = fillet_all(_L4, shape="CIRCULAR", radius=0.3, rail_type="CHORD")
_re = 0.3 / math.sqrt(2.0)
check("edge fillet by chord: ball radius chord / sqrt 2 on a 90 deg edge (area change r L (2 - pi/2))",
      _r11 == {'FINISHED'} and abs((_ea0 - A(_L4)) - _re * 4 * (2 - math.pi / 2)) < 2e-3,
      f"dA {_ea0 - A(_L4):.5f} vs {_re * 4 * (2 - math.pi / 2):.5f}")

# G3 blend fillets: degree 7 sections whose fourth CV carries each surface's curvature rate (exact third
# derivatives). A straight crossing of two curved surfaces: G3 to both at rounding, rolling ball and chord;
# a G2 blend leaves G3 open; a history rebuild keeps it
from blender_nurbs import contcheck as _ccf, surfeval as _sev
def _fmk(name, G, Uu, Uv):
    pts_ = [tuple(map(float, x_)) for x_ in G.reshape(-1, 3)]
    return store.new_object(ctx, name, ext.surface_from_cvs(3, 3, pts_, list(Uu), list(Uv)))
def _fw(tag, y0, vary=False):
    X_, Y_ = np.meshgrid(np.linspace(0, 4, 7), np.linspace(-2, 2, 9), indexing="ij")
    Zf = 0.25 * np.sin(1.1 * X_) + (0.12 * (1 - (Y_ / 2) ** 2) ** 2 * (X_ - 2) if vary else 0.0)
    fl_ = _fmk(tag + "F", np.stack([X_, Y_ + y0, Zf], -1), geom.clamped_uniform_knots(7, 3), geom.clamped_uniform_knots(9, 3))
    Yw, Zw = np.meshgrid(np.linspace(-2, 2, 9), np.linspace(-1, 2, 6), indexing="ij")
    Xw = 2.2 + 0.25 * np.sin(1.3 * Zw) * ((1 + 0.4 * (1 - (Yw / 2) ** 2) ** 2) if vary else 1.0)
    wa_ = _fmk(tag + "W", np.stack([Xw, Yw + y0, Zw], -1), geom.clamped_uniform_knots(9, 3), geom.clamped_uniform_knots(6, 3))
    return fl_, wa_
def _fil_pairs(fil, base):
    bf, bb = store.get_brep(fil), store.get_brep(base)
    Mf, Mb = geom.world_matrix(fil), geom.world_matrix(base)
    mid = lambda M_, e_: np.array(M_ @ Vector(e_.point_at(0.5 * sum(e_.parameter_range()))))
    out = []
    for i_, e_ in enumerate(bf.edges()):
        m_ = mid(Mf, e_)
        j_, d_ = min(((j_, np.linalg.norm(mid(Mb, f_) - m_)) for j_, f_ in enumerate(bb.edges())), key=lambda t_: t_[1])
        if d_ < 1e-3:
            out.append((f"E{i_}", f"E{j_}"))
    return out
def _fil_g3(fil, bases):
    worst = [0.0, 0.0, 0.0]
    n_ = 0
    for base in bases:
        for ef, eb in _fil_pairs(fil, base):
            r_ = _ccf.analyze({"a": _ccf.side_key(fil, ef), "b": _ccf.side_key(base, eb)}, 60)
            worst = [max(worst[0], float(r_["ang"].max())), max(worst[1], float(r_["crel"].max())),
                     max(worst[2], float(np.nanmax(r_["g3rel"])))]
            n_ += 1
    return n_, worst
_fg = {}
for _k, (_c, _rt) in enumerate((("3", "RADIUS"), ("3", "CHORD"), ("2", "RADIUS"))):
    _f, _w = _fw(f"FG3_{_k}", 1100 + 10 * _k)
    r = bpy.ops.nproto.fillet(surf_a=_f.name, surf_b=_w.name, radius=0.3, shape="BLEND", continuity=_c, rail_type=_rt,
                              trim_bases=True)
    _fo = ctx.active_object
    _fg[_c, _rt] = (r, ext.surface_cvs(store.get_surfaces(_fo)[0])["degree_v"], _fil_g3(_fo, (_f, _w)), _fo, _f, _w)
_fo, _f, _w = _fg["3", "RADIUS"][3:]
features.rebuild(_fo, ctx, force=True)
_fre = _fil_g3(_fo, (_f, _w))
check("surface fillet G3 blend: degree 7 across, G3 to both surfaces at rounding (rolling ball and chord); a G2 blend "
      "leaves G3 open; a history rebuild keeps it",
      all(v_[0] == {'FINISHED'} for v_ in _fg.values())
      and all(_fg[k_][1] == 7 and _fg[k_][2][0] == 2 and _fg[k_][2][1][1] < 1e-6 and _fg[k_][2][1][2] < 1e-8
              for k_ in (("3", "RADIUS"), ("3", "CHORD")))
      and _fg["2", "RADIUS"][1] == 5 and _fg["2", "RADIUS"][2][1][1] < 1e-6 and _fg["2", "RADIUS"][2][1][2] > 0.1
      and _fre[0] == 2 and _fre[1][2] < 1e-8 and features.history(_fo)["params"].get("blend") == 3,
      f"{[(k_, v_[:3]) for k_, v_ in _fg.items()]} rebuilt {_fre}")
# a crossing whose sections change along it: G3 between the sections as close as G2 (the skin through them)
def _rail_err(fil, base):
    sf, sb = _sev.Surf(store.get_surfaces(fil)[0]), _sev.Surf(store.get_surfaces(base)[0])
    best = None
    for v_ in sf.v_domain:
        us = np.linspace(*sf.u_domain, 41)[2:-2]
        Df = sf.at(us, np.full_like(us, v_), 3)
        nf, Qf, Cf = _sev.forms(Df)
        uv = [_kl().surface_closest_point(store.get_surfaces(base)[0], tuple(map(float, q_)))[1][0] for q_ in Df[0, 0]]
        uv = np.array([(a_.x, a_.y) if hasattr(a_, "x") else (a_[0], a_[1]) for a_ in uv])
        Db = sb.at(uv[:, 0], uv[:, 1], 3)
        nb, Qb, Cb = _sev.forms(Db)
        gap = float(np.linalg.norm(Df[0, 0] - Db[0, 0], axis=1).max())
        sg = np.sign(np.sum(nf * nb, axis=1))
        x_ = Df[0, 1] - np.sum(Df[0, 1] * nf, axis=1, keepdims=True) * nf
        x_ /= np.linalg.norm(x_, axis=1, keepdims=True)
        kf, kb = np.einsum("ni,nij,nj->n", x_, Qf, x_), sg * np.einsum("ni,nij,nj->n", x_, Qb, x_)
        rf, rb = _sev.cubic(Cf, x_), sg * _sev.cubic(Cb, x_)
        e_ = (gap, float(np.abs(kf - kb).max() / max(np.abs(kb).max(), 1e-9)),
              float(np.abs(rf - rb).max() / max(np.abs(rb).max(), 1e-9)))
        if best is None or e_[0] < best[0]:
            best = e_
    return best
_f, _w = _fw("FG3_v", 1140, vary=True)
r = bpy.ops.nproto.fillet(surf_a=_f.name, surf_b=_w.name, radius=0.3, shape="BLEND", continuity="3", trim_bases=False)
_fo = ctx.active_object
_fv = [_rail_err(_fo, b_) for b_ in (_f, _w)]
check("surface fillet G3 blend along a crossing whose sections change: G2 and G3 within 0.1 % at both rails",
      r == {'FINISHED'} and all(g_ < 1e-4 and c_ < 1e-3 and r_ < 1e-3 for g_, c_, r_ in _fv), f"{_fv}")
# Edge Fillet's chord blend takes the same G3 sections
_L5 = make_L("FG3_L", loc=(0, 1160, 0))
r = fillet_all(_L5, shape="BLEND", radius=0.3, rail_type="CHORD", continuity="3")
_degs = sorted({int(ext.surface_cvs(f_.surface())["degree_v"]) for f_ in store.get_brep(_L5).faces()} |
               {int(ext.surface_cvs(f_.surface())["degree_u"]) for f_ in store.get_brep(_L5).faces()})
check("edge fillet by chord, G3 blend: exact sections of degree 7", r == {'FINISHED'} and 7 in _degs, f"{r} {_degs}")

from blender_nurbs import history_ui as _hui
_hui.sync_step(_L4, 0)
_pg = _L4.nproto_step
_shown = (_pg.rail_type, round(_pg.radius, 6), _hui.STEP_FIELDS["edge_fillet"][1])
_lbl = steps.LABEL["edge_fillet"](features.stack(_L4)[0]["params"])
_pg.rail_type = "RADIUS"                      # step editor: back to a rolling ball of 0.3
_dA_r = _ea0 - A(_L4)
check("edge fillet step shows Rail = Chord (label 'chord'); switching it to Rolling Ball rebuilds with radius 0.3",
      _shown == ("CHORD", 0.3, "rail_type") and "chord 0.3" in _lbl
      and abs(_dA_r - 0.3 * 4 * (2 - math.pi / 2)) < 2e-3 and not features.stack(_L4)[0]["params"].get("chord"),
      f"{_shown} {_lbl} dA {_dA_r:.5f}")

# --- Boundary Fill: the solid a wall and two sheets enclose ---
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _bc = ctx.active_object
_bc.location = (0, 1000, -1); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((_bc, "S0")), distance=2.0, direction="Z"); _bw = ctx.active_object
_bl = poly("BF_lo_ln", [(-2, 998, -0.5), (-2, 1002, -0.5)])
bpy.ops.nproto.extrude(inputs=ins((_bl, "S0")), distance=4.0, direction="X"); _blo = ctx.active_object
_bh = poly("BF_hi_ln", [(-2, 998, 0.5), (-2, 1002, 0.5)])
bpy.ops.nproto.extrude(inputs=ins((_bh, "S0")), distance=4.0, direction="X"); _bhi = ctx.active_object
_sel(_bw, _blo, _bhi)
_rf = bpy.ops.nproto.boundary_fill(); _sol = ctx.active_object
_v0 = Vol(_sol)
_bh.location.z += 0.5; ctx.view_layer.update()
features.rebuild(_bhi, ctx); features.rebuild(_sol, ctx)
_v1 = Vol(_sol)
features.release_inputs(ctx, [_blo, _bhi])
_sel(_blo, _bhi)
try:
    _bad = bpy.ops.nproto.boundary_fill()
except RuntimeError as _e:
    _bad = str(_e)
check("boundary fill: wall + two sheets -> closed solid pi r^2 h, inputs hidden, follows a moved sheet; "
      "surfaces that enclose nothing are refused",
      _rf == {'FINISHED'} and abs(_v0 - math.pi) < 0.01 * math.pi and abs(_v1 - 1.5 * math.pi) < 0.015 * math.pi
      and _bw.hide_get() and features.history(_sol)["op"] == "boundary_fill"
      and ("enclose" in str(_bad) or "does not cross" in str(_bad)),
      f"v {_v0:.4f} -> {_v1:.4f} bad={str(_bad)[:100]}")

# --- kept inputs live in the hidden NURBS History collection; Delete History removes freed ones ---
_ka = patch("KI_a", (0, 1100, 0), (0, 1104, 0), "X", 2.0)
_kb = patch("KI_b", (2, 1100, 0), (2, 1104, 0), "Z", 2.0)
_home = _ka.users_collection[0].name
_sel(_ka, _kb); bpy.ops.nproto.join(); _kj = ctx.active_object
_hc = features.history_collection(ctx, create=False)
_in_coll = _hc is not None and all(o.name in _hc.objects and len(o.users_collection) == 1 for o in (_ka, _kb))
for _o in ctx.view_layer.objects: _o.hide_set(False)  # what Unhide All does
_still_hidden = not _ka.visible_get() and not _kb.visible_get() and _kj.visible_get()
_sel(_kj); bpy.ops.nproto.restore_inputs()
_back = (_ka.users_collection[0].name == _home and _ka.visible_get() and _kb.visible_get()
         and _ka.name not in _hc.objects and "KI_a_joined" not in [o.name for o in bpy.data.objects])
_sel(_ka, _kb); bpy.ops.nproto.join(); _kj = ctx.active_object
_names = (_ka.name, _kb.name)
_sel(_kj); bpy.ops.nproto.delete_history(scope='SELECTED')
_purged = all(n not in bpy.data.objects for n in _names) and not features.has_history(_kj) and _kj.name in bpy.data.objects
# an older scene: the inputs sit hidden in the main collection -> Collect Inputs tidies them
_kc = patch("KI_c", (0, 1110, 0), (0, 1114, 0), "X", 2.0)
_kd = patch("KI_d", (2, 1110, 0), (2, 1114, 0), "Z", 2.0)
_sel(_kc, _kd); bpy.ops.nproto.join(); _kj2 = ctx.active_object
features.release_inputs(ctx, [_kc, _kd])              # as if made before the collection existed
_rc = bpy.ops.nproto.collect_inputs()
_collected = _kc.name in _hc.objects and _kd.name in _hc.objects and not _kc.visible_get()
check("kept inputs: Join puts them in the hidden NURBS History collection (stay hidden after Unhide All), "
      "Restore Inputs puts them back, Delete History deletes the ones nothing uses, Collect Inputs tidies old scenes",
      _in_coll and _still_hidden and _back and _purged and _rc == {'FINISHED'} and _collected,
      f"coll={_in_coll} hidden={_still_hidden} back={_back} purged={_purged} collected={_collected}")

# --- Construction Tree lists history steps (fillets ...) on their own rows; eye works on kept inputs ---
import types as _types
class _FL:
    def __init__(self, log): self.log, self.active = log, True
    def split(self, **k): return _FL(self.log)
    def row(self, **k): return _FL(self.log)
    def column(self, **k): return _FL(self.log)
    def box(self): return _FL(self.log)
    def separator(self, **k): pass
    def prop(self, *a, **k): pass
    def label(self, text="", **k): self.log.append(("label", text))
    def operator(self, idname, text="", **k):
        o = _types.SimpleNamespace(); self.log.append(("op", idname, text, o)); return o
_ta = patch("TR_a", (0, 1120, 0), (0, 1124, 0), "X", 2.0)
_tb = patch("TR_b", (2, 1120, 0), (2, 1124, 0), "Z", 2.0)
_sel(_ta, _tb); bpy.ops.nproto.join(); _tj = ctx.active_object
fillet_all(_tj, shape="CIRCULAR", radius=0.3, rail_type="CHORD")
_log = []
_hui.draw_tree(_FL(_log), ctx, _tj)
_steps = [e for e in _log if e[0] == "op" and e[1] == "nproto.tree_step"]
_labels = [e[1] for e in _log if e[0] == "label"]
_objrows = [e[2] for e in _log if e[0] == "op" and e[1] == "nproto.history_focus"]
_ok_rows = len(_steps) == 1 and _steps[0][2].startswith("1. Edge Circular") and any("chord 0.3" in t for t in _labels)
_order = [o.name for o in _hui.build_order(_tj)]
_ok_rows = _ok_rows and _objrows[-1] == _tj.name and _order[-1] == _tj.name and len(_objrows) == len(set(_objrows)) \
    and _order.index("TR_a") < _order.index(_tj.name) and _order.index("TR_b") < _order.index(_tj.name)
_sel(_ta)
bpy.ops.nproto.tree_step(name=_tj.name, root=_tj.name, index=0)
_opened = ctx.view_layer.objects.active == _tj and _tj.nproto_step.index == 0 and _tj.nproto_step.rail_type == "CHORD"
bpy.ops.nproto.history_toggle_hide(name=_ta.name)
_vis_on = _ta.visible_get()
bpy.ops.nproto.history_toggle_hide(name=_ta.name)
_vis_off = not _ta.visible_get() and _ta.name not in ctx.scene.collection.objects
check("history list: built order (inputs first, the body last, each once), a fillet step has its own row "
      "(click opens it); the eye shows / hides a kept input",
      _ok_rows and _opened and _vis_on and _vis_off,
      f"rows={[e[2] for e in _steps]} labels={_labels} opened={_opened} eye={_vis_on}/{_vis_off}")

# --- step focus: a step opened from the tree shows only that step; the object row shows the feature again ---
_log = []
features.mark_built(_tj)
_ctx_feat = _types.SimpleNamespace(view_layer=ctx.view_layer, scene=ctx.scene, window_manager=ctx.window_manager)
from blender_nurbs import feature_ui as _fui
from blender_nurbs import subsel as _ssf
_ssf.clear_tree_pick()             # (still picked from above: clicking its row again would unpick it)
bpy.ops.nproto.tree_step(name=_tj.name, root=_tj.name, index=0)
_fui.draw_feature(_FL(_log), ctx)
_lab1 = [e[1] for e in _log if e[0] == "label"]
from blender_nurbs import subsel as _ssf
_sf = _ssf.step_faces(_tj)
_tjf = store.get_brep(_tj).faces()
_sf_ok = len(_sf) == 1 and _ssf.faces_shown(_tj) == set(_sf) and _tjf[_sf[0]].area() < min(f.area() for k, f in enumerate(_tjf) if k != _sf[0])
_mute_ops = [e for e in _log if e[0] == "op" and e[1] == "nproto.mute_step"]
bpy.ops.nproto.history_focus(name=_tj.name, root=_tj.name)
_log = []
_fui.draw_feature(_FL(_log), ctx)
_lab2 = [e[1] for e in _log if e[0] == "label"]
check("step focus: a step picked in the tree reads like a feature (Feature: <step>, its own settings); "
      "the object shows only its own feature (its steps are tree rows, each with its on / off switch)",
      any(t.startswith("Feature: Edge") for t in _lab1) and not any(t.startswith("Feature: Join") for t in _lab1)
      and not any(t.startswith("Step 1 of") for t in _lab1)
      and any(t.startswith("Feature: Join") for t in _lab2) and not any("History steps" in t for t in _lab2)
      and not _tj.nproto_step.focus and len(_mute_ops) == 1 and _mute_ops[0][3].name == _tj.name,
      f"{_lab1[:4]} | {_lab2[:4]} | mute {len(_mute_ops)}")
check("step focus: the fillet step knows its face (highlighted while open; CVs / Hulls shows just its net)",
      _sf_ok and _ssf.step_faces(_tj) == [], f"{_sf} areas {[round(f.area(), 3) for f in _tjf]}")

# --- an earlier fillet step still finds its face after a later step cut away the point stored for it
# (a chamfer through the fillet's end): it is found by its surface; the open row is drawn pressed ---
def _edge_near(o, p):
    _pl = _ssf._edge_polys(o)
    return int(np.argmin([_ssf._dist_poly(P, np.array(p, float)) if P is not None else 1e9 for P in _pl]))
bpy.ops.nproto.add_primitive(kind="BOX", width=2, depth=2, height=2, degree="0")
_eb = ctx.active_object; ctx.view_layer.update()
bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_eb.name, f"E{_edge_near(_eb, (1, 0, 1))}"]]), shape="CIRCULAR", radius=0.5)
_eb = ctx.active_object
_ech = [_edge_near(_eb, p) for p in ((0, -1, 1), (-1, -1, 0), (-1, 0, 1))]
bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_eb.name, f"E{i}"] for i in _ech]), shape="CHAMFER", radius=0.3)
_eb = ctx.active_object
bpy.ops.nproto.tree_step(name=_eb.name, root=_eb.name, index=0)
_es0 = _ssf.step_faces(_eb)
_ebf = store.get_brep(_eb).faces()
_es0_ok = len(_es0) == 1 and abs(_ebf[_es0[0]].area() - 0.5 * math.pi * 0.5 * 2.0) < 0.25   # a quarter cylinder, end cut
bpy.ops.nproto.tree_step(name=_eb.name, root=_eb.name, index=1)
_es1 = _ssf.step_faces(_eb)
class _FLD(_FL):
    def split(self, **k): return _FLD(self.log)
    def row(self, **k): return _FLD(self.log)
    def column(self, **k): return _FLD(self.log)
    def box(self): return _FLD(self.log)
    def operator(self, idname, text="", **k):
        o = _types.SimpleNamespace(); self.log.append(("op", idname, text, o, k.get("depress"), k.get("emboss"))); return o
_log = []
_hui.draw_tree(_FLD(_log), ctx, _eb)
_pressed = [e[2] for e in _log if e[0] == "op" and e[4] and e[5]]
check("step focus: an earlier fillet step finds its face after a later chamfer cut its stored point away "
      "(by its surface); the open step's row is drawn pressed (and only it)",
      _es0_ok and len(_es1) == 4 and not set(_es0) & set(_es1) and len(_pressed) == 1 and _pressed[0].startswith("2. "),
      f"step0={_es0} area={[round(_ebf[i].area(), 3) for i in _es0]} step1={_es1} pressed={_pressed}")

# --- a feature clicked in the History list is drawn blue (the body filled from its own mesh, a sketch
# as its curve) while it stays active and selected; an open step shows only its faces ---
bpy.ops.nproto.history_focus(name=_eb.name, root=_eb.name)
_tp1 = _ssf.tree_pick(ctx) == _eb
_tpt = _ssf._mesh_tris(_eb)
_tp_mesh = len(_tpt[1]) == len(_eb.data.loop_triangles) > 0
bpy.ops.nproto.tree_step(name=_eb.name, root=_eb.name, index=0)
_tp2 = _ssf.tree_pick(ctx) is None
bpy.ops.nproto.history_focus(name=_eb.name, root=_eb.name)
_eb.select_set(False)
_tp3 = _ssf.tree_pick(ctx) is None
_tpc = bezier("TP_sketch", [(0, 1300), (1, 1300.5), (2, 1300)], 0.0)
bpy.ops.nproto.history_focus(name=_tpc.name, root=_tpc.name)
_tpp = _ssf._curve_polys(_tpc)
_tp4 = _ssf.tree_pick(ctx) == _tpc and len(_tpp) == 1 and len(_tpp[0]) == 200 \
    and np.allclose(_tpp[0][0], (0, 1300, 0), atol=1e-6) and np.allclose(_tpp[0][-1], (2, 1300, 0), atol=1e-6)
check("history pick: the clicked feature is drawn blue while active and selected (not while one of its "
      "steps is open, not once deselected); a clicked sketch is drawn as its curve",
      _tp1 and _tp_mesh and _tp2 and _tp3 and _tp4, f"{_tp1} {_tp_mesh} {_tp2} {_tp3} {_tp4}")

# --- nothing picked: a body selected in the viewport presses no History row; clicking the pressed row
# again unpicks it; selecting something else ends the pick (an open step closes) ---
def _pressed_rows(o):
    _lg = []
    _hui.draw_tree(_FLD(_lg), ctx, o)
    return [e[2] for e in _lg if e[0] == "op" and e[4] and e[5]]
def _viewport_select(o):
    for _o in ctx.selected_objects:
        _o.select_set(False)
    o.select_set(True); ctx.view_layer.objects.active = o
    ctx.view_layer.update(); _ssf._tree_pick_watch(ctx.scene)
bpy.ops.nproto.history_focus(name=_eb.name, root=_eb.name)
_np1 = _pressed_rows(_eb) == [_eb.name]
bpy.ops.nproto.history_focus(name=_eb.name, root=_eb.name)          # again: unpicked, still selected
_np2 = _pressed_rows(_eb) == [] and _ssf.tree_pick(ctx) is None and _eb.select_get()
bpy.ops.nproto.tree_step(name=_eb.name, root=_eb.name, index=0)
_np3 = _eb.nproto_step.focus and [t[:3] for t in _pressed_rows(_eb)] == ["1. "]
_viewport_select(_tpc)                                               # something else: the pick ends
_viewport_select(_eb)                                                # the body again, from the viewport
_np4 = _ssf.TREE_PICK["name"] is None and not _eb.nproto_step.focus and _pressed_rows(_eb) == [] \
    and _ssf.step_faces(_eb) == [] and _ssf.tree_pick(ctx) is None
bpy.ops.nproto.tree_step(name=_eb.name, root=_eb.name, index=1)
bpy.ops.nproto.tree_step(name=_eb.name, root=_eb.name, index=1)    # again: the step closes
_np5 = not _eb.nproto_step.focus and _pressed_rows(_eb) == [] and _ssf.step_faces(_eb) == []
check("history pick: a body selected in the viewport presses no row (nothing picked); clicking a pressed "
      "row again unpicks it; selecting something else ends the pick and closes an open step",
      _np1 and _np2 and _np3 and _np4 and _np5, f"{_np1} {_np2} {_np3} {_np4} {_np5}")
bpy.data.objects.remove(_tpc)

# --- two surfaces trimmed by each other follow their inputs (the rebuild order does not matter) ---
from blender_nurbs import history as _hist
from blender_nurbs.contcheck import edge_kind as _ek_kind
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _mca = ctx.active_object
_mca.location = (0, 2000, 0); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((_mca, "S0")), distance=3.0, direction="Z"); _mA = ctx.active_object; _mA.name = "MT_A"
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.4); _mcb = ctx.active_object
_mcb.location = (0, 2000, 1.5); _mcb.rotation_euler = (0, math.radians(90), 0); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((_mcb, "S0")), distance=2.0, direction="X"); _mB = ctx.active_object; _mB.name = "MT_B"
r1 = bpy.ops.nproto.trim(target="MT_A", cutters=json.dumps([["MT_B", "SURF"]]), ref_point=(1, 2000, 1.5),
                         normal=(0, 0, 1), action="DELETE", scope="THROUGH")
r2 = bpy.ops.nproto.trim(target="MT_B", cutters=json.dumps([["MT_A", "SURF"]]), ref_point=(0.5, 2000, 1.9),
                         normal=(0, 0, 1), action="DELETE", scope="THROUGH")
_sel(_mA, _mB); bpy.ops.nproto.join(tolerance=1e-3); _mj = ctx.active_object
def _open_n(o):
    return sum(1 for i in range(len(store.get_brep(o).edges())) if _ek_kind(o, f"E{i}") == "open")
_mj_area0, _mj_open0, _mA_area0 = area(_mj), _open_n(_mj), area(_mA)
_hp = json.loads(_mcb[features.PROP_HIST]); _hp["params"]["radius"] = 0.55
_mcb[features.PROP_HIST] = json.dumps(_hp)
features.rebuild(_mcb, ctx, force=True)
for _o in (_mA, _mB, _mj):     # the order that broke: A first, while B still has its old shape
    features.rebuild(_o, ctx)
_mj_area1, _mj_open1, _mA_area1 = area(_mj), _open_n(_mj), area(_mA)
# the hole in A is now the radius-0.55 crossing: A lost more area, the body still closes up along it
check("two surfaces trimmed by each other, joined: changing one circle's radius re-trims BOTH (whichever rebuilds "
      "first) and the joined body follows, still closed along the crossing",
      r1 == {'FINISHED'} and r2 == {'FINISHED'} and _mj_open0 == 3 and _mj_open1 == 3
      and _mA_area1 < _mA_area0 - 0.2 and abs(_mj_area1 - _mj_area0) > 0.05,
      f"open {_mj_open0}->{_mj_open1} A {_mA_area0:.3f}->{_mA_area1:.3f} body {_mj_area0:.3f}->{_mj_area1:.3f}")

# --- Bake / Delete (Alias' Delete History): a feature stops following its inputs; a step bakes with what is under it ---
_bk_a = patch("BK_a", (0, 1140, 0), (0, 1144, 0), "X", 2.0)
_bk_b = patch("BK_b", (2, 1140, 0), (2, 1144, 0), "Z", 2.0)
_sel(_bk_a, _bk_b); bpy.ops.nproto.join(); _bk = ctx.active_object
fillet_all(_bk, shape="CIRCULAR", radius=0.3)
_area_bk = area(_bk)
_bk2 = _bk.copy(); _bk2.data = _bk.data.copy(); ctx.collection.objects.link(_bk2)
r1 = bpy.ops.nproto.bake_feature(name=_bk.name, step=-1)          # the Join only: the fillet stays live
_st1 = features.stack(_bk)
_live_fillet = features.history(_bk) is None and len(_st1) == 1 and features.PROP_BASE in _bk
features.rebuild(_bk, ctx, force=True)
_same1 = abs(area(_bk) - _area_bk) < 1e-6
r2 = bpy.ops.nproto.bake_feature(name=_bk2.name, step=0)          # up to the fillet: all of it
_none = not features.has_history(_bk2) and abs(area(_bk2) - _area_bk) < 1e-6
check("bake: the feature alone (its steps stay live, the result unchanged); up to a step (nothing left, the shape kept)",
      r1 == {'FINISHED'} and r2 == {'FINISHED'} and _live_fillet and _same1 and _none,
      f"stack {len(_st1)} same {_same1} none {_none}")
_dk_a = patch("DK_a", (0, 1150, 0), (0, 1154, 0), "X", 2.0)
_dk_b = patch("DK_b", (2, 1150, 0), (2, 1154, 0), "Z", 2.0)
_sel(_dk_a, _dk_b); bpy.ops.nproto.join(); _dk = ctx.active_object
_dk_name = _dk.name
bpy.ops.nproto.delete_feature(name=_dk_name, mode="ALL", keep_inputs=False)     # Shift+click: inputs go too
_gone_all = _dk_name not in bpy.data.objects and "DK_a" not in bpy.data.objects and "DK_b" not in bpy.data.objects
_ek_a = patch("EK_a", (0, 1160, 0), (0, 1164, 0), "X", 2.0)
_ek_b = patch("EK_b", (2, 1160, 0), (2, 1164, 0), "Z", 2.0)
_sel(_ek_a, _ek_b); bpy.ops.nproto.join(); _ek = ctx.active_object
_ek_area = area(_ek)
_feeds = _hui.from_feeds(_ek_a, ctx.scene)
bpy.ops.nproto.delete_feature(name="EK_a", mode="BAKE")          # an input: what is built from it is baked
_kept_shape = _ek.name in bpy.data.objects and not features.has_history(_ek) and abs(area(_ek) - _ek_area) < 1e-6
check("delete (Shift+click): a body takes its unused hidden inputs with it; deleting an input bakes what was built "
      "from it (it keeps its shape) and tidies the inputs nothing uses now",
      _gone_all and _kept_shape and "EK_a" not in bpy.data.objects and "EK_b" not in bpy.data.objects
      and "Feeds: " in _feeds,
      f"gone_all {_gone_all} kept {_kept_shape} feeds '{_feeds}' left {[n for n in ('EK_a', 'EK_b') if n in bpy.data.objects]}")

_rk_a = patch("RK_a", (0, 1170, 0), (0, 1174, 0), "X", 2.0)
_rk_b = patch("RK_b", (2, 1170, 0), (2, 1174, 0), "Z", 2.0)
_sel(_rk_a, _rk_b); bpy.ops.nproto.join(); _rk = ctx.active_object
_rk_name = _rk.name
_sel(_rk); bpy.ops.nproto.thicken(distance=0.05); _rt = ctx.active_object
_rt_name = _rt.name
_hidden0 = _rk.hide_get()
bpy.ops.nproto.delete_feature(name=_rt_name)
_step1 = (_rt_name not in bpy.data.objects and _rk_name in bpy.data.objects and not _rk.hide_get()
          and _rk.select_get() and ctx.view_layer.objects.active == _rk and features.has_history(_rk))
bpy.ops.nproto.delete_feature(name=_rk_name)
_step2 = (_rk_name not in bpy.data.objects and all(n_ in bpy.data.objects for n_ in ("RK_a", "RK_b"))
          and not bpy.data.objects["RK_a"].hide_get() and not bpy.data.objects["RK_b"].hide_get()
          and bpy.data.objects["RK_a"].select_get() and bpy.data.objects["RK_b"].select_get())
check("delete (the trash, a plain click): one step back -- removing a Thicken brings back the joined body it was made "
      "from (shown, selected, its history kept); removing that Join brings back its surfaces",
      _hidden0 and _step1 and _step2, f"hidden {_hidden0} step1 {_step1} step2 {_step2}")

# a Join of two surfaces and the loft between them: removing the Join brings all three back -- the loft still
# uses the two, but a loft doesn't keep its inputs hidden (only Join, Thicken, ... do)
_ja = patch("JL_a", (0, 1180, 0), (0, 1184, 0), "X", 2.0)
_jb = patch("JL_b", (4, 1180, 0), (4, 1184, 0), "X", 2.0)
def _edge_at_x(o, x):
    s_ = store.get_surfaces(o)[0]; M_ = geom.world_matrix(o)
    return min(("U0", "U1", "V0", "V1"), key=lambda e_: abs(np.mean([(M_ @ Vector(p_)).x for p_ in geom.boundary_points(s_, e_, 5)]) - x))
bpy.ops.nproto.loft(inputs=json.dumps([[_ja.name, _edge_at_x(_ja, 2.0)], [_jb.name, _edge_at_x(_jb, 4.0)]]),
                    cont_a="G1", cont_b="G1")
_jc = ctx.active_object
_sel(_ja, _jb, _jc); bpy.ops.nproto.join(); _jj = ctx.active_object
_jj_name, _parts3 = _jj.name, [_ja.name, _jb.name, _jc.name]
_hid3 = all(bpy.data.objects[n_].hide_get() for n_ in _parts3)
bpy.ops.nproto.delete_feature(name=_jj_name)
_hc = features.history_collection(ctx, create=False)
_back3 = all(not bpy.data.objects[n_].hide_get() and (_hc is None or n_ not in _hc.objects)
             and bpy.data.objects[n_].select_get() for n_ in _parts3)
# an input stranded in the History collection (nothing keeps it any more): its eye brings it home, not a peek
_orph = patch("JL_orphan", (0, 1190, 0), (0, 1194, 0), "X", 2.0)
features.keep_inputs(ctx, [_orph])
features.set_shown(ctx, _orph, True)
_hc = features.history_collection(ctx, create=False)
_home = (not _orph.hide_get() and _orph.name not in _hc.objects and features.PROP_HOME not in _orph
         and features.PROP_PEEK not in _orph)
check("delete: removing a Join brings back every part it hid, also ones a loft is built from (a loft doesn't hide "
      "them); a hidden input nothing keeps any more goes home when its eye is clicked",
      _hid3 and _back3 and _home, f"hid {_hid3} back {_back3} home {_home}")

# --- Class-A on a closed rim of two edges (a tilted cap on a round wall): light patches, no kernel fallback ---
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _cc2 = ctx.active_object
_cc2.location = (0, 1200, -1); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((_cc2, "S0")), distance=2.0, direction="Z"); _cw2 = ctx.active_object
_cl = poly("CA_lo_ln", [(-2, 1198, -0.5), (-2, 1202, -0.5)])
bpy.ops.nproto.extrude(inputs=ins((_cl, "S0")), distance=4.0, direction="X"); _clo = ctx.active_object
_ch = poly("CA_hi_ln", [(-2.5, 1198, 0.2), (-2.5, 1202, 0.2)])
bpy.ops.nproto.extrude(inputs=ins((_ch, "S0")), distance=5.0, direction="VECTOR", dir_vec=(1.0, 0.0, 0.15)); _chi = ctx.active_object
_sel(_cw2, _clo, _chi); bpy.ops.nproto.boundary_fill(); _cs = ctx.active_object
_bb = store.get_brep(_cs)
_top = [i for i in store.shared_edges(_cs)
        if (lambda e: (lambda t: e.point_at(0.5 * (t[0] + t[1]))[2])(e.parameter_range()))(_bb.edges()[i]) > -0.2
        and _bb.edges()[i].length() > 0.5]
_cs_plain = _cs.copy(); _cs_plain.data = _cs.data.copy(); ctx.collection.objects.link(_cs_plain)
_rca = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_cs.name, f"E{i}"] for i in _top]), shape="BLEND",
                                  continuity="3", radius=0.15, class_a=True, ca_tolerance=1e-3)
# (rolling ball: a chord blend, G1 - G3, is built from exact sections, which Class-A leaves as they are)
_rep = json.loads(_cs.get("nproto_class_a", "{}"))
_bf2 = store.get_brep(_cs)
_dense = [f for f in _bf2.faces() if max(ext.surface_cvs(f.surface())["count_u"], ext.surface_cvs(f.surface())["count_v"]) > 69]
from blender_nurbs.ops_trim import class_a_summary as _cas
_rings_ca = _rep.get("rings") or []
check("class-A on a closed (not round) rim, blend G3: the rim is ONE even periodic ring (as without Class-A), "
      "no patches, no kernel fallback, and the report says so",
      _rca == {'FINISHED'} and len(_top) >= 1 and "error" not in _rep and _rep.get("patches", 1) == 0
      and len(_rings_ca) == 1 and _rings_ca[0]["ring"] == "periodic" and _rings_ca[0]["round"] <= 69 and not _dense
      and "one even ring" in _cas(_rep)[0] and _cas(_rep)[1],
      f"edges {_top} report {str(_rep)[:300]} dense {len(_dense)} · {_cas(_rep)[0]}")

# --- Blend / chamfer on a closed rim (cap on a round wall): the kernel's face has no seam and
#     covers a sliver; it must come back as the full ring, joined to cap and wall ---
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _rc = ctx.active_object
_rc.location = (0, 1300, 0); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((_rc, "S0")), distance=2.0, direction="Z"); _rw = ctx.active_object
_rtop = max(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([q[2] for q in geom.boundary_points(store.get_surfaces(_rw)[0], w, 8)]))
bpy.ops.nproto.planar(inputs=ins((_rw, _rtop))); _rcap = ctx.active_object
_sel(_rw, _rcap); bpy.ops.nproto.join(tolerance=1e-3); _rj = ctx.active_object
_ring_area = {}
for _shape, _cont in (("CIRCULAR", "1"), ("BLEND", "2"), ("BLEND", "3"), ("CHAMFER", "1")):
    _o = _rj.copy(); _o.data = _rj.data.copy(); ctx.collection.objects.link(_o)
    r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_o.name, f"E{i}"] for i in store.shared_edges(_o)]),
                                   shape=_shape, continuity=_cont, radius=0.2)
    _b = store.get_brep(_o)
    _ring_area[_shape + _cont] = (r == {'FINISHED'}, _b.face_count(), round(min(f.area() for f in _b.faces()), 4),
                                  sum(1 for e in _b.edges() if len(e.faces()) == 1))
_ca_ref = _ring_area["CIRCULAR1"][2]
check("blend / chamfer of a closed rim: full ring face (not a sliver), joined like the circular fillet",
      all(v[0] and v[1] == 3 and v[2] > 0.7 * _ca_ref and v[3] == _ring_area["CIRCULAR1"][3] for v in _ring_area.values()),
      str(_ring_area))

# --- A fillet round a closed rim is ONE smooth face: on a round part (a drafted wall) the exact surface of
#     revolution of its section (9 rational CVs round, like the wall); otherwise an even periodic degree-5 ring ---
def _skey(s_):
    d_ = ext.surface_cvs(s_)
    return (d_["count_u"], d_["count_v"], tuple(round(c_, 7) for p_ in d_["points"] for c_ in p_))
def _new_faces(o_, before_keys):
    return [f for f in store.get_brep(o_).faces() if _skey(f.surface()) not in before_keys]
def _ring_iso(s_, along, at, n=721):
    (u0_, u1_), (v0_, v1_) = s_.u_domain(), s_.v_domain()
    ts_ = np.linspace(0, 1, n)
    if along == "u":
        return np.array([s_.evaluate(u0_ + (u1_ - u0_) * t_, v0_ + (v1_ - v0_) * at) for t_ in ts_])
    return np.array([s_.evaluate(u0_ + (u1_ - u0_) * at, v0_ + (v1_ - v0_) * t_) for t_ in ts_])
def _round_dir(s_):
    """The parameter direction that runs round the ring (its iso closes on itself)."""
    for along in ("u", "v"):
        q_ = _ring_iso(s_, along, 0.5, 41)
        if np.linalg.norm(q_[0] - q_[-1]) < 1e-6 and np.ptp(q_, axis=0).max() > 1e-3:
            return along
    return None
from blender_nurbs.contcheck import edge_kind as _cc_kind
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _dc = ctx.active_object
_dc.location = (0, 1350, 0); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((_dc, "S0")), distance=2.0, direction="Z", draft=-2.0); _dw = ctx.active_object
_dtop = max(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([q[2] for q in geom.boundary_points(store.get_surfaces(_dw)[0], w, 8)]))
bpy.ops.nproto.planar(inputs=ins((_dw, _dtop))); _dcap = ctx.active_object
_sel(_dw, _dcap); bpy.ops.nproto.join(tolerance=1e-3); _dj = ctx.active_object
_dkeys = {_skey(f.surface()) for f in store.get_brep(_dj).faces()}
_revs = {}
for _shape, _cont in (("BLEND", "2"), ("BLEND", "3"), ("CIRCULAR", "1"), ("CHAMFER", "1")):
    _o = _dj.copy(); _o.data = _dj.data.copy(); ctx.collection.objects.link(_o)
    r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_o.name, f"E{i}"] for i in store.shared_edges(_o)]),
                                   shape=_shape, continuity=_cont, radius=0.3)
    _nf = _new_faces(_o, _dkeys)
    _info = {"ok": r == {'FINISHED'}, "new": len(_nf)}
    if len(_nf) == 1:
        _s = _nf[0].surface(); _d = ext.surface_cvs(_s); _rd = _round_dir(_s)
        _info["round"] = _rd
        if _rd:
            _n_round = _d["count_u"] if _rd == "u" else _d["count_v"]
            _deg_round = _d["degree_u"] if _rd == "u" else _d["degree_v"]
            _sp = []
            for _at in (0.0, 0.25, 0.5, 0.75, 1.0):
                _q = _ring_iso(_s, _rd, _at)
                _sp.append(max(float(np.ptp(np.linalg.norm(_q[:, :2] - np.array([0, 1350]), axis=1))), float(np.ptp(_q[:, 2]))))
            _info.update(cvs=_n_round, deg=_deg_round, rational=bool(_d["weights"]), spread=max(_sp))
    _info["open"] = sum(1 for i in range(len(store.get_brep(_o).edges())) if _cc_kind(_o, f"E{i}") == "open")
    _revs[_shape + _cont] = (_info, _o)
check("drafted rim, every fillet shape: ONE ring face, an exact surface of revolution (9 rational CVs round, "
      "degree 2, every iso round it a true circle to 1e-6), stitched in with no new open edge",
      all(v[0]["ok"] and v[0]["new"] == 1 and v[0].get("cvs") == 9 and v[0].get("deg") == 2 and v[0].get("rational")
          and v[0].get("spread", 1) < 1e-6 and v[0]["open"] == 1 for v in _revs.values()),
      str({k: v[0] for k, v in _revs.items()}))
_oca = _dj.copy(); _oca.data = _dj.data.copy(); ctx.collection.objects.link(_oca)
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_oca.name, f"E{i}"] for i in store.shared_edges(_oca)]),
                               shape="BLEND", continuity="2", radius=0.3, class_a=True)
_nfa = _new_faces(_oca, _dkeys)
_repa = json.loads(_oca.get("nproto_class_a", "{}"))
_da = ext.surface_cvs(_nfa[0].surface()) if len(_nfa) == 1 else None
check("drafted rim with Class-A Patches on: the same exact revolution (9 CVs round), reported as such",
      r == {'FINISHED'} and len(_nfa) == 1 and _da is not None and 9 in (_da["count_u"], _da["count_v"])
      and [g["ring"] for g in _repa.get("rings", [])] == ["revolution"] and "exact revolution" in _cas(_repa)[0],
      f"{len(_nfa)} new faces, {_da and (_da['count_u'], _da['count_v'])} · {_cas(_repa)[0]}")
_st_ca = features.stack(_oca)
_rims_ca = steps.rims_of(_oca, _st_ca[-1]["params"]["mids_local"]) if _st_ca else None
check("the fillet step records what its rim became (its panel shows it, and hides the Class-A settings: "
      "every fillet face is the ring)", _rims_ca is not None and _rims_ca["only"]
      and [g["ring"] for g in _rims_ca["rings"]] == ["revolution"], str(_rims_ca))
from blender_nurbs import contcheck as _cc0
_cc0.clear()
_bo = _revs["BLEND2"][1]
for i in store.shared_edges(_bo):
    _cc0.add(_bo, f"E{i}")
_dj2 = [_cc0.judge(_cc0.result(p_), ctx.window_manager.nproto_view) for p_ in _cc0.PAIRS]
_cc0.clear()
check("drafted rim, G2 blend ring: G2 to wall and cap all round (gap < 1e-6, < 0.01 deg, curvature within 0.5%)",
      len(_dj2) == 2 and all(isinstance(j, dict) and j["G0"][0] < 1e-6 and j["G1"][0] < 0.01 and j["G2"][0] < 0.005
                             for j in _dj2), str(_dj2))
# the tilted rim of the Class-A test without Class-A: not round, so one even periodic degree-5 ring
_pkeys = {_skey(f.surface()) for f in store.get_brep(_cs_plain).faces()}
r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_cs_plain.name, f"E{i}"] for i in _top]), shape="BLEND",
                               continuity="2", radius=0.15)
_pf = _new_faces(_cs_plain, _pkeys)
_pinfo = {"ok": r == {'FINISHED'}, "new": len(_pf)}
if len(_pf) == 1:
    _s = _pf[0].surface(); _d = ext.surface_cvs(_s); _rd = _round_dir(_s)
    _kr = np.array(_d["knots_u"] if _rd == "u" else _d["knots_v"]) if _rd else np.zeros(0)
    _inner = _kr[(_kr > _kr[0]) & (_kr < _kr[-1])] if len(_kr) else _kr
    _pinfo.update(round=_rd, deg=(_d["degree_u"] if _rd == "u" else _d["degree_v"]) if _rd else None,
                  cvs=(_d["count_u"] if _rd == "u" else _d["count_v"]) if _rd else None,
                  even=bool(len(_inner) and len(np.unique(np.round(_inner, 12))) == len(_inner)
                            and np.ptp(np.diff(np.r_[_kr[0], _inner, _kr[-1]])) < 1e-9))
_cc0.clear()
_pk = _skey(_pf[0].surface()) if len(_pf) == 1 else None
for i, e_ in enumerate(store.get_brep(_cs_plain).edges()):      # the ring's rails and its seam
    if any(_skey(f_.surface()) == _pk for f_ in e_.faces()) and _cc_kind(_cs_plain, f"E{i}") in ("shared", "seam"):
        _cc0.add(_cs_plain, f"E{i}")
_pj = [_cc0.judge(_cc0.result(p_), ctx.window_manager.nproto_view) for p_ in _cc0.PAIRS]
_cc0.clear()
_pinfo["cont"] = [{k_: round(v_[0], 6) for k_, v_ in j_.items()} if isinstance(j_, dict) else j_ for j_ in _pj]
check("tilted (not round) rim, G2 blend: ONE even periodic ring (degree 5, uniform simple knots, <= 69 CVs round), "
      "G2 across its seam and to both faces",
      _pinfo["ok"] and _pinfo["new"] == 1 and _pinfo.get("deg") == 5 and _pinfo.get("even") and _pinfo.get("cvs", 999) <= 69
      and len(_pj) >= 3 and all(isinstance(j_, dict) and j_["G0"][0] < 1e-5 and j_["G1"][0] < 0.05 and j_["G2"][0] < 0.01
                                for j_ in _pj),
      str(_pinfo))

# --- Continuity Check on picked edges: one stitched edge of a joined body, or two surfaces' edges ---
from blender_nurbs import contcheck as _cc
from blender_nurbs.ops import _pick_candidates as _pc
_vw = ctx.window_manager.nproto_view
_ccres = {}
for _shape, _cont in (("CIRCULAR", "1"), ("BLEND", "2")):
    _o = _rj.copy(); _o.data = _rj.data.copy(); ctx.collection.objects.link(_o)
    bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_o.name, f"E{i}"] for i in store.shared_edges(_o)]),
                               shape=_shape, continuity=_cont, radius=0.2)
    _cc.clear()
    for i in store.shared_edges(_o):
        _cc.add(_o, f"E{i}")
    _rs = [_cc.result(p_) for p_ in _cc.PAIRS]
    _ccres[_shape] = [r_ if isinstance(r_, str) else {lv: (round(v_[0], 6), v_[1]) for lv, v_ in _cc.judge(r_, _vw).items()}
                      for r_ in _rs]
_rim = [_cc.result(p_) for p_ in _cc.PAIRS]
from blender_nurbs.kernel import load as _kl2
_cvmax = max(max(_kl2().blender_ext.surface_cvs(f.surface())["count_u"], _kl2().blender_ext.surface_cvs(f.surface())["count_v"])
             for f in store.get_brep(_o).faces())
check("continuity check: a closed rim is sampled all the way round (seam included), with exactly the Samples set, "
      "and the samples a true comb needs are recommended (4 per CV)",
      all(not isinstance(r_, str) and r_["closed"] and len(r_["ang"]) == _vw.cont_samples
          and r_["recommended"] == min(4 * _cvmax, 1200) for r_ in _rim),
      f"{[(r_['closed'], len(r_['ang']), r_['recommended']) for r_ in _rim if not isinstance(r_, str)]} cvs {_cvmax}")
_s0 = _vw.cont_samples
_vw.cont_samples = 40
_rim40 = [_cc.result(p_) for p_ in _cc.PAIRS]
_vw.cont_samples = _s0
check("continuity check: changing Samples changes the measurement (and so the comb)",
      all(not isinstance(r_, str) and len(r_["ang"]) == 40 for r_ in _rim40))
_g2 = [j for j in _ccres["BLEND"] if isinstance(j, dict)]
check("continuity check, stitched edges: circular fillet holds G1 but not G2; the G2 blend's curvature matches "
      "(within 1%) and it sits within 0.1 mm / 0.3 deg of wall and cap",
      all(isinstance(j, dict) and j["G1"][1] and not j["G2"][1] for j in _ccres["CIRCULAR"]) and len(_g2) == 2
      and all(j["G2"][0] < 0.01 and j["G0"][0] < 1e-4 and j["G1"][0] < 0.3 for j in _g2), str(_ccres))
_cc.clear()
_k1 = patch("CK_floor", (0, 1400, 0), (0, 1404, 0), "X", 2.0)
_k2 = patch("CK_wall", (2, 1400, 0), (2, 1404, 0), "Z", 2.0)
_cands = [c for c in _pc(ctx, True) if not c[1].startswith("S") and store.is_nurbs(c[0])]
_ea = [c for c in _cands if c[0] == _k1 and all(abs(p_[0] - 2.0) < 1e-4 for p_ in c[2])]
_pa = _cc.find_partner(_k1, _ea[0][1], _ea[0][2], _cands) if _ea else None
if _pa:
    _cc.add(_k1, _ea[0][1], _pa)
_r90 = _cc.result(_cc.PAIRS[-1]) if _cc.PAIRS else "no pair"
_j90 = _cc.judge(_r90, _vw) if not isinstance(_r90, str) else {}
check("continuity check, two surfaces: the partner edge is found; G0 holds, G1 fails at the 90 deg crease",
      _pa is not None and _pa[0] == _k2 and _j90 and _j90["G0"][1] and not _j90["G1"][1] and abs(_j90["G1"][0] - 90) < 1,
      f"{_pa} {_j90 or _r90}")
_seams = [f"E{i}" for i in range(len(store.get_brep(_o).edges())) if _cc.edge_kind(_o, f"E{i}") == "seam"]
_cc.clear()
for _r in _seams:
    _cc.add(_o, _r)
_sr = [_cc.result(p_) for p_ in _cc.PAIRS]
_sj = [_cc.judge(r_, _vw) if not isinstance(r_, str) else r_ for r_ in _sr]
check("continuity check, seams: a joined body's seam edges (a closed face meeting itself) are measured with one click, "
      "between the face's two ends", len(_seams) >= 2 and all(isinstance(j, dict) and j["G0"][0] < 1e-4 and j["G1"][0] < 0.5
                                                            for j in _sj),
      f"{_seams} {[j if isinstance(j, str) else {k: round(x[0], 6) for k, x in j.items()} for j in _sj]}")
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _sc = ctx.active_object
_sc.location = (0, 1500, 0); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((_sc, "S0")), distance=2.0, direction="Z"); _sw = ctx.active_object
_cands = [c for c in _pc(ctx, True) if not c[1].startswith("S") and store.is_nurbs(c[0])]
_own = [c for c in _cands if c[0] == _sw]
_hit = None
for c in _own:
    _pp = _cc.find_partner(_sw, c[1], c[2], _cands)
    if _pp is not None and _pp[0] == _sw:
        _hit = (c[1], _pp[1]); break
_cc.clear()
if _hit:
    _cc.add(_sw, _hit[0], (_sw, _hit[1]))
_ns = _cc.result(_cc.PAIRS[-1]) if _cc.PAIRS else "no pair"
_nj = _cc.judge(_ns, _vw) if not isinstance(_ns, str) else _ns
check("continuity check, untrimmed closed surface: its two ends are paired and measured end against end",
      _hit is not None and isinstance(_nj, dict) and _nj["G0"][0] < 1e-6 and _nj["G1"][0] < 1.0, f"{_hit} {_nj}")
# STEP export of that blended rim: every edge loop must close (a closed trim curve that ends ~1e-6 from
# its start is still ONE vertex), or other programs drop the face -- Plasticity lost the fillet
from blender_nurbs import step_export as _se2, step_import as _si2
_rim_step = os.path.join(tempfile.mkdtemp(), "rim.step")
_rst = _se2.export_step(_rim_step, [_o], unit="MM")
with open(_rim_step) as _fh:
    _rtxt = _fh.read()
_rres, _rrep = _si2.read_step(_rim_step, 1.0)
_bo = store.get_brep(_o)
check("step export: the blended rim's loops all close, one vertex per kernel vertex, and it reads back with every face",
      _rst["open_loops"] == [] and _rtxt.count("VERTEX_POINT") == len(list(_bo.vertices()))
      and _rrep["failed"] == 0 and _rres and _rres[0][1].face_count() == _bo.face_count(),
      f"open {_rst.get('open_loops')} verts {_rtxt.count('VERTEX_POINT')}/{len(list(_bo.vertices()))} {_rrep}")
_cc.clear(); _cc.add(_k1, _ea[0][1], _pa) if _ea and _pa else None
_vw.cont_show_checks = True
_a1 = _cc.active(ctx)
_cc.stop(ctx)
_a2 = _cc.active(ctx)
check("continuity check: its button toggles it like Zebra -- off hides the checks (kept for when it is on again)",
      _a1 and not _a2 and len(_cc.PAIRS) == 1 and not _vw.cont_show_checks)
_vw.cont_show_checks = True
_n_before = len(_cc.PAIRS)
bpy.ops.nproto.continuity_clear(index=-1)
check("continuity check: Clear All empties the list", _n_before == 1 and not _cc.PAIRS)

# --- Dynamic Section: exact plane sections with exact curvature combs ---
from blender_nurbs import sections as _sx
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.add_primitive(kind="SPHERE", radius=1.0, degree="0"); _sph2 = ctx.active_object
_sph2.location = (0, 1600, 0); ctx.view_layer.update()
bpy.ops.object.select_all(action='DESELECT'); _sph2.select_set(True); ctx.view_layer.objects.active = _sph2
r = bpy.ops.nproto.section_add()
_sec = ctx.active_object
_sec.nproto_section.count = 3; _sec.nproto_section.step = 0.4
_sec.location = (0, 1600, 0)
def _sect_err(sec, ctr):
    errs, n_c = [], 0
    for sd in _sx.compute(sec):
        p0, nn = sd["plane"]
        d = float((p0 - ctr) @ nn)
        rho = math.sqrt(max(1.0 - d * d, 1e-12))
        cc = ctr + d * nn                       # the circle's centre
        for c in sd["curves"]:
            n_c += 1
            if c["K"] is None:
                return None, n_c
            for P_, K_ in zip(c["P"], c["K"]):
                k = np.linalg.norm(K_)
                inward = (cc - P_) / (np.linalg.norm(cc - P_) or 1.0)
                errs.append(max(abs(k * rho - 1.0), 1.0 - float(K_ @ inward) / (k or 1.0)))
    return (max(errs) if errs else None), n_c
_ctr = np.array([0.0, 1600.0, 0.0])
for _ax, _rot in (("Z", (0, 0, 0)), ("tilted 35 deg", (math.radians(35), math.radians(20), 0))):
    _sec.rotation_euler = _rot; ctx.view_layer.update()
    _e, _nc = _sect_err(_sec, _ctr)
    check(f"dynamic section ({_ax}): 3 planes cut a sphere in 3 circles; every comb hair has the circle's exact curvature "
          "1/sqrt(R^2-d^2) and points to its centre", r == {'FINISHED'} and _sx.is_section(_sec) and _nc == 3
          and _e is not None and _e < 1e-3, f"circles {_nc} worst rel. error {_e}")
_sec.rotation_euler = (0, 0, 0); ctx.view_layer.update()
_k0 = _sx._cache.get(_sec.name, (None,))[0]
_f0 = _sx.comb_factor(_sec.nproto_section, _sx.targets(_sec), _sx._size(_sx.targets(_sec))[1])
_sec.location.z += 0.1; ctx.view_layer.update(); _sx.compute(_sec)
_f1 = _sx.comb_factor(_sec.nproto_section, _sx.targets(_sec), _sx._size(_sx.targets(_sec))[1])
check("dynamic section: Scale works like the other combs (a sphere's curvature 1/R at Scale 1 makes a hair a fifth of "
      "its size) and stays the same while the section moves", abs(_f0 - _f1) < 1e-12
      and abs(_f0 * 1.0 - 0.2 * _sx._size(_sx.targets(_sec))[1]) < 1e-3 * _sx._size(_sx.targets(_sec))[1], f"{_f0} {_f1}")
_mid = _sx.planes(_sec)[1][0]
check("dynamic section: moving the section moves its planes (centred on it) and recomputes the sections",
      _sx._cache[_sec.name][0] != _k0 and abs(_mid.z - 0.1) < 1e-6)
_sec.location.z = 0.0; ctx.view_layer.update()
_n_sec = len([o for o in ctx.scene.objects if _sx.is_section(o)])
bpy.ops.object.select_all(action='DESELECT'); _sph2.select_set(True); ctx.view_layer.objects.active = _sph2
_on0 = _sx.section_active(ctx)
r2 = bpy.ops.nproto.section_add()
_off1 = not _sx.section_active(ctx) and _sec.hide_get()
r3 = bpy.ops.nproto.section_add()
check("dynamic section: the button toggles it like Zebra -- off hides the section, on brings the same one back "
      "selected (one per scene, never a second)",
      r2 == r3 == {'FINISHED'} and _on0 and _off1 and _sx.section_active(ctx) and not _sec.hide_get()
      and len([o for o in ctx.scene.objects if _sx.is_section(o)]) == _n_sec == 1
      and ctx.view_layer.objects.active == _sec and _sec.select_get())
r = bpy.ops.nproto.section_curves()
_scv = ctx.scene.objects.get("Section Curves")
check("dynamic section: Create Curves makes the sections real curves", r == {'FINISHED'} and _scv is not None
      and _scv.type == 'CURVE' and len(_scv.data.splines) == 3, str(_scv and len(_scv.data.splines)))

# --- Visual Clip: a Geometry Nodes modifier drops the faces past the middle plane ---
_sec.rotation_euler = (0, 0, 0); _sec.location = (0, 1600, 0); _sec.nproto_section.offset = 0.0
ctx.view_layer.update()
def _kept(o):
    dg = ctx.evaluated_depsgraph_get()
    me = o.evaluated_get(dg).to_mesh()
    M = o.matrix_world
    zs = [(M @ pl.center).z for pl in me.polygons]
    vz = [(M @ vv.co).z for vv in me.vertices]
    n = len(me.polygons)
    o.evaluated_get(dg).to_mesh_clear()
    _kept.verts = (min(vz) if vz else None, max(vz) if vz else None)
    return n, (min(zs) if zs else None), (max(zs) if zs else None)
_n_all = len(_sph2.data.polygons)
_sec.nproto_section.clip = True; ctx.view_layer.update()
_c1 = _kept(_sph2)
_sec.nproto_section.flip = True; ctx.view_layer.update()
_c2 = _kept(_sph2)
_sec.nproto_section.flip = False; _sec.nproto_section.slab = True; _sec.nproto_section.slab_width = 0.3
ctx.view_layer.update()
_c3 = _kept(_sph2)
_v3 = _kept.verts
_planes = _sx.clip_planes(_sph2)
_sec.nproto_section.slab = False; ctx.view_layer.update()
_kept(_sph2); _v1 = _kept.verts
_sec.nproto_section.slab = True; ctx.view_layer.update()
check("visual clip cuts cleanly: every vertex left is on the kept side and the cut reaches the plane exactly "
      "(no sawtooth of whole triangles); Clip Offset the same at its second plane",
      abs(_v1[1]) < 1e-5 and abs(_v3[1]) < 1e-5 and abs(_v3[0] + 0.3) < 1e-5, f"behind {_v1} slab {_v3}")
from mathutils import Vector as _V
_seg = _sx.keep_segments([_V((0, 0, -1)), _V((0, 0, 1)), _V((1, 0, 2)), _V((1, 0, 3))],
                         [(_V((0, 0, 0)), _V((0, 0, 1)), None)])
check("visual clip: B-rep edge lines are cut exactly at the plane", len(_seg) == 2 and abs(_seg[1].z) < 1e-12
      and abs(_seg[0].z + 1) < 1e-12, str(_seg))
check("visual clip: only the faces behind the middle plane stay (Flip: the other side; Clip Offset: a slab)",
      _sph2.modifiers.get(_sx.CLIP) is not None and 0 < _c1[0] < _n_all and _c1[2] <= 1e-6
      and _c2[1] >= -1e-6 and 0 < _c3[0] < _c1[0] and _c3[1] >= -0.3 - 1e-6 and _planes,
      f"all {_n_all} behind {_c1} flip {_c2} slab {_c3}")
_sec.nproto_section.slab = False; ctx.view_layer.update()
_c4 = _kept(_sph2)
bpy.ops.nproto.section_add(); ctx.view_layer.update()          # switched off: the clip pauses
_c5 = _kept(_sph2)
bpy.ops.nproto.section_add(); ctx.view_layer.update()          # on again: it clips as before
_c6 = _kept(_sph2)
check("visual clip pauses while the section is switched off and comes back with it",
      _c5[0] == _n_all and _c6[0] == _c4[0] < _n_all, f"{_c4[0]} -> off {_c5[0]} -> on {_c6[0]}")
_sec.nproto_section.clip = False; ctx.view_layer.update()
_off = _sph2.modifiers.get(_sx.CLIP) is None
_sec.nproto_section.clip = True; ctx.view_layer.update()
ctx.view_layer.objects.active = _sph2
bpy.ops.nproto.section_remove(name=_sec.name); ctx.view_layer.update(); _sx._clip_cleanup(ctx.scene)
check("visual clip: turning it off, or removing the section (Remove), removes the modifier and the section",
      _off and _sph2.modifiers.get(_sx.CLIP) is None and not any(_sx.is_section(o) for o in ctx.scene.objects) and len(_sph2.evaluated_get(ctx.evaluated_depsgraph_get()).data.polygons) == _n_all)

# --- Surface Combs: exact curvature combs on iso-curves ---
from blender_nurbs import isocombs as _ic
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=0.5, height=1.0, caps=False); _icy = ctx.active_object
_icy.location = (0, 1700, 0); ctx.view_layer.update()
_vw.iso_count_u = 3; _vw.iso_count_v = 3; _vw.iso_samples = 40
_runs, _k95, _isz = _ic.compute(_icy, _vw)
_round, _flat, _bad = 0, 0, []
for _P, _K in _runs:
    _k = np.linalg.norm(_K, axis=1)
    if _k.max() < 1e-6:
        _flat += 1
        continue
    _round += 1
    _ax = np.array([0.0, 1700.0])
    for _p, _kv in zip(_P, _K):
        _in = np.array([_ax[0] - _p[0], _ax[1] - _p[1], 0.0]); _in /= np.linalg.norm(_in)
        _bad.append(max(abs(np.linalg.norm(_kv) * 0.5 - 1.0), 1.0 - float(_kv @ _in) / np.linalg.norm(_kv)))
check("surface combs: on a cylinder the iso-curves around it have curvature exactly 1/r pointing to the axis, "
      "those along it none", _round == 3 and _flat == 3 and max(_bad) < 1e-4, f"round {_round} flat {_flat} err {max(_bad) if _bad else None}")

# --- Visual Clip on a straight wall (long triangles) cut by a tilted plane: the cut stays ON the surface ---
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _tc = ctx.active_object
_tc.location = (0, 1800, 0); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((_tc, "S0")), distance=2.0, direction="Z"); _tw = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); _tw.select_set(True); ctx.view_layer.objects.active = _tw
bpy.ops.nproto.section_add(); _ts = ctx.active_object
_ts.location = (0, 1800, 1.0); _ts.rotation_euler = (math.radians(35), 0, 0)
_ts.nproto_section.clip = True; ctx.view_layer.update()
_dg = ctx.evaluated_depsgraph_get()
_me = _tw.evaluated_get(_dg).to_mesh()
_Mw = _tw.matrix_world
_vs = [(_Mw @ vv.co) for vv in _me.vertices]
_tw.evaluated_get(_dg).to_mesh_clear()
_nrm_t = (_ts.matrix_world.to_3x3() @ Vector((0, 0, 1))).normalized()
_rad = max(abs(math.hypot(p_.x, p_.y - 1800) - 1.0) for p_ in _vs)
_sd = max((p_ - _ts.matrix_world.translation).dot(_nrm_t) for p_ in _vs)
_edge = max((_Mw @ _tw.data.vertices[e.vertices[0]].co - _Mw @ _tw.data.vertices[e.vertices[1]].co).length
            for e in _tw.data.edges)
check("visual clip, tilted plane through a straight wall: the cut edge lies on the plane AND on the surface "
      "(corners move along the surface, not off it; float32 mesh at y = 1800)", _sd < 2e-4 and _rad < 2e-3,
      f"off the cylinder {_rad:.2e}, past the plane {_sd:.2e}, longest mesh edge {_edge:.3f}")
bpy.ops.nproto.section_remove(name=_ts.name)

# --- Face / Edge selection modes (picks on a body; CVs / Hulls and Continuity Check follow them) ---
from blender_nurbs import subsel as _ssel
from blender_nurbs import contcheck as _ccs
_vw = ctx.window_manager.nproto_view
_sb = _revs["BLEND2"][1]
bpy.ops.object.select_all(action='DESELECT'); _sb.select_set(True); ctx.view_layer.objects.active = _sb
_sbf = store.get_brep(_sb).faces()
_ri = next(k for k, f in enumerate(_sbf) if _skey(f.surface()) not in _dkeys)        # the ring face
_sb[_ssel.PROP] = "x"
_vw.sel_mode = {"FACE"}
_cleared = _ssel.PROP not in _sb
_ssel.set_picks(_sb, "FACE", [_ri])
_nets = store.display(_sb).get("nets", [])
_shown = [n for k, n in enumerate(_nets) if n.get("f", k) in (_ssel.faces_shown(_sb) or set())]
_pts, _tris, _polys = _ssel.face_tris(_sb, _ri)
_tri_area = float(sum(np.linalg.norm(np.cross(_pts[b] - _pts[a], _pts[c] - _pts[a])) / 2 for a, b, c in _tris))
check("Face mode: switching mode clears old picks; a picked face is kept on the body, CVs / Hulls shows only its net, "
      "and its own tessellation covers it (area within 2%)",
      _cleared and _ssel.get(_sb) == ("FACE", [_ri]) and len(_nets) == 3 and len(_shown) == 1
      and _shown[0]["f"] == _ri and abs(_tri_area - _sbf[_ri].area()) < 0.02 * _sbf[_ri].area() and len(_polys) >= 2,
      f"get {_ssel.get(_sb)} nets {len(_nets)} shown {len(_shown)} area {_tri_area:.4f} vs {_sbf[_ri].area():.4f}")
# a stored pick whose index no longer matches (a rebuild renumbered the faces) is found again by its point
_d0 = json.loads(_sb[_ssel.PROP]); _wrong = (_ri + 1) % len(_sbf)
_sb[_ssel.PROP] = json.dumps({"kind": "FACE", "items": [[_wrong, _d0["FACE"][0][1]]]})   # older files' format
_remap = _ssel.get(_sb)
_sb.select_set(False)
_desel = _ssel.get(_sb)
_sb.select_set(True)
check("Face mode: a pick survives renumbering (found again by its point) and only counts while its body is selected",
      _remap == ("FACE", [_ri]) and _desel == (None, []), f"{_remap} {_desel}")
_vw.sel_mode = {"EDGE"}
_sh = store.shared_edges(_sb)
_ssel.set_picks(_sb, "EDGE", _sh)
_cleared_f = _ssel.faces_shown(_sb) is None
_ccs.clear()
_nadd = _ccs.add_selected(ctx)
_jj = [_ccs.judge(_ccs.result(p_), _vw) for p_ in _ccs.PAIRS]
_ccs.clear()
check("Edge mode: picked edges are what Edge Fillet / Continuity Check take; the check measures them at once "
      "(the drafted G2 rim: G2 both sides)",
      _cleared_f and _ssel.selected_edges(ctx) == [(_sb, f"E{i}") for i in _sh] and _nadd == len(_sh) == 2
      and all(isinstance(j, dict) and j["G2"][1] for j in _jj), f"{_ssel.selected_edges(ctx)} {_nadd} {_jj}")
_vw.sel_mode = {"BODY"}
check("Body mode: the picks go", _ssel.PROP not in _sb and _ssel.get(_sb) == (None, []))

# --- Dome: close an open rim, G1 / G2 (exact revolution on a round rim, the rim's own curve otherwise) ---
from blender_nurbs import dome as _dome
from blender_nurbs import contcheck as _ccd
from blender_nurbs.ops import _pick_candidates as _pcd
def _top_rim(o):
    return max(("U0", "U1", "V0", "V1"), key=lambda w: np.mean([q[2] for q in geom.boundary_points(store.get_surfaces(o)[0], w, 8)]))
def _dome_cont(base, ref):
    _ccd.clear()
    cands = [c for c in _pcd(ctx, True) if not c[1].startswith("S")]
    own = next(c for c in cands if c[0] == base and c[1] == ref)
    partner = _ccd.find_partner(base, ref, own[2], cands)
    if partner is None:
        return None
    _ccd.add(base, ref, partner)
    j = _ccd.judge(_ccd.result(_ccd.PAIRS[0]), ctx.window_manager.nproto_view)
    _ccd.clear()
    return {k: float(v[0]) for k, v in j.items()}
_dres = {}
for _i, (_shape, _cont) in enumerate((("CIRCLE", "G1"), ("CIRCLE", "G2"), ("ELLIPSE", "G2"))):
    bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _dc = ctx.active_object
    if _shape == "ELLIPSE":
        _h = json.loads(_dc[features.PROP_HIST]); _h["params"]["elliptic"] = True; _h["params"]["radius2"] = 0.6
        _dc[features.PROP_HIST] = json.dumps(_h); features.rebuild(_dc, ctx, force=True)
    _dc.location = (4.0 * _i, 2100, 0); ctx.view_layer.update()
    bpy.ops.nproto.extrude(inputs=ins((_dc, "S0")), distance=2.0, direction="Z", draft=-8.0); _dbase = ctx.active_object
    _dref = _top_rim(_dbase)
    r = bpy.ops.nproto.dome(inputs=ins((_dbase, _dref)), dome_cont=_cont, dome_height=0.75)
    _dd = ctx.active_object
    _dcv = ext.surface_cvs(store.get_surfaces(_dd)[0])
    _dres[_shape + _cont] = {"ok": r == {'FINISHED'} and _dd.name.startswith("Dome"), "deg": (_dcv["degree_u"], _dcv["degree_v"]),
                             "n": (_dcv["count_u"], _dcv["count_v"]), "j": _dome_cont(_dbase, _dref),
                             "round": _dome.rim_data(_dbase, _dref)["round"], "obj": _dd, "base": _dbase, "ref": _dref}
_g1, _g2, _el = _dres["CIRCLEG1"], _dres["CIRCLEG2"], _dres["ELLIPSEG2"]
check("dome on a round rim (a drafted cone): an exact surface of revolution, 9 rational CVs round; G1 tangent, "
      "G2 also with the wall's curvature",
      _g1["ok"] and _g2["ok"] and _g1["round"] and _g1["deg"] == (3, 2) and _g1["n"] == (4, 9) and _g2["deg"] == (5, 2)
      and _g1["j"] and _g1["j"]["G0"] < 1e-6 and _g1["j"]["G1"] < 0.01
      and _g2["j"] and _g2["j"]["G0"] < 1e-6 and _g2["j"]["G1"] < 0.01 and _g2["j"]["G2"] < 0.005,
      str({k: (v["deg"], v["n"], v["j"]) for k, v in _dres.items()}))
check("dome on a planar rim that is not round (a drafted ellipse): on the rim's own curve, G2 to it",
      _el["ok"] and not _el["round"] and _el["j"] and _el["j"]["G0"] < 1e-5 and _el["j"]["G1"] < 0.05
      and _el["j"]["G2"] < 0.02, str(_el["j"]))
_h = json.loads(_g2["base"][features.PROP_HIST]); _h["params"]["draft"] = -3.0
_g2["base"][features.PROP_HIST] = json.dumps(_h)
features.rebuild(_g2["base"], ctx, force=True); features.rebuild(_g2["obj"], ctx)
_jj = _dome_cont(_g2["base"], _g2["ref"])
check("dome history: change the surface's draft and the dome follows (still G2)",
      _jj and _jj["G0"] < 1e-6 and _jj["G1"] < 0.01 and _jj["G2"] < 0.005 and features.history(_g2["obj"])["op"] == "dome",
      str(_jj))

# a barrel loft through three ellipses, its top rim leaning in (the fit used to stall at the loft's seam)
_bl = []
for _z, _r in ((0.0, 1.0), (1.0, 1.25), (2.0, 0.8)):
    bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=_r); _bc = ctx.active_object
    _h = json.loads(_bc[features.PROP_HIST]); _h["params"]["elliptic"] = True; _h["params"]["radius2"] = 0.7 * _r
    _bc[features.PROP_HIST] = json.dumps(_h); features.rebuild(_bc, ctx, force=True)
    _bc.location = (0, 2120, _z); ctx.view_layer.update()
    _bl.append(_bc)
bpy.ops.nproto.loft(inputs=ins(*[(c_, "S0") for c_ in _bl])); _bar = ctx.active_object
_bref = _top_rim(_bar)
r = bpy.ops.nproto.dome(inputs=ins((_bar, _bref)), dome_cont="G2")
_bd = ctx.active_object
_bj = _dome_cont(_bar, _bref) if r == {'FINISHED'} else None
check("dome on a barrel loft's top rim (an ellipse leaning in): built, G2 to it",
      r == {'FINISHED'} and _bd.name.startswith("Dome") and _bj and _bj["G0"] < 1e-5 and _bj["G1"] < 0.05 and _bj["G2"] < 0.02,
      str(_bj))

# draped: one even four-sided surface over the opening, trimmed to the rim (round, a flat spatula, an ellipse)
_drp = {}
for _i, (_nm, _r2, _dr) in enumerate((("round", None, -8.0), ("spatula", 0.25, 0.0), ("ellipse", 0.6, -5.0))):
    bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _dc = ctx.active_object
    if _r2:
        _h = json.loads(_dc[features.PROP_HIST]); _h["params"]["elliptic"] = True; _h["params"]["radius2"] = _r2
        _dc[features.PROP_HIST] = json.dumps(_h); features.rebuild(_dc, ctx, force=True)
    _dc.location = (4.0 * _i, 2140, 0); ctx.view_layer.update()
    bpy.ops.nproto.extrude(inputs=ins((_dc, "S0")), distance=2.0, direction="Z", draft=_dr); _db = ctx.active_object
    _rf = _top_rim(_db)
    r = bpy.ops.nproto.dome(inputs=ins((_db, _rf)), dome_cont="G2", dome_layout="DRAPED")
    _do = ctx.active_object
    _f = store.get_brep(_do).faces()
    _sd = ext.surface_cvs(_f[0].surface()) if _f else None
    _drp[_nm] = {"ok": r == {'FINISHED'} and _do.name.startswith("Dome"), "faces": len(_f),
                 "net": _sd and (_sd["count_u"], _sd["count_v"], _sd["degree_u"]), "j": _dome_cont(_db, _rf)}
check("draped dome: one even 20 x 20 degree-5 surface trimmed to the rim (no pole), G2 to it -- round, a flat "
      "spatula (4:1) and a drafted ellipse",
      all(v["ok"] and v["faces"] == 1 and v["net"] == (20, 20, 5) and v["j"] and v["j"]["G0"] < 1e-5
          and v["j"]["G1"] < 0.1 and v["j"]["G2"] < 0.05 for v in _drp.values()),
      str({k: (v["net"], v["j"]) for k, v in _drp.items()}))

# the G2 dome is round at its top (it used to go flat there: its top three CV rows at one height)
def _apex_k(obj, frac=0.04):
    s = store.get_surfaces(obj)[0]
    (u0, u1), (v0, v1) = s.u_domain(), s.v_domain()
    vs = np.linspace(v0, v1, 16, endpoint=False)
    def spread(u):
        return float(np.ptp(np.array([s.evaluate(u, v) for v in vs]), axis=0).max())
    ue, ui = (u1, u1 - frac * (u1 - u0)) if spread(u1) < spread(u0) else (u0, u0 + frac * (u1 - u0))
    top = np.array(s.evaluate(ue, v0))
    ks = []
    for v in vs:                                   # circles through the top, centred on the axis
        q = np.array(s.evaluate(ui, v))
        d, dz = float(np.hypot(*(q - top)[:2])), float(top[2] - q[2])
        ks.append(2.0 * dz / (d * d + dz * dz))
    return min(ks), max(ks)
r = bpy.ops.nproto.dome(inputs=ins((_g2["base"], _g2["ref"])), dome_cont="G1", dome_height=0.75)   # (its draft changed above)
_ka1, _ka2 = _apex_k(ctx.active_object), _apex_k(_g2["obj"])
r = bpy.ops.nproto.dome(inputs=ins((_el["base"], _el["ref"])), dome_cont="G1", dome_height=0.75)
_ke1, _ke2 = _apex_k(ctx.active_object), _apex_k(_el["obj"])
check("G2 dome: as round at its top as the G1 one (no flat tip) -- round rim and ellipse",
      _ka2[0] > 0.9 * _ka1[0] and _ka2[0] > 0.5 and _ke2[0] > 0.85 * _ke1[0] and _ke2[0] > 0.3,
      f"round G1 {_ka1} G2 {_ka2}; ellipse G1 {_ke1} G2 {_ke2}")

# Dome's live stage: the preview lies on the dome it makes, the arrow's tip is its top, the diamond is on it and
# moves along its line as Fullness changes; both handles give their values back (round rim and ellipse)
_dm_rep = []
for _dbase, _dref in ((_g1["base"], _g1["ref"]), (_el["base"], _el["ref"])):
    for _cont, _hh, _ff in (("G2", 1.0, 1.0), ("G1", 0.7, 1.4), ("G2", 1.6, 0.6)):
        _x = _plain(_dome.NPROTO_OT_dome, inputs=ins((_dbase, _dref)), dome_cont=_cont, dome_height=_hh,
                    dome_fullness=_ff, dome_layout="POLAR", dome_cvs=20)
        _ok = _x.drag_setup(ctx, None)
        _pv, _hs = _x.drag_preview(), _x.drag_handles()
        _hf = _hs[1]
        _x.dome_fullness = _hf["to_value"](1.1 * _hf["param"])
        _lin = float(np.linalg.norm(_vd.ValueDrag._pos(_x.drag_handles()[1]) - (_hf["origin"] + 1.1 * _hf["param"] * _hf["dir"])))
        _x.dome_fullness = _ff
        r = bpy.ops.nproto.dome(inputs=ins((_dbase, _dref)), dome_cont=_cont, dome_height=_hh, dome_fullness=_ff)
        _res = ctx.active_object; ctx.view_layer.update(); _t = _tree(_res)
        _gap = max(_t.find_nearest(Vector(q_))[3] for q_ in np.vstack([_pv["tris"]] + list(_pv["lines"])))
        _on = max(_t.find_nearest(Vector(_vd.ValueDrag._pos(h_)))[3] for h_ in _hs)
        _dm_rep.append((_ok and r == {'FINISHED'}, round(_gap, 5), round(_on, 5), round(_lin, 5),
                        round(_hs[0]["to_value"](_hs[0]["param"]) - _hh, 9), round(_hf["to_value"](_hf["param"]) - _ff, 9)))
        bpy.data.objects.remove(_res)
check("dome live stage: preview on the dome it makes, height arrow's tip its top, fullness diamond on it and moving "
      "along its line, values back from the handles",
      all(v[0] and v[1] < 2e-3 and v[2] < 2e-3 and v[3] < 0.01 and v[4] == 0 and v[5] == 0 for v in _dm_rep), str(_dm_rep))

# Dome's continuity label (as Patch's): one on the rim, G0 / G1 / G2, the polar dome holds each; G0 ignores the wall
# (a hemisphere-like cap: the drafted cone's 8 degrees show as a kink); a click steps it through the history
from blender_nurbs import contlabels as _cl
# (the ellipse without draft: Extrude's draft on its rational circle creases the wall 0.1 degrees at the double
# knots, which the label rightly shows for any dome)
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=1.0); _dc = ctx.active_object
_h = json.loads(_dc[features.PROP_HIST]); _h["params"]["elliptic"] = True; _h["params"]["radius2"] = 0.6
_dc[features.PROP_HIST] = json.dumps(_h); features.rebuild(_dc, ctx, force=True)
_dc.location = (12.0, 2100, 0); ctx.view_layer.update()
bpy.ops.nproto.extrude(inputs=ins((_dc, "S0")), distance=2.0, direction="Z"); _el0 = ctx.active_object
_dl_rep, _dl_ok = [], True
for _dbase, _dref in ((_g1["base"], _g1["ref"]), (_el0, _top_rim(_el0))):
    for _cont in ("G0", "G1", "G2"):
        r = bpy.ops.nproto.dome(inputs=ins((_dbase, _dref)), dome_cont=_cont, dome_height=1.0, dome_fullness=1.0,
                                dome_layout="POLAR")
        _do = ctx.active_object
        _L = _cl.labels(_do)
        _kink = _cl.measure(_do, _dbase, _dref, 1)["angle"]
        _dl_rep.append((_cont, len(_L), _L and _L[0]["value"], _L and _L[0]["check"]["ok"], round(_kink, 3)))
        _dl_ok &= (r == {'FINISHED'} and len(_L) == 1 and _L[0]["values"] == ["G0", "G1", "G2"] and _L[0]["value"] == _cont
                   and _L[0]["check"]["ok"] and ((_kink > 5.0) if _cont == "G0" else (_kink < 0.1)))
        if _cont == "G2" and _dbase is _g1["base"]:
            bpy.ops.nproto.cycle_continuity('EXEC_DEFAULT', obj_name=_do.name, index=0, step=1)
            _L2 = _cl.labels(_do)
            _dl_ok &= features.history(_do)["params"]["dome_cont"] == "G0" and _L2[0]["value"] == "G0" and _L2[0]["check"]["ok"]
            _dl_rep.append(("click", features.history(_do)["params"]["dome_cont"]))
        bpy.data.objects.remove(_do)
check("dome continuity label: one on the rim (G0 / G1 / G2), polar dome holds each on a round rim and an ellipse; "
      "G0 meets the rim with the wall's kink; a click steps it G2 -> G0", _dl_ok, str(_dl_rep))

# a loft of selected curves goes along them (a selection has no order: bottom, top, middle folded back)
from blender_nurbs.ops import loft_selection_order as _lso
_ls = []
for _z, _r in ((0.0, 1.0), (2.0, 1.0), (1.0, 1.3)):
    bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=_r); _lc = ctx.active_object
    _lc.location = (0, 2160, _z); ctx.view_layer.update(); _ls.append(_lc)
_lo = _lso(_ls, _ls[2])
bpy.ops.nproto.loft(inputs=ins(*[(c_, "S0") for c_ in _lo])); _lf = ctx.active_object
_Ml = geom.world_matrix(_lf)
_lz = sorted(float(np.mean([(_Ml @ Vector(q))[2] for q in geom.boundary_points(store.get_surfaces(_lf)[0], w, 16)]))
             for w in ("U0", "U1", "V0", "V1"))
check("loft of a selection: along the sections (bottom, middle, top), not in the selection's order; picks keep "
      "theirs; two sections keep the active one last",
      [o.name for o in _lo] == [_ls[0].name, _ls[2].name, _ls[1].name] and abs(_lz[0]) < 1e-3 and abs(_lz[-1] - 2.0) < 1e-3
      and [o.name for o in _lso(_ls[:2], _ls[0])] == [_ls[1].name, _ls[0].name]
      and [o.name for o in _lso(_ls, _ls[2], closed=True)][-1] == _ls[2].name,
      f"{[o.name for o in _lo]} ends {_lz}")

# loft through sections of different layouts (two circles with Bezier knots, a reshaped one with uniform knots
# that starts elsewhere): the kernel's skin rose past the top section and folded back down to it, and twisted
def _cyc(name, pts, z, endpoint, s=1.0):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('NURBS'); sp.points.add(len(pts) - 1)
    for p_, c_ in zip(sp.points, pts): p_.co = (s * c_[0], s * c_[1], 0.0, c_[2])
    sp.order_u = 3; sp.use_cyclic_u = True; sp.use_endpoint_u = endpoint
    o_ = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o_); o_.location = (0, 2180, z); return o_
_ring = [(0, -.161, 1), (-.181, -.161, .707), (-.181, 0, 1), (-.181, .161, .707), (0, .161, 1), (.181, .161, .707), (.181, 0, 1), (.181, -.161, .707)]
_bent = [(0, -.174, 1), (-.16, -.135, .707), (-.206, 0, 1), (-.081, .135, .707), (0, .167, 1), (.081, .135, .707), (.206, 0, 1), (.16, -.135, .707)]
_secs = [_cyc("LsA", _ring, 0.0, True), _cyc("LsB", _ring, 0.087, True), _cyc("LsC", _bent, 0.188, False, 0.749)]
ctx.view_layer.update()
_lres = {}
for _aa in (True, False):
    _g = features._build_loft([(o_, "S0") for o_ in _secs], {"auto_align": _aa, "closed": False})
    (_u0, _u1), (_v0, _v1) = _g.u_domain(), _g.v_domain()
    _Z = np.array([[_g.evaluate(u, v)[2] for u in np.linspace(_u0, _u1, 41)] for v in np.linspace(_v0, _v1, 24, endpoint=False)])
    _st = []
    for u in np.linspace(_u0, _u1, 9):
        _P = np.array([_g.evaluate(u, v) for v in np.linspace(_v0, _v1, 12, endpoint=False)]); _c = _P.mean(axis=0)
        _st.append(float(np.degrees(np.arctan2(_P[0, 1] - _c[1], _P[0, 0] - _c[0]))))
    _cp = lambda q: features._k()[0].surface_closest_point(_g, tuple(map(float, q)))[0]
    _sec_err = max(float(np.linalg.norm(np.array([_cp(q).x, _cp(q).y, _cp(q).z]) - q))
                   for o_ in _secs for q in geom.sample_curve(features.input_curve(o_, "S0"), 30))
    _lres[_aa] = {"rise": float(np.diff(_Z, axis=1).min()), "top": float(_Z.max()), "twist": max(_st) - min(_st), "err": _sec_err}
check("loft through sections of different layouts: rises steadily (no overshoot past the top and back), holds "
      "every section; Auto Align starts the closed sections alike (no twist)",
      all(v["rise"] > 0 and v["top"] < 0.188 + 1e-6 and v["err"] < 1e-6 for v in _lres.values()) and _lres[True]["twist"] < 1.0
      and _lres[False]["twist"] > 15.0, str(_lres))
_whole = features.run("loft", [(o_, "S0") for o_ in _secs], {"auto_align": True, "closed": False})
check("...and stays one face: between sections of different layouts it turns a degree at the circle's quarter "
      "knots, which is no fold (at the sections it is smooth)",
      isinstance(_whole, features._k()[0].Surface), type(_whole).__name__)
# a closed section reseated: the same curve, starting elsewhere
_kc = features.input_curve(_secs[2], "S0")
_t0, _t1 = _kc.parameter_range()
_rs = geom.reseat_closed(_kc, _t0 + 0.37 * (_t1 - _t0))
_Pa, _Pb = geom.sample_curve(_kc, 300), geom.sample_curve(_rs, 300)
check("reseat a closed curve: the same curve, starting at the new point",
      np.linalg.norm(_Pb[0] - np.array(_kc.evaluate(_t0 + 0.37 * (_t1 - _t0)))) < 1e-9
      and max(np.min(np.linalg.norm(_Pa - q, axis=1)) for q in _Pb) < 0.01 * 0.2
      and abs(np.linalg.norm(np.diff(_Pa, axis=0), axis=1).sum() - np.linalg.norm(np.diff(_Pb, axis=0), axis=1).sum()) < 1e-4)
# curve evaluation a hair past the end of the range works on every kernel build (Linux / Windows refuse
# the end point by a rounding step where macOS takes it); really out of range still raises
from blender_nurbs.kernel import load as _klh
_smh = _klh()
_ce = _smh.create_curve_interp_points([(float(x), float(np.sin(x)), 0.0) for x in np.linspace(0.0, 3.0, 8)])
_ct0, _ct1 = _ce.parameter_range()
_hair_ok = np.linalg.norm(np.array(_ce.evaluate(_ct1 + 1e-13 * (_ct1 - _ct0))) - np.array(_ce.evaluate(_ct1))) < 1e-9
try:
    _ce.evaluate(_ct1 + 0.5 * (_ct1 - _ct0)); _far_raises = False
except Exception:
    _far_raises = True
check("kernel guard: Edge.point_at / Curve.evaluate are wrapped; a hair past the end evaluates the end",
      getattr(type(_ce).evaluate, "_nproto_safe", False) and getattr(_smh.Edge.point_at, "_nproto_safe", False) and _hair_ok,
      f"far past the end raises: {_far_raises}")

# --- Line: Perpendicular / Tangent end snaps, lines tangent to two circles, square / parallel directions
from blender_nurbs import linesolve as _lsv


def _circ(c, r):
    c = np.asarray(c, float)
    return _lsv.Curve(lambda t: (c + r * np.array([math.cos(t), math.sin(t), 0.0]),
                                 r * np.array([-math.sin(t), math.cos(t), 0.0]),
                                 -r * np.array([math.cos(t), math.sin(t), 0.0])), 0.0, 2 * math.pi)


_C1, _C2 = _circ((0, 0, 0), 1.0), _circ((4, 0, 0), 0.5)
_tp = _lsv.from_point(_C1, (3, 0, 0), "tan")
check("line solve: the two tangent points from a point to a circle (x = r^2 / d)", len(_tp) == 2 and all(
    abs(Q[0] - 1 / 3) < 1e-12 and abs(abs(Q[1]) - math.sqrt(8 / 9)) < 1e-12 for _, Q in _tp), str(_tp))
_seeds = [s_ for a_ in (0.0, 1.5, 3.1, 4.7) for s_ in _lsv.seeds_from(_C1, a_, _C2)]
_tt = _lsv.between(_C1, "tan", _C2, "tan", _seeds)
_exp = []                       # outer tangents: cos = (r1 - r2) / d; inner: (r1 + r2) / d
for _cth, _sg in ((0.125, 1), (0.375, -1)):
    for _sy in (1, -1):
        _sth = _sy * math.sqrt(1 - _cth * _cth)
        _exp.append((np.array([_cth, _sth, 0]), np.array([4 + _sg * 0.5 * _cth, _sg * 0.5 * _sth, 0])))
check("line solve: two circles have four lines tangent to both, all found exactly", len(_tt) == 4 and all(
    any(np.allclose(Q1, a_, atol=1e-12) and np.allclose(Q2, b_, atol=1e-12) for _, _, Q1, Q2 in _tt)
    for a_, b_ in _exp), str([(s_[2], s_[3]) for s_ in _tt]))
_pp = _lsv.between(_C1, "perp", _C2, "perp", _seeds)
_tq = _lsv.between(_C1, "tan", _C2, "perp", _seeds)
check("line solve: square to both circles runs along the line of centres; tangent to one, square to the other",
      len(_pp) == 4 and all(abs(Q1[1]) < 1e-12 and abs(Q2[1]) < 1e-12 for _, _, Q1, Q2 in _pp)
      and len(_tq) == 4 and all(abs(np.linalg.norm(Q1) - 1) < 1e-12 and abs((Q2 - Q1) @ Q1) < 1e-12
                                and abs(np.cross(Q2 - Q1, Q2 - (4, 0, 0))[2]) < 1e-12 for _, _, Q1, Q2 in _tq))
_ln = _lsv.Curve(lambda t: (np.array([-2.0, -3, 0]) + t * np.array([7.0, 4, 0]), np.array([7.0, 4, 0]), np.zeros(3)),
                 0.0, 1.0)
_ft = _lsv.from_point(_ln, (0, 0, 0), "perp")
check("line solve: the foot of the perpendicular on a line; a straight line has no tangent points",
      _ln.straight and len(_ft) == 1 and np.allclose(_ft[0][1], (0.8, -1.4, 0), atol=1e-12)
      and _lsv.from_point(_ln, (0, 0, 0), "tan") == [], str(_ft))
check("line solve: no tangent from a point off the circle's plane (a right angle still works)",
      _lsv.from_point(_C1, (3, 0, 0.5), "tan") == [] and len(_lsv.from_point(_C1, (3, 0, 0.5), "perp")) == 2)
check("line solve: square to a circle lies in its plane, square to a line in the drawing plane",
      np.allclose(np.abs(_lsv.side_direction(_C1, 0.0)), (1, 0, 0))
      and np.allclose(np.abs(_lsv.side_direction(_ln, 0.3, (0, 0, 1))), np.abs(np.array([-4.0, 7, 0]) / math.sqrt(65))))
# the add-on's own circles, through the kernel
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.6)
_KA = ctx.active_object
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.3)
_KB = ctx.active_object
_KA.location, _KB.location = (-1, 0, 0), (1, 0, 0)
ctx.view_layer.update()
_smk = features._k()[0]
_kA = _lsv.kernel_curve(features.input_curve(_KA, "S0"), _smk)
_kB = _lsv.kernel_curve(features.input_curve(_KB, "S0"), _smk)
_ktt = _lsv.between(_kA, "tan", _kB, "tan", _lsv.seeds_from(_kA, _lsv.nearest_t(_kA, (-0.9, 0.6, 0)), _kB))


def _ksin(c, t, V):
    D = c.at(t)[2]
    return float(np.linalg.norm(np.cross(D, V)) / (np.linalg.norm(D) * np.linalg.norm(V)))


check("line solve on kernel circles: both lines tangent to both on the side clicked, exact to rounding",
      len(_ktt) == 2 and all(Q1[1] > 0 and _ksin(_kA, a_, Q2 - Q1) < 1e-9 and _ksin(_kB, b_, Q2 - Q1) < 1e-9
                             for a_, b_, Q1, Q2 in _ktt), str([(s_[2], s_[3]) for s_ in _ktt]))
_Tk = _sn.collect([_KA, _KB])
_nl = _Tk.near_lines(_snap_view(), _px(-0.4, 0.0) + (5, 0), 2 * _sn.SNAP_PX)
check("snap: the curves passing near the mouse, nearest first",
      len(_nl) >= 1 and _Tk.line_info(_nl[0][0])[1][:2] == (_KA.name, "S0") and abs(_nl[0][1] - 5.0) < 0.1, str(_nl))


# the Line tool's own logic, on a stand-in (a modal operator cannot run headless)
class _FakeLine(_dt.SketchMixin):
    start_mode, line_mode = "FREE", "EDGE"


for _nm in ("_curve", "_start_curve", "_solutions", "_apply_start"):
    setattr(_FakeLine, _nm, _dt.NPROTO_OT_draw_line.__dict__[_nm])
_fl = _FakeLine()
_fl._curves, _fl._solved, _fl._lock = {}, {}, None
_fl._plane = (np.zeros(3), np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]))
_fl._start, _fl._start_refs = np.array([0.8, 0.0, 0.0]), None
_sol = _fl._solutions((_KA.name, "S0"), _kA)
_lab = sorted(s_[0] for s_ in _sol)
check("Line: from a free start the end snaps Perpendicular (2) and Tangent (2) to a circle",
      _lab == ["Perpendicular"] * 2 + ["Tangent"] * 2
      and all(abs(np.linalg.norm(Q - (-1, 0, 0)) - 0.6) < 1e-6 for _, _, Q in _sol), str(_lab))
_ta = _lsv.nearest_t(_kA, (-1.0, 0.6, 0))
_fl._start = _kA.at(_ta)[1]
_fl._start_refs = {"cands": [{"src": (_KA.name, "S0", "NEAREST", None)}], "i": 0}
_fl.start_mode = "TANGENT"
_sol2 = _fl._solutions((_KB.name, "S0"), _kB)
_both = [s_ for s_ in _sol2 if s_[0] == "Tangent to both"]
check("Line: a Tangent start slides along its circle to a line tangent to both circles",
      len(_both) == 2 and all(np.linalg.norm(A_ - _fl._start) > 1e-3 and A_[1] > 0 for _, A_, _ in _both)
      and any(s_[0] == "Tangent · Perpendicular" for s_ in _sol2), str([s_[0] for s_ in _sol2]))
_fl._apply_start()
_tan_ok = _fl._lock == "T" and abs(_fl._tvec[1]) < 1e-6 * np.linalg.norm(_fl._tvec)
_fl.start_mode = "PERP"
_fl._apply_start()
check("Line: a Tangent start leaves along the circle, a Perpendicular one square to it (in its plane)",
      _tan_ok and _fl._lock == "T" and _fl._tkind == "square" and np.allclose(np.abs(_fl._tvec), (0, 1, 0), atol=1e-6),
      f"{_fl._tvec}")
_fl._lock, _fl._akind = "A", "square"
check("Line: the A lock reads 'square to the line'", "square to the line" in _fl.lock_text())

# ...and through the modal code: mouse moves and clicks in the fake top view (1 unit = 250 px)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.6)
_MA = ctx.active_object
bpy.ops.nproto.add_primitive(kind="CIRCLE", radius=0.3)
_MB = ctx.active_object
_MA.location, _MB.location = (-1, 0, 0), (1, 0, 0)
ctx.view_layer.update()
bpy.ops.nproto.draw_line(data=json.dumps({"lines": [[[-1.5, -1.3, 0.0], [1.5, -0.7, 0.0]]], "edit": ""}))
_ML = ctx.active_object


def _ln_tool(**props):
    f = _fake(_dt.NPROTO_OT_draw_line, **dict(dict(polyline=False, start_mode="FREE", line_mode="EDGE"), **props))
    f._edit, f._lines, f._start, f._lock, f._start_refs = None, [], None, None, None
    f._reset_typing()
    f.sketch_start(fc)
    return f


def _at(f, q, px=(0, 0), **kw):
    xy = np.array(_wpx(q[0], q[1])) + px
    f.modal(fc, _ev('MOUSEMOVE', xy, 'NOTHING', **kw))
    return xy


def _click(f, q, px=(0, 0)):
    xy = _at(f, q, px)
    f.modal(fc, _ev('LEFTMOUSE', xy))


_c1, _r1, _c2, _r2 = np.array([-1.0, 0, 0]), 0.6, np.array([1.0, 0, 0]), 0.3
# a free start, the end Tangent to a circle
_S = np.array([0.0, 1.5, 0])
_d = _S - _c1
_dh, _ca = _d / np.linalg.norm(_d), _r1 / np.linalg.norm(_d)
_Qt = _c1 + _r1 * (_ca * _dh + math.sqrt(1 - _ca * _ca) * np.array([-_dh[1], _dh[0], 0]))
f = _ln_tool()
_click(f, _S)
_at(f, _Qt, (3, 2))
_lab = _drw.sketch_overlay["snap_label"]
f.modal(fc, _ev('LEFTMOUSE', np.array(_wpx(*_Qt[:2])) + (3, 2)))
_L0 = f._lines[0] if f._lines else (np.zeros(3), np.zeros(3))
check("Line modal: from a free start the end snaps Tangent to a circle (exact tangent point)",
      _lab == "Tangent" and np.linalg.norm(_L0[0] - _S) < 1e-9 and np.linalg.norm(_L0[1] - _Qt) < 1e-6,
      f"{_lab} {_L0} want {_Qt}")
f.modal(fc, _ev('ESC'))
# a Tangent start on the big circle's top (a CV on the curve): it slides to the line tangent to both circles
_ct = (_r1 - _r2) / np.linalg.norm(_c2 - _c1)
_nt = np.array([_ct, math.sqrt(1 - _ct * _ct), 0])
_Q1, _Q2 = _c1 + _r1 * _nt, _c2 + _r2 * _nt
f = _ln_tool(start_mode="TANGENT")
_click(f, (-1.0, 0.6))
_lk = f._lock
_at(f, _Q2, (2, 3))
_lab = _drw.sketch_overlay["snap_label"]
f.modal(fc, _ev('LEFTMOUSE', np.array(_wpx(*_Q2[:2])) + (2, 3)))
_L0 = f._lines[0] if f._lines else (np.zeros(3), np.zeros(3))
check("Line modal: a Tangent start on a circle's quarter point (a CV) leaves tangent, then slides to the line "
      "tangent to both circles", _lk == "T" and _lab == "Tangent to both" and np.linalg.norm(_L0[0] - _Q1) < 1e-6
      and np.linalg.norm(_L0[1] - _Q2) < 1e-6, f"{_lk} {_lab} {_L0} want {_Q1} {_Q2}")
f.modal(fc, _ev('ESC'))
# a free start, the end Perpendicular to a line
_S = np.array([-0.2, 1.2, 0])
_LA, _LD = np.array([-1.5, -1.3, 0]), np.array([3.0, 0.6, 0])
_foot = _LA + _LD * ((_S - _LA) @ _LD) / (_LD @ _LD)
f = _ln_tool()
_click(f, _S)
_at(f, _foot, (3, -2))
_lab = _drw.sketch_overlay["snap_label"]
f.modal(fc, _ev('LEFTMOUSE', np.array(_wpx(*_foot[:2])) + (3, -2)))
_L0 = f._lines[0] if f._lines else (np.zeros(3), np.zeros(3))
check("Line modal: the end snaps Perpendicular to a line (its foot)",
      _lab == "Perpendicular" and np.linalg.norm(_L0[1] - _foot) < 1e-6, f"{_lab} {_L0} want {_foot}")   # curves are float32
f.modal(fc, _ev('ESC'))
# A / Shift+A over the line: the next line runs parallel / square to it
_dirs = {}
for _sh in (False, True):
    f = _ln_tool()
    _at(f, (-1.0, -1.2))
    f.modal(fc, _ev('A', _wpx(-1.0, -1.2), shift=_sh))
    _kind = (f._lock, getattr(f, "_akind", None))
    _click(f, (-1.5, 1.6))
    _click(f, (0.8, 1.95) if not _sh else (-1.2, 0.2))
    a_, b_ = f._lines[0] if f._lines else (np.zeros(3), np.ones(3))
    _dirs[_sh] = (_kind, (b_ - a_) / np.linalg.norm(b_ - a_))
    f.modal(fc, _ev('ESC'))
_ud = _LD / np.linalg.norm(_LD)
check("Line modal: A draws parallel to the line under the mouse, Shift+A square to it",
      _dirs[False][0] == ("A", "parallel") and abs(abs(_dirs[False][1] @ _ud) - 1) < 1e-9
      and _dirs[True][0] == ("A", "square") and abs(_dirs[True][1] @ _ud) < 1e-6, str(_dirs))

# --- Key hints: a short header, a grouped panel that wraps to the width, the mouse basics in the status bar
from blender_nurbs import hints as _hn
_meas = lambda t: (7.0 * len(t), 12.0)
_rows = [("Place", [_hn.key("0-9", "length"), _hn.key("Tab", "angle"), _hn.key("Enter", "place")]),
         ("Direction", [_hn.key("H V", "horizontal / vertical", on=True), _hn.key("X Y Z", "axis"),
                        _hn.key("Shift+X/Y/Z", "plane"), _hn.key("Shift", "15°"), _hn.key("Ctrl", "grid")]),
         ("Start", [_hn.cycle("S", ["free", "tangent", "perpendicular"], "tangent")]),
         ("Polyline", [_hn.toggle("P", "", False)]),
         ("", [_hn.note("Tangent chains are picked together; click a picked edge again to un-pick it.")])]
_wW, _hW, _opsW = _hn.layout(_rows, _meas, 2000)
_wN, _hN, _opsN = _hn.layout(_rows, _meas, 380)
_lines = lambda ops: len({round(o[2]) for o in ops if o[0] == "text"})
_right = lambda ops: max((o[1] + o[3]) if o[0] == "cap" else (o[1] + _meas(o[3])[0]) for o in ops)
check("hints: one line per group on a wide view; on a small one the rows wrap and stay inside the width",
      _lines(_opsW) == 5 and _lines(_opsN) > 5 and _right(_opsN) <= 380 - 8 + 1e-9 and _wN <= 380 and _hN > _hW,
      f"{_lines(_opsW)} / {_lines(_opsN)} lines, right edge {_right(_opsN):.0f}, {_wN:.0f} x {_hN:.0f}")
_sty = {o[3]: o[4] for o in _opsW if o[0] == "text"}
_cap = {o[5]: o[6] for o in _opsW if o[0] == "cap"}
check("hints: a cycling key shows every option with the current one lit, a toggle shows off dimmed, an active "
      "lock is lit", _sty.get("tangent") == "on" and _sty.get("free") == "off" and _sty.get("perpendicular") == "off"
      and _sty.get("off") == "off" and _sty.get("horizontal / vertical") == "on" and _cap.get("H") == "on"
      and _cap.get("X") == "cap" and _sty.get("Place") == "label", f"{_sty} {_cap}")


class _HArea:
    def __init__(self):
        self.text, self.regions = "unset", []
    def header_text_set(self, t):
        self.text = t
    def tag_redraw(self):
        pass
    def as_pointer(self):
        return 4242


_hc = _types.SimpleNamespace(area=_HArea(), workspace=None)
_head = _hn.show(_hc, "Line", ["place the end", "", "Length 2 ft"], _rows, [("MOUSE_LMB", "Place")])
_spec_ok = _hn.HUD["spec"] is not None and _hn.HUD["area"] == 4242
_hn.clear(_hc)
check("hints: the header holds only the live state (no key list); clear() drops header and panel",
      _head == "Line  ·  place the end  ·  Length 2 ft" and _spec_ok and _hc.area.text is None
      and _hn.HUD["spec"] is None, _head)
_was = _hn.collapsed()
_hn.handle(_types.SimpleNamespace(type='F1', value='PRESS', unicode=""))
_c1_ = _hn.collapsed()
_hn.handle(_types.SimpleNamespace(type='SLASH', value='PRESS', unicode="?"))
check("hints: F1 or ? collapses the panel and brings it back", _c1_ != _was and _hn.collapsed() == _was)
_reg = lambda t, x, y, w, h: _types.SimpleNamespace(type=t, x=x, y=y, width=w, height=h)
_win = _reg('WINDOW', 0, 21, 1200, 800)
_side = [_reg('TOOLS', 0, 21, 44, 800), _reg('UI', 900, 21, 300, 800)]
_ins_hidden = _hn._insets(_types.SimpleNamespace(regions=[_win] + _side + [_reg('HUD', 0, 0, 280, 287)]), _win)
_ins = _hn._insets(_types.SimpleNamespace(regions=[_win] + _side + [_reg('HUD', 54, 31, 260, 120)]), _win)
check("hints: the free width excludes the toolbar and the sidebar; a hidden redo panel (parked below the view, "
      "still sized) is ignored, a shown one is found", _ins_hidden == (44, 300, None) and _ins == (44, 300, (54, 314, 130)),
      f"{_ins_hidden} {_ins}")
_p0 = _hn.place(400, 90, 1200, _ins_hidden)
_p1 = _hn.place(400, 90, 1200, _ins)
_p2 = _hn.place(700, 90, 1200, _ins)
check("hints: the panel sits centred at the bottom of the free width; beside a shown redo panel when there is room, "
      "else above it", _p0 == (272, 10) and _p1 == (407, 10) and _p2 == (122, 140), f"{_p0} {_p1} {_p2}")
# every converted tool fills the panel: the sketch tools through their modal code
f = _ln_tool(start_mode="TANGENT")
_at(f, (1.6, 1.7))
_sp = _hn.HUD["spec"]
_grp = [r[0] for r in _sp["rows"]] if _sp else []
_start = [i for r in _sp["rows"] if r[0] == "Start" for i in r[1]] if _sp else []
check("hints: Draw Line shows its groups, the start mode lit, a short header",
      _grp == ["Type", "Direction", "Snap", "Start", "Reference", "Polyline"] and _start and _start[0]["current"] == "tangent"
      and fc.area.text.startswith("Line  ·  click the start") and "Backspace" not in fc.area.text, f"{_grp} {fc.area.text}")
f.modal(fc, _ev('ESC'))
check("hints: leaving the tool clears the header and the panel", fc.area.text is None and _hn.HUD["spec"] is None)
_cvh = _fake(_dt.NPROTO_OT_draw_curve, mode="CV", degree="3", closed=False, start_cont="G1", end_cont="G2",
             edge_mode="EDGE", keep_aligned=True)
_cvh._edit, _cvh._pts, _cvh._snapinfo, _cvh._frames, _cvh._start_refs = None, [], [], {}, None
_cvh.sketch_start(fc)
_cvh._status(fc)
_ends = [i for r in _hn.HUD["spec"]["rows"] if r[0] == "Ends" for i in r[1]]
check("hints: CV Curve shows start / end continuity as choices with the current ones lit, and Keep Aligned (A)",
      [e["current"] for e in _ends] == ["G1", "G2", "on"] and _ends[2]["keys"] == ["A"]
      and fc.area.text.startswith("CV Curve"), str(_ends))
_cvh.sketch_end(fc)
from blender_nurbs import ops_trim as _otm
_efr = _otm.NPROTO_OT_edge_fillet.pick_rows(_types.SimpleNamespace(
    _picks=[], _ties=[], pick_min=1, extra_keys=_otm.NPROTO_OT_edge_fillet.extra_keys,
    extra_help=_otm.NPROTO_OT_edge_fillet.extra_help))
check("hints: a picking tool lists its own keys and its advice (Edge Fillet: Alt+click, A)",
      [r[0] for r in _efr] == ["Pick", "Also", ""] and [i["keys"] for i in _efr[1][1]] == [["Alt+LMB"], ["A"]]
      and "un-pick" in _efr[2][1][0]["note"], str(_efr))

# number-pad digits (and the top-row ones under Emulate Numpad, which arrive as number-pad keys): they change the
# view until the first click, then they type the value
f = _ln_tool()
_pass0 = _dt.SketchMixin.passthrough(f, fc, _ev('NUMPAD_3'))
_click(f, (-1.5, 1.6))
_pass1 = _dt.SketchMixin.passthrough(f, fc, _ev('NUMPAD_3'))
for _t in ('NUMPAD_3', 'NUMPAD_PERIOD', 'NUMPAD_5'):
    f.modal(fc, _ev(_t, _wpx(0.5, 1.6)))
_typed = f._buf
f.modal(fc, _ev('RET', _wpx(0.5, 1.6)))
_L0 = f._lines[0] if f._lines else (np.zeros(3), np.zeros(3))
check("Line: number-pad digits change the view before the first click, then type the length (3.5 placed)",
      _pass0 and not _pass1 and _typed == "3.5" and abs(np.linalg.norm(_L0[1] - _L0[0]) - _dt.parse_length(ctx, "3.5")) < 1e-9,
      f"{_pass0} {_pass1} '{_typed}' {_L0}")
f.modal(fc, _ev('ESC'))

# --- G2 loft to the untrimmed side of a trimmed surface (a spoon's bowl to its handle): that side is exact,
#     whole or cut short by the trim, so the loft is exact at any Samples; two exact layouts are matched exactly
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
from blender_nurbs import contlabels as _clb
_tb = [nurbs(f"TB{k}", [(x, -w, 0.0), (x, 0, 0.5 + 0.3 * (k == 1)), (x, w, 0.0)], order=3)
       for k, (x, w) in enumerate(((-2.0, 1.0), (0.0, 1.5), (2.0, 1.0)))]
bpy.ops.nproto.loft(inputs=ins(*[(c_, "S0") for c_ in _tb])); _tbw = ctx.active_object; _tbw.name = "TB_Bowl"
_tcut = poly("TB_cut", [(1.2, -3, 5), (1.2, 3, 5)])
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.nproto.trim(action="KEEP", target="TB_Bowl", cutters=ins((_tcut, "S0")), ref_point=(0, 0, 0.6),
                    normal=(0, 0, 1), view_dir=(0, 0, -1), projection="Z")
_e_end = _trim_edge(_tbw, lambda m: abs(m.x + 2.0) < 1e-4)                 # the first section: a whole side
_e_long = _trim_edge(_tbw, lambda m: m.y < -0.9 and m.x < 1.0)             # a long side, cut short at x = 1.2
_e_cut = _trim_edge(_tbw, lambda m: abs(m.x - 1.2) < 1e-3)                 # the trim curve itself
_iso_end, _iso_long = features.brep_edge_iso(_tbw, _e_end), features.brep_edge_iso(_tbw, _e_long)
check("a trimmed surface's untouched side is found as that side of its surface (whole, or the stretch the trim "
      "left), the trim curve is not", not store.is_natural(_tbw) and _iso_end is not None and _iso_end[2:] == (0.0, 1.0)
      and _iso_long is not None and 0.0 <= _iso_long[2] < _iso_long[3] < 1.0 and (_iso_long[3] - _iso_long[2]) > 0.7
      and features.brep_edge_iso(_tbw, _e_cut) is None, f"{_iso_end and _iso_end[1:]} {_iso_long and _iso_long[1:]}")


def _plate(name, a, b, d, direction):
    pl_ = poly(name, [a, b])
    bpy.ops.nproto.extrude(inputs=ins((pl_, "S0")), distance=d, direction=direction)
    return ctx.active_object


def _near_side(o, test):
    s_ = store.get_surfaces(o)[0]; M_ = geom.world_matrix(o)
    for e_ in ("U0", "U1", "V0", "V1"):
        if test(M_ @ Vector(geom.boundary_points(s_, e_, 3)[1])):
            return e_


_pe = _plate("TB_pe", (-5, -0.8, 0.3), (-5, 0.8, 0.3), 2.0, "X")             # up to x = -3, facing the first section
_pl = _plate("TB_pl", (-1.5, -4.5, 0.2), (0.5, -4.5, 0.2), 1.5, "Y")         # up to y = -3, facing the long side
_res = {}
for _nm, _p_, _side, _e in (("whole side", _pe, _near_side(_pe, lambda m: abs(m.x + 3) < 1e-6), _e_end),
                            ("side cut short", _pl, _near_side(_pl, lambda m: abs(m.y + 3) < 1e-6), _e_long)):
    _lay = set()
    for _smp in (5, 9, 24):
        r = bpy.ops.nproto.loft(inputs=json.dumps([[_p_.name, _side], [_tbw.name, _e]]), cont_a="G2", cont_b="G2",
                                samples=_smp, edges_a="FREE", edges_b="FREE")     # the square rows: exact
        _lo = ctx.active_object
        _d = ext.surface_cvs(store.get_surfaces(_lo)[0])
        _lay.add((_d["degree_v"], _d["count_v"], tuple(round(k_, 9) for k_ in _d["knots_v"])))
        _m = [_clb.measure(_lo, _p_, _side, 2), _clb.measure(_lo, _tbw, _e, 2)]
        _res.setdefault(_nm, []).append((r, max(m_["angle"] for m_ in _m), max(m_["curv"] for m_ in _m),
                                         max(m_["gap"] for m_ in _m)))
    _res[_nm + " layouts"] = _lay
_Gl = np.array(ext.surface_cvs(store.get_surfaces(_lo)[0])["points"]).reshape(_d["count_u"], _d["count_v"], 3)
_gap = lambda i: float(np.mean(np.linalg.norm(_Gl[i + 1] - _Gl[i], axis=1)))
_ratio = _gap(0) / _gap(_d["count_u"] - 2)
check("G2 loft to a trimmed surface's side: its tangent rows as long as the plate's (both tangent 1) -- they were "
      "11x too short, the loft hugged the bowl and turned sharply away", 0.5 < _ratio < 2.0,
      f"first gaps {_gap(0):.4f} / {_gap(_d['count_u'] - 2):.4f}")
check("G2 loft to a trimmed bowl's untouched side (whole, and cut short by the trim): exact at any Samples -- one "
      "layout (the bowl's, the plate's line raised to it), angle and curvature at rounding",
      all(len(_res[k_ + " layouts"]) == 1 and next(iter(_res[k_ + " layouts"]))[0] == 2
          and all(r_ == {'FINISHED'} and a_ < 1e-4 and c_ < 1e-5 and g_ < 1e-9 for r_, a_, c_, g_ in _res[k_])
          for k_ in ("whole side", "side cut short")), str(_res))
# exactness of the tools: a side cut to a stretch, two sides put on one layout -- the same curves
_rng = np.random.default_rng(7)
_side3 = {"deg": 3, "knots": [0, 0, 0, 0, 0.3, 0.6, 1, 1, 1, 1], "P": _rng.normal(size=(6, 3)),
          "D": _rng.normal(size=(6, 3)), "DD": _rng.normal(size=(6, 3))}
_side2 = {"deg": 2, "knots": [0, 0, 0, 0.5, 1, 1, 1], "P": _rng.normal(size=(4, 3)), "D": _rng.normal(size=(4, 3)),
          "DD": None}
def _ev_side(sd_, key, t):
    c_ = ext.curve_from_cvs(int(sd_["deg"]), [tuple(map(float, p_)) for p_ in sd_[key]], [float(k_) for k_ in sd_["knots"]], None)
    return np.array(c_.evaluate(float(t)), float)
_sub = geom.sub_side(_side3, 0.2, 0.75)
_err_sub = max(np.linalg.norm(_ev_side(_sub, k_, t_) - _ev_side(_side3, k_, 0.2 + 0.55 * t_))
               for k_ in ("P", "D", "DD") for t_ in np.linspace(0, 1, 11))
_c2, _c3 = geom.compatible_sides(_side2, _side3)
_err_cmp = max(max(np.linalg.norm(_ev_side(_c2, k_, t_) - _ev_side(_side2, k_, t_)) for k_ in ("P", "D") for t_ in np.linspace(0, 1, 11)),
               max(np.linalg.norm(_ev_side(_c3, k_, t_) - _ev_side(_side3, k_, t_)) for k_ in ("P", "D", "DD") for t_ in np.linspace(0, 1, 11)))
check("exact sides: cut to a stretch, and two layouts (degree 2 and 3) put on one, the curves don't move",
      _err_sub < 1e-12 and _err_cmp < 1e-12 and _c2["deg"] == _c3["deg"] == 3 and _c2["knots"] == _c3["knots"]
      and _c2["DD"] is None, f"{_err_sub:.1e} {_err_cmp:.1e}")
# second derivatives at a surface's border: z = -(1 - u)^3 has z_uu = 0 at u = 1, z_uuu = 6 (the old one-sided
# difference gave the value half a step inside: -3e-5)
from blender_nurbs import square as _sqr
_cub = ext.surface_from_cvs(3, 1, [(x_, y_, z_) for x_, z_ in ((0, -1), (1 / 3, 0), (2 / 3, 0), (1, 0)) for y_ in (0, 1)],
                            [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1])
_d2 = _sqr._second(_cub, 1.0, 0.5)[2]
_d2i = _sqr._second(_cub, 0.5, 0.5)[2]
check("second derivatives at a surface's border are second-order accurate (z_uu = 0 at u = 1, -3 at u = 0.5)",
      abs(_d2[2]) < 1e-7 and abs(_d2i[2] + 3.0) < 1e-6, f"{_d2} {_d2i}")

# --- Degree, not Order: the Curve box (degree, CVs, spans; Degree buttons keep the CVs) and knot ticks on the comb
from blender_nurbs import curvetools as _ctl, diagnostics as _dgn
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
_q2 = nurbs("DG_Q2", [(0, 0, 0), (1, 1.5, 0), (3, 1.2, 0), (4.5, 0, 0)], order=3)   # degree 2, two spans
_q3 = nurbs("DG_Q3", [(0, 0, 0), (1, 1.5, 0), (2, 2.0, 0), (3, 1.5, 0), (4, 0, 0)], order=4)
_pl = poly("DG_PL", [(0, 0, 0), (2, 0, 0), (3, 1, 0)])
_i2, _i3 = _ctl.spline_info(_q2.data.splines[0]), _ctl.spline_info(_q3.data.splines[0])
check("Curve box: an Order U 3 spline reads as degree 2 with 2 spans and a note on its comb; degree 3 has none",
      (_i2["degree"], _i2["cvs"], _i2["spans"], _i2["kind"]) == (2, 4, 2, "open") and _ctl.degree_note(_i2)
      and (_i3["degree"], _i3["spans"]) == (3, 2) and _ctl.degree_note(_i3) is None, f"{_i2} {_i3}")
_mk = lambda o: _dgn.comb_knots(o, 64)[0]
_k2, _k3, _kp = _mk(_q2), _mk(_q3), _mk(_pl)
_P2, _K2 = _dgn.comb_data(_q2, 64)[0]
_kap2 = np.linalg.norm(_K2, axis=1)
_step = float(np.max(np.abs(np.diff(_kap2))))
check("comb: a tick at every knot -- orange where a degree-2 curve's curvature jumps, grey on degree 3, red at a "
      "polyline's bend; the jump is drawn as a step", [k_[2] for k_ in _k2] == ["jump"]
      and [k_[2] for k_ in _k3] == ["smooth"] and [k_[2] for k_ in _kp] == ["corner"]
      and _step > 0.05 * float(_kap2.max()), f"{[k_[2] for k_ in _k2]} {[k_[2] for k_ in _k3]} {[k_[2] for k_ in _kp]} step {_step:.3f}")
ctx.view_layer.objects.active = _q2
r = bpy.ops.nproto.curve_degree(degree=3, spline=0)
_ord = _q2.data.splines[0].order_u
r2 = bpy.ops.nproto.curve_degree(degree=4, spline=0)
check("Degree buttons: degree 3 sets Order U 4 on the same 4 CVs; degree 4 needs 5 CVs and is refused",
      r == {'FINISHED'} and _ord == 4 and len(_q2.data.splines[0].points) == 4 and r2 == {'CANCELLED'}
      and _q2.data.splines[0].order_u == 4)
_q2.data.splines[0].order_u = 3


class _RecLay:
    def __init__(self, rec): object.__setattr__(self, "_rec", rec)
    def __setattr__(self, k, v): pass
    def __getattr__(self, name):
        rec = self._rec
        def f(*a, **k):
            if name == "operator":
                e = {"op": a[0], "text": k.get("text"), "depress": k.get("depress"), "props": {}}
                rec.append(e)
                return _FakeProps(e["props"])
            if name == "label":
                rec.append({"label": k.get("text", a[0] if a else "")})
            return _RecLay(rec)
        return f


_rec = []
_ctl.draw_curve_info(_RecLay(_rec), ctx, _q2)
_labels = [e["label"] for e in _rec if "label" in e]
_degs = [e for e in _rec if e.get("op") == "nproto.curve_degree"]
_reb = [e for e in _rec if e.get("op") == "nproto.rebuild_curve"]
check("Curve box draws: degree / CVs / spans, Degree 1-5 with 2 pressed, Blender's Order U beside it, the note and "
      "Rebuild as degree 3", any("degree 2 · 4 CVs · 2 span(s)" in l_ for l_ in _labels)
      and [e["text"] for e in _degs] == ["1", "2", "3", "4", "5"] and [e["depress"] for e in _degs].index(True) == 1
      and any("Order U 3" in l_ for l_ in _labels) and any(l_.startswith("Degree 2: the curvature jumps or kinks") for l_ in _labels)
      and len(_reb) == 1 and _reb[0]["props"] == {"rc_degree": "3", "rc_cvs": 0}, str(_rec))
ctx.view_layer.objects.active = _q2
for o in ctx.selected_objects:
    o.select_set(False)
_q2.select_set(True)
r = bpy.ops.nproto.rebuild_curve(rc_degree="3", rc_cvs=0)
_kr = _dgn.comb_knots(_q2, 64)[0]
check("Rebuild as degree 3 (the box's button) leaves no curvature jump on the comb",
      r == {'FINISHED'} and _q2.data.splines[0].order_u == 4 and all(k_[2] == "smooth" for k_ in _kr),
      f"{r} order {_q2.data.splines[0].order_u} {[k_[2] for k_ in _kr]}")

# --- While drawing: the comb on the CV Curve preview (K), the sidebar usable (mouse moves pass through),
#     Emulate Numpad's top-row digits set the degree; Body / Face / Edge at the right of the tool header;
#     a G-label click while the feature's redo panel is up changes the value there
_v = ctx.window_manager.nproto_view
_comb0 = _v.show_comb
_v.show_comb = False
_cvk = _fake(_dt.NPROTO_OT_draw_curve, mode="CV", degree="2", closed=False, start_cont="G0", end_cont="G0",
             edge_mode="EDGE", keep_aligned=False)
_cvk._edit, _cvk._pts, _cvk._snapinfo, _cvk._frames, _cvk._start_refs = None, [], [], {}, None
_cvk.sketch_start(fc)
for _q in ((-1.5, -1.5), (-0.5, 1.2), (0.7, 1.0), (1.6, -1.2)):
    _cvk.modal(fc, _ev('MOUSEMOVE', _wpx(*_q), 'NOTHING'))
    _cvk.modal(fc, _ev('LEFTMOUSE', _wpx(*_q)))
_cvk.modal(fc, _ev('MOUSEMOVE', _wpx(1.8, -1.8), 'NOTHING'))
_no_comb = _drw.sketch_overlay.get("comb")
_cvk.modal(fc, _ev('K', _wpx(1.8, -1.8)))
_cg = _drw.sketch_overlay.get("comb")
_show = _cvk.show_comb and not _v.show_comb      # the tool's own Comb, not Analyze's for every curve
_khint = [i for r in _hn.HUD["spec"]["rows"] if r[0] == "Show" for i in r[1]] if _hn.HUD["spec"] else []
check("CV Curve: K turns the curvature comb on the curve being drawn on (the tool's own Comb), with its knot ticks",
      _no_comb is None and _show and _cg is not None and len(_cg[0]) > 20 and len(_cg[1]) > 20
      and len(_cg[2]["jump"]) >= 2 and _khint and _khint[0]["current"] == "on",
      f"{_no_comb is None} {_show} {None if _cg is None else (len(_cg[0]), {k: len(v_) for k, v_ in _cg[2].items()})}")
_out = _types.SimpleNamespace(type='MOUSEMOVE', value='NOTHING', mouse_x=1500.0, mouse_y=500.0, shift=False, ctrl=False,
                              alt=False, oskey=False, unicode="")
_inn = _types.SimpleNamespace(type='MOUSEMOVE', value='NOTHING', mouse_x=500.0, mouse_y=500.0, shift=False, ctrl=False,
                              alt=False, oskey=False, unicode="")
check("sketch tools: a mouse move over the sidebar goes to the UI (its buttons respond), one in the view does not",
      _dt.SketchMixin.passthrough(_cvk, fc, _out) and not _dt.SketchMixin.passthrough(_cvk, fc, _inn))
_emu = ctx.preferences.inputs.use_emulate_numpad
ctx.preferences.inputs.use_emulate_numpad = True
_np_mine = _dt.SketchMixin.passthrough(_cvk, fc, _ev('NUMPAD_4'))
_cvk.modal(fc, _ev('NUMPAD_4', _wpx(1.8, -1.8)))
_deg4 = _cvk.degree
ctx.preferences.inputs.use_emulate_numpad = False
_np_view = _dt.SketchMixin.passthrough(_cvk, fc, _ev('NUMPAD_4'))
ctx.preferences.inputs.use_emulate_numpad = _emu
check("CV Curve: with Emulate Numpad the top-row digits (sent as number-pad keys) set the degree; without it the "
      "number pad still changes the view", not _np_mine and _deg4 == "4" and _np_view, f"{_np_mine} {_deg4} {_np_view}")
_cvk.modal(fc, _ev('ESC'))
_v.show_comb = _comb0
check("leaving CV Curve clears the comb preview", _drw.sketch_overlay.get("comb") is None)
from blender_nurbs import subsel as _ssl
_hdr = bpy.types.VIEW3D_HT_tool_header._dyn_ui_initialize()
_dts = bpy.types.VIEW3D_HT_tool_header.draw_tool_settings
_calls = []
_orig_dts = getattr(_dts, "_nproto_orig", None)
if _orig_dts is not None:
    _dts._nproto_orig = lambda self, c: _calls.append("blender")
    _sv = _ssl.draw_tool_header
    _ssl.draw_tool_header = lambda self, c: _calls.append("nurbs")
    try:
        _dts(None, ctx)
    finally:
        _dts._nproto_orig, _ssl.draw_tool_header = _orig_dts, _sv
check("Body / Face / Edge come right after Blender's tool settings (before the spacer and Options), not at either "
      "end of the tool header", getattr(_dts, "_nproto_wrapped", False) and _calls == ["blender", "nurbs"]
      and _ssl.draw_tool_header not in _hdr, f"{_calls} {[f.__name__ for f in _hdr]}")
# the redo panel's operator: the last one run, when it made the active feature
_lft = bpy.data.objects.new("RD_loft", None)
ctx.collection.objects.link(_lft)
_lft[features.PROP_HIST] = json.dumps({"op": "loft", "refs": [], "params": {"cont_a": "G0"}})
ctx.view_layer.objects.active = _lft


class _FOpProps:
    def __init__(self):
        self.cont_a = "G0"
        self.bl_rna = _types.SimpleNamespace(properties={"cont_a": None})


_fop = _types.SimpleNamespace(bl_idname="NPROTO_OT_loft", name="Loft", properties=_FOpProps())
_fctx = _types.SimpleNamespace(window_manager=_types.SimpleNamespace(operators=[_fop]), view_layer=ctx.view_layer)
_got = _clb.redo_operator(_fctx, _lft)
_fctx2 = _types.SimpleNamespace(window_manager=_types.SimpleNamespace(
    operators=[_types.SimpleNamespace(bl_idname="VIEW3D_OT_select")]), view_layer=ctx.view_layer)
check("label click: the redo panel's operator is found only while it is the last one run and made the active "
      "feature", _got is _fop and _clb.redo_operator(_fctx2, _lft) is None)
bpy.data.objects.remove(_lft)
# _in_view: a hidden redo panel (parked below the view, still sized) does not hide the view's corner
from blender_nurbs.ops import PickMixin as _PM
_reg2 = lambda t, x, y, w, h: _types.SimpleNamespace(type=t, x=x, y=y, width=w, height=h)
_ar = _types.SimpleNamespace(regions=[_reg2('WINDOW', 0, 21, 1200, 800), _reg2('HUD', 0, 0, 280, 287)])
_ar2 = _types.SimpleNamespace(regions=[_reg2('WINDOW', 0, 21, 1200, 800), _reg2('HUD', 10, 31, 280, 287)])
_evc = _types.SimpleNamespace(mouse_x=100.0, mouse_y=100.0)
check("a click in the view's bottom-left corner reaches the tool when the redo panel is hidden, not when it shows",
      _PM._in_view(_types.SimpleNamespace(area=_ar), _evc) and not _PM._in_view(_types.SimpleNamespace(area=_ar2), _evc))

# --- Clip Start: too small for the view -> a warning with a one-click fix (Display panel, header icon)
_spc = lambda n: _types.SimpleNamespace(clip_start=n, type='VIEW_3D')
_rvp = lambda z, p='PERSP': _types.SimpleNamespace(view_distance=z, view_perspective=p)
_c_bad = _ui.clip_check(_spc(1e-6), _rvp(0.37))
check("Clip Start: 1 um at 0.37 m warns (depth steps ~8 mm) and suggests 0.2 mm; 1 mm, the default and an "
      "orthographic view don't", _c_bad is not None and abs(_c_bad["step"] - 6e-8 * 0.37 ** 2 / 1e-6) < 1e-12
      and abs(_c_bad["suggest"] - 2e-4) < 1e-12 and _ui.clip_check(_spc(1e-3), _rvp(0.37)) is None
      and _ui.clip_check(_spc(0.01), _rvp(0.37)) is None and _ui.clip_check(_spc(1e-6), _rvp(0.37, 'ORTHO')) is None,
      str(_c_bad))
_wrec = []
_wctx = _types.SimpleNamespace(space_data=_types.SimpleNamespace(type='VIEW_3D', clip_start=1e-6,
                                                                 region_3d=_rvp(0.37)), scene=ctx.scene)
_shown_w = _ui.draw_clip_warning(_FakeLay(_wrec), _wctx)
_fix = [e for e in _wrec if e.get("op") == "nproto.fix_clip_start"]
check("Clip Start warning: the Display panel shows it with a button that sets the suggested value",
      _shown_w and len(_fix) == 1 and abs(_fix[0]["props"]["value"] - 2e-4) < 1e-12, str(_wrec))

# --- Face + Edge together, picks feeding tools, gizmo clicks pass through, picked faces split off / deleted
from blender_nurbs import subsel as _sx2, ops as _opx
_st = _sx2.settle_modes
check("selection modes: Face and Edge combine (Shift+click), Body stands alone, none left is Body",
      _st({"BODY", "FACE"}, {"BODY"}) == {"FACE"} and _st({"FACE", "BODY"}, {"FACE"}) == {"BODY"}
      and _st({"FACE", "EDGE"}, {"FACE"}) == {"FACE", "EDGE"} and _st(set(), {"EDGE"}) == {"BODY"})
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
_vw.sel_mode = {"BODY"}
bpy.ops.nproto.add_primitive(kind="BOX")
_bx = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); _bx.select_set(True); ctx.view_layer.objects.active = _bx
_vw.sel_mode = {"FACE"}
_sx2.set_picks(_bx, "FACE", [0])
_vw.sel_mode = {"FACE", "EDGE"}
_kept = _sx2.picks(_bx, "FACE") == [0]
_sx2.set_picks(_bx, "EDGE", [3, 5])
_all = _sx2.get_all(_bx)
_sum = _sx2.summary(_bx)
_vw.sel_mode = {"EDGE"}
check("Face + Edge: switching Edge on keeps the face picks; both kinds are kept per body (summary names both); "
      "switching Face off drops only the face picks", _kept and _all == {"FACE": [0], "EDGE": [3, 5]}
      and _sum.startswith("1 face and 2 edges picked") and _sx2.kinds(ctx) == {"EDGE"}
      and _sx2.get_all(_bx) == {"FACE": [], "EDGE": [3, 5]}, f"{_all} '{_sum}'")
# the navigation gizmo's corner (and the continuity labels) take their own clicks
_rgn = lambda t_, x, y, w, h: _types.SimpleNamespace(type=t_, x=x, y=y, width=w, height=h)
_gw = _rgn('WINDOW', 0, 21, 1200, 800)
_gar = _types.SimpleNamespace(regions=[_gw, _rgn('HEADER', 0, 795, 1200, 26), _rgn('UI', 900, 21, 300, 800)],
                              spaces=_types.SimpleNamespace(active=_types.SimpleNamespace(show_gizmo=True,
                                                                                          show_gizmo_navigate=True)))
_gctx = _types.SimpleNamespace(preferences=_types.SimpleNamespace(
    view=_types.SimpleNamespace(gizmo_size_navigate_v3d=80, mini_axis_type='GIZMO', show_navigate_ui=True),
    system=_types.SimpleNamespace(ui_scale=1.0)))
from blender_nurbs import contlabels as _clh
_h0 = _clh.hover.get("key")
_clh.hover["key"] = None
_on_ball, _on_view = _sx2._over_gizmo(_gctx, _gar, _gw, 860, 760), _sx2._over_gizmo(_gctx, _gar, _gw, 500, 400)
_clh.hover["key"] = ("x", 0)
_on_label = _sx2._over_gizmo(_gctx, _gar, _gw, 500, 400)
_clh.hover["key"] = _h0
check("Face / Edge mode: clicks on the navigation gizmo (beside the sidebar, under the header) and on continuity "
      "labels go to them; the rest of the view picks", _on_ball and not _on_view and _on_label)
from blender_nurbs.ops import PickMixin as _PMg
_ivc = _types.SimpleNamespace(area=_gar, preferences=_gctx.preferences)
_evg = lambda x, y: _types.SimpleNamespace(mouse_x=x, mouse_y=y)
_clh.hover["key"] = ("x", 0)                    # a hovered label still takes a sketch tool's click
_iv = (_PMg._in_view(_ivc, _evg(860, 760)), _PMg._in_view(_ivc, _evg(500, 400)))
_clh.hover["key"] = _h0
check("modal tools (sketch tools, picking, trims): the navigation gizmo is not 'in the view', so its clicks and "
      "drags go to Blender; the rest of the view (a label too) stays the tool's", _iv == (False, True), str(_iv))
from blender_nurbs import ops as _opsg
_opsg.GIZMO["held"] = False
_evt = lambda t, v, x, y: _types.SimpleNamespace(type=t, value=v, mouse_x=x, mouse_y=y)
_seq = [_PMg._in_view(_ivc, _evt('LEFTMOUSE', 'PRESS', 860, 760)),       # press on the axis ball
        _PMg._in_view(_ivc, _evt('MOUSEMOVE', 'NOTHING', 500, 400)),     # dragging off it, over the view
        _PMg._in_view(_ivc, _evt('LEFTMOUSE', 'CLICK_DRAG', 860, 760)),  # Blender starts the rotate
        _PMg._in_view(_ivc, _evt('MOUSEMOVE', 'NOTHING', 500, 400))]     # (after it) the tool again
_seq2 = [_PMg._in_view(_ivc, _evt('LEFTMOUSE', 'PRESS', 860, 760)),
         _PMg._in_view(_ivc, _evt('LEFTMOUSE', 'RELEASE', 860, 760)),    # a click: released on the gizmo
         _PMg._in_view(_ivc, _evt('MOUSEMOVE', 'NOTHING', 500, 400))]
check("gizmo drag while a tool runs: after a press on the gizmo the mouse moves go to Blender (so its drag starts), "
      "until the drag starts or the button comes up", _seq == [False, False, False, True]
      and _seq2 == [False, False, True] and not _opsg.GIZMO["held"], f"{_seq} {_seq2}")
# tools take the picked edges -- through their real invoke (background Blender has no window: bpy.ops would run
# execute straight away), on a stand-in like the sketch tools'
def _fake_pick(op_cls, **props):
    body = {k: v for k, v in op_cls.__dict__.items() if callable(v) and not k.startswith("bl_")
            and k not in ("__init__", "pick_candidates")}      # (its super() needs the real class)
    for k in ("accepts_picks", "pick_min", "pick_max", "finish_on_min", "bl_label", "bl_idname"):
        if hasattr(op_cls, k):
            body[k] = getattr(op_cls, k)
    bases = tuple(b for b in op_cls.__mro__[1:] if b not in (bpy.types.Operator, object) and not b.__name__.startswith("bpy"))
    f = type("FakePick" + op_cls.__name__, bases or (_opx.PickMixin,), body)()
    f.msgs, f.inputs = [], ""
    f.report = lambda t_, m: f.msgs.append(m)
    for k, v in props.items():
        setattr(f, k, v)
    return f


_ev0 = _types.SimpleNamespace(type='LEFTMOUSE', value='PRESS', alt=False, shift=False, ctrl=False, oskey=False)
_sx2.set_picks(_bx, "EDGE", [3, 5])
_n0 = len(bpy.data.objects)
_fd = _fake_pick(_opx.NPROTO_OT_dup_edge if hasattr(_opx, "NPROTO_OT_dup_edge") else
                 __import__("blender_nurbs.ops_sweep", fromlist=["x"]).NPROTO_OT_dup_edge)
r = _fd.invoke(ctx, _ev0)
_dup_made = len(bpy.data.objects) - _n0
_dup_in = json.loads(_fd.inputs or "[]")
bpy.ops.object.select_all(action='DESELECT'); _bx.select_set(True); ctx.view_layer.objects.active = _bx
_sx2.set_picks(_bx, "EDGE", [3])
_fp = _fake_pick(_opx.NPROTO_OT_planar, chain=True)
r2 = _fp.invoke(ctx, _ev0)
_pl = ctx.active_object
_pl_in = json.loads(_fp.inputs or "[]")
check("tools take picked edges: Duplicate Edge makes their two curves at once, Planar fills the picked edge's "
      "planar loop (4 edges)", r == {'FINISHED'} and [x[1] for x in _dup_in] == ["E3", "E5"] and _dup_made >= 1
      and r2 == {'FINISHED'} and len(_pl_in) == 4 and _pl is not _bx and store.is_nurbs(_pl)
      and store.get_brep(_pl).face_count() == 1, f"{r} {_dup_in} {_dup_made} {r2} {_pl_in} {_fp.msgs} {_fd.msgs}")
_fake_self = _types.SimpleNamespace(pick_min=2, pick_max=32, expand_pick=lambda c: [c],
                                    _cands=_opx._pick_candidates(ctx, True))
bpy.ops.object.select_all(action='DESELECT'); _bx.select_set(True); ctx.view_layer.objects.active = _bx
_sx2.set_picks(_bx, "EDGE", [3])
_one = _opx.PickMixin.picked_inputs(_fake_self, ctx)
_sx2.set_picks(_bx, "EDGE", [3, 5])
_two = _opx.PickMixin.picked_inputs(_fake_self, ctx)
check("picked edges that don't fit a tool (too few for a Loft) leave it to pick by hand",
      _one is None and _two == [(_bx, "E3"), (_bx, "E5")], f"{_one} {_two}")
# faces: Unjoin splits the picked ones off, X / Delete removes them (all of them: the body goes)
_vw.sel_mode = {"FACE"}
bpy.ops.object.select_all(action='DESELECT'); _bx.select_set(True); ctx.view_layer.objects.active = _bx
_nf = store.get_brep(_bx).face_count()
_sx2.set_picks(_bx, "FACE", [0, 2])
r = bpy.ops.nproto.unjoin()
_after = store.get_brep(_bx).face_count()
_parts = [o for o in ctx.selected_objects if o is not _bx]
check("Unjoin with picked faces: just those come off (one object each), the rest stays one body",
      r == {'FINISHED'} and _nf == 6 and _after == 4 and len(_parts) == 2
      and all(store.get_brep(o).face_count() == 1 for o in _parts), f"{_nf} -> {_after}, {len(_parts)} parts")
bpy.ops.object.select_all(action='DESELECT'); _bx.select_set(True); ctx.view_layer.objects.active = _bx
_sx2.set_picks(_bx, "FACE", [1])
r = bpy.ops.nproto.delete_faces()
_left = store.get_brep(_bx).face_count()
_p0 = _parts[0]
bpy.ops.object.select_all(action='DESELECT'); _p0.select_set(True); ctx.view_layer.objects.active = _p0
_sx2.set_picks(_p0, "FACE", [0])
_pname = _p0.name
r2 = bpy.ops.nproto.delete_faces()
check("Delete Faces (X / Delete in Face mode): picked faces go; a body whose faces are all picked is deleted",
      r == {'FINISHED'} and _left == 3 and r2 == {'FINISHED'} and _pname not in bpy.data.objects, f"{_left}")
_vw.sel_mode = {"BODY"}

# --- Trim with a cornered cutter (a notch: line, arc, line): imprinted one smooth piece at a time, the loop
#     closes exactly (one creased sheet let the intersection lose a sliver at a corner)
import importlib.util as _ilu
_kdir = os.path.join(os.path.dirname(features.__file__), "kernel")
_ksm = features._k()[0]
check("extension policy: the kernel loads inside the add-on's package (no top-level module, no sys.path entry)",
      _ksm.__name__.endswith(".kernel._nproto_kernel") and "_nproto_kernel" not in sys.modules
      and not any(os.path.normpath(p) == os.path.normpath(_kdir) for p in sys.path), _ksm.__name__)
# The worker runs in its own Python and imports the kernel from its folder. Loaded here, in this process, it
# gets the kernel already loaded (a compiled module cannot load twice); both are taken back afterwards.
_path0 = list(sys.path)
sys.modules["_nproto_kernel"] = _ksm
try:
    _wspec = _ilu.spec_from_file_location("nproto_worker_t", os.path.join(_kdir, "worker.py"))
    _wk = _ilu.module_from_spec(_wspec)
    _wspec.loader.exec_module(_wk)
finally:
    sys.path[:] = _path0
    for _m in ("_nproto_kernel", "harden"):
        sys.modules.pop(_m, None)
_d = {"degree": 2, "points": [[0, 0, 0], [0.5, 0, 0], [1, 0, 0], [1.5, 0.5, 0], [1, 1, 0], [0.5, 1, 0], [0, 1, 0]],
      "knots": [0, 0, 0, 1, 1, 2, 2, 3, 3, 3], "weights": None}     # line, a turn, line: corners at 1 and 2
_pcs = _wk.corner_pieces(_d)
_kc0 = _wk.curve(_d)
_err = 0.0
for _pc in _pcs:
    _kp = _wk.curve(_pc)
    _a, _b = _kp.parameter_range()
    for _t in np.linspace(_a, _b, 9):
        _err = max(_err, float(np.linalg.norm(np.array(_kp.evaluate(_t)) - np.array(_kc0.evaluate(_t)))))
check("trim cutter split at its sharp corners, exactly (the same curve, piece by piece)",
      len(_pcs) == 3 and _err < 1e-12 and len(_wk.corner_pieces({**_d, "points": [[0, 0, 0], [0.5, 0, 0], [1, 0, 0],
                                                                                 [1.5, 0, 0], [2, 0, 0], [2.5, 0, 0],
                                                                                 [3, 0, 0]]})) == 1, f"{len(_pcs)} {_err:.1e}")
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
_ns = [nurbs(f"NT{k}", [(x, -w, 0.0), (x, 0, 0.5 + 0.3 * (k == 1)), (x, w, 0.0)], order=3)
       for k, (x, w) in enumerate(((-2.0, 1.0), (0.0, 1.5), (2.0, 1.0)))]
bpy.ops.nproto.loft(inputs=ins(*[(c_, "S0") for c_ in _ns])); _nb = ctx.active_object; _nb.name = "NT_Bowl"
_notch = poly("NT_notch", [(-3, -0.6, 2), (-1.2, -0.6, 2), (-1.2, 0.6, 2), (-3, 0.6, 2)])      # a squared notch
bpy.ops.object.select_all(action='DESELECT')
r = bpy.ops.nproto.trim(action="KEEP", target="NT_Bowl", cutters=ins((_notch, "S0")), ref_point=(1, 0, 0.6),
                        normal=(0, 0, 1), view_dir=(0, 0, -1), projection="Z")
_bn = store.get_brep(_nb)
_ends = []
for e_ in _bn.edges():
    t0_, t1_ = e_.parameter_range()
    _ends.append((np.array(e_.point_at(t0_)), np.array(e_.point_at(t1_))))
_open = max(min(np.linalg.norm(p_ - q_) for j_, (a2_, z2_) in enumerate(_ends) if j_ != i_ for q_ in (a2_, z2_))
            for i_, (a_, z_) in enumerate(_ends) for p_ in (a_, z_))
check("trim with a squared notch (a polyline cutter with corners): one face, its loop closes exactly",
      r == {'FINISHED'} and _bn.face_count() == 1 and len(_ends) >= 6 and _open < 1e-9, f"{len(_ends)} edges, open {_open:.1e}")

# --- A rebuild that numbers B-rep edges differently: features built on them follow their edge (by place)
_mk = features._edge_marks(_nb)
_n = len(_mk)
_sz = float(np.ptp(np.concatenate(_mk), axis=0).max())
_sw35 = list(_mk); _sw35[3], _sw35[5] = _mk[5], _mk[3]
check("edge renumbering: same place keeps its number, a rotation / a swap moves, a nudge does not",
      features.edge_moves(_mk, _mk) == {} and features.edge_moves(_mk, [m + 1e-4 * _sz for m in _mk]) == {}
      and features.edge_moves(_mk, _mk[1:] + _mk[:1]) == {i: (i - 1) % _n for i in range(_n)}
      and features.edge_moves(_mk, _sw35) == {3: 5, 5: 3}
      and features.edge_moves(_mk, [_mk[0]] * _n) == {}, str(features.edge_moves(_mk, _sw35)))
_k_mid = min(range(_n), key=lambda i: abs(_mk[i][1][0] - (-1.2)) + abs(_mk[i][1][1]))     # the notch's middle edge
_other = (_k_mid + 2) % _n
bpy.ops.object.select_all(action='DESELECT')
r = bpy.ops.nproto.dup_edge(inputs=ins((_nb, f"E{_k_mid}"))); _ec = ctx.active_object
_ref0 = features.history(_ec)["inputs"][0][1]
features.rebuild(_nb, ctx, force=True)                       # same numbering: nothing renamed
_ref1 = features.history(_ec)["inputs"][0][1]
_old = list(_mk); _old[_k_mid], _old[_other] = _mk[_other], _mk[_k_mid]      # before, that edge had another number
features._remap_dependents(_nb, _old, turned=False)
_ref2 = features.history(_ec)["inputs"][0][1]
check("edge renumbering: a feature on a trimmed edge keeps its edge (same numbers stay, a renumbered one follows)",
      r == {'FINISHED'} and _ref0 == _ref1 == f"E{_k_mid}" and _ref2 == f"E{_other}", f"{_ref0} {_ref1} {_ref2}")

# --- Sketch snapping to plain meshes (Snap to Meshes): vertex, edge, edge centre, face, face centre;
#     evaluated (modifiers applied), only what faces the viewer, loose wire edges, a dense mesh stays fast
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
_ALL = set(_sn.MESH_ELEMENTS)
_sf0 = _sn.SNAP_STATE["surface"]
_sn.SNAP_STATE["surface"] = False
_e_off = _sn.sketch_mesh_elements()
_sn.SNAP_STATE["surface"] = True
_e_on = _sn.sketch_mesh_elements()
_sn.SNAP_STATE["surface"] = _sf0
check("mesh snap is always on for the sketch tools (vertex, edge, edge / face centre, like curves), Face only with O",
      _e_off == {"VERTEX", "EDGE", "EDGE_MIDPOINT", "FACE_MIDPOINT"} and _e_on == _e_off | {"FACE"}, f"{_e_off} {_e_on}")
bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0)); MC = ctx.active_object; MC.name = "SnapCube"
bpy.ops.mesh.primitive_cube_add(size=0.2, location=(0.5, 0.5, -5)); MB = ctx.active_object; MB.name = "Behind"
bpy.ops.nproto.add_primitive(kind='CYLINDER', radius=0.3, height=0.5, caps=True); _ncy = ctx.active_object
_ncy.location = (6, 0, 0)
top = _snap_view()
_els = [_ALL]
TM = _sn.collect([MC, MB, _ncy], mesh_elements=lambda: _els[0])
check("mesh snap: plain meshes are mesh targets, a NURBS object is not", TM.counts["meshes"] == 2, str(TM.counts))
MC.hide_select = True
_ncy.hide_select = True
TX = _sn.collect([MC, _ncy], mesh_elements=lambda: _ALL)
_hx = TX.pick(top, _px(1, 1) + (3, 2))
MC.hide_select = False
_ncy.hide_select = False
check("snap: an object that can't be selected never snaps (a locked scan or reference), mesh or NURBS",
      _hx is None and TX.counts["meshes"] == 0 and TX.counts.get("end", 0) == 0, f"{_hx} {TX.counts}")
h = TM.pick(top, _px(1, 1) + (3, 2))
check("mesh snap: vertex (the top one, nearest the viewer)", h is not None and h[1] == "vertex"
      and np.allclose(h[0], (1, 1, 1), atol=1e-6) and h[3] is None, f"{h}")
h = TM.pick(top, _px(1, 0) + (-3, 4))
check("mesh snap: edge centre", h is not None and h[1] == "mid" and np.allclose(h[0], (1, 0, 1), atol=1e-6), f"{h}")
h = TM.pick(top, _px(0, 0) + (4, 3))
check("mesh snap: face centre", h is not None and h[1] == "center" and np.allclose(h[0], (0, 0, 1), atol=1e-6), f"{h}")
h = TM.pick(top, _px(1, 0.5) + (5, 0))
check("mesh snap: a point on an edge lies on it", h is not None and h[1] == "edge"
      and abs(h[0][0] - 1) < 1e-6 and abs(h[0][2] - 1) < 1e-6 and abs(h[0][1] - 0.5) < 0.03, f"{h}")
h = TM.pick(top, _px(0.4, -0.3))
check("mesh snap: Face puts the point on the face under the mouse", h is not None and h[1] == "face"
      and np.allclose(h[0], (0.4, -0.3, 1), atol=1e-6), f"{h}")
h2 = TM.pick(top, _px(0.4, -0.3), edges=False)
check("mesh snap: a lock, Ctrl or Shift leaves only the point snaps (no edge, no face)", h2 is None, f"{h2}")
_hf = TM.pick(top, _px(1, 0.5) + (5, 0))
_fr = _hf[2] if _hf is not None else None
check("mesh snap: an edge carries its frame (direction, its face's normal) for a profile",
      _fr is not None and np.shape(_fr) == (2, 3) and abs(abs(_fr[0][1]) - 1) < 1e-9
      and np.allclose(_fr[1], (0, 0, 1), atol=1e-9), f"{_fr}")
import json as _jsm
_prof_n = len(bpy.data.objects)
r = bpy.ops.nproto.draw_profile('EXEC_DEFAULT', shape='CIRCLE', radius=0.2, rotation=0.0,
                                 data=_jsm.dumps({"mesh": True, "P": [1.0, 0.5, 1.0], "T": [0.0, 1.0, 0.0],
                                                  "up": [0.0, 0.0, 1.0]}))
_pf = ctx.active_object
_Mp = np.array(_pf.matrix_world)
_hp = features.history(_pf)
check("Profile on a mesh edge: perpendicular to the edge, standing square to its face, at the click; a one-off "
      "(no history on the mesh)", r == {'FINISHED'} and len(bpy.data.objects) == _prof_n + 1
      and np.allclose(_Mp[:3, 2], (0, 1, 0), atol=1e-6) and np.allclose(_Mp[:3, 1], (0, 0, 1), atol=1e-6)
      and np.allclose(_Mp[:3, 3], (1, 0.5, 1), atol=1e-6) and _hp is not None and _hp["op"] == "prim_circle",
      f"{r} {_Mp[:3, 1:].T.round(6).tolist()} {_hp and _hp['op']} objects {_prof_n} -> {len(bpy.data.objects)}")
bpy.data.objects.remove(_pf)
_els[0] = _ALL - {"FACE"}
h = TM.pick(top, _px(0.4, -0.3))
check("mesh snap: Face off: the point stays free (the drawing plane)", h is None, f"{h}")
h = TM.pick(top, _px(0.6, 0.6) + (1, 1))
check("mesh snap: a vertex hidden behind a face never snaps", h is None or h[1] != "vertex", f"{h}")
_els[0] = set()
h = TM.pick(top, _px(1, 1) + (3, 2))
check("mesh snap: Snap to Meshes off: no mesh snaps at all", h is None, f"{h}")
_els[0] = {"EDGE"}
h = TM.pick(top, _px(1, 1) + (3, 2))
check("mesh snap: only the switched-on elements (Edge only: no vertex)", h is not None and h[1] == "edge", f"{h}")
# read once per session: the next tool run reuses it; a geometry change drops it
_d1 = _sn.MESH_CACHE.get(MC.session_uid)
T2 = _sn.collect([MC], mesh_elements=lambda: _ALL)
T2.pick(top, _px(1, 1) + (3, 2))
_reused = _d1 is not None and _sn.MESH_CACHE.get(MC.session_uid) is _d1
MC.location.x += 0.25                           # a move: same mesh, the next run snaps in the new place
ctx.view_layer.update()
T3 = _sn.collect([MC], mesh_elements=lambda: _ALL)
h = T3.pick(top, _px(1.25, 1) + (3, 2))
_moved = _sn.MESH_CACHE.get(MC.session_uid) is _d1 and h is not None and np.allclose(h[0], (1.25, 1, 1), atol=1e-6)
MC.location.x -= 0.25
MC.data.vertices[0].co.z -= 0.5                 # a geometry change
MC.data.update()
ctx.view_layer.update()
_dropped = MC.session_uid not in _sn.MESH_CACHE
check("mesh snap: a mesh is read once per session (the next tool reuses it, also after a move); a geometry change "
      "drops it", _reused and _moved and _dropped, f"{_reused} {_moved} {_dropped} {h}")
# evaluated geometry: a subdivided cube snaps on its smooth surface, not its cage
bpy.ops.object.select_all(action='DESELECT')
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
bpy.ops.mesh.primitive_cube_add(size=2); MS = ctx.active_object
_md = MS.modifiers.new("Subdivision", 'SUBSURF'); _md.levels = 2
_dg = ctx.evaluated_depsgraph_get()
_me = MS.evaluated_get(_dg).to_mesh()
_ztop = max(v_.co.z for v_ in _me.vertices if abs(v_.co.x) < 1e-6 and abs(v_.co.y) < 1e-6)
MS.evaluated_get(_dg).to_mesh_clear()
TS = _sn.collect([MS], mesh_elements=lambda: _ALL)
h = TS.pick(top, _px(0, 0) + (2, 2))
check("mesh snap: a subdivided mesh snaps to what the view shows (modifiers applied)",
      h is not None and h[1] == "vertex" and np.linalg.norm(h[0][:2]) < 1e-6 and abs(h[0][2] - _ztop) < 1e-6
      and _ztop < 0.99, f"{h} top {_ztop:.4f}")
h = TS.pick(top, _px(0.35, -0.2) + (1, 0))
_hit = MS.ray_cast((0.35 + 1 / 250, -0.2, 5), (0, 0, -1))
check("mesh snap: the face point lies on the smooth surface", h is not None and (h[1] != "face" or
      abs(h[0][2] - _hit[1][2]) < 1e-5), f"{h} ray {_hit[1][:] if _hit[0] else None}")
# loose edges (a wire outline: no face for a ray to hit)
_mw = bpy.data.meshes.new("Wire"); _mw.from_pydata([(3, 0, 0), (4, 0, 0), (4, 1, 0)], [(0, 1), (1, 2)], [])
MW = bpy.data.objects.new("Wire", _mw); ctx.collection.objects.link(MW)
TW = _sn.collect([MW], mesh_elements=lambda: _ALL)
hm = TW.pick(top, _px(3.5, 0) + (2, 1))
he = TW.pick(top, _px(3.25, 0) + (0, 5))
hv = TW.pick(top, _px(4, 1) + (-2, -2))
check("mesh snap: loose wire edges snap (centre, a point on the edge, vertex)",
      hm is not None and hm[1] == "mid" and np.allclose(hm[0], (3.5, 0, 0), atol=1e-6)
      and he is not None and he[1] == "edge" and abs(he[0][1]) < 1e-9 and abs(he[0][0] - 3.25) < 0.03
      and hv is not None and hv[1] == "vertex" and np.allclose(hv[0], (4, 1, 0), atol=1e-6), f"{hm} | {he} | {hv}")
# a point cloud (more loose points than LOOSE_MAX) doesn't snap point by point: it would cost every mouse move
_lm0 = _sn.LOOSE_MAX
_sn.LOOSE_MAX = 50
_mp = bpy.data.meshes.new("Cloud"); _mp.from_pydata([(5 + 0.01 * k, 0, 0) for k in range(80)], [], [])
MP = bpy.data.objects.new("Cloud", _mp); ctx.collection.objects.link(MP)
TP = _sn.collect([MP], mesh_elements=lambda: _ALL)
_hp = TP.pick(top, _px(5.2, 0) + (1, 1))
_skip = _sn.MESH_CACHE[MP.session_uid].loose_skipped
_sn.LOOSE_MAX = _lm0
check("mesh snap: a point cloud past LOOSE_MAX snaps only on its faces (no per-point cost)", _skip and _hp is None,
      f"{_skip} {_hp}")
# a dense mesh (160k faces) snaps at mouse-move speed
bpy.ops.mesh.primitive_grid_add(x_subdivisions=400, y_subdivisions=400, size=3, location=(0, 0, 0.5)); MG = ctx.active_object
TG = _sn.collect([MG], mesh_elements=lambda: _ALL)
TG.pick(top, _px(0.1, 0.1))                    # the first move reads the mesh
import time as _tm
_t0 = _tm.perf_counter()
_hs = [TG.pick(top, _px(0.1 + 0.01 * k, 0.1)) for k in range(20)]
_dtm = (_tm.perf_counter() - _t0) / 20
check("mesh snap: a dense mesh (160k faces) snaps at mouse-move speed",
      _dtm < 0.03 and all(h_ is not None and h_[1] in ("vertex", "mid", "center", "edge") for h_ in _hs),
      f"{_dtm * 1000:.1f} ms per move, {len(MG.data.polygons)} faces")

# --- Ctrl grid snapping: the grid the view draws (its unit levels, faded in by spacing), never a finer one
from blender_nurbs import drawtools as _dtg
_gi = _dtg.grid_steps("IMPERIAL", 1.0)
_gm = _dtg.grid_steps("METRIC", 1.0)
_gn = _dtg.grid_steps("NONE", 1.0, 1.0, 10)
_snap_in = lambda ppi: round(_dtg.visible_grid_step(_gi, ppi / 0.0254) / 0.0254, 6)
_snap_cm = lambda ppcm: round(_dtg.visible_grid_step(_gm, ppcm / 0.01) * 100, 6)
check("grid snap: Imperial levels are its units (thou, inch, foot, yard...), no inch fractions; the step is the finest "
      "level drawn at least a quarter strong",
      _snap_in(74) == 1.0 and _snap_in(2000) == 1.0 and _snap_in(10) == 12.0 and _snap_in(25) == 1.0
      and len(_gi) == 8 and abs(_gi[1] - 0.0254) < 1e-12, f"{_snap_in(74)} {_snap_in(2000)} {_snap_in(10)}")
check("grid snap: Metric steps by tens with the zoom (mm, cm, 10 cm, m); no units: Grid Scale x Subdivisions",
      _snap_cm(200) == 0.1 and _snap_cm(40) == 1.0 and _snap_cm(5) == 10.0 and _snap_cm(1) == 10.0 and _snap_cm(0.6) == 100.0
      and _gn[:3] == [1.0, 10.0, 100.0], f"{_snap_cm(200)} {_snap_cm(40)} {_snap_cm(5)} {_snap_cm(1)} {_snap_cm(0.6)}")
check("grid snap: the scene's Unit Scale and the overlay's Grid Scale move the levels",
      abs(_dtg.grid_steps("METRIC", 0.001)[4] - 1000.0) < 1e-9 and abs(_dtg.grid_steps("METRIC", 1.0, 2.0)[4] - 2.0) < 1e-12)
check("grid snap: lines a third strong (mm 12 px apart) are snapped to, not skipped for the next level",
      _snap_cm(120) == 0.1 and _snap_cm(9) == 10.0, f"{_snap_cm(120)} {_snap_cm(9)}")

# Blender 5.1+'s own grid (overlay_grid.hh): strengths as measured in 5.3's drawing (Top view, a 1147 px wide
# region: mm lines 13 px apart ~0.6 of a full line, 7 px ~0.15, 5 px none; perspective at 50 deg from straight
# down: 1 m lines ~0.65 at 30 m eye height, ~0.3 at 50 m, gone at 80 m)
from types import SimpleNamespace as _NS
from mathutils import Matrix as _Mx, Euler as _Eu
def _top_rv3d(ppu, w=1147, h=800):
    W = _Mx.Diagonal((2.0 * ppu / w, 2.0 * ppu / h, -0.01, 1.0))
    return _NS(is_perspective=False, is_orthographic_side_view=True, window_matrix=W, perspective_matrix=W,
               view_distance=1.0), _NS(width=w, height=h)
def _persp_rv3d(height, tilt=50.0):
    R = _Eu((math.radians(tilt), 0, 0)).to_matrix().to_4x4()
    eye = R @ Vector((0, 0, height / math.cos(math.radians(tilt))))
    return _NS(is_perspective=True, is_orthographic_side_view=False,
               view_matrix=(_Mx.Translation(eye) @ R).inverted()), _NS(width=1147, height=800)
_bm = _dtg.blender_grid_steps("METRIC", 1.0, 1.0, 10, True)
_bu = _dtg.blender_grid_steps("METRIC", 1.0, 1.0, 10, False)
_ta = lambda p: _dtg.blender_level_alphas(_bm, *reversed(_top_rv3d(p * 1000.0)))[1]
_pa = lambda hh: _dtg.blender_level_alphas(_bu, *reversed(_persp_rv3d(hh)))[0]
check("grid snap: Blender's grid levels -- every unit in Top / Front / Side, from 1 m up elsewhere; no units: "
      "three Subdivisions finer in Top",
      _bm == _dtg.grid_steps("METRIC", 1.0) and _bu[:4] == [1.0, 10.0, 100.0, 1000.0]
      and abs(_dtg.blender_grid_steps("NONE", 1.0, 1.0, 10, True)[0] - 0.001) < 1e-12
      and _dtg.blender_grid_steps("NONE", 1.0, 1.0, 10, False)[0] == 1.0)
check("grid snap: Blender 5.1+ grid strengths follow 5.3's drawing (Top: mm 13 / 7 / 5 px; floor: 1 m at 30 / 50 / 80 m)",
      0.4 < _ta(13.2) < 0.7 and 0.03 < _ta(7.2) < 0.25 and _ta(5.0) < 0.05
      and 0.5 < _pa(30) < 0.8 and 0.2 < _pa(50) < 0.45 and _pa(80) == 0.0,
      f"{_ta(13.2):.2f} {_ta(7.2):.2f} {_ta(5.0):.2f} | {_pa(30):.2f} {_pa(50):.2f} {_pa(80):.2f}")
_gov = _NS(show_overlays=True, show_floor=True, show_ortho_grid=True, grid_scale=1.0, grid_subdivisions=10)
_gctx = _NS(scene=ctx.scene, space_data=_NS(type='VIEW_3D', overlay=_gov), area=None,
            preferences=ctx.preferences, window_manager=ctx.window_manager)
def _dstep(rv, rg, ppu, n=(0, 0, 1)):
    return _dtg.drawn_grid_step(_gctx, rg, rv, ppu, np.array(n, float))
_t13 = _dstep(*_top_rv3d(13200.0), 13200.0)
_t5 = _dstep(*_top_rv3d(5000.0), 5000.0)
_p30 = _dstep(*_persp_rv3d(30), 6.0)         # 1 m lines 6 px apart (squeezed), 0.65 strong: the old rule took 10 m
_p80 = _dstep(*_persp_rv3d(80), 4.1)         # 1 m not drawn: 10 m
_pup = _dstep(*_persp_rv3d(30), 6.0, (0, 1, 0))   # an upright plane: nothing drawn on it, the add-on's rule
check("grid snap: Ctrl snaps to the finest level drawn plainly where you draw -- Top: mm lines 13 px apart "
      "(they show), not 5 px; perspective floor: Blender's 1 m lines while they show",
      (_t13, _t5, _p30, _p80, _pup) == (0.001, 0.01, 1.0, 10.0, 10.0) if _dtg.NEW_BLENDER_GRID else True,
      f"{_t13} {_t5} {_p30} {_p80} {_pup}")

# --- Draw Line: the lines drawn so far snap (their ends, middles, any point on them); Alt places a free point;
#     O switches snapping onto mesh faces
_TL = _sn.collect([])
_segs = [(np.array([0.0, 0, 0]), np.array([2.0, 0, 0])), (np.array([2.0, 0, 0]), np.array([2.0, 1.0, 0]))]
hm = _TL.pick(top, _px(1, 0) + (3, -2), segments=_segs)
hl = _TL.pick(top, _px(0.4, 0) + (0, 5), segments=_segs)
hv = _TL.pick(top, _px(2, 0.75) + (-4, 0), segments=_segs)
check("drawn lines snap: a middle, and any point on a line (exactly on it)",
      hm is not None and hm[1] == "mid" and np.allclose(hm[0], (1, 0, 0)) and hl is not None and hl[1] == "line"
      and abs(hl[0][1]) < 1e-12 and abs(hl[0][0] - 0.4) < 0.02 and hv is not None and hv[1] == "line"
      and abs(hv[0][0] - 2) < 1e-12, f"{hm} | {hl} | {hv}")
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
f = _ln_tool(polyline=True)
_click(f, (0.0, 0.0, 0))
_click(f, (2.0, 0.0, 0))
_click(f, (2.0, 1.0, 0))
_at(f, (1.0, 0.0, 0), (2, 3))
_lm = _drw.sketch_overlay["snap_label"]
_at(f, (0.5, 0.0, 0), (0, 5))
_ll = _drw.sketch_overlay["snap_label"]
_evA = lambda xy: _types.SimpleNamespace(type='MOUSEMOVE', value='NOTHING', mouse_x=float(xy[0]), mouse_y=float(xy[1]),
                                         shift=False, ctrl=False, alt=True, oskey=False, unicode="")
f.modal(fc, _evA(np.array(_wpx(1.0, 0.0)) + (2, 3)))
_la = _drw.sketch_overlay["snap_label"]
_sf0 = _sn.SNAP_STATE["surface"]
_sn.SNAP_STATE["surface"] = False
_evO = _types.SimpleNamespace(type='O', value='PRESS', mouse_x=600.0, mouse_y=600.0, shift=False, ctrl=False,
                              alt=False, oskey=False, unicode="o")
f.modal(fc, _evO)
_on = _sn.SNAP_STATE["surface"]
_row = dict(_hn.HUD["spec"]["rows"]).get("Snap") if _hn.HUD.get("spec") else None
f.modal(fc, _evO)
_off = _sn.SNAP_STATE["surface"]
_sn.SNAP_STATE["surface"] = _sf0
f.modal(fc, _ev('ESC', (600, 600)))
check("Draw Line: the polyline drawn so far snaps (Mid, On Line); Alt: no snapping; O toggles on-surface snapping "
      "(shown in the key hints)", _lm == "Mid" and _ll == "On Line" and _la is None and _on and not _off
      and _row is not None and [it for it in _row if it["keys"] == ["O"]][0].get("current") == "on",
      f"{_lm} {_ll} {_la} {_on} {_off} {_row}")

# --- Draw Line: tangent arcs (R) -- a slot: line, arc, line, arc, one exact closed curve
_m = _dt.tangent_arc((0, 0, 0), (1, 0, 0), (0, 2, 0))
_C, _r, _x, _y, _th = _dt.arc_circle((0, 0, 0), _m, (0, 2, 0))
_te = _dt.seg_end_tangent((0, 0, 0), (0, 2, 0), _m)
_m3 = _dt.tangent_arc((0, 0, 0), (1, 0, 0), (-1, 1, 0))
_th3 = _dt.arc_circle((0, 0, 0), _m3, (-1, 1, 0))[4]
check("tangent arc: leaves along the last direction, sweeps twice the tangent-chord angle (180 / 270 degrees), "
      "ends in the mirrored direction; none along the tangent line",
      np.allclose(_m, (1, 1, 0)) and np.allclose(_C, (0, 1, 0)) and abs(_r - 1) < 1e-12 and abs(_th - math.pi) < 1e-12
      and np.allclose(_te, (-1, 0, 0)) and abs(_th3 - 1.5 * math.pi) < 1e-12
      and np.allclose(_m3, (math.cos(math.pi / 4), 1 + math.sin(math.pi / 4), 0))
      and _dt.tangent_arc((0, 0, 0), (1, 0, 0), (3, 0, 0)) is None, f"{_m} {_C} {_th} {_te} {_m3} {_th3}")
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
f = _ln_tool(polyline=True)
_click(f, (0.0, 0.0, 0))
_click(f, (2.0, 0.0, 0))
f.modal(fc, _ev('R', _wpx(2.0, 0.5)))
_st_arc = fc.area.text
_click(f, (2.0, 1.0, 0))
f.modal(fc, _ev('R', _wpx(1.0, 1.0)))
_click(f, (0.0, 1.0, 0))
f.modal(fc, _ev('R', _wpx(0.0, 0.5)))
_click(f, (0.0, 0.0, 0))
_arcs = list(f._arc_list())
_n0 = set(bpy.data.objects)
f.modal(fc, _ev('RET', (600, 600)))
_new = [o for o in bpy.data.objects if o not in _n0 and o.type == 'CURVE']
_sl = _new[0] if len(_new) == 1 else None
_spl = _sl.data.splines[0] if _sl is not None else None
_dev = 1.0
if _spl is not None:
    _kc = features.input_curve(_sl, "S0")
    _t0, _t1 = _kc.parameter_range()
    def _slot_d(p):                       # distance to the exact slot: two lines and two half circles
        x, y = p[0], p[1]
        if 0 <= x <= 2:
            return min(abs(y), abs(y - 1)) if not (0 < y < 1) or True else 0
        c = (2.0, 0.5) if x > 2 else (0.0, 0.5)
        return abs(math.hypot(x - c[0], y - c[1]) - 0.5)
    _dev = max(_slot_d(_kc.evaluate(_t0 + (_t1 - _t0) * k / 400)) for k in range(401))
check("Draw Line + R: a slot (line, tangent arc, line, tangent arc back to the start) is one closed exact curve "
      "(order 3, Bezier knots), on the slot's lines and half circles (to Blender's single-precision CVs)",
      [m_ is not None for m_ in _arcs] == [False, True, False, True] and "arc ends" in _st_arc
      and _spl is not None and _spl.order_u == 3 and _spl.use_bezier_u and _spl.use_cyclic_u
      and len(_spl.points) == 12 and _dev < 1e-7, f"{_arcs} | {_st_arc} | {_spl and len(_spl.points)} dev {_dev:.1e}")

# --- Draw Line arcs: typed Radius / Angle (90: a half circle); smart guides align to the sketch
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
_glin = nurbs("GuideRef", [(3.0, 3.0, 0.0), (4.0, 3.0, 0.0)], order=2)
f = _ln_tool(polyline=True)
_click(f, (0.0, 0.0, 0))
_click(f, (2.0, 0.0, 0))
f.modal(fc, _ev('R', _wpx(2.3, 0.6)))
_xy = _at(f, (2.3, 0.6, 0))
for ch_, ty_ in (("0", "ZERO"), (".", "PERIOD"), ("5", "FIVE")):
    _type(f, ch_, ty_, _xy)
f.modal(fc, _ev('TAB', _xy))
for ch_, ty_ in (("9", "NINE"), ("0", "ZERO")):
    _type(f, ch_, ty_, _xy)
_hdr = fc.area.text
_type(f, "", "RET", _xy)
_seg, _mid = f._lines[-1], f._arc_list()[-1]
check("Draw Line arc: typed Radius 0.5 and Angle 90 (from the direction it leaves) make an exact half circle; "
      "the header shows Radius, Angle and the arc's sweep",
      np.allclose(_seg[1], (2, 1, 0), atol=1e-12) and _mid is not None and np.allclose(_mid, (2.5, 0.5, 0), atol=1e-12)
      and "Radius" in _hdr and "arc 180" in _hdr, f"{_seg[1]} {_mid} | {_hdr}")
f.modal(fc, _ev('R', _xy))                      # lines again
_at(f, (1.0, 1.0, 0), (0, 4))
_g1 = (_drw.sketch_overlay["snap_label"], f._target(fc, _ev('MOUSEMOVE', np.array(_wpx(1.0, 1.0)) + (0, 4), 'NOTHING'))[0])
_at(f, (0.0, 1.0, 0), (3, 3))
_g2_label, _g2_guides = _drw.sketch_overlay["snap_label"], len(_drw.sketch_overlay["guides"])
_g2 = f._target(fc, _ev('MOUSEMOVE', np.array(_wpx(0.0, 1.0)) + (3, 3), 'NOTHING'))[0]
_click(f, (0.0, 1.0, 0), (3, 3))
_end = f._lines[-1][1]
_at(f, (4.0, 3.0, 0), (2, 2))                   # hover an old line's end: guides align to it from now on
_at(f, (4.0, 1.0, 0), (3, -4))
_g3 = (_drw.sketch_overlay["snap_label"], f._target(fc, _ev('MOUSEMOVE', np.array(_wpx(4.0, 1.0)) + (3, -4), 'NOTHING'))[0])
_snap_row = [it["keys"][0] for it in dict(_hn.HUD["spec"]["rows"])["Snap"]]
f.modal(fc, _ev('ESC', (600, 600)))
check("smart guides: the line back stays on the arc's tangent line; where it crosses the first point's vertical "
      "it snaps exactly (dashed guides drawn), so the slot's sides end square; a hovered point's guides align too",
      _g1[0] in ("Tangent", "Aligned", "Aligned · Tangent", "Tangent · Aligned") and abs(_g1[1][1] - 1) < 1e-12
      and "Aligned" in (_g2_label or "") and _g2_guides > 0 and np.allclose(_g2, (0, 1, 0), atol=1e-12)
      and np.allclose(_end, (0, 1, 0), atol=1e-12) and "Aligned" in (_g3[0] or "")
      and np.allclose(_g3[1], (4, 1, 0), atol=1e-12), f"{_g1} | {_g2_label} {_g2} {_end} | {_g3}")
check("key hints: the Snap row reads Ctrl grid, Alt no snapping, O on surface", _snap_row == ["Ctrl", "Alt", "O"],
      str(_snap_row))

# --- smart guides follow the sketch: perpendicular / parallel to its lines, not only horizontal / vertical
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
f = _ln_tool(polyline=True)
_click(f, (0.0, 0.0, 0))
_click(f, (2.0, 1.0, 0))                        # a slanted line
f.modal(fc, _ev('R', _wpx(2.4, 0.2)))
_dl = np.array([2.0, 1.0, 0.0]) / math.sqrt(5)
_Qp = np.array([2.0, 1.0, 0.0]) + np.array([1.0, -2.0, 0.0]) / math.sqrt(5)      # on the perpendicular through its end
_off = np.array([2.0, 1.0]) / math.sqrt(5) * 3                                  # 3 px off that guide
_at(f, _Qp, _off)
_gl = _drw.sketch_overlay["snap_label"]
_Pp = f._target(fc, _ev('MOUSEMOVE', np.array(_wpx(*_Qp[:2])) + _off, 'NOTHING'))[0]
_click(f, _Qp, _off)
_th = _dt.arc_circle(f._lines[-1][0], f._arc_list()[-1], f._lines[-1][1])[4] if f._arc_list()[-1] is not None else 0
f.modal(fc, _ev('ESC', (600, 600)))
check("smart guides: perpendicular to a slanted sketch line through its end -- the arc ending there is a half circle",
      _gl is not None and "Perpendicular" in _gl and abs(float((_Pp - np.array([2.0, 1.0, 0.0])) @ _dl)) < 1e-9
      and abs(_th - math.pi) < 1e-9, f"{_gl} {_Pp} sweep {math.degrees(_th):.6f}")

# --- no H / V any more: X / Y / Z lock a world axis and stop where it crosses a smart guide
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
f = _ln_tool(polyline=True)
_click(f, (0.0, 0.0, 0))
_click(f, (2.0, 1.0, 0))
f.modal(fc, _ev('H', _wpx(1.0, 1.0)))
_hlock = f._lock
f.modal(fc, _ev('X', _wpx(1.0, 1.0)))
_xy = _at(f, (0.0, 1.0, 0), (2, 3))
_lx = _drw.sketch_overlay["snap_label"]
_Px = f._target(fc, _ev('MOUSEMOVE', _xy, 'NOTHING'))[0]
_dir_keys = [it["keys"][0] for it in dict(_hn.HUD["spec"]["rows"])["Direction"]]
f.modal(fc, _ev('ESC', (600, 600)))
check("H / V are gone (the guides do horizontal / vertical); an X lock stops where it crosses a guide (above the "
      "first point)", _hlock is None and "H V" not in _dir_keys and "Aligned" in (_lx or "")
      and np.allclose(_Px, (0, 1, 0), atol=1e-12), f"{_hlock} {_dir_keys} {_lx} {_Px}")

# --- new shapes are sized to the view (a quarter of its width, on the visible grid), not a fixed 2 m
from mathutils import Matrix as _Mx
from blender_nurbs import feature_ui as _fui
def _vs_ctx(width_units, system):
    VM = _Mx.Translation((0, 0, -10))
    WM = _Mx.Diagonal((2.0 / width_units, 2.0 / width_units * 1000 / 800, -0.01, 1.0))
    rv = _types.SimpleNamespace(view_matrix=VM, window_matrix=WM, perspective_matrix=WM @ VM, is_perspective=False,
                                view_perspective='ORTHO')
    rg = _types.SimpleNamespace(type='WINDOW', width=1000, height=800, x=0, y=0)
    sp = _types.SimpleNamespace(region_3d=rv, overlay=_types.SimpleNamespace(grid_scale=1.0, grid_subdivisions=10))
    ar = _types.SimpleNamespace(type='VIEW_3D', regions=[rg], spaces=_types.SimpleNamespace(active=sp))
    ctx.scene.unit_settings.system = system
    return _types.SimpleNamespace(area=ar, scene=ctx.scene)
_us0 = (ctx.scene.unit_settings.system, ctx.scene.unit_settings.scale_length)
ctx.scene.cursor.location = (0, 0, 0)
ctx.scene.unit_settings.scale_length = 1.0
_vm = _fui.view_size(_vs_ctx(0.2, 'METRIC'))
_vi = _fui.view_size(_vs_ctx(0.2, 'IMPERIAL'))
_vbig = _fui.view_size(_vs_ctx(40.0, 'METRIC'))
ctx.scene.unit_settings.system, ctx.scene.unit_settings.scale_length = _us0
_none = _fui.view_size(_types.SimpleNamespace(area=None, scene=ctx.scene))
check("shape size from the view: a 20 cm wide view gives 5 cm (metric) / exactly 2 in (imperial); a 40 m view 10 m; "
      "no 3D view: the panel defaults", abs(_vm - 0.05) < 1e-12 and abs(_vi - 0.0508) < 1e-12 and abs(_vbig - 10.0) < 1e-9
      and _none is None, f"{_vm} {_vi} {_vbig} {_none}")
_cg = [sec[0] for sec in [g for g in __import__("blender_nurbs.toolgroups", fromlist=["GROUPS"]).GROUPS if g[0] == "CURVES"][0][2]]
check("Create Curves: Blend Curve right under the drawing tools", _cg[:2] == ["Draw", "Between Curves"], str(_cg))

# --- Circle and Arc drawing tools: modes, snapping, typed values; exact prim_circle / prim_arc features
from blender_nurbs import drawshapes as _ds
def _fake_shape(op_cls, **props):
    base_ = next(b_ for b_ in op_cls.__mro__[1:] if isinstance(b_, type) and issubclass(b_, _ds._ShapeTool))
    cls = type("Fake" + op_cls.__name__, (base_,), {})
    for k_, v_ in op_cls.__dict__.items():
        if ((callable(v_) or isinstance(v_, (staticmethod, property))) and k_ not in ("invoke", "__init__")
                and not k_.startswith("bl_")) or k_ in ("MODES", "TITLE", "STEPS", "PARAMS", "FIELD_NAMES", "FIELD_HINTS", "NUMBERS"):
            setattr(cls, k_, v_)
    cls.passthrough = lambda self, c, e: False
    cls.report = lambda self, t, m: self.msgs.append(m)
    f_ = cls()
    f_.msgs, f_.data = [], ""
    for k_, v_ in props.items():
        setattr(f_, k_, v_)
    f_._reset()
    f_.sketch_start(fc)
    return f_
def _typ(f_, text, xy):
    names = {".": "PERIOD", "0": "ZERO", "1": "ONE", "2": "TWO", "3": "THREE", "4": "FOUR", "5": "FIVE",
             "6": "SIX", "7": "SEVEN", "8": "EIGHT", "9": "NINE"}
    for ch_ in text:
        _type(f_, ch_, names[ch_], xy)
def _newest_curve(before):
    new_ = [o for o in bpy.data.objects if o not in before and o.type == 'CURVE']
    return new_[-1] if new_ else None
def _ends(o):
    kc_ = features.input_curve(o, "S0")
    t0_, t1_ = kc_.parameter_range()
    return (np.array(kc_.evaluate(t0_)), np.array(kc_.evaluate(0.5 * (t0_ + t1_))), np.array(kc_.evaluate(t1_)),
            [np.array(kc_.evaluate(t0_ + (t1_ - t0_) * k / 40)) for k in range(41)])
def _on_circle(pts, C, r):
    return max(abs(np.linalg.norm(q - np.asarray(C)) - r) for q in pts)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
_res = {}
# Circle: centre + typed radius
f = _fake_shape(_ds.NPROTO_OT_draw_circle, mode="CENTER", radius=0.5)
_b = set(bpy.data.objects)
_click(f, (0.0, 0.0, 0))
_xy = _at(f, (0.8, 0.3, 0))
_typ(f, "0.5", _xy)
_type(f, "", "RET", _xy)
_o = _newest_curve(_b)
_res["centre"] = None if _o is None else (_on_circle(_ends(_o)[3], (0, 0, 0), 0.5), features.history(_o)["op"])
# Circle: 2 points, 3 points
f = _fake_shape(_ds.NPROTO_OT_draw_circle, mode="TWO", radius=0.5)
_b = set(bpy.data.objects)
_click(f, (0.0, 0.0, 0))
_click(f, (2.0, 0.0, 0))
_o = _newest_curve(_b)
_res["two"] = None if _o is None else _on_circle(_ends(_o)[3], (1, 0, 0), 1.0)
f = _fake_shape(_ds.NPROTO_OT_draw_circle, mode="THREE", radius=0.5)
_b = set(bpy.data.objects)
for q_ in ((1.0, 0.0, 0), (0.0, 1.0, 0), (-1.0, 0.0, 0)):
    _click(f, q_)
_o = _newest_curve(_b)
_res["three"] = None if _o is None else _on_circle(_ends(_o)[3], (0, 0, 0), 1.0)
check("Circle tool: centre + typed radius, 2 points (diameter), 3 points -- exact circles, prim_circle features",
      _res["centre"] is not None and _res["centre"][0] < 1e-6 and _res["centre"][1] == "prim_circle"
      and _res["two"] is not None and _res["two"] < 1e-6 and _res["three"] is not None and _res["three"] < 1e-6,
      str(_res))
# two straight lines meeting at the origin: tangent circle / fillet arc in the corner the mouse is in
_lx = nurbs("TanX", [(-2.0, 0.0, 0.0), (2.0, 0.0, 0.0)], order=2)
_ly = nurbs("TanY", [(0.0, -2.0, 0.0), (0.0, 2.0, 0.0)], order=2)
f = _fake_shape(_ds.NPROTO_OT_draw_circle, mode="TANGENT", radius=0.5)
_b = set(bpy.data.objects)
_click(f, (1.5, 0.0, 0))
_click(f, (0.0, 1.5, 0))
_xy = _at(f, (0.8, 0.6, 0))
_typ(f, "0.5", _xy)
_type(f, "", "RET", _xy)
_o = _newest_curve(_b)
_res["tan"] = None if _o is None else _on_circle(_ends(_o)[3], (0.5, 0.5, 0), 0.5)
f = _fake_shape(_ds.NPROTO_OT_draw_arc, mode="FILLET", radius=0.5, angle=90.0)
_b = set(bpy.data.objects)
_click(f, (1.5, 0.0, 0))
_click(f, (0.0, 1.5, 0))
_xy = _at(f, (0.8, 0.6, 0))
_typ(f, "0.5", _xy)
_type(f, "", "RET", _xy)
_o = _newest_curve(_b)
_fe = _ends(_o) if _o is not None else None
_res["fillet"] = None if _fe is None else (_on_circle(_fe[3], (0.5, 0.5, 0), 0.5),
                                           sorted([tuple(np.round(_fe[0], 6)), tuple(np.round(_fe[2], 6))]))
check("Circle tool Tangent / Arc tool Fillet: tangent to two lines, in the mouse's corner, typed radius",
      _res["tan"] is not None and _res["tan"] < 1e-6 and _res["fillet"] is not None and _res["fillet"][0] < 1e-6
      and _res["fillet"][1] == [(0.0, 0.5, 0.0), (0.5, 0.0, 0.0)], f"{_res['tan']} {_res['fillet']}")
bpy.data.objects.remove(_lx)
bpy.data.objects.remove(_ly)
# Arc: 3 points (start, end, through), centre (the mouse sweeps either way, past 180), tangent from a line's end
f = _fake_shape(_ds.NPROTO_OT_draw_arc, mode="THREE", radius=0.5, angle=90.0)
_b = set(bpy.data.objects)
for q_ in ((1.0, 0.0, 0), (-1.0, 0.0, 0), (0.6, 0.8, 0)):
    _click(f, q_)
_o = _newest_curve(_b)
_a3 = _ends(_o) if _o is not None else None
f = _fake_shape(_ds.NPROTO_OT_draw_arc, mode="CENTER", radius=0.5, angle=90.0)
_b = set(bpy.data.objects)
_click(f, (0.0, 0.0, 0))
_click(f, (1.0, 0.0, 0))
for q_ in ((0.9, 0.4, 0), (0.6, 0.8, 0), (0.0, 1.0, 0), (-0.7, 0.7, 0), (-1.0, 0.0, 0), (-0.6, -0.8, 0)):
    _at(f, q_)
_click(f, (0.0, -1.0, 0))
_o = _newest_curve(_b)
_ac = (_ends(_o), features.history(_o)["params"]["angle"]) if _o is not None else None
_ln = nurbs("TanStart", [(-2.0, 0.0, 0.0), (0.0, 0.0, 0.0)], order=2)
f = _fake_shape(_ds.NPROTO_OT_draw_arc, mode="TANGENT", radius=0.5, angle=90.0)
_b = set(bpy.data.objects)
_click(f, (0.0, 0.0, 0), (2, 1))                 # the line's end
_xy = _at(f, (0.3, 0.9, 0))
_typ(f, "0.5", _xy)
f.modal(fc, _ev('TAB', _xy))
_typ(f, "90", _xy)
_type(f, "", "RET", _xy)
_o = _newest_curve(_b)
_at_ = _ends(_o) if _o is not None else None
check("Arc tool: 3 points through a point, centre sweeping 270 degrees round, tangent from a line's end "
      "(typed 0.5 and 90: a half circle) -- exact arcs, prim_arc features",
      _a3 is not None and _on_circle(_a3[3], (0, 0, 0), 1.0) < 1e-6 and np.allclose(_a3[1], (0, 1, 0), atol=1e-6)
      and _ac is not None and abs(_ac[1] - 270.0) < 1e-4 and np.allclose(_ac[0][2], (0, -1, 0), atol=1e-6)
      and np.allclose(_ac[0][1], (-math.sqrt(0.5), math.sqrt(0.5), 0), atol=1e-6)
      and _at_ is not None and np.allclose(_at_[0], (0, 0, 0), atol=1e-6) and np.allclose(_at_[2], (0, 1, 0), atol=1e-6)
      and np.allclose(_at_[1], (0.5, 0.5, 0), atol=1e-6),
      f"{_a3 and _a3[1]} | {_ac and (_ac[1], _ac[0][2])} | {_at_ and (_at_[0], _at_[1], _at_[2])}")

# --- a Tangent arc from a curve's end always leaves outward (a slot's end behind the start used to flip it)
_mids = []
for _pts in (((-2.0, 0.0, 0.0), (0.0, 0.0, 0.0)), ((0.0, 0.0, 0.0), (-2.0, 0.0, 0.0))):      # end, then start
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    nurbs("TanOut", list(_pts), order=2)
    f = _fake_shape(_ds.NPROTO_OT_draw_arc, mode="TANGENT", radius=0.5, angle=90.0)
    _b = set(bpy.data.objects)
    _click(f, (0.0, 0.0, 0), (2, 1))
    for q_ in ((0.2, 0.6, 0), (0.0, 1.0, 0), (-0.05, 1.0, 0)):
        _at(f, q_)
    _click(f, (-0.05, 1.0, 0))
    _o = _newest_curve(_b)
    _mids.append(_ends(_o)[1] if _o is not None else None)
check("Arc Tangent: from a curve's end (its end or its start) the arc carries the curve on, outward, even when "
      "its end lies a little behind -- no flip",
      all(m_ is not None and m_[0] > 0.3 for m_ in _mids), str(_mids))

# --- Ellipse, Rectangle (rounded corners), Slot, Polygon drawing tools: exact, editable features
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
def _rbox(p_, c_, hw_, hh_, r_):                  # distance to a rounded box's outline (0: on it)
    qx, qy = abs(p_[0] - c_[0]) - (hw_ - r_), abs(p_[1] - c_[1]) - (hh_ - r_)
    return abs(math.hypot(max(qx, 0.0), max(qy, 0.0)) + min(max(qx, qy), 0.0) - r_)
_res = {}
f = _fake_shape(_ds.NPROTO_OT_draw_ellipse, mode="CENTER", radius=0.5, radius2=0.25)
_b = set(bpy.data.objects)
_click(f, (0.0, 0.0, 0))
_click(f, (1.0, 0.0, 0))
_click(f, (0.3, 0.4, 0))
_o = _newest_curve(_b)
_p = features.history(_o)["params"] if _o is not None else {}
_res["ell"] = None if _o is None else (max(abs((q[0] / 1.0) ** 2 + (q[1] / 0.4) ** 2 - 1) for q in _ends(_o)[3]),
                                       _p.get("elliptic"), _p.get("radius2"))
f = _fake_shape(_ds.NPROTO_OT_draw_ellipse, mode="AXIS", radius=0.5, radius2=0.25)
_b = set(bpy.data.objects)
_click(f, (-1.0, 0.0, 0))
_xy = _at(f, (0.6, 0.0, 0))
_typ(f, "2", _xy)
_type(f, "", "RET", _xy)
_xy = _at(f, (0.0, 0.3, 0))
_typ(f, "0.5", _xy)
_type(f, "", "RET", _xy)
_o = _newest_curve(_b)
_res["ell2"] = None if _o is None else max(abs(q[0] ** 2 + (q[1] / 0.5) ** 2 - 1) for q in _ends(_o)[3])
check("Ellipse tool: centre / major end / minor size; both axis ends + typed sizes -- exact ellipses",
      _res["ell"] is not None and _res["ell"][0] < 1e-5 and _res["ell"][1] and abs(_res["ell"][2] - 0.4) < 1e-6
      and _res["ell2"] is not None and _res["ell2"] < 1e-5, str(_res))
f = _fake_shape(_ds.NPROTO_OT_draw_rectangle, mode="CORNERS", width=1.0, depth=0.5, corner=0.0)
_b = set(bpy.data.objects)
_click(f, (0.0, 0.0, 0))
_xy = _at(f, (2.0, 1.0, 0))
f.modal(fc, _ev('TAB', _xy))
f.modal(fc, _ev('TAB', _xy))
_typ(f, "0.25", _xy)
_type(f, "", "RET", _xy)
_o = _newest_curve(_b)
_rp = features.history(_o)["params"] if _o is not None else {}
_res["rect"] = None if _o is None else (max(_rbox(q, (1.0, 0.5), 1.0, 0.5, 0.25) for q in _ends(_o)[3]), _rp.get("corner"))
f = _fake_shape(_ds.NPROTO_OT_draw_rectangle, mode="THREE", width=1.0, depth=0.5, corner=0.0)
_b = set(bpy.data.objects)
_click(f, (0.0, 0.0, 0))
_click(f, (1.0, 1.0, 0))
_click(f, (-0.5, 0.5, 0))
_o = _newest_curve(_b)
_rp3 = features.history(_o)["params"] if _o is not None else {}
check("Rectangle tool: two corners with a typed corner radius (exact rounded corners); three points (rotated)",
      _res["rect"] is not None and _res["rect"][0] < 1e-5 and abs(_res["rect"][1] - 0.25) < 1e-9
      and abs(_rp3.get("width", 0) - math.sqrt(2)) < 1e-6 and abs(_rp3.get("depth", 0) - math.sqrt(0.5)) < 1e-6,
      f"{_res['rect']} {_rp3}")
f = _fake_shape(_ds.NPROTO_OT_draw_slot, mode="CENTERS", length=1.0, width=0.4)
_b = set(bpy.data.objects)
_click(f, (0.0, 0.0, 0))
_click(f, (2.0, 0.0, 0))
_click(f, (1.0, 0.25, 0))
_o = _newest_curve(_b)
def _seg_d(p_):
    x_ = min(max(p_[0], 0.0), 2.0)
    return math.hypot(p_[0] - x_, p_[1])
_res["slot"] = None if _o is None else (max(abs(_seg_d(q) - 0.25) for q in _ends(_o)[3]),
                                        features.history(_o)["op"], features.history(_o)["params"])
check("Slot tool: both arc centres, then the width -- an exact slot (lines and half circles), a prim_slot feature",
      _res["slot"] is not None and _res["slot"][0] < 1e-5 and _res["slot"][1] == "prim_slot"
      and abs(_res["slot"][2]["length"] - 2) < 1e-9 and abs(_res["slot"][2]["width"] - 0.5) < 1e-6, str(_res["slot"]))
f = _fake_shape(_ds.NPROTO_OT_draw_polygon, mode="SIDE", radius=0.5, sides=6)
_b = set(bpy.data.objects)
_click(f, (0.0, 0.0, 0))
f.modal(fc, _ev('UP_ARROW', _wpx(1.0, 0.0)))
f.modal(fc, _ev('UP_ARROW', _wpx(1.0, 0.0)))
_click(f, (1.0, 0.0, 0))
_o = _newest_curve(_b)
_pp = features.history(_o)["params"] if _o is not None else {}
_kc = features.input_curve(_o, "S0") if _o is not None else None
_dmid = min(np.linalg.norm(np.array(_kc.evaluate(_kc.parameter_range()[0] + (_kc.parameter_range()[1] - _kc.parameter_range()[0]) * k / 4000)) - (1, 0, 0))
            for k in range(4001)) if _kc else 1
check("Polygon tool: Up / Down set the sides (8); 'to a side' puts the clicked point at a side's middle",
      _pp.get("sides") == 8 and abs(_pp.get("radius", 0) - 1 / math.cos(math.pi / 8)) < 1e-6 and _dmid < 1e-3,
      f"{_pp} {_dmid:.2e}")

# --- shape tools: guides along the shape's own axes (a rotated ellipse's minor axis through its centre)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
f = _fake_shape(_ds.NPROTO_OT_draw_ellipse, mode="CENTER", radius=0.5, radius2=0.25)
_click(f, (0.0, 0.0, 0))
_click(f, (1.0, 1.0, 0))                        # the major axis at 45 degrees
_q = np.array([-0.3, 0.3, 0.0])                 # on the minor axis
_off = np.array([1.0, 1.0]) / math.sqrt(2) * 3  # 3 px off it
_at(f, _q, _off)
_gl = _drw.sketch_overlay["snap_label"]
_gp = f._point(fc, _ev('MOUSEMOVE', np.array(_wpx(*_q[:2])) + _off, 'NOTHING'))[0]
f.modal(fc, _ev('ESC', (600, 600)))
check("shape tools: guides follow the shape's own axes -- a rotated ellipse's minor axis through its centre",
      _gl is not None and "Perpendicular" in _gl and abs(float(_gp @ np.array([1.0, 1.0, 0.0]))) < 1e-9, f"{_gl} {_gp}")

# --- Fillet (curves): exact arc tangent to both, trims to the tangent points, joins; a polyline's corner in place
from blender_nurbs import fillet as _fl
_ln = lambda a, b: [np.array([[*a, 1.0], [*b, 1.0]])]
_sol = _fl.solve_fillet(_ln((-2, 0, 0), (0, 0, 0)), _ln((0, 0, 0), (0, 2, 0)), (-1, 0, 0), (0, 1, 0), 0.5)
_circ = _fl.arc_spans((1, 1.2, 0), (0, 2.2, 0), (-1, 1.2, 0)) + _fl.arc_spans((-1, 1.2, 0), (0, 0.2, 0), (1, 1.2, 0))
_sol2 = _fl.solve_fillet(_ln((-3, 0, 0), (3, 0, 0)), _circ, (2.5, 0, 0), (1.0, 1.2, 0), 0.5)
check("Fillet geometry: two lines -- the exact corner circle; a line and a circle -- solved on the curves "
      "(tangent to both: 0.5 from the line, 1.5 from the circle's centre)",
      _sol is not None and np.allclose(_sol["centre"], (-0.5, 0.5, 0), atol=1e-12)
      and np.allclose(_sol["t1"], (-0.5, 0, 0), atol=1e-12) and np.allclose(_sol["t2"], (0, 0.5, 0), atol=1e-12)
      and _sol2 is not None and abs(_sol2["centre"][1] - 0.5) < 1e-9
      and abs(np.linalg.norm(_sol2["centre"] - np.array([0, 1.2, 0])) - 1.5) < 1e-9 and _sol2["centre"][0] > 0,
      f"{_sol and _sol['centre']} {_sol2 and _sol2['centre']}")
def _fil_setup(join=False):
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    a_ = nurbs("FilA", [(-2.0, 0.0, 0.0), (0.0, 0.0, 0.0)], order=2)
    b_ = nurbs("FilB", [(0.0, 0.0, 0.0), (0.0, 2.0, 0.0)], order=2)
    f_ = _fake_shape(_fl.NPROTO_OT_fillet_curves, mode="RADIUS", radius=0.25, join=join, trim=True)
    _click(f_, (-1.0, 0.0, 0), (0, 2))
    _click(f_, (0.0, 1.0, 0), (2, 0))
    xy_ = _at(f_, (-0.3, 0.3, 0))
    _typ(f_, "0.5", xy_)
    _type(f_, "", "RET", xy_)
    return a_, b_
def _sp_ends(o):
    kc_ = features.input_curve(o, "S0")
    t0_, t1_ = kc_.parameter_range()
    return np.array(kc_.evaluate(t0_)), np.array(kc_.evaluate(t1_)), \
        [np.array(kc_.evaluate(t0_ + (t1_ - t0_) * k / 200)) for k in range(201)]
def _lfil_d(p_):          # distance to the L (x-axis from -2, y-axis to 2) rounded with r 0.5 at the corner
    x_, y_ = p_[0], p_[1]
    if x_ <= -0.5 and abs(y_) < 0.3:
        return abs(y_)
    if y_ >= 0.5 and abs(x_) < 0.3:
        return abs(x_)
    return abs(math.hypot(x_ + 0.5, y_ - 0.5) - 0.5)
_a, _b = _fil_setup()
_arc = [o for o in bpy.data.objects if o.name.startswith("Fillet")]
_ea, _eb = _sp_ends(_a), _sp_ends(_b)
_earc = _sp_ends(_arc[0]) if _arc else None
check("Fillet tool: two lines, typed radius -- both trimmed to the tangent points, an exact arc between",
      np.allclose(sorted([tuple(np.round(_ea[0], 9)), tuple(np.round(_ea[1], 9))]), [(-2, 0, 0), (-0.5, 0, 0)])
      and np.allclose(sorted([tuple(np.round(_eb[0], 9)), tuple(np.round(_eb[1], 9))]), [(0, 0.5, 0), (0, 2, 0)])
      and _earc is not None and max(abs(np.linalg.norm(q - np.array([-0.5, 0.5, 0])) - 0.5) for q in _earc[2]) < 1e-7,
      f"{_ea[:2]} {_eb[:2]} {len(_arc)}")
_a, _b = _fil_setup(join=True)
_objs = [o for o in bpy.data.objects if o.type == 'CURVE']
_ej = _sp_ends(_objs[0]) if len(_objs) == 1 else None
check("Fillet tool with Join: one curve -- line, arc, line -- exact all along",
      _ej is not None and max(_lfil_d(q) for q in _ej[2]) < 1e-7
      and sorted([tuple(np.round(_ej[0], 9)), tuple(np.round(_ej[1], 9))]) == [(-2.0, 0.0, 0.0), (0.0, 2.0, 0.0)],
      f"{len(_objs)} {_ej and (_ej[0], _ej[1])}")
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
_pl = nurbs("FilPoly", [(-2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 2.0, 0.0)], order=2)
f = _fake_shape(_fl.NPROTO_OT_fillet_curves, mode="RADIUS", radius=0.25, join=False, trim=True)
_click(f, (-1.0, 0.0, 0), (0, 2))
_click(f, (0.0, 1.0, 0), (2, 0))
_xy = _at(f, (-0.3, 0.3, 0))
_typ(f, "0.5", _xy)
_type(f, "", "RET", _xy)
_ep = _sp_ends(_pl)
check("Fillet tool on one polyline: its corner is rounded in place (one curve, exact)",
      len([o for o in bpy.data.objects if o.type == 'CURVE']) == 1 and max(_lfil_d(q) for q in _ep[2]) < 1e-7
      and sorted([tuple(np.round(_ep[0], 9)), tuple(np.round(_ep[1], 9))]) == [(-2.0, 0.0, 0.0), (0.0, 2.0, 0.0)],
      f"{_ep[0]} {_ep[1]} {max(_lfil_d(q) for q in _ep[2]):.1e}")

# --- Fillet: clicks close to the corner (the tangent points beyond them) still keep the outer parts
_near = {}
for _kind in ("poly", "two", "join"):
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    if _kind == "poly":
        nurbs("FilNear", [(-2.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 2.0, 0.0)], order=2)
    else:
        nurbs("FilNA", [(-2.0, 0.0, 0.0), (0.0, 0.0, 0.0)], order=2)
        nurbs("FilNB", [(0.0, 0.0, 0.0), (0.0, 2.0, 0.0)], order=2)
    f = _fake_shape(_fl.NPROTO_OT_fillet_curves, mode="RADIUS", radius=0.25, join=_kind == "join", trim=True)
    _click(f, (-0.2, 0.0, 0), (0, 2))
    _click(f, (0.0, 0.2, 0), (2, 0))
    _xy = _at(f, (-0.4, 0.4, 0))
    _typ(f, "0.5", _xy)
    _type(f, "", "RET", _xy)
    _cs = [o for o in bpy.data.objects if o.type == 'CURVE']
    _near[_kind] = (len(_cs), max(max(_lfil_d(q) for q in _sp_ends(o)[2]) for o in _cs),
                    sorted(tuple(np.round(e, 6)) for o in _cs for e in _sp_ends(o)[:2]))
check("Fillet: clicked close to the corner (inside the tangent points) -- the outer parts are kept, the corner goes "
      "(one polyline, two lines, two lines joined)",
      _near["poly"][0] == 1 and _near["poly"][1] < 1e-7 and _near["poly"][2] == [(-2.0, 0.0, 0.0), (0.0, 2.0, 0.0)]
      and _near["two"][0] == 3 and _near["two"][1] < 1e-7
      and (-2.0, 0.0, 0.0) in _near["two"][2] and (0.0, 2.0, 0.0) in _near["two"][2]
      and (0.0, 0.0, 0.0) not in _near["two"][2]
      and _near["join"][0] == 1 and _near["join"][1] < 1e-7 and _near["join"][2] == [(-2.0, 0.0, 0.0), (0.0, 2.0, 0.0)],
      str(_near))

# --- Helix Taper: narrower (or wider) on top, like a thread on a cone
_hx = {}
for _tp in (10.0, 0.0, -10.0):
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    ctx.scene.cursor.location = (0, 0, 0)
    bpy.ops.nproto.add_primitive(kind='HELIX', radius=1.0, height=2.0, turns=3.0, taper=_tp, spans_u=8)
    _ho = ctx.active_object
    _kc = features.input_curve(_ho, "S0")
    _t0, _t1 = _kc.parameter_range()
    _pts = [np.array(_kc.evaluate(_t0 + (_t1 - _t0) * k / 300)) for k in range(301)]
    _hx[_tp] = (max(abs(math.hypot(q[0], q[1]) - (1.0 - q[2] * math.tan(math.radians(_tp)))) for q in _pts),
                math.hypot(*_pts[-1][:2]), features.history(_ho)["params"].get("taper"))
check("Helix Taper: the helix follows the cone (10 degrees: narrower on top, -10: wider), 0 stays straight; "
      "in the Feature panel", all(v_[0] < 5e-3 for v_ in _hx.values()) and abs(_hx[10.0][1] - (1 - 2 * math.tan(math.radians(10)))) < 5e-3
      and abs(_hx[0.0][1] - 1) < 5e-3 and _hx[-10.0][1] > 1.3 and _hx[10.0][2] == 10.0, str(_hx))

# --- Helix drawing tool: centre, start, height in 3D; turns / pitch / taper typed; left-handed; around a circle
check("helix height: the axis point nearest the view ray; none when the ray runs down the axis",
      abs(_ds.axis_height((0, 0, 0), (0, 0, 1), (5, 0, 1.5), (-1, 0, 0)) - 1.5) < 1e-12
      and _ds.axis_height((0, 0, 0), (0, 0, 1), (0, 0, 9), (0, 0, -1)) is None)
def _helix_pts(o, k=200):
    kc_ = features.input_curve(o, "S0")
    t0_, t1_ = kc_.parameter_range()
    return [np.array(kc_.evaluate(t0_ + (t1_ - t0_) * j / k)) for j in range(k + 1)]
_hres = {}
for _case in ("turns", "pitch", "left"):
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    f = _fake_shape(_ds.NPROTO_OT_draw_helix, mode="CENTER", radius=0.5, height=1.0, turns=3.0, taper=0.0, left=False)
    _b = set(bpy.data.objects)
    _click(f, (0.0, 0.0, 0))
    _click(f, (1.0, 0.0, 0))
    _xy = _at(f, (1.2, 0.4, 0))
    if _case == "left":
        f.modal(fc, _ev('L', _xy))
    _typ(f, "2", _xy)
    f.modal(fc, _ev('TAB', _xy))
    if _case == "pitch":
        f.modal(fc, _ev('TAB', _xy))
        _typ(f, "0.5", _xy)
    else:
        _typ(f, "4", _xy)
    _type(f, "", "RET", _xy)
    _o = _newest_curve(_b)
    _hp = features.history(_o)["params"] if _o is not None else {}
    _pt = _helix_pts(_o) if _o is not None else []
    _hres[_case] = (_hp.get("radius"), _hp.get("height"), _hp.get("turns"), _hp.get("left"),
                    max((abs(math.hypot(q[0], q[1]) - 1.0) for q in _pt), default=1),
                    _pt[0] if _pt else None, _pt[-1] if _pt else None, _pt[3][1] if _pt else 0)
_rt, _rp, _rl = _hres["turns"], _hres["pitch"], _hres["left"]
check("Helix tool: centre, start (radius 1), typed height 2 and 4 turns -- or a pitch of 0.5 (4 turns); "
      "right-handed rising, L left-handed",
      abs(_rt[0] - 1) < 1e-6 and abs(_rt[1] - 2) < 1e-9 and abs(_rt[2] - 4) < 1e-9 and not _rt[3] and _rt[4] < 5e-3
      and np.allclose(_rt[5], (1, 0, 0), atol=1e-3) and abs(_rt[6][2] - 2) < 1e-3 and _rt[7] > 0
      and abs(_rp[2] - 4) < 1e-9 and _rl[3] and _rl[7] < 0, str(_hres))
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
ctx.scene.cursor.location = (0, 0, 0)
bpy.ops.nproto.add_primitive(kind='CIRCLE', radius=0.5, degree="0")
f = _fake_shape(_ds.NPROTO_OT_draw_helix, mode="AROUND", radius=0.5, height=1.0, turns=3.0, taper=0.0, left=False)
_b = set(bpy.data.objects)
_click(f, (0.5, 0.0, 0), (2, 1))                 # the circle
_click(f, (0.0, 0.7, 0))                        # where it starts: on the circle, a quarter turn round
_xy = _at(f, (0.8, 0.3, 0))
_typ(f, "1", _xy)
_type(f, "", "RET", _xy)
_o = _newest_curve(_b)
_ha = features.history(_o)["params"] if _o is not None else {}
_pa = _helix_pts(_o) if _o is not None else []
check("Helix tool Around: a circle gives the centre, radius and axis; then the start on it; typed height 1",
      abs(_ha.get("radius", 0) - 0.5) < 1e-6 and abs(_ha.get("height", 0) - 1) < 1e-9
      and max((abs(math.hypot(q[0], q[1]) - 0.5) for q in _pa), default=1) < 5e-3
      and _pa and abs(_pa[-1][2] - 1) < 1e-3 and np.allclose(_pa[0], (0, 0.5, 0), atol=1e-3),
      f"{_ha} {_pa[0] if _pa else None} {_pa[-1] if _pa else None}")

# --- Helix around a cone: the height snapped to the cone's other rim gives the taper -- it follows the cone
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
ctx.scene.cursor.location = (0, 0, 0)
bpy.ops.nproto.add_primitive(kind='CONE', radius=1.0, radius2=0.5, height=2.0, caps=False, degree="0")
_cone = ctx.active_object
_rims = []
for e_ in store.get_brep(_cone).edges():
    t0_, t1_ = e_.parameter_range()
    q_ = np.array([e_.point_at(t0_ + (t1_ - t0_) * k / 8) for k in range(9)])
    if np.ptp(q_[:, 2]) < 1e-9:
        _rims.append((float(q_[0, 2]), float(np.linalg.norm(q_[0, :2]))))
_rims.sort(key=lambda zr: -zr[1])               # the wide rim first
f = _fake_shape(_ds.NPROTO_OT_draw_helix, mode="AROUND", radius=0.5, height=1.0, turns=3.0, taper=0.0, left=False)
_b = set(bpy.data.objects)
_click(f, (1.0, 0.0, 0), (2, 1))                 # the wide rim
_click(f, (0.0, 1.3, 0))                        # the start, on it
_xy = _at(f, (0.0, 0.5, 0), (-1, 2))            # the height: onto the narrow rim
_note = fc.area.text
_click(f, (0.0, 0.5, 0), (-1, 2))
_o = _newest_curve(_b)
_hc = features.history(_o)["params"] if _o is not None else {}
_dz = abs(_rims[1][0] - _rims[0][0]) if len(_rims) == 2 else 0
_pc = _helix_pts(_o) if _o is not None else []
_dev = max((abs(math.hypot(q[0], q[1]) - (1.0 - abs(q[2] - _rims[0][0]) * 0.5 / _dz)) for q in _pc), default=1) \
    if _dz else 1
check("Helix around a cone: snapping the height to its other rim takes that rim's level and the cone's taper -- "
      "the helix follows the cone", len(_rims) == 2 and abs(_hc.get("height", 0) - _dz) < 1e-6
      and abs(_hc.get("taper", 0) - math.degrees(math.atan(0.5 / _dz))) < 1e-6 and _dev < 5e-3
      and "from the cone" in _note, f"{_rims} {_hc} dev {_dev:.1e} | {_note}")

# --- Helix around a pointed cone: its tip, or its seam line, gives the taper (no other rim to snap to)
_pc_res = {}
for _where in ("tip", "seam"):
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    ctx.scene.cursor.location = (0, 0, 0)
    bpy.ops.nproto.add_primitive(kind='CONE', radius=1.0, radius2=0.0, height=2.0, caps=False, degree="0")
    _cn = ctx.active_object
    _seam, _zb = None, None
    for e_ in store.get_brep(_cn).edges():
        t0_, t1_ = e_.parameter_range()
        q_ = np.array([e_.point_at(t0_ + (t1_ - t0_) * k / 8) for k in range(9)])
        if np.ptp(q_[:, 2]) < 1e-9 and np.linalg.norm(q_[0, :2]) > 0.5:
            _zb = float(q_[0, 2])                      # the base rim
        elif np.ptp(q_[:, 2]) > 1e-6:
            _seam = (q_[0], q_[-1])                    # the seam line: base to tip
    _tip = max(_seam, key=lambda q: q[2]) if _seam is not None else np.array([0, 0, 2.0])
    _mid = 0.5 * (_seam[0] + _seam[1]) if _seam is not None else np.array([0.5, 0, 1.0])
    f = _fake_shape(_ds.NPROTO_OT_draw_helix, mode="AROUND", radius=0.5, height=1.0, turns=3.0, taper=0.0, left=False)
    _b = set(bpy.data.objects)
    _click(f, (0.0, -1.0, 0), (1, -2))              # the base rim
    _click(f, (0.0, -1.3, 0))                       # the start
    _q = _tip if _where == "tip" else _mid
    _xy = _at(f, (_q[0], _q[1], 0), (2, 1) if _where == "tip" else (0, 0))
    _nt = fc.area.text
    _click(f, (_q[0], _q[1], 0), (2, 1) if _where == "tip" else (0, 0))
    _o = _newest_curve(_b)
    _hh = features.history(_o)["params"] if _o is not None else {}
    _pts_ = _helix_pts(_o) if _o is not None else []
    _dev = max((abs(math.hypot(q[0], q[1]) - max(0.0, 1.0 - (q[2] - _zb) * 0.5)) for q in _pts_), default=1)
    _pc_res[_where] = (_hh.get("height"), _hh.get("taper"), _dev, "from the cone" in _nt)
_tp = math.degrees(math.atan(0.5))
check("Helix around a pointed cone: snapping the height to its tip, or along its seam line, takes the cone's taper "
      "(no other rim needed) -- the helix follows the cone",
      abs(_pc_res["tip"][0] - 2) < 1e-6 and abs(_pc_res["tip"][1] - _tp) < 1e-6 and _pc_res["tip"][2] < 1e-2
      and _pc_res["tip"][3] and abs(_pc_res["seam"][0] - 1) < 1e-6 and abs(_pc_res["seam"][1] - _tp) < 1e-6
      and _pc_res["seam"][2] < 5e-3, str(_pc_res))

# --- Center snap on circles and arcs (curves) and circular surface edges (a cylinder's rim)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
ctx.scene.cursor.location = (2.0, 1.0, 0.0)
bpy.ops.nproto.add_primitive(kind='CIRCLE', radius=0.5, degree="0")
ctx.scene.cursor.location = (-2.0, 1.0, 0.0)
bpy.ops.nproto.add_primitive(kind='ARC', radius=0.5, angle=90.0, degree="0")
ctx.scene.cursor.location = (0.0, -2.0, 0.0)
bpy.ops.nproto.add_primitive(kind='ELLIPSE', radius=0.6, radius2=0.3, degree="0")
ctx.scene.cursor.location = (0.0, 2.0, 0.0)
bpy.ops.nproto.add_primitive(kind='CYLINDER', radius=0.4, height=1.0, caps=False)
ctx.scene.cursor.location = (0, 0, 0)
_Tc = _sn.collect(list(ctx.scene.objects))
_vw4 = _snap_view(s=4.0)
_px4 = lambda x, y: np.array([500 + 500 * x / 4.0, 500 + 500 * y / 4.0])
_hc = _Tc.pick(_vw4, _px4(2.0, 1.0) + (2, 1))
_ha = _Tc.pick(_vw4, _px4(-2.0, 1.0) + (2, 1))
_he = _Tc.pick(_vw4, _px4(0.0, -2.0) + (2, 1))
_hr = _Tc.pick(_vw4, _px4(0.0, 2.0) + (2, 1))
_zr = max(v_.co.z for v_ in [o for o in ctx.scene.objects if o.type == 'MESH'][0].data.vertices) \
    if [o for o in ctx.scene.objects if o.type == 'MESH'] else None
check("Center snap: a circle's and an arc's centre (curves), a cylinder rim's centre (the top one); not an ellipse",
      _hc is not None and _hc[1] == "center" and np.allclose(_hc[0], (2, 1, 0), atol=1e-9)
      and _ha is not None and _ha[1] == "center" and np.allclose(_ha[0], (-2, 1, 0), atol=1e-9)
      and (_he is None or _he[1] != "center")
      and _hr is not None and _hr[1] == "center" and np.allclose(_hr[0][:2], (0, 2), atol=1e-9) and _hr[0][2] > 0.1,
      f"{_hc} | {_ha} | {_he} | {_hr}")


# --- Drawing on a flat face: the first point on a flat face (plain mesh or NURBS) draws on its plane,
#     axes along its edges; a facet of a curved surface (cylinder side, sphere) does not
import bmesh as _bm
def _ff_clear():
    for o_ in list(bpy.data.objects):
        bpy.data.objects.remove(o_)
def _ff(obj, view, xy):
    return _sn.collect([obj]).flat_face(view, np.asarray(xy, float))
def _prism(name="Chamfer"):
    """A box (+-1) with its top front edge (y = -1, z = 1, along X) chamfered at 45 degrees."""
    pent = [(-1, -1), (1, -1), (1, 1), (-0.2, 1), (-1, 0.2)]
    V = [(x, y, z) for x in (-1.0, 1.0) for (y, z) in pent]
    F = [(0, 1, 2, 3, 4), (9, 8, 7, 6, 5)] + [(i, (i + 1) % 5, 5 + (i + 1) % 5, 5 + i) for i in range(5)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(V, [], F)
    me.update()
    o_ = bpy.data.objects.new(name, me)
    ctx.scene.collection.objects.link(o_)
    ctx.view_layer.update()
    return o_
top, front = _snap_view(), _snap_view(True)
_ffr = {}
_ff_clear()
bpy.ops.mesh.primitive_cube_add(size=2)
_c = ctx.active_object
_f = _ff(_c, top, _px(0.3, 0.2))
_ffr["cube"] = _f is not None and np.allclose(_f["P"], (0.3, 0.2, 1), atol=1e-6) and np.allclose(_f["n"], (0, 0, 1), atol=1e-9) \
    and np.allclose(_f["u"], (1, 0, 0), atol=1e-9) and np.allclose(_f["v"], (0, 1, 0), atol=1e-9) and len(_f["seg"]) == 4
_c.rotation_euler = (0, 0, math.radians(30))
ctx.view_layer.update()
_f = _ff(_c, top, _px(0.1, 0.1))
_ffr["rotated"] = _f is not None and np.allclose(_f["u"], (math.cos(math.radians(30)), math.sin(math.radians(30)), 0), atol=1e-6)
_ff_clear()
bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=1, depth=2)
_c = ctx.active_object
_ffr["cyl side"] = _ff(_c, front, _px(0.1, 0.2)) is None
_f = _ff(_c, top, _px(0.2, 0.1))
_ffr["cyl cap"] = _f is not None and np.allclose(_f["n"], (0, 0, 1), atol=1e-9) and np.allclose(_f["u"], (1, 0, 0), atol=1e-9)
_ff_clear()
bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=1, depth=2, end_fill_type='TRIFAN')
_c = ctx.active_object
_f = _ff(_c, top, _px(0.2, 0.1))
_ffr["fan cap"] = _f is not None and len(_f["seg"]) == 32 and abs(_f["P"][2] - 1) < 1e-6
_bmc = _bm.new()
_bmc.from_mesh(_c.data)
_vert_e = [e_ for e_ in _bmc.edges if abs(e_.verts[0].co.z - e_.verts[1].co.z) > 1.0]
_bm.ops.subdivide_edges(_bmc, edges=_vert_e, cuts=3, use_grid_fill=True)
_bmc.to_mesh(_c.data)
_bmc.free()
_c.data.update()
ctx.view_layer.update()
_sn.forget_meshes()
_ffr["cut side"] = _ff(_c, front, _px(0.1, 0.2)) is None and len(_c.data.polygons) > 100
_ff_clear()
bpy.ops.mesh.primitive_cylinder_add(vertices=8, radius=1, depth=2)
_ffr["octagon side"] = _ff(ctx.active_object, front, _px(0.05, 0.2)) is not None
_ff_clear()
bpy.ops.mesh.primitive_uv_sphere_add(radius=1)
_ffr["sphere"] = _ff(ctx.active_object, top, _px(0.1, 0.1)) is None
_ff_clear()
bpy.ops.mesh.primitive_cube_add(size=2)
_c = ctx.active_object
_bv = _c.modifiers.new("Bevel", 'BEVEL')
_bv.width, _bv.segments = 0.1, 4
ctx.view_layer.update()
_f = _ff(_c, top, _px(0.2, 0.1))
_ffr["bevelled"] = _f is not None and np.allclose(_f["n"], (0, 0, 1), atol=1e-9) and np.allclose(_f["u"], (1, 0, 0), atol=1e-9)
_c.modifiers.remove(_bv)
_c.modifiers.new("Tri", 'TRIANGULATE')
ctx.view_layer.update()
_sn.forget_meshes()
_f = _ff(_c, top, _px(0.2, 0.1))
_ffr["triangulated"] = _f is not None and len(_f["seg"]) == 4 and np.allclose(_f["u"], (1, 0, 0), atol=1e-9)
_ff_clear()
bpy.ops.mesh.primitive_grid_add(x_subdivisions=200, y_subdivisions=200, size=2)
_c = ctx.active_object
_t0 = _time.perf_counter()
_f = _ff(_c, top, _px(0.3, -0.4))
_tgrid = _time.perf_counter() - _t0
_ffr["grid"] = _f is not None and np.allclose(np.ptp(_f["seg"].reshape(-1, 3), axis=0), (2, 2, 0), atol=1e-6) \
    and len(_f["seg"]) == 4 * (round(math.sqrt(len(_c.data.polygons))))
_ffr_grid = (None if _f is None else (len(_f["seg"]), len(_c.data.polygons), np.ptp(_f["seg"].reshape(-1, 3), axis=0)))
_ff_clear()
_ch = _prism()
_f = _ff(_ch, top, _px(0.1, -0.6))
_n45 = np.array([0, -1, 1]) / math.sqrt(2)
_ffr["chamfer"] = _f is not None and np.allclose(_f["n"], _n45, atol=1e-9) and np.allclose(_f["u"], (1, 0, 0), atol=1e-9) \
    and abs(float(_f["P"] @ _n45) - 1.2 / math.sqrt(2)) < 1e-6 and np.allclose(_f["P"][:2], (0.1, -0.6), atol=1e-6)
_f = _ff(_ch, top, _px(0.1, 0.4))
_ffr["chamfer top"] = _f is not None and np.allclose(_f["n"], (0, 0, 1), atol=1e-9)
_m0 = _sn.FLAT_MAX_LOOPS
_sn.FLAT_MAX_LOOPS = 10
_sn.forget_meshes()
_ffr["dense"] = _ff(_ch, top, _px(0.1, 0.4)) is None
_sn.FLAT_MAX_LOOPS = _m0
_sn.forget_meshes()
_ff_clear()
bpy.ops.nproto.add_primitive(kind="BOX", width=2, depth=2, height=2, degree="0")
_nb = ctx.active_object
_ztop = max((_nb.matrix_world @ Vector(c_)).z for c_ in _nb.bound_box)
_f = _ff(_nb, top, _px(0.3, 0.2))
_ffr["nurbs box"] = _f is not None and np.allclose(_f["n"], (0, 0, 1), atol=1e-9) and abs(_f["P"][2] - _ztop) < 1e-6 \
    and np.allclose(_f["u"], (1, 0, 0), atol=1e-6)
_ff_clear()
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0, caps=True)
_ffr["nurbs cylinder side"] = _ff(ctx.active_object, front, _px(0.1, 0.0)) is None
check("flat faces: box, rotated box (axes along its edges), cylinder cap (n-gon / fan), octagon side, bevelled and "
      "triangulated box, subdivided grid, chamfer, NURBS box; not a cylinder side (cut or not), sphere, dense mesh",
      all(_ffr.values()), str({k_: v_ for k_, v_ in _ffr.items() if not v_}) + f" grid {_ffr_grid}")
check("flat faces: a 40,000-face grid's flat face is found fast", _tgrid < 0.5, f"{_tgrid:.3f} s")

# ...and the sketch tools draw on it (fake top view over the chamfered box)
_ff_clear()
_ch = _prism()
ctx.scene.cursor.location = (0, 0, 0)
def _eva(type_, q, value='PRESS', alt=False):
    e_ = _ev(type_, _wpx(q[0], q[1]), value)
    e_.alt = alt
    return e_
f = _fake_shape(_ds.NPROTO_OT_draw_rectangle, mode="CORNERS", width=1.0, depth=0.5, corner=0.0)
_at(f, (0.0, -0.6, 0))
_outl = [np.asarray(s_) for s_ in _drw.sketch_overlay["face"]]
_b = set(bpy.data.objects)
_click(f, (-0.5, -0.8, 0))
_click(f, (0.5, -0.4, 0))
_o = _newest_curve(_b)
_pts = _ends(_o)[3] if _o is not None else []
_off = max(abs(float(q @ _n45) - 1.2 / math.sqrt(2)) for q in _pts) if len(_pts) else 1
_xr = (min(q[0] for q in _pts), max(q[0] for q in _pts)) if len(_pts) else None
check("Rectangle on a flat face: the face outlined before the click, the rectangle lies on the chamfer, sides along its edges",
      len(_outl) == 4 and all(abs(float(s_[k_] @ _n45) - 1.2 / math.sqrt(2)) < 1e-6 for s_ in _outl for k_ in (0, 1))
      and _off < 1e-6 and _xr is not None and abs(_xr[0] + 0.5) < 1e-6 and abs(_xr[1] - 0.5) < 1e-6,
      f"{len(_outl)} off {_off:.1e} x {_xr}")
f = _fake_shape(_ds.NPROTO_OT_draw_rectangle, mode="CORNERS", width=1.0, depth=0.5, corner=0.0)
_b = set(bpy.data.objects)
for q_ in ((-0.5, -0.8), (0.5, -0.4)):
    f.modal(fc, _eva('MOUSEMOVE', q_, 'NOTHING', alt=True))
    f.modal(fc, _eva('LEFTMOUSE', q_, alt=True))
_o = _newest_curve(_b)
_pts = _ends(_o)[3] if _o is not None else []
check("Rectangle with Alt: the view (grid) plane, as before", len(_pts) and max(abs(q[2]) for q in _pts) < 1e-9,
      f"{[round(q[2], 3) for q in _pts[:3]]}")
f = _ln_tool()
_click(f, (0.0, 0.3))
_click(f, (0.6, 0.6))
_click(f, (-0.5, -0.6))
_click(f, (0.4, -0.5))
_L = f._lines
check("Line: each new line starts on the flat face under it (the top, then the chamfer), its free end on that face",
      len(_L) == 2 and all(abs(q[2] - 1) < 1e-6 for q in _L[0])
      and all(abs(float(np.asarray(q) @ _n45) - 1.2 / math.sqrt(2)) < 1e-6 for q in _L[1]), f"{_L}")
f.modal(fc, _ev('ESC'))
_ff_clear()

# the face's corners and edge middles: drawn as small squares, and on a mesh they snap (a run of collinear edges too)
_fm = {}
bpy.ops.mesh.primitive_cube_add(size=2)
_f = _ff(ctx.active_object, top, _px(0.3, 0.2))
_mk = lambda P: sorted(tuple(round(float(x), 6) for x in q) for q in P)
_fm["cube"] = _f is not None and _mk(_f["corners"]) == _mk([(x, y, 1) for x in (-1, 1) for y in (-1, 1)]) \
    and _mk(_f["mids"]) == _mk([(0, -1, 1), (0, 1, 1), (-1, 0, 1), (1, 0, 1)]) and len(_f["snap"]) == 8
_ff_clear()
bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=1, depth=2)
_f = _ff(ctx.active_object, top, _px(0.2, 0.1))
_fm["32-gon"] = _f is not None and len(_f["corners"]) == 0 and len(_f["mids"]) == 0
_ff_clear()
bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=1, depth=2)
_f = _ff(ctx.active_object, top, _px(0.2, 0.1))
_fm["16-gon"] = _f is not None and len(_f["corners"]) == 16 and len(_f["mids"]) == 16
_ff_clear()
bpy.ops.mesh.primitive_grid_add(x_subdivisions=7, y_subdivisions=7, size=2)
_f = _ff(ctx.active_object, top, _px(0.3, -0.4))
_fm["grid"] = _f is not None and len(_f["corners"]) == 4 and _mk(_f["mids"]) == _mk([(0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0)])
_ff_clear()
_ch = _prism()
_f = _ff(_ch, top, _px(0.1, -0.6))
_fm["chamfer"] = _f is not None and len(_f["corners"]) == 4 and len(_f["mids"]) == 4 \
    and _mk(_f["mids"]) == _mk([(0, -1, 0.2), (0, -0.2, 1), (-1, -0.6, 0.6), (1, -0.6, 0.6)])
_ff_clear()
bpy.ops.nproto.add_primitive(kind="BOX", width=2, depth=2, height=2, degree="0")
_nb = ctx.active_object
_f = _ff(_nb, top, _px(0.3, 0.2))
_ztop = max((_nb.matrix_world @ Vector(c_)).z for c_ in _nb.bound_box)
_fm["nurbs box"] = _f is not None and len(_f["corners"]) == 4 and len(_f["mids"]) == 4 and not _f["snap"] \
    and any(np.allclose(q, (0, 1, _ztop), atol=1e-6) for q in _f["mids"])
_ff_clear()
_me = bpy.data.meshes.new("Split")
_me.from_pydata([(-1, -1, 0), (-0.4, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3, 4)])
_me.update()
_so = bpy.data.objects.new("Split", _me)
ctx.scene.collection.objects.link(_so)
ctx.view_layer.update()
f = _ln_tool()
_at(f, (0.0, -0.98))
_sq = list(_drw.sketch_overlay["face_pts"])
_fm["run middle snaps"] = _drw.sketch_overlay["snap_label"] == "Mid" and _drw.sketch_overlay["snap"] is not None \
    and np.allclose(np.asarray(_drw.sketch_overlay["snap"]), (0, -1, 0), atol=1e-6) and len(_sq) == 8
f.modal(fc, _ev('ESC'))
check("face squares: corners and edge middles (box, grid, chamfer, 16-gon, NURBS box; none on a 32-gon); "
      "a mesh face's snap, even the middle of a side split unevenly",
      all(_fm.values()), str({k_: v_ for k_, v_ in _fm.items() if not v_}) + f" {_drw.sketch_overlay['snap_label']} {_drw.sketch_overlay['snap']}")
_ff_clear()


# --- BREP to Mesh: plain meshes meshed afresh (Light: n-gons on flat faces, quads on curved ones), welded,
#     exact corner normals, UVs, seams, sharp creases, the brep_face attribute; the operator
import bmesh as _bmx
from blender_nurbs import brep_mesh as _bmm
def _bm_clear():
    for o_ in list(bpy.data.objects):
        bpy.data.objects.remove(o_)
def _bm_stats(me):
    b_ = _bmx.new()
    b_.from_mesh(me)
    open_ = sum(1 for e_ in b_.edges if len(e_.link_faces) != 2)
    area_ = sum(f_.calc_area() for f_ in b_.faces)
    vol_ = b_.calc_volume(signed=True)
    b_.free()
    return open_, area_, vol_
def _bm_of(o, **kw):
    me_, m_ = _bmm.mesh_from_brep(store.get_brep(o), "T_" + o.name, **kw)
    return me_, m_, _bmm.counts(me_), _bm_stats(me_)
_bmr = {}
_bm_clear()
bpy.ops.nproto.add_primitive(kind="BOX", width=2, depth=2, height=2, degree="0")
_bx = ctx.active_object
me, m, c, (op_, ar_, vo_) = _bm_of(_bx)
_fa = np.empty(len(me.polygons), np.int64)
me.attributes["brep_face"].data.foreach_get("value", _fa)
_bmr["box: 6 quads, closed, outward, one polygon per B-rep face"] = \
    c["faces"] == 6 and c["quads"] == 6 and c["verts"] == 8 and op_ == 0 and abs(vo_ - 8.0) < 1e-9 \
    and sorted(_fa.tolist()) == list(range(6))
_bmr["box: its 12 edges are seams and sharp"] = sum(e_.use_seam for e_ in me.edges) == 12 \
    and sum(not e_.use_edge_sharp for e_ in me.edges) == 0
me, m, c, (op_, ar_, vo_) = _bm_of(_bx, light=False)
_bmr["box, Triangles: 12 triangles"] = c["faces"] == 12 and c["tris"] == 12 and op_ == 0
_bm_clear()
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0, caps=True)
_cy = ctx.active_object
me, m, c, (op_, ar_, vo_) = _bm_of(_cy)
_ba = store.get_brep(_cy).area(1e-7)
_sides = [p_ for p_ in me.polygons if len(p_.vertices) == 4]
_rad = [math.hypot(*np.mean([me.vertices[v_].co[:2] for v_ in p_.vertices], axis=0)) for p_ in _sides]
_diag = math.sqrt(2 * 2 + 2 * 2 + 2 * 2)
_bmr["cylinder: 2 n-gon caps + one strip of quads, closed, within the chord"] = \
    c["ngons"] == 2 and c["tris"] == 0 and 40 <= c["quads"] <= 64 and op_ == 0 and vo_ > 0 \
    and max(1 - r_ for r_ in _rad) <= 0.001 * _diag * 1.01 and c["faces"] * 50 < len(_cy.data.polygons)
_uv = me.uv_layers["UVMap"]
_suv = np.array([_uv.data[li].uv[:] for p_ in _sides for li in p_.loop_indices])
_span = np.ptp(_suv, axis=0)
_bmr["cylinder: Real Size UVs (side 2 pi r by 2 m), seam + rims are seams, rims sharp, side smooth"] = \
    abs(_span.max() - 2 * math.pi) < 0.05 * 2 * math.pi and abs(_span.min() - 2.0) < 1e-3 \
    and sum(e_.use_seam for e_ in me.edges) == c["quads"] * 2 + 1 \
    and sum(e_.use_edge_sharp for e_ in me.edges) == 2 * c["quads"]
_ok_n = True
for p_ in me.polygons:
    for li in p_.loop_indices:
        if Vector(me.corner_normals[li].vector).dot(p_.normal) <= 0:
            _ok_n = False
_bmr["normals: every corner normal on its polygon's outer side"] = _ok_n
me, m, c, (op_, ar_, vo_) = _bm_of(_cy, uv_mode="NORMALIZED")
_suv = np.array([d_.uv[:] for d_ in me.uv_layers["UVMap"].data])
_bmr["0-1 per Face UVs"] = _suv.min() > -1e-9 and _suv.max() < 1 + 1e-9
_bm_clear()
bpy.ops.nproto.add_primitive(kind="SPHERE")
_sp = ctx.active_object
_cnt = {}
for d_ in ("LOW", "MEDIUM", "HIGH"):
    _t0 = _time.perf_counter()
    me, m, c, (op_, ar_, vo_) = _bm_of(_sp, detail=d_)
    _cnt[d_] = (c["faces"], op_, _time.perf_counter() - _t0)
    if d_ == "MEDIUM":
        _poles = {i_ for i_, v_ in enumerate(me.vertices) if abs(abs(v_.co.z) - 1) < 1e-6 and abs(v_.co.x) < 1e-6}
        _tri_pole = all(_poles & set(p_.vertices) for p_ in me.polygons if len(p_.vertices) == 3)
        _sp_q = c["quads"] / c["faces"]
_bmr["sphere: closed at every detail, more faces with more detail, triangles only at the poles"] = \
    all(v_[1] == 0 for v_ in _cnt.values()) and _cnt["LOW"][0] < _cnt["MEDIUM"][0] < _cnt["HIGH"][0] \
    and _tri_pole and _sp_q > 0.9 and _cnt["HIGH"][2] < 3.0
_bm_clear()
bpy.ops.nproto.add_primitive(kind="TORUS")
me, m, c, (op_, ar_, vo_) = _bm_of(ctx.active_object)
_bmr["torus (no edges): closed, nearly all quads"] = op_ == 0 and c["quads"] > 0.95 * c["faces"] and vo_ > 0
_bm_clear()
bpy.ops.nproto.add_primitive(kind="BOX", width=2, depth=2, height=2, degree="0"); _a = ctx.active_object
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=0.4, height=4.0, caps=True); _c = ctx.active_object
_c.location = (0.3, 0.2, -1.0); ctx.view_layer.update()
bpy.ops.nproto.boolean(operation="DIFFERENCE", names=json.dumps([_a.name, _c.name]))
_bo = ctx.active_object
me, m, c, (op_, ar_, vo_) = _bm_of(_bo)
_bab = store.get_brep(_bo).area(1e-7)
_bmr["box with a hole: closed, n-gons around the hole, quads in it, area within 0.1%"] = \
    op_ == 0 and c["ngons"] == 4 and c["tris"] == 0 and abs(ar_ - _bab) < 1e-3 * _bab
# what each part measured, and the B-rep's topology (edges per loop of each face): a failure says which part
# went wrong and whether the boolean's result differed (it failed once on the Windows runner only)
_bmr_hole = {"open": op_, "ngons": c["ngons"], "tris": c["tris"], "quads": c["quads"], "faces": c["faces"],
             "area": round(ar_, 6), "brep area": round(_bab, 6), "problems": m.problems,
             "loops": [[len(l_) for l_ in f_.boundary_loops()] for f_ in store.get_brep(_bo).faces()]}
_bm_clear()
bpy.ops.nproto.add_primitive(kind="SPHERE"); _s = ctx.active_object
bpy.ops.nproto.add_primitive(kind="BOX", width=1.2, depth=1.2, height=1.2, degree="0"); _b = ctx.active_object
_b.location = (0.7, 0.6, 0.5); ctx.view_layer.update()
bpy.ops.nproto.boolean(operation="DIFFERENCE", names=json.dumps([_s.name, _b.name]))
_so = ctx.active_object
me, m, c, (op_, ar_, vo_) = _bm_of(_so)
_bas = store.get_brep(_so).area(1e-7)
_bmr["sphere cut by a box (trimmed curved face): closed, mostly quads, area within 0.5%"] = \
    op_ == 0 and c["quads"] > 0.8 * c["faces"] and abs(ar_ - _bas) < 5e-3 * _bas and not m.problems
_bm_clear()
bpy.ops.nproto.add_primitive(kind="BOX", width=2, depth=2, height=2, degree="0"); _f = ctx.active_object
bpy.ops.nproto.edge_fillet(inputs=json.dumps([[_f.name, f"E{i}"] for i in store.shared_edges(_f)]), radius=0.3)
_fo = ctx.active_object
me, m, c, (op_, ar_, vo_) = _bm_of(_fo)
_baf = store.get_brep(_fo).area(1e-7)
_shp = sum(e_.use_edge_sharp for e_ in me.edges)
_bmr["filleted box: closed, area within 0.1%, no sharp edges (all tangent)"] = \
    op_ == 0 and abs(ar_ - _baf) < 1e-3 * _baf and _shp == 0
check("BREP to Mesh: " + ", ".join(k_ for k_ in _bmr), all(_bmr.values()),
      str({k_: v_ for k_, v_ in _bmr.items() if not v_}) + f" counts {_cnt} box with a hole {_bmr_hole}")

# even, symmetric lines: a surface symmetric left to right (its curvature strongest at the sides) meshes into a
# mirror-symmetric grid of quads (lines picked one by one from an end left a short piece at the far end);
# an untrimmed face's grid sets its edges' points, so its two rails get the same columns
_xs = np.linspace(-2.0, 2.0, 7)
_ys = np.linspace(0.0, 1.5, 5)
_pts = [(float(x), float(y), float(0.25 * x ** 4 + 0.1 * y * y)) for x in _xs for y in _ys]
_sym = store.new_object(ctx, "SymPatch", ext.surface_from_cvs(3, 3, _pts, geom.clamped_uniform_knots(7, 3),
                                                              geom.clamped_uniform_knots(5, 3)))
_symr = {}
for d_ in ("LOW", "MEDIUM", "HIGH"):
    me, m, c, _st = _bm_of(_sym, detail=d_)
    V_ = np.array([v_.co[:] for v_ in me.vertices])
    R_ = V_ * (-1, 1, 1)
    _err = float(np.sqrt(((R_[:, None] - V_[None]) ** 2).sum(axis=2)).min(axis=1).max())
    _symr[d_] = (c["faces"], c["quads"], c["tris"], round(_err, 9))
check("BREP to Mesh: a symmetric surface meshes into a mirror-symmetric grid of quads at every detail",
      all(v_[0] == v_[1] and v_[2] == 0 and v_[3] < 1e-6 for v_ in _symr.values())
      and _symr["LOW"][0] < _symr["MEDIUM"][0] < _symr["HIGH"][0], str(_symr))
bpy.data.objects.remove(_sym)

# the operator: a plain mesh object beside the NURBS object (hidden), materials and modifiers kept; Join; Pack
_bm_clear()
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0, caps=True)
_o1 = ctx.active_object
_mat = bpy.data.materials.new("BMTest")
_o1.data.materials.append(_mat)
_o1.modifiers.new("Bev", 'BEVEL').width = 0.02
bpy.ops.nproto.add_primitive(kind="BOX", width=1, depth=1, height=1, degree="0")
_o2 = ctx.active_object
_o2.location = (4, 0, 0)
ctx.view_layer.update()
bpy.ops.object.select_all(action='DESELECT')
_o1.select_set(True)
ctx.view_layer.objects.active = _o1
_r = bpy.ops.nproto.brep_to_mesh(detail="LOW", pack=True)
_d1 = ctx.active_object
_puv = np.array([d_.uv[:] for d_ in _d1.data.uv_layers["UVMap"].data])
_res = {"ran": _r == {'FINISHED'}, "plain": not store.is_nurbs(_d1) and _d1.type == 'MESH' and _d1.name == _o1.name + " Mesh",
        "material": list(_d1.data.materials) == [_mat], "modifier": [m_.type for m_ in _d1.modifiers] == ['BEVEL']
        and abs(_d1.modifiers[0].width - 0.02) < 1e-9,
        "hidden": _o1.hide_get() and _o1.hide_render, "same place": np.allclose(np.array(_d1.matrix_world), np.array(_o1.matrix_world)),
        "packed": _puv.min() > -1e-9 and _puv.max() < 1 + 1e-9, "counts": _bmm.LAST.get("counts", {}).get("faces") == len(_d1.data.polygons)}
bpy.ops.object.select_all(action='DESELECT')
_o1.hide_set(False)
_o1.select_set(True)
_o2.select_set(True)
ctx.view_layer.objects.active = _o2
_n0 = len(bpy.data.objects)
_r = bpy.ops.nproto.brep_to_mesh(join=True, hide_original=False)
_res["join"] = _r == {'FINISHED'} and len(bpy.data.objects) == _n0 + 1 and not _o2.hide_get()
bpy.ops.object.select_all(action='DESELECT')
_res["poll needs a NURBS object"] = not bpy.ops.nproto.brep_to_mesh.poll()
_res["Object > Convert lists it"] = _bmm.menu_convert in getattr(bpy.types.VIEW3D_MT_object_convert.draw, "_draw_funcs", [])
check("BREP to Mesh operator: plain mesh beside the hidden NURBS object, material + modifier kept, packed UVs, "
      "counts; Join; poll; Object > Convert menu", all(_res.values()), str({k_: v_ for k_, v_ in _res.items() if not v_}))
_bm_clear()


# --- Ctrl grid snap in a Side view: the step is measured along the drawing plane (world X faces the viewer
#     there, which made the step huge before the first point)
class _FCtxSide(_FCtx):
    """bpy.context with a Right orthographic view (looking along -X: screen x = Y, y = Z), 1000 px = 4 units."""
    def __init__(self):
        # (not _FCtx.__init__: a later test reuses the name it builds its view rotation with)
        self.region = _types.SimpleNamespace(x=0, y=0, width=1000, height=1000, type='WINDOW')
        R_ = _Mx(((0, 0, 1), (1, 0, 0), (0, 1, 0)))                 # view axes in world
        V_ = _Mx.Translation((0, 0, -10)) @ R_.transposed().to_4x4()
        Wm = _Mx(((0.5, 0, 0, 0), (0, 0.5, 0, 0), (0, 0, -0.01, 0), (0, 0, 0, 1)))
        self.region_data = _types.SimpleNamespace(view_matrix=V_, perspective_matrix=Wm @ V_, window_matrix=Wm,
                                                  view_rotation=R_.to_quaternion(), is_perspective=False,
                                                  view_distance=10.0, view_perspective='ORTHO')
        self.area = _FArea(self.region, self.region_data)
_bm_clear()
fcs = _FCtxSide()
ctx.scene.cursor.location = (0, 0, 0)
f = _fake(_dt.NPROTO_OT_draw_line, polyline=False, start_mode="FREE", line_mode="EDGE")
f._edit, f._lines, f._start, f._lock, f._start_refs = None, [], None, None, None
f._reset_typing()
f.sketch_start(fcs)
_q = (0.37, 0.52)                                           # world (y, z)
_P, _ = f.pick_point(fcs, _ev('MOUSEMOVE', (500 + 250 * _q[0], 500 + 250 * _q[1]), 'NOTHING', ctrl=True))
_st = f._grid_step(fcs, np.zeros(3), (0, 1, 0))
_on = _P is not None and abs(_P[0]) < 1e-9 and all(abs(_P[k] / _st - round(_P[k] / _st)) < 1e-6 for k in (1, 2))
check("Ctrl grid in a Side view: the first point lands on the visible grid in the YZ plane (step <= 1)",
      _on and _st <= 1.0 and abs(_P[1] - _q[0]) <= 0.5 * _st + 1e-9 and abs(_P[2] - _q[1]) <= 0.5 * _st + 1e-9,
      f"P {_P} step {_st}")
f.sketch_end(fcs)


# --- In any view that isn't Top / Front / Side the sketch lies on the world's XY plane (the floor grid) through
#     the origin, Ctrl snaps to the world grid there; nearly level with it, the upright world plane facing the
#     view; the 3D cursor (moved, rotated) plays no part
from mathutils import Euler as _Eu
class _FCtxView(_FCtxSide):
    """bpy.context with an orthographic view turned by R (view axes in world), 1000 px = 4 units."""
    def __init__(self, R_):
        self.region = _types.SimpleNamespace(x=0, y=0, width=1000, height=1000, type='WINDOW')
        V_ = _Mx.Translation((0, 0, -10)) @ R_.transposed().to_4x4()
        Wm = _Mx(((0.5, 0, 0, 0), (0, 0.5, 0, 0), (0, 0, -0.01, 0), (0, 0, 0, 1)))
        self.region_data = _types.SimpleNamespace(view_matrix=V_, perspective_matrix=Wm @ V_, window_matrix=Wm,
                                                  view_rotation=R_.to_quaternion(), is_perspective=False,
                                                  view_distance=10.0, view_perspective='ORTHO')
        self.area = _FArea(self.region, self.region_data)
_bm_clear()
ctx.scene.cursor.location = (0.3, -0.7, 0.45)                 # moved and turned: it must not matter
ctx.scene.cursor.rotation_euler = (math.radians(30), 0, math.radians(15))
fcv = _FCtxView(_Eu((math.radians(55), 0, math.radians(35))).to_matrix())
f = _fake(_dt.NPROTO_OT_draw_line, polyline=False, start_mode="FREE", line_mode="EDGE")
f._edit, f._lines, f._start, f._lock, f._start_refs = None, [], None, None, None
f._reset_typing()
f.sketch_start(fcv)
_pl = f._view_plane(fcv, np.zeros(3))
_P, _ = f.pick_point(fcv, _ev('MOUSEMOVE', (561, 437), 'NOTHING', ctrl=True))
_P0, _ = f.pick_point(fcv, _ev('MOUSEMOVE', (561, 437), 'NOTHING'))
_st = f._grid_step(fcv, np.zeros(3), (1, 0, 0), (0, 1, 0))
_floor = np.allclose(_pl[1], (0, 0, 1), atol=1e-9) and np.allclose(_pl[2], (1, 0, 0), atol=1e-9) \
    and _P is not None and abs(_P[2]) < 1e-9 and all(abs(_P[k] / _st - round(_P[k] / _st)) < 1e-6 for k in (0, 1)) \
    and _st <= 1.0 and _P0 is not None and abs(_P0[2]) < 1e-9
_lvl = f._view_plane(_FCtxView(_Eu((math.radians(85), 0, math.radians(20))).to_matrix()), np.zeros(3))
_upright = np.allclose(_lvl[1], (0, -1, 0), atol=1e-9) and np.allclose(_lvl[3], (0, 0, 1), atol=1e-9) \
    and np.allclose(_lvl[2], (1, 0, 0), atol=1e-9)
f._lock = "X"                                               # an axis lock's first point: along world X, through the origin
_PX, _ = f.pick_point(fcv, _ev('MOUSEMOVE', (600, 450), 'NOTHING'))
f._lock = None
_axis0 = _PX is not None and abs(_PX[1]) < 1e-9 and abs(_PX[2]) < 1e-9
f.sketch_end(fcv)
ctx.scene.cursor.location = (0, 0, 0)
ctx.scene.cursor.rotation_euler = (0, 0, 0)
check("Free view: the sketch lies on the world floor through the origin (wherever the 3D cursor is, however turned), "
      "Ctrl lands on world grid points; nearly level with it the upright world plane facing the view; an X lock's "
      "first point runs along world X",
      _floor and _upright and _axis0, f"floor {_pl[1]} {_P} {_P0} step {_st} | level {_lvl[1:]} | X lock {_PX}")

# --- Ctrl on a slanted face: the grid sits square on the face, its lines through the face's centre
_fc = {"u": np.array([1.0, 0, 0]), "v": np.array([0, 0.6, 0.8]), "c": np.array([0.13, 0.2, 0.4])}
_fa = {"u": np.array([1.0, 0, 0]), "v": np.array([0, 1.0, 0]), "c": np.array([0.13, 0.2, 0.4])}
check("Ctrl grid origin: a slanted face's centre; a level face's, and no face, the world origin",
      np.allclose(f._grid_origin(ctx, np.ones(3), _fc), _fc["c"])
      and np.allclose(f._grid_origin(ctx, np.ones(3), _fa), 0)
      and np.allclose(f._grid_origin(ctx, np.ones(3), None), 0))

# --- An active work plane: sketches lie exactly on it, Ctrl snaps to its own grid, X / Y / Z lock to its axes,
#     a face lying in it uses its axes and grid; seen edge-on the World's planes stand in
from blender_nurbs import workplanes as _wpk
_stK = ctx.scene.nproto_workplanes
_frK = _wpk.frame_from_points((0.2, 0.1, 0.3), (1.2, 0.1, 0.6), (0.2, 1.1, 0.9))
_wK = _wpk.add_plane(ctx.scene, "Tilted", _frK)
_wK.auto_spacing, _wK.spacing, _wK.major = False, 0.025, 4
_frK = _wpk.frame_of(_wK)                                   # as stored (Euler round trip)
_Rm = np.array(_Eu(_wK.rotation, 'XYZ').to_matrix(), float)
_orth = np.abs(np.array([_frK[1], _frK[2], _frK[3]]) @ np.array([_frK[1], _frK[2], _frK[3]]).T - np.eye(3)).max()
check("work plane: its axes are orthonormal in double precision (1e-15) and match Blender's XYZ Euler",
      _orth < 1e-14 and np.allclose(_Rm, np.array([_frK[1], _frK[2], _frK[3]]).T, atol=1e-6), f"{_orth:.1e}")
_OK, _xK, _yK, _nK = _frK
f = _fake(_dt.NPROTO_OT_draw_line, polyline=False, start_mode="FREE", line_mode="EDGE")
f._edit, f._lines, f._start, f._lock, f._start_refs = None, [], None, None, None
f._reset_typing()
f.sketch_start(fcv)
_plK = f._view_plane(fcv)
_exact_plane = all(np.allclose(a_, b_, atol=1e-12) for a_, b_ in zip(_plK, (_OK, _nK, _xK, _yK)))
_offs, _grid_ok = [], True
for _m in ((561, 437), (300, 700), (820, 150), (512.3, 498.7)):
    _P0, _ = f.pick_point(fcv, _ev('MOUSEMOVE', _m, 'NOTHING'))
    _Pc, _ = f.pick_point(fcv, _ev('MOUSEMOVE', _m, 'NOTHING', ctrl=True))
    _st = f._grid_step(fcv, _Pc, _xK, _yK)
    _offs += [abs(float((_P0 - _OK) @ _nK)), abs(float((_Pc - _OK) @ _nK))]
    _cx, _cy = float((_Pc - _OK) @ _xK) / _st, float((_Pc - _OK) @ _yK) / _st
    _grid_ok &= abs(_cx - round(_cx)) < 1e-6 and abs(_cy - round(_cy)) < 1e-6 \
        and any(abs(_st - l_) < 1e-12 for l_ in _wpk.fixed_levels(_wK.spacing, 4))
f._lock = "X"
_PXk, _ = f.pick_point(fcv, _ev('MOUSEMOVE', (600, 450), 'NOTHING'))
f._lock = None
_dX = _PXk - _OK
_xlock = np.linalg.norm(_dX - float(_dX @ _xK) * _xK) < 1e-9
_faceK = {"n": -_nK, "P": _OK + 0.3 * _xK - 0.2 * _yK, "u": _yK, "v": _xK, "c": _OK + 0.3 * _xK, "size": 1.0}
_faceOff = dict(_faceK, P=_faceK["P"] + 0.01 * _nK)
_onw = f._on_work(fcv, _faceK) and not f._on_work(fcv, _faceOff)
f._face_hover = _faceK
f._plane = None
f.fix_plane(fcv, _faceK["P"])
_fixed = all(np.allclose(a_, b_, atol=1e-12) for a_, b_ in zip(f._plane, (_OK, _nK, _xK, _yK))) \
    and np.allclose(f._grid_origin(fcv, f._plane[0], _faceK), _OK)
f._face_hover, f._plane = None, None
f.sketch_end(fcv)
# edge-on: looking along the plane's X (the normal is square to the view)
_Re = _Mx((tuple(_yK), tuple(_nK), tuple(-_xK))).transposed()      # view x = plane Y, view y = normal, looking along X
_ple = f._view_plane(_FCtxView(_Re), None)
_edge = not np.allclose(abs(float(_ple[1] @ _nK)), 1.0)
_stK.planes.clear(); _stK.active = -1; _stK.index = 0
check("work plane sketching: the drawing plane is exactly the work plane; points (Ctrl or not) lie on it, Ctrl on "
      "its own grid (spacing 0.025, major 4); an X lock runs along its X from its origin; a face in it uses its "
      "axes and grid, a face 0.01 off does not; seen edge-on the World's planes stand in",
      _exact_plane and max(_offs) < 1e-12 and _grid_ok and _xlock and _onw and _fixed and _edge,
      f"plane {_exact_plane} off {max(_offs):.1e} grid {_grid_ok} xlock {_xlock} face {_onw} {_fixed} edge {_edge}")

# --- Ctrl step in a close perspective view: from the projection's derivative (a whole-unit step went behind
#     the camera and gave 1 m), the fewer pixels per unit of the plane's two axes
_Rp = _Eu((math.radians(60), 0, 0)).to_matrix()
_Vp = (_Mx.Translation((0, -0.1, 0.06)) @ _Rp.to_4x4()).inverted()
_Wp = _Mx(((2.78, 0, 0, 0), (0, 2.78, 0, 0), (0, 0, -1.0002, -0.020002), (0, 0, -1, 0)))
_rvp = _types.SimpleNamespace(perspective_matrix=_Wp @ _Vp)
_rgp = _types.SimpleNamespace(width=1000, height=1000)
_ppx = _dt.pixels_per_unit(_rgp, _rvp, np.zeros(3), [np.array((1.0, 0, 0))])
_ppy = _dt.pixels_per_unit(_rgp, _rvp, np.zeros(3), [np.array((0, 1.0, 0))])
_ppb = _dt.pixels_per_unit(_rgp, _rvp, np.zeros(3), [np.array((1.0, 0, 0)), np.array((0, 1.0, 0))])
_hx = 1e-6
_q0 = _Wp @ _Vp @ Vector((0, 0, 0, 1)); _q1 = _Wp @ _Vp @ Vector((_hx, 0, 0, 1))
_fd = np.linalg.norm((np.array(_q1[:2]) / _q1[3] - np.array(_q0[:2]) / _q0[3]) * 500) / _hx
check("Ctrl step in a close perspective view: pixels per unit from the projection's derivative (matches a tiny "
      "step), the fewer of the plane's two axes",
      abs(_ppx - _fd) < 1e-3 * _fd and _ppy < _ppx and abs(_ppb - _ppy) < 1e-9 * _ppy and _ppx > 5000,
      f"x {_ppx:.0f} (fd {_fd:.0f}) y {_ppy:.0f}")
# looking across the axes (view turned 45 degrees about Z): the density in the
# most squeezed direction, the same however the plane's axes point (the grids
# fade the same looking along an axis or diagonally across)
_Rz45 = _Eu((0, 0, math.radians(45))).to_matrix()     # the same view, turned about Z round the origin
_Vd = (_Mx.Translation(_Rz45 @ Vector((0, -0.1, 0.06))) @ (_Rz45 @ _Rp).to_4x4()).inverted()
_rvd = _types.SimpleNamespace(perspective_matrix=_Wp @ _Vd)
_c45, _s45 = math.cos(math.radians(45)), math.sin(math.radians(45))
_pd_xy = _dt.pixels_per_unit(_rgp, _rvd, np.zeros(3), [np.array((1.0, 0, 0)), np.array((0, 1.0, 0))])
_pd_rot = _dt.pixels_per_unit(_rgp, _rvd, np.zeros(3), [np.array((_c45, _s45, 0)), np.array((-_s45, _c45, 0))])
_pd_x = _dt.pixels_per_unit(_rgp, _rvd, np.zeros(3), [np.array((1.0, 0, 0))])
check("grid density looking diagonally across the axes: the most squeezed direction, the same for any axes "
      "(and below what either axis alone gives)",
      abs(_pd_xy - _pd_rot) < 1e-9 * _pd_rot and _pd_xy < 0.9 * _pd_x and abs(_pd_xy - _ppy) < 1e-6 * _ppy,
      f"xy {_pd_xy:.1f} rotated {_pd_rot:.1f} x alone {_pd_x:.1f} looking along y {_ppy:.1f}")

# --- Fine floor grid (draw.py): a turned orthographic view gets the levels below 1 m / 1 ft a Top view draws
from blender_nurbs import draw as _drw
_flm = _dt.fine_grid_levels("METRIC", 1.0, 1.0, 10)
_fli = _dt.fine_grid_levels("IMPERIAL", 1.0, 1.0, 10)
_fl2 = _dt.fine_grid_levels("METRIC", 1.0, 2.0, 10)
_lv_ok = _flm is not None and np.allclose(_flm[0], [0.001, 0.01, 0.1, 1.0]) and _flm[1] == 1.0 \
    and _fli is not None and np.allclose(_fli[0], [0.0000254, 0.0254, 0.3048, 0.9144]) and abs(_fli[1] - 0.3048) < 1e-12 \
    and _fl2 is not None and abs(_fl2[1] - 2.0) < 1e-12 and _dt.fine_grid_levels("NONE", 1.0, 1.0, 10) is None
_tv = _FCtxView(_Eu((math.radians(30), 0, math.radians(20))).to_matrix())
_rvo = _types.SimpleNamespace(is_perspective=False, window_matrix=_Mx.Diagonal((2 * 1640 / 1000, 2 * 1640 / 1000, -0.01, 1)),
                              view_rotation=_tv.region_data.view_rotation, view_location=Vector((0.2, 0.1, 0.0)))
_rgo = _types.SimpleNamespace(width=1000, height=1000)
_sqo = _drw.fine_grid_square(_rvo, _rgo, 1000.0, _fli[0], _fli[1], _dt.GRID_FADE_PX)
_rvf = _types.SimpleNamespace(is_perspective=False, window_matrix=_Mx.Diagonal((0.02, 0.02, -0.01, 1)),
                              view_rotation=_tv.region_data.view_rotation, view_location=Vector((0, 0, 0)))
_sqf = _drw.fine_grid_square(_rvf, _rgo, 1000.0, _fli[0], _fli[1], _dt.GRID_FADE_PX)
_rve = _types.SimpleNamespace(is_perspective=False, window_matrix=_rvo.window_matrix, view_location=Vector((0, 0, 0)),
                              view_rotation=_Eu((math.radians(90), 0, 0)).to_quaternion())
_sqe = _drw.fine_grid_square(_rve, _rgo, 1000.0, _fli[0], _fli[1], _dt.GRID_FADE_PX)
_hw = 1000 / (2 * 1640)
_camp = _Mx.Translation((0.4, -0.3, 0.8)) @ _Eu((math.radians(50), 0, 0)).to_matrix().to_4x4()
_rvp2 = _types.SimpleNamespace(is_perspective=True, view_matrix=_camp.inverted(), window_matrix=_Wp)
_rvp3 = _types.SimpleNamespace(is_perspective=True, view_matrix=(_Mx.Translation((0, 0, 900.0)) @ _camp).inverted(),
                               window_matrix=_Wp)
_sqp = _drw.fine_grid_square(_rvp2, _rgo, 1000.0, _flm[0], _flm[1], _dt.GRID_FADE_PX)
_sqh = _drw.fine_grid_square(_rvp3, _rgo, 1000.0, _flm[0], _flm[1], _dt.GRID_FADE_PX)
_Rp_exp = 1.5 * 0.1 * 2.78 * 500 / _dt.GRID_FADE_PX
_vw = bpy.context.window_manager.nproto_view
check("Fine Grid toggle: on by default, a Display toggle and a header icon",
      _vw.show_fine_grid is True and ("show_fine_grid", "Fine Grid", 'GRID') in _ui.DISPLAY_TOGGLES)
check("Fine floor grid: levels mm / cm / 10 cm (thou / inch) under 1 m (1 ft), none without units; a turned "
      "orthographic view gets a square covering what it sees of the floor, none zoomed out or edge-on; a "
      "perspective view a square around the eye out to where 10 cm lines get too dense, none from high up",
      _lv_ok and _sqo is not None and abs(_sqo[0] - 0.2) < 1e-6 and abs(_sqo[1] - 0.1) < 1e-6
      and _sqo[2] >= _hw / math.cos(math.radians(30)) and _sqo[2] < 1.0 and _sqf is None and _sqe is None
      and _sqp is not None and abs(_sqp[0] - 0.4) < 1e-6 and abs(_sqp[1] + 0.3) < 1e-6
      and abs(_sqp[2] - _Rp_exp) < 1e-6 * _Rp_exp and _sqh is None,
      f"{_flm} {_fli} | square {_sqo} far {_sqf} edge-on {_sqe} | persp {_sqp} high {_sqh}")

# --- Mesh to BREP: planes, bevels (Profile 0.5) as fillets of the bevel's radius, holes and cylinders exact,
#     the fillets as Edge Fillet history steps; unrecognised areas keep their facets; open meshes become sheets
from blender_nurbs import mesh_brep as _m2b
_bm_clear()
def _m2b_cube(name, a=0.6, bevel=None, w=0.1, seg=6, profile=0.5):
    """A cube of side a about the origin, the edges bevel(edge midpoint) picks bevelled by w (Blender's bevel)."""
    me_ = bpy.data.meshes.new(name)
    b_ = _bmx.new()
    _bmx.ops.create_cube(b_, size=a)
    if bevel is not None:
        es_ = [e_ for e_ in b_.edges if bevel(0.5 * (e_.verts[0].co + e_.verts[1].co))]
        _bmx.ops.bevel(b_, geom=es_, offset=w, offset_type='OFFSET', segments=seg, profile=profile, affect='EDGES')
    b_.to_mesh(me_)
    b_.free()
    o_ = bpy.data.objects.new(name, me_)
    ctx.scene.collection.objects.link(o_)
    return o_
def _m2b_run(o_, **kw):
    new_, summ_, note_ = _m2b.convert(ctx, o_, **kw)
    b_ = store.get_brep(new_)
    return new_, summ_, note_, b_, abs(b_.volume(1e-6)), bool(b_.is_manifold_solid())
_a, _w = 0.6, 0.1
_top_x = lambda m_: abs(m_.x) < 1e-6 and m_.y > 0.29 and m_.z > 0.29          # the top edge along X at +Y
_corner3 = lambda m_: sum(abs(abs(c_) - 0.3) < 1e-6 for c_ in m_) == 2 and all(c_ > -1e-6 for c_ in m_)
_m2r = {}
n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_cube("M2B_Cube"))
_m2r["plain cube: 6 planes, a closed solid of the mesh's volume"] = \
    s_ == "6 planes" and b_.face_count() == 6 and sol_ and abs(v_ - _a ** 3) < 1e-6 * _a ** 3 and no_ is None
def _m2b_whole(f_):
    """Whether a face is its whole surface: the surface's four corners are the face's edge ends."""
    s_ = f_.surface()
    (u0_, u1_), (v0_, v1_) = s_.u_domain(), s_.v_domain()
    C_ = np.array([tuple(s_.evaluate(u_, w_)) for u_ in (u0_, u1_) for w_ in (v0_, v1_)], float)
    E_ = np.array([tuple(e_.point_at(t_)) for e_ in f_.edges() for t_ in e_.parameter_range()], float)
    return max(np.linalg.norm(E_ - c_, axis=1).min() for c_ in C_) < 1e-6
_m2r["plain cube: each four-sided face is an untrimmed bilinear patch on its corners, not a trimmed plane"] = \
    all(_m2b_whole(f_) for f_ in b_.faces())
def _m2b_notched(name, cuts, verts=48):
    """A 3 x 1.4 x 0.5 plate at the origin (a mesh's float32 vertices far out are too coarse) with vertical
    holes (x, y, r) cut by Blender's Boolean."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0)); o_ = ctx.active_object; o_.name = name
    o_.scale = (3, 1.4, 0.5); bpy.ops.object.transform_apply(location=False, scale=True)
    for x_, y_, r_ in cuts:
        bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r_, depth=2, location=(x_, y_, 0)); c_ = ctx.active_object
        m_ = o_.modifiers.new("b", 'BOOLEAN'); m_.object = c_; m_.operation = 'DIFFERENCE'
        ctx.view_layer.objects.active = o_; bpy.ops.object.modifier_apply(modifier="b"); bpy.data.objects.remove(c_)
    return o_
def _m2b_cyl_dev(b_, cuts):
    """Largest distance of the curved faces from the nearest cut's true cylinder."""
    out_ = 0.0
    for f_ in b_.faces():
        s2_ = f_.surface()
        if s2_.is_planar(1e-9):
            continue
        (u0_, u1_), (v0_, v1_) = s2_.u_domain(), s2_.v_domain()
        P_ = np.array([tuple(s2_.evaluate(u_, w_)) for u_ in np.linspace(u0_, u1_, 41) for w_ in np.linspace(v0_, v1_, 5)])
        out_ = max(out_, min(float(np.abs(np.hypot(P_[:, 0] - x_, P_[:, 1] - y_) - r_).max()) for x_, y_, r_ in cuts))
    return out_
for _nm, _cuts, _vn, _planes in (("through the end (half a cylinder)", [(1.5, 0.0, 0.4)], 48, 7),
                                 ("off-centre through the end", [(1.6, 0.0, 0.4)], 48, 7),
                                 ("off-centre, a 16-sided cylinder (facets cut short at its sides)", [(1.75, 0.0, 0.4)], 16, 7),
                                 ("off-centre, 48 sides (the boolean splits it in pieces)", [(1.57, 0.0, 0.4)], 48, 7),
                                 ("through a corner", [(1.5, 0.7, 0.4)], 48, 6)):
    n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_notched("M2B_Notch", _cuts, _vn))
    _m2r[f"a hole cut {_nm}: one exact cylinder face, no flat strips or free-form patch, a closed solid"] = \
        sol_ and no_ is None and s_ == f"{_planes} planes · 1 hole ⌀ 80 cm" and b_.face_count() == _planes + 1 \
        and _m2b_cyl_dev(b_, _cuts) < 1e-6
for seg_ in (2, 3, 6, 12):
    n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_cube(f"M2B_Edge{seg_}", bevel=_top_x, seg=seg_))
    st_ = features.stack(n_)
    exp_ = _a ** 3 - (1 - math.pi / 4) * _w ** 2 * _a
    _m2r[f"one edge bevelled ({seg_} segments): a fillet of the bevel's radius, exact volume, an Edge Fillet step"] = \
        b_.face_count() == 7 and sol_ and abs(v_ - exp_) < 1e-6 * exp_ and len(st_) == 1 \
        and st_[0]["op"] == "edge_fillet" and abs(st_[0]["params"]["radius"] - _w) < 1e-6 and "1 fillet" in s_
n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_cube("M2B_Chamfer", bevel=_top_x, seg=1))
_m2r["a chamfer (1 segment) stays a plane"] = s_ == "7 planes" and b_.face_count() == 7 and sol_ \
    and abs(v_ - (_a ** 3 - 0.5 * _w ** 2 * _a)) < 1e-6 * _a ** 3 and not features.stack(n_)
n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_cube("M2B_Corner", bevel=_corner3, seg=6))
_m2r["three bevelled edges at a corner: three fillets, one radius, a corner patch, closed"] = \
    b_.face_count() == 10 and sol_ and "3 fillets" in s_ and len(features.stack(n_)) == 1 \
    and store.brep_from_text(n_[features.PROP_BASE]).face_count() == 6
n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_cube("M2B_All", bevel=lambda m_: True, seg=5))
_bb = _a - 2 * _w
_exp = _bb ** 3 + 6 * _bb ** 2 * _w + 3 * math.pi * _bb * _w ** 2 + 4 / 3 * math.pi * _w ** 3
_m2r["all twelve edges bevelled: 12 fillets with sphere corners, the rounded box's exact volume"] = \
    b_.face_count() == 26 and sol_ and abs(v_ - _exp) < 1e-6 * _exp and "12 fillets" in s_
# a plate with a vertical and a horizontal hole (Boolean modifiers, exact solver)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(3.0, 0.0, 0.0), scale=(1.0, 0.6, 0.2))
_pl = ctx.active_object
for i_, (loc_, rot_, r_, h_) in enumerate((((3.2, 0.0, 0.0), (0, 0, 0), 0.1, 0.5),
                                           ((2.75, 0.0, 0.0), (math.radians(90), 0, 0), 0.05, 1.0))):
    bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=r_, depth=h_, location=loc_, rotation=rot_)
    md_ = _pl.modifiers.new(f"Hole{i_}", 'BOOLEAN')
    md_.object, md_.solver, md_.operation = ctx.active_object, 'EXACT', 'DIFFERENCE'
n_, s_, no_, b_, v_, sol_ = _m2b_run(_pl)
_exp = 1.0 * 0.6 * 0.2 - math.pi * 0.1 ** 2 * 0.2 - math.pi * 0.05 ** 2 * 0.6
_m2r["a plate with a vertical and a horizontal hole: both exact (diameter, volume), the plate's 6 planes"] = \
    sol_ and abs(v_ - _exp) < 1e-6 * _exp and "6 planes" in s_ and "2 holes" not in s_ and s_.count("hole") == 2 \
    and b_.face_count() == 8
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(3.0, 2.0, 0.0), scale=(0.6, 0.6, 0.2))
_pb = ctx.active_object
bpy.ops.mesh.primitive_cylinder_add(vertices=40, radius=0.08, depth=0.2, location=(3.0, 2.0, 0.1))  # 0.1 deep
md_ = _pb.modifiers.new("Blind", 'BOOLEAN')
md_.object, md_.solver, md_.operation = ctx.active_object, 'EXACT', 'DIFFERENCE'
n_, s_, no_, b_, v_, sol_ = _m2b_run(_pb)
_exp = 0.6 * 0.6 * 0.2 - math.pi * 0.08 ** 2 * 0.1
_m2r["a blind hole: exact diameter and depth (its flat bottom made by the cylinder)"] = \
    sol_ and abs(v_ - _exp) < 1e-6 * _exp and s_.count("hole") == 1 and b_.face_count() == 8
bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.3, depth=0.6, location=(5.0, 0.0, 0.0),
                                    rotation=(math.radians(90), 0, 0))
n_, s_, no_, b_, v_, sol_ = _m2b_run(ctx.active_object)
_m2r["a cylinder (turned on its side): an exact cylinder with trimmed caps"] = \
    "1 cylinder" in s_ and b_.face_count() == 3 and sol_ and abs(v_ - math.pi * 0.09 * 0.6) < 1e-6
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.5, location=(7.0, 0.0, 0.0))
_nf = len(ctx.active_object.data.polygons)
n_, s_, no_, b_, v_, sol_ = _m2b_run(ctx.active_object)
_m2r["an icosphere (nothing to recognise): a solid of its flat facets"] = b_.face_count() == _nf == 80 and sol_
bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3, size=1.0, location=(9.0, 0.0, 0.0))
n_, s_, no_, b_, v_, sol_ = _m2b_run(ctx.active_object)
_m2r["an open mesh: a sheet, said so"] = b_.face_count() == 1 and not sol_ and no_ is not None and "open" in no_
# a shelled cap (Pool Table.blend): a profile spun to the axis by Screw, thickened outward by Solidify. Solidify's
# corner offset is not exact: the outer wall leans ~0.1 deg (it stopped growing a third of the way round) and the
# spun top's tip sits above its rim (64 triangles, no plane)
def _m2b_shell(name, x_, ngon=False):
    me_ = bpy.data.meshes.new(name)
    me_.from_pydata([(-0.238125, 0, 0.40005), (-0.238125, 0, 0.45085), (0, 0, 0.45085)], [(1, 0), (2, 1)], [])   # (edge order: Solidify thickens outward)
    o_ = bpy.data.objects.new(name, me_)
    ctx.scene.collection.objects.link(o_)
    o_.location = (x_, 0.0, 0.0)
    m_ = o_.modifiers.new("Screw", 'SCREW')
    m_.steps = m_.render_steps = 64
    m_ = o_.modifiers.new("Solidify", 'SOLIDIFY')
    m_.thickness, m_.offset, m_.use_even_offset = 0.00635, 1.0, True
    if ngon:                                   # applied, each spun cap dissolved into one n-gon
        dg_ = ctx.evaluated_depsgraph_get()
        me2_ = bpy.data.meshes.new_from_object(o_.evaluated_get(dg_))
        o_.modifiers.clear()
        b_ = _bmx.new()
        b_.from_mesh(me2_)
        _bmx.ops.remove_doubles(b_, verts=b_.verts, dist=1e-7)
        for z_ in (0.45085, 0.4572):
            _bmx.ops.dissolve_faces(b_, faces=[f_ for f_ in b_.faces if len(f_.verts) == 3
                                               and all(abs(v_.co.z - z_) < 2e-4 for v_ in f_.verts)])
        b_.to_mesh(me_)
        b_.free()
    return o_
for _ng in (False, True):
    _o = _m2b_shell("M2B_Shell", 11.0, _ng)
    _dg = ctx.evaluated_depsgraph_get()
    _ev = _o.evaluated_get(_dg).to_mesh()
    _P = np.array([tuple(v_.co) for v_ in _ev.vertices])
    _o.evaluated_get(_dg).to_mesh_clear()
    _ro = float(np.hypot(_P[:, 0], _P[:, 1]).max() + np.hypot(_P[:, 0], _P[:, 1])[_P[:, 2] < 0.4001].max()) / 2
    n_, s_, no_, b_, v_, sol_ = _m2b_run(_o)
    _exp = math.pi * (_ro ** 2 * (0.45711 - 0.40005) - 0.238125 ** 2 * (0.45085 - 0.40005))   # (top: the outer rim)
    _dev = _m2b_cyl_dev(b_, [(0.0, 0.0, 0.238125), (0.0, 0.0, _ro)])   # (the body is local)
    _m2r["a shelled cap (Screw + Solidify" + (", n-gon caps" if _ng else "") + "): one hole, one cylinder, 3 planes, "
         "a closed solid of 5 faces"] = \
        "1 hole" in s_ and "1 cylinder" in s_ and s_.startswith("3 planes") and b_.face_count() == 5 and sol_ \
        and _dev < 6e-5 and abs(v_ - _exp) < 1e-3 * _exp and no_ is None
    bpy.data.objects.remove(_o)
# a tube: its hole runs through both end faces (annuli), which shut only the outer cylinder
_tb = _bmx.new()
_n = 48
_vs = [[_tb.verts.new((r_ * math.cos(2 * math.pi * k_ / _n), r_ * math.sin(2 * math.pi * k_ / _n), z_))
        for k_ in range(_n)] for r_, z_ in ((0.4, 0.0), (0.4, 0.5), (0.3, 0.5), (0.3, 0.0))]
for j_ in range(4):
    for k_ in range(_n):
        _tb.faces.new((_vs[j_][k_], _vs[j_][(k_ + 1) % _n], _vs[(j_ + 1) % 4][(k_ + 1) % _n], _vs[(j_ + 1) % 4][k_]))
_bmx.ops.recalc_face_normals(_tb, faces=_tb.faces)
_me = bpy.data.meshes.new("M2B_Tube")
_tb.to_mesh(_me)
_tb.free()
_o = bpy.data.objects.new("M2B_Tube", _me)
ctx.scene.collection.objects.link(_o)
_o.location = (13.0, 0.0, 0.0)
n_, s_, no_, b_, v_, sol_ = _m2b_run(_o)
_m2r["a tube (quads only): an exact cylinder and hole, the hole through both annuli, a closed solid of 4 faces"] = \
    "1 hole" in s_ and "1 cylinder" in s_ and b_.face_count() == 4 and sol_ \
    and abs(v_ - math.pi * (0.16 - 0.09) * 0.5) < 1e-6 and _m2b_cyl_dev(b_, [(0.0, 0.0, 0.3), (0.0, 0.0, 0.4)]) < 1e-6
# cones: a turned profile (closed loop of (r, z), r 0 a pole) spun into quads and pole fans; a cone is rebuilt as
# a face ruled between its end circles and sewn to what closes its ends (the stool's lid in "mesh to brep.blend")
def _m2b_turned(name, prof, x_, n_=48):
    tb_ = _bmx.new()
    rows_ = [[tb_.verts.new((r_ * math.cos(2 * math.pi * k_ / n_), r_ * math.sin(2 * math.pi * k_ / n_), z_))
              for k_ in range(n_)] if r_ > 0 else [tb_.verts.new((0.0, 0.0, z_))] * n_ for r_, z_ in prof]
    for j_ in range(len(prof)):
        a_, c_ = rows_[j_], rows_[(j_ + 1) % len(prof)]
        for k_ in range(n_):
            q_ = [a_[k_], a_[(k_ + 1) % n_], c_[(k_ + 1) % n_], c_[k_]]
            q_ = [v_ for i_, v_ in enumerate(q_) if v_ not in q_[:i_]]
            if len(q_) >= 3:
                tb_.faces.new(q_)
    _bmx.ops.recalc_face_normals(tb_, faces=tb_.faces)
    me_ = bpy.data.meshes.new(name)
    tb_.to_mesh(me_)
    tb_.free()
    o_ = bpy.data.objects.new(name, me_)
    ctx.scene.collection.objects.link(o_)
    o_.location = (x_, 4.0, 0.0)
    return o_
_frus = lambda R_, r_, h_: math.pi * h_ / 3 * (R_ * R_ + R_ * r_ + r_ * r_)
n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_turned("M2B_Chamfered", [(0, 0), (0.4, 0), (0.4, 0.4), (0.3, 0.5), (0, 0.5)], 0.0))
_m2r["a chamfered puck (turned): an exact cylinder and a 45 deg cone sewn to the caps, a closed solid of 4 faces"] = \
    "1 cylinder" in s_ and "1 cone 45°" in s_ and b_.face_count() == 4 and sol_ \
    and abs(v_ - math.pi * 0.16 * 0.4 - _frus(0.4, 0.3, 0.1)) < 1e-6
n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_turned("M2B_Countersink", [(0.1, 0), (0.5, 0), (0.5, 0.3), (0.2, 0.3), (0.1, 0.2)], 1.5))
_m2r["a countersunk ring: the hole and its cone sewn, the outer wall and both annuli with it, a closed solid of 5 faces"] = \
    "1 hole" in s_ and "1 cylinder" in s_ and "1 cone" in s_ and b_.face_count() == 5 and sol_ \
    and abs(v_ - (math.pi * 0.25 * 0.3 - math.pi * 0.01 * 0.2 - _frus(0.2, 0.1, 0.1))) < 1e-6
n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_turned("M2B_Taper", [(0, 0), (0.4, 0), (0.2, 0.5), (0, 0.5)], 3.0, n_=24))
_m2r["a tapered plug (24 sides): one cone between two discs, the frustum's exact volume"] = \
    "2 planes" in s_ and s_.count("cone") == 1 and b_.face_count() == 3 and sol_ and abs(v_ - _frus(0.4, 0.2, 0.5)) < 1e-6
def _m2b_lid(name, x_):                    # the lid: a slanted rim and top spun by Screw, Solidify (complex) outward
    me_ = bpy.data.meshes.new(name)
    me_.from_pydata([(-0.264934, 0, 0.40005), (-0.238125, 0, 0.45085), (0, 0, 0.45085)], [(1, 0), (2, 1)], [])
    o_ = bpy.data.objects.new(name, me_)
    ctx.scene.collection.objects.link(o_)
    o_.location = (x_, 4.0, 0.0)
    sc_ = o_.modifiers.new("Screw", 'SCREW')
    sc_.steps = sc_.render_steps = 32
    so_ = o_.modifiers.new("Solidify", 'SOLIDIFY')
    so_.solidify_mode, so_.thickness, so_.offset, so_.use_even_offset = 'NON_MANIFOLD', 0.00635, 1.0, True
    return o_
n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_lid("M2B_Lid", 4.5))
_m2r["a lid (Screw + Solidify, slanted rim): three cones (outside, inside, the rim between), 2 planes, 5 faces"] = \
    "2 planes" in s_ and s_.count("cone") == 3 and b_.face_count() == 5 and sol_ and no_ is None
n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_lid("M2B_LidNoCones", 5.5), cones=False)
_m2r["a lid with cones off: its facets, as before"] = "cone" not in s_ and b_.face_count() == 98 and sol_
# boxy bevels (Profile above 0.5): an exact superellipse strip, G2 to both faces, where it ends on flat faces
def _boxy_vol(p_, w_=0.1, a_=0.6):
    r_ = -2 * math.log(2) / math.log(p_)
    return a_ ** 3 - a_ * w_ * w_ * (1 - math.gamma(1 + 1 / r_) ** 2 / math.gamma(1 + 2 / r_))
for p_, seg_ in ((0.6, 8), (0.75, 4), (0.75, 8), (0.9, 8), (0.9, 16), (0.93, 12)):
    n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_cube(f"M2B_Boxy{p_}_{seg_}", bevel=_top_x, seg=seg_, profile=p_))
    exp_ = _boxy_vol(p_)
    _m2r[f"a boxy bevel (profile {p_}, {seg_} segments): one exact strip, the profile read back, exact volume"] = \
        b_.face_count() == 7 and sol_ and abs(v_ - exp_) < 1e-5 * exp_ and f"1 bevel profile {p_:.3g}" in s_ \
        and not features.stack(n_) and no_ is None
def _m2b_plane_fit(b_):
    """Largest ratio of a planar face's surface (its bilinear CV net) to the face's own area."""
    out_ = 1.0
    for f_ in b_.faces():
        if not f_.surface().is_planar(1e-9):
            continue
        C_ = np.array(ext.surface_cvs(f_.surface())["points"], float)
        if len(C_) == 4:
            out_ = max(out_, float(np.linalg.norm(np.cross(C_[2] - C_[0], C_[1] - C_[0]))) / f_.area(1e-6))
    return out_
_m2r["a boxy bevel's end faces (a curved trim): their planes cut down to the outline, not the kernel's oversized plane"] = \
    _m2b_plane_fit(_m2b_run(_m2b_cube("M2B_BoxyTight", bevel=_top_x, seg=8, profile=0.75))[3]) < 1.3
n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_cube("M2B_FilletTight", bevel=_top_x, seg=6))
_ft0 = _m2b_plane_fit(b_)
features.rebuild(n_, ctx, force=True)
_m2r["a filleted edge: the planes beside it cut down to what the fillet leaves, also when the history rebuilds it"] = \
    sol_ and _ft0 < 1.1 and _m2b_plane_fit(store.get_brep(n_)) < 1.1 and bool(store.get_brep(n_).is_manifold_solid())
n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_cube("M2B_BoxyG1", bevel=_top_x, seg=8, profile=0.8))
_sw = [f_ for f_ in b_.faces() if not f_.surface().is_planar(1e-9)]
_tan = 0
if len(_sw) == 1:
    s_g = _sw[0].surface()
    (u0_, u1_), (v0_, v1_) = s_g.u_domain(), s_g.v_domain()
    for u_, v_g in ((u0_, 0.5 * (v0_ + v1_)), (u1_, 0.5 * (v0_ + v1_)), (0.5 * (u0_ + u1_), v0_), (0.5 * (u0_ + u1_), v1_)):
        nn_ = np.array(s_g.normal(u_, v_g), float)
        nn_ /= np.linalg.norm(nn_)
        _tan += int(max(abs(nn_[1]), abs(nn_[2])) > 1 - 1e-9)
_m2r["the boxy strip meets both faces tangentially (its normal on each rail is the face's)"] = len(_sw) == 1 and _tan == 2
n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_cube("M2B_Boxy95", bevel=_top_x, seg=16, profile=0.96))
_m2r["profile 0.96 (Blender makes a square corner from about 0.95): the plain cube it is"] = \
    s_ == "6 planes" and sol_ and abs(v_ - 0.6 ** 3) < 1e-6
n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_cube("M2B_Boxy3", bevel=_top_x, seg=3, profile=0.8))
_m2r["a boxy bevel of 3 segments keeps its facets (any 3 facets fit some profile): a valid solid"] = \
    b_.face_count() == 9 and sol_ and "profile" not in s_
_top_xx = lambda m_: abs(m_.x) < 1e-6 and abs(m_.y) > 0.29 and m_.z > 0.29        # both top edges along X
n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_cube("M2B_Boxy2", bevel=_top_xx, seg=8, profile=0.8))
_exp = _boxy_vol(0.8) * 2 - 0.6 ** 3
_m2r["two parallel boxy bevels: both exact"] = b_.face_count() == 8 and sol_ and "2 bevels profile 0.8" in s_ \
    and abs(v_ - _exp) < 1e-5 * _exp
_corner2 = lambda m_: (abs(m_.x) < 1e-6 and m_.y < -0.29 and m_.z > 0.29) or (abs(m_.z) < 1e-6 and m_.x > 0.29 and m_.y < -0.29)
def _m2b_seams(b_):
    """Largest normal jump (degrees) across seams that touch a curved face: (tangent joins, creases excluded;
    patch-to-patch median)."""
    tang, pp = [0.0], []
    for e_ in b_.edges():
        fs_ = e_.faces()
        if len(fs_) != 2:
            continue
        curved = [not f_.surface().is_planar(1e-9) for f_ in fs_]
        if not any(curved):
            continue
        t0_, t1_ = e_.parameter_range()
        js_ = []
        for t_ in np.linspace(t0_, t1_, 9)[1:-1]:
            q_ = e_.point_at(t_)
            q_ = (q_.x, q_.y, q_.z) if hasattr(q_, "x") else tuple(q_)
            ns_ = []
            for f_ in fs_:
                w_ = _sm_k.surface_closest_point(f_.surface(), q_)[1][0]
                n2_ = np.array(f_.outward_normal_at_uv(float(w_.x), float(w_.y)), float)
                ns_.append(n2_ / np.linalg.norm(n2_))
            js_.append(math.degrees(math.acos(min(1.0, abs(float(ns_[0] @ ns_[1]))))))
        if max(js_) > 80:
            continue                                        # a crease (the bevel's end against a face)
        (pp if all(curved) else tang).append(max(js_) if not all(curved) else float(np.median(js_)))
    return max(tang), (float(np.median(pp)) if pp else 0.0)
from blender_nurbs.kernel import load as _klm
_sm_k = _klm()
n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_cube("M2B_BoxyCorner", bevel=_corner2, seg=6, profile=0.8))
_tg = _m2b_seams(b_)[0]
_m2r["two boxy bevels meeting at a box corner: exact strips mitred where they meet, tangent to the faces"] = \
    sol_ and b_.face_count() == 8 and "2 bevels profile 0.8" in s_ and no_ is None and _tg < 0.01
_ring = lambda m_: m_.z > 0.29 and (abs(m_.x) < 1e-6 or abs(m_.y) < 1e-6)
n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_cube("M2B_BoxyRing", bevel=_ring, seg=6, profile=0.8))
_m2r["the four top edges boxy: four exact strips mitred round the top"] = \
    sol_ and b_.face_count() == 10 and "4 bevels profile 0.8" in s_
for seg_ in (6, 12):
    n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_cube(f"M2B_Boxy3c{seg_}", bevel=_corner3, seg=seg_, profile=0.8))
    _tg, _pp = _m2b_seams(b_)
    _m2r[f"three boxy bevels at a corner ({seg_} segments): free-form patches through the corner, tangent to the "
         f"faces and to each other"] = sol_ and "free-form patch" in s_ and no_ is None and _tg < 0.01 and _pp < 0.01
n_, s_, no_, b_, v_, sol_ = _m2b_run(_m2b_cube("M2B_BoxyAll", bevel=lambda m_: True, seg=6, profile=0.8))
_m2r["all twelve edges boxy: a closed solid of the six faces and free-form patches"] = \
    sol_ and s_.startswith("6 planes") and "free-form patch" in s_ and no_ is None
# a bevel swept along a boxy bevel's profile (bevel an edge, then the outline of the face at its end)
_me = bpy.data.meshes.new("M2B_Swept")
_bs = _bmx.new()
_bmx.ops.create_cube(_bs, size=0.6)
_bmx.ops.bevel(_bs, geom=[e_ for e_ in _bs.edges if _top_x(0.5 * (e_.verts[0].co + e_.verts[1].co))], offset=0.15,
               offset_type='OFFSET', segments=8, profile=0.75, affect='EDGES')
_bs.faces.ensure_lookup_table()
_endf = max(_bs.faces, key=lambda f_: f_.calc_center_median().x)
_rim = [e_ for e_ in _endf.edges if not (max(v_.co.z for v_ in e_.verts) < -0.29 or max(v_.co.y for v_ in e_.verts) < -0.29)]
_bmx.ops.bevel(_bs, geom=_rim, offset=0.05, offset_type='OFFSET', segments=6, profile=0.5, affect='EDGES')
_bs.to_mesh(_me)
_bs.free()
_os = bpy.data.objects.new("M2B_Swept", _me)
ctx.scene.collection.objects.link(_os)
n_, s_, no_, b_, v_, sol_ = _m2b_run(_os)
_tg, _pp = _m2b_seams(b_)
_m2r["a bevel swept along a boxy bevel's profile: free-form patches, a closed solid, tangent where the mesh is smooth"] = \
    sol_ and "free-form patch" in s_ and _tg < 0.01 and _pp < 0.01
# the operator: the selected meshes, originals hidden, the new bodies selected; not for NURBS objects
_srcs = [bpy.data.objects["M2B_Cube"], bpy.data.objects["M2B_Corner"]]
for o_ in ctx.view_layer.objects:
    o_.select_set(o_ in _srcs)
ctx.view_layer.objects.active = _srcs[0]
_r = bpy.ops.nproto.mesh_to_brep()
_sel = [o_ for o_ in ctx.selected_objects]
_m2r["the operator converts the selected meshes, hides them and selects the new bodies"] = \
    _r == {'FINISHED'} and len(_sel) == 2 and all(store.is_nurbs(o_) for o_ in _sel) \
    and all(o_.hide_get() and o_.hide_render for o_ in _srcs)
_m2r["the operator needs a mesh selected (not a NURBS body)"] = not bpy.ops.nproto.mesh_to_brep.poll()
for k_, v_ in _m2r.items():
    check("Mesh to BREP: " + k_, v_)

# -- SubD to BREP ------------------------------------------------------------------------
from blender_nurbs import subd_brep as _sdb
def _sd_obj(name, build, crease=None, levels=2):
    """A mesh object built by build(bmesh) with a Subdivision Surface modifier; edges crease(mesh, edge) picks creased 1.0."""
    me_ = bpy.data.meshes.new(name)
    b_ = _bmx.new()
    build(b_)
    b_.to_mesh(me_)
    b_.free()
    o_ = bpy.data.objects.new(name, me_)
    ctx.scene.collection.objects.link(o_)
    if crease:
        a_ = me_.attributes.new("crease_edge", 'FLOAT', 'EDGE')
        for e_ in me_.edges:
            a_.data[e_.index].value = 1.0 if crease(me_, e_) else 0.0
    m_ = o_.modifiers.new("Subdivision", 'SUBSURF')
    m_.levels = m_.render_levels = levels
    return o_
def _sd_run(o_, **kw):
    new_, summ_, note_ = _sdb.convert(ctx, o_, **kw)
    return new_, summ_, note_, store.get_brep(new_)
def _sd_seams(b_, crease_deg=20.0):
    """Largest normal jump (degrees) along the smooth edges, their ends (the star points) included; the
    sharp ones (over crease_deg) left out and counted."""
    worst_, sharp_ = 0.0, 0
    for e_ in b_.edges():
        fs_ = e_.faces()
        if len(fs_) != 2:
            continue
        t0_, t1_ = e_.parameter_range()
        js_ = []
        for t_ in np.linspace(t0_, t1_, 13):
            q_ = e_.point_at(t_)
            q_ = (q_.x, q_.y, q_.z) if hasattr(q_, "x") else tuple(q_)
            ns_ = []
            for f_ in fs_:
                w_ = _sm_k.surface_closest_point(f_.surface(), q_)[1][0]
                n2_ = np.array(f_.outward_normal_at_uv(float(w_.x), float(w_.y)), float)
                ns_.append(n2_ / np.linalg.norm(n2_))
            js_.append(math.degrees(math.acos(min(1.0, float(ns_[0] @ ns_[1])))))
        if max(js_) > crease_deg:
            sharp_ += 1
        else:
            worst_ = max(worst_, max(js_))
    return worst_, sharp_
def _sd_dev(o_, b_, k_=250):
    """Largest distance from Blender's own limit surface (the modifier's vertices) to the body."""
    dg_ = ctx.evaluated_depsgraph_get()
    me_ = o_.evaluated_get(dg_).to_mesh()
    P_ = np.array([v_.co for v_ in me_.vertices], float)
    o_.evaluated_get(dg_).to_mesh_clear()
    P_ = P_[np.random.default_rng(1).choice(len(P_), min(k_, len(P_)), replace=False)]
    surfs_ = [f_.surface() for f_ in b_.faces()]
    out_ = 0.0
    for q_ in P_:
        out_ = max(out_, min(float(_sm_k.surface_closest_point(s2_, tuple(map(float, q_)))[2]) for s2_ in surfs_))
    return out_
_sdr = {}
_ocube = _sd_obj("SD_Cube", lambda b_: _bmx.ops.create_cube(b_, size=1.0), levels=4)
n_, s_, no_, b_ = _sd_run(_ocube)
_g1, _sh = _sd_seams(b_)
_dv = _sd_dev(_ocube, b_)
_cvs = [ext.surface_cvs(f_.surface()) for f_ in b_.faces()]
_sdr["a SubD cube (eight star points), G2 by default: six faces, a closed solid, G1 within 0.01 degrees along "
     "every edge and at the star points, within 0.1% of its size of the SubD surface"] = \
    b_.face_count() == 6 and b_.is_manifold_solid() and _g1 < 0.01 and _sh == 0 and _dv < 1e-3 * math.sqrt(3) \
    and no_ is None and "G2 within 0.00%" in s_
_sdr["... its faces by star points: degree 6, 12 x 12 CVs, knots [0^7, .5^5, 1^7] (C1 between 2 x 2 pieces)"] = \
    all(c_["degree_u"] == 6 and c_["degree_v"] == 6 and c_["count_u"] == 12 and c_["count_v"] == 12
        and np.allclose(c_["knots_u"], [0.0] * 7 + [0.5] * 5 + [1.0] * 7) for c_ in _cvs)
print("   SubD cube:", s_, f"| measured G1 {_g1:.4f} deg, deviation {_dv * 1000:.3f} mm")
def _torus(b_, R_=1.0, r_=0.4, nu_=8, nv_=6):
    vs_ = [b_.verts.new(((R_ + r_ * math.cos(v2_)) * math.cos(u2_), (R_ + r_ * math.cos(v2_)) * math.sin(u2_), r_ * math.sin(v2_)))
           for u2_ in np.linspace(0, 2 * math.pi, nu_ + 1)[:-1] for v2_ in np.linspace(0, 2 * math.pi, nv_ + 1)[:-1]]
    for i_ in range(nu_):
        for j_ in range(nv_):
            b_.faces.new([vs_[i_ * nv_ + j_], vs_[((i_ + 1) % nu_) * nv_ + j_], vs_[((i_ + 1) % nu_) * nv_ + (j_ + 1) % nv_],
                          vs_[i_ * nv_ + (j_ + 1) % nv_]])
_otor = _sd_obj("SD_Torus", _torus, levels=4)
n_, s_, no_, b_ = _sd_run(_otor)
_g1, _sh = _sd_seams(b_)
_sdr["an all-ordinary cage (a torus): every face one bicubic span (16 CVs), the SubD exactly, C2 across edges"] = \
    b_.face_count() == 48 and b_.is_manifold_solid() and "one bicubic span each" in s_ and _g1 < 1e-3 \
    and all(ext.surface_cvs(f_.surface())["count_u"] == 4 and ext.surface_cvs(f_.surface())["count_v"] == 4 for f_ in b_.faces()) \
    and _sd_dev(_otor, b_, 120) < 1e-5
_ocr = _sd_obj("SD_Crease", lambda b_: _bmx.ops.create_cube(b_, size=1.0),
               crease=lambda me_, e_: all(me_.vertices[v_].co.z > 0.4 for v_ in e_.vertices), levels=4)
n_, s_, no_, b_ = _sd_run(_ocr)
_g1, _sh = _sd_seams(b_)
_sdr["a creased top ring (crease 1.0): its four edges sharp, the others G1 and G2, a closed solid near the SubD surface"] = \
    b_.face_count() == 6 and b_.is_manifold_solid() and _sh == 4 and _g1 < 0.01 and _sd_dev(_ocr, b_) < 1e-3 * math.sqrt(3) \
    and no_ is None
print("   creased cube:", s_, f"| measured G1 {_g1:.4f} deg, {_sh} sharp edges")
_ocyl = _sd_obj("SD_Cyl", lambda b_: _bmx.ops.create_cone(b_, cap_ends=True, segments=8, radius1=0.5, radius2=0.5, depth=1.0))
n_, s_, no_, b_ = _sd_run(_ocyl)
_g1, _sh = _sd_seams(b_)
_sdr["n-gon caps (a cylinder): one face per side quad, the caps split in eight (16 T-junctions where they meet), "
     "a closed G2 solid, G1 within 0.01 degrees along every edge (the T-junctions' too)"] = \
    b_.face_count() == 24 and b_.is_manifold_solid() and _g1 < 0.01 and _sh == 0 and no_ is None \
    and "16 T-junctions" in s_ and _sd_dev(_ocyl, b_) < 3e-3 * math.sqrt(2)
print("   n-gon cylinder:", s_, f"| measured G1 {_g1:.4f} deg")
_ouvs = _sd_obj("SD_UVSphere", lambda b_: _bmx.ops.create_uvsphere(b_, u_segments=8, v_segments=6, radius=0.5))
n_, s_, no_, b_ = _sd_run(_ouvs)
_g1, _sh = _sd_seams(b_)
_sdr["triangle fans (a UV sphere): 32 quad faces, the 16 triangles in three each, a closed G2 solid, G1 within "
     "0.01 degrees"] = b_.face_count() == 80 and b_.is_manifold_solid() and _g1 < 0.01 and _sh == 0 and no_ is None
def _open_cyl(b_):
    _bmx.ops.create_cone(b_, cap_ends=True, segments=6, radius1=0.5, radius2=0.5, depth=1.0)
    _bmx.ops.delete(b_, geom=[f_ for f_ in b_.faces if len(f_.verts) > 4 and f_.calc_center_median().z > 0],
                    context='FACES_ONLY')
_oop = _sd_obj("SD_OpenCyl", _open_cyl)
_V, _F, _S, _M = _sdb.cage(_oop, ctx.evaluated_depsgraph_get())
_n2, _k2, _s2, _c2 = _sdb.build(_V, _F, _S)
_sdr["an n-gon by an open border: one face per quad, the n-gon split (12 faces), not the whole cage split (30)"] = \
    len(_n2) == 12 and _s2["layout_fallback"] is None and _s2["normal"] < 1e-3 and _s2["curvature"] < 1e-4
# open grids: a border running straight through is no star point; Keep Corners (the modifier's Boundary Smooth)
# pins the corners, so the grid is all bicubic, Blender's own surface exactly
def _sd_grid(b_, n_=3, m_=2):
    vs_ = [[b_.verts.new((i_, j_, 0.4 * math.sin(1.3 * i_) * math.cos(0.9 * j_) + 0.1 * i_ * j_)) for j_ in range(m_ + 1)]
           for i_ in range(n_ + 1)]
    for i_ in range(n_):
        for j_ in range(m_):
            b_.faces.new([vs_[i_][j_], vs_[i_ + 1][j_], vs_[i_ + 1][j_ + 1], vs_[i_][j_ + 1]])
_ogr = _sd_obj("SD_Grid", _sd_grid, levels=4)
n_, s_, no_, b_ = _sd_run(_ogr)
_sdr["an open grid, borders smoothed (All): only its four corners are star points"] = \
    b_.face_count() == 6 and "by the 4 star points" in s_ and "G2 within 0.00%" in s_
_ogr.modifiers[0].boundary_smooth = 'PRESERVE_CORNERS'
n_, s_, no_, b_ = _sd_run(_ogr)
_cvs = [ext.surface_cvs(f_.surface()) for f_ in b_.faces()]
_sdr["... with Keep Corners (as its modifier says): six bicubic faces of one span, Blender's SubD surface exactly"] = \
    b_.face_count() == 6 and "one bicubic span each" in s_ and no_ is None \
    and all(c_["degree_u"] == 3 and c_["count_u"] == 4 and c_["count_v"] == 4 for c_ in _cvs) and _sd_dev(_ogr, b_) < 1e-5
n_, s_, no_, b_ = _sd_run(_ogr, keep_corners=False)
_sdr["... the modifier's setting overridden (keep_corners=False): the corners smoothed, star points again"] = \
    "by the 4 star points" in s_
_V, _F, _S, _M = _sdb.cage(_ogr, ctx.evaluated_depsgraph_get())
_c0 = next(v_ for v_ in range(len(_V)) if sum(v_ in f_ for f_ in _F) == 1)
_f0 = next(f_ for f_ in _F if _c0 in f_)
_k0 = _f0.index(_c0)
_F3 = [(_c0, _f0[(_k0 + 1) % 4], _f0[(_k0 + 3) % 4]), tuple(_f0[(_k0 + i_) % 4] for i_ in (1, 2, 3))] \
    + [f_ for f_ in _F if f_ != _f0]
_n2, _k2, _s2, _c2 = _sdb.build(_V, _F3, set(), keep_corners=True)
_sdr["... a triangle in a corner with Keep Corners: its corner stays on the cage vertex"] = \
    min(float(np.linalg.norm(np.asarray(net_)[c_] - _V[_c0])) for net_ in _n2
        for c_ in ((0, 0), (-1, 0), (0, -1), (-1, -1))) < 1e-9
# G1 and G2 on the converter's own measures (Continuity Check's): exact along every edge and at the star points,
# close to the SubD; a face with no star corner keeps the SubD's own bicubic span, so its joins are C2
for _nm, _ob in (("a SubD cube", _ocube), ("n-gon caps (a cylinder)", _ocyl), ("a creased cube", _ocr)):
    _V, _F, _S, _M = _sdb.cage(_ob, ctx.evaluated_depsgraph_get())
    _res, _out = {}, {}
    for _g2 in (False, True):
        _n2, _k2, _s2, _c2 = _sdb.build(_V, _F, _S, g2=_g2)
        _res[_g2], _out[_g2] = _s2, (_n2, _k2, _c2)
        print("  ", "G2" if _g2 else "G1", _nm, {k_: v_ for k_, v_ in _s2.items() if k_ != "solver"},
              "newton", len(_s2["solver"].get("newton", [])), f"{_s2['solver'].get('seconds', 0):.2f}s")
        _ok = _s2["normal"] <= 0.001 and _s2["normal_all"] <= 0.01 and _s2["deviation"] < 3e-3 * _c2.size \
            and _s2["fallback"] is None
        if _g2:
            _ok = _ok and _s2["curvature"] <= 1e-4
        _sdr[f"{'G2' if _g2 else 'G1'} on {_nm}: G1 within 0.001 deg across every edge (0.01 at the star points)"
             + (", G2 within 0.01 %" if _g2 else "") + ", within 0.3 % of its size of the SubD surface"] = _ok
    _n1, _k1, _c1 = _out[False]
    _six = {f_ for f_, (ku_, _) in enumerate(_k1) if len(ku_) != 8}
    _ed = {e_ for e_, u_ in _c1.edges.items() if len(u_) == 2 and e_ not in _S and any(f_ in _six for f_, _, _ in u_)}
    _cg1 = max((j_ for j_, _ in _c1.curvature_jumps(_n1, _k1, edges=_ed).values()), default=0.0)
    _sdr[f"G2 on {_nm} bends alike across the edges at least as well as G1 (G1 {100 * _cg1:.2f} %)"] = \
        _res[True]["curvature"] <= _cg1 + 1e-9
    _sdr[f"G1 on {_nm} is the lighter net ({_res[False]['cvs']} CVs, G2 {_res[True]['cvs']})"] = \
        _res[False]["cvs"] < 0.8 * _res[True]["cvs"]
new_, s_, no_, b_ = _sd_run(_ocube, g2=False)
_g1, _sh = _sd_seams(b_)
_cvs = [ext.surface_cvs(f_.surface()) for f_ in b_.faces()]
_sdr["G1 builds a closed solid of degree-5 faces by the star points (10 x 10 CVs, knots [0^6, .5^4, 1^6]), "
     "its summary leaves G2 out"] = \
    b_.is_manifold_solid() and "G1 within" in s_ and "G2" not in s_ and _g1 < 0.01 and no_ is None \
    and all(c_["degree_u"] == 5 and c_["count_u"] == 10 and c_["count_v"] == 10
            and np.allclose(c_["knots_v"], [0.0] * 6 + [0.5] * 4 + [1.0] * 6) for c_ in _cvs)
# G2 at degree 5 (Degree: 5 · Lighter): G1's net, still exactly G2, a little further from the SubD than degree 6
for _nm, _ob in (("a SubD cube", _ocube), ("n-gon caps (a cylinder)", _ocyl)):
    _V, _F, _S, _M = _sdb.cage(_ob, ctx.evaluated_depsgraph_get())
    _n5, _k5, _s5, _c5 = _sdb.build(_V, _F, _S, g2=True, degree=5)
    _n6, _k6, _s6, _c6 = _sdb.build(_V, _F, _S, g2=True)
    _sdr[f"G2 at degree 5 on {_nm}: degree-5 faces by star points, G1 within 0.001 deg (0.01 at the star points), "
         f"G2 within 0.01 %, within 0.3 % of its size of the SubD, fewer CVs than degree 6 "
         f"({_s5['cvs']} vs {_s6['cvs']})"] = \
        _s5["degree"] == 5 and _s5["normal"] <= 0.001 and _s5["normal_all"] <= 0.01 and _s5["curvature"] <= 1e-4 \
        and _s5["deviation"] < 3e-3 * _c5.size and _s5["fallback"] is None and _s5["cvs"] < 0.75 * _s6["cvs"]
new_, s_, no_, b_ = _sd_run(_ocube, g2=True, degree=5)
_cvs = [ext.surface_cvs(f_.surface()) for f_ in b_.faces()]
_sdr["G2 at degree 5 builds a closed solid of 10 x 10 degree-5 faces, its summary reports G2"] = \
    b_.is_manifold_solid() and "G2 within 0.00%" in s_ and no_ is None \
    and all(c_["degree_u"] == 5 and c_["count_u"] == 10 for c_ in _cvs)
# Fast (G2): the star points' first rings shaped directly and held, the solve only the rest; n-gon cages ignore it
_V, _F, _S, _M = _sdb.cage(_ocube, ctx.evaluated_depsgraph_get())
_nf, _kf, _sf, _cf = _sdb.build(_V, _F, _S, g2=True, fast=True)
_n6, _k6, _s6, _c6 = _sdb.build(_V, _F, _S, g2=True)
_sdr[f"Fast on a SubD cube: degree-6 faces, G1 within 0.001 deg (0.01 at the star points), G2 within 0.01 %, within "
     f"0.3 % of its size of the SubD ({100 * _sf['deviation'] / _cf.size:.3f} %), fewer unknowns "
     f"({_sf['solver']['unknowns']} vs {_s6['solver']['unknowns']})"] = \
    _sf["degree"] == 6 and _sf["normal"] <= 0.001 and _sf["normal_all"] <= 0.01 and _sf["curvature"] <= 1e-4 \
    and _sf["deviation"] < 3e-3 * _cf.size and _sf["fallback"] is None \
    and _sf["solver"]["unknowns"] < _s6["solver"]["unknowns"]
for _d in (6, 5):
    _V, _F, _S, _M = _sdb.cage(_ocyl, ctx.evaluated_depsgraph_get())
    _nf, _kf, _sf, _cf = _sdb.build(_V, _F, _S, g2=True, degree=_d, fast=True)
    _n6, _k6, _s6, _c6 = _sdb.build(_V, _F, _S, g2=True, degree=_d)
    _sdr[f"Fast on n-gon caps (a cylinder), degree {_d}: ignored, the same faces as without it"] = \
        len(_nf) == len(_n6) and all(np.allclose(a_, b_, rtol=0.0, atol=1e-7 * _cf.size) for a_, b_ in zip(_nf, _n6))   # (the sparse solve repeats only to ~1e-9)
_sdr["the operator offers Fast for G2, off by default"] = \
    bpy.ops.nproto.subd_to_brep.get_rna_type().properties["fast"].default is False
# Local Re-solve: the same cage with a vertex moved solves again only the faces near it, from the last solve
_lrV, _lrF, _lrS, _lrM = _sdb.cage(_ocube, ctx.evaluated_depsgraph_get())
for _ in range(2):                                                 # a finer cage: star points far apart
    _lrV, _lrF, _lrS = _sdb.refine_any(np.asarray(_lrV, float), _lrF, _lrS, False)
    _lrF = [tuple(int(x) for x in f_) for f_ in _lrF]
    _lrS = _sdb.with_borders(_lrF, _lrS)
_lrn0, _lrk0, _lrs0, _lrc0 = _sdb.build(_lrV, _lrF, _lrS, g2=True)
_lrx = sorted(_lrc0.irr)[0]
_lry = [y_ for f_ in _lrc0.vf[_lrx] for y_ in _lrF[f_] if y_ not in _lrc0.irr][0]   # a vertex next to a star point
_lrV2 = np.array(_lrV, float)
_lrV2[_lry] += 0.03 * _lrc0.size * _lrc0.cage_normals[_lry]
_lrnl, _lrkl, _lrsl, _lrcl = _sdb.build(_lrV2, _lrF, _lrS, g2=True, prev=_lrs0["solver"]["state"])
_lrnf, _lrkf, _lrsf, _lrcf = _sdb.build(_lrV2, _lrF, _lrS, g2=True)
_lrlo = _lrsl["solver"]["local"]
_lrfar = [f_ for f_ in range(len(_lrF)) if min(np.linalg.norm(np.asarray(_lrV)[list(_lrF[f_])] - _lrV2[_lry], axis=1)) > 0.3 * _lrc0.size]
_sdr[f"Local Re-solve on a finer cube ({len(_lrF)} faces), a vertex by a star point moved: "
     f"{_lrlo and _lrlo['faces']} of {_lrlo and _lrlo['of']} star faces solved again, G1 within 0.001 deg (0.01 at the star "
     f"points), G2 within 0.01 %, as near the SubD as a full solve, faces far off unchanged, within 0.1 % of the "
     f"full solve's CVs"] = \
    _lrlo is not None and not _lrlo["fallback"] and _lrlo["faces"] < 0.5 * _lrlo["of"] \
    and _lrsl["normal"] <= 0.001 and _lrsl["normal_all"] <= 0.01 and _lrsl["curvature"] <= 1e-4 \
    and _lrsl["deviation"] <= 1.1 * _lrsf["deviation"] + 1e-6 * _lrcf.size and bool(_lrfar) \
    and all(np.array_equal(_lrnl[f_], _lrn0[f_]) for f_ in _lrfar) \
    and max(float(np.abs(a_ - b_).max()) for a_, b_ in zip(_lrnl, _lrnf)) < 1e-3 * _lrcf.size
_sdr["the operator offers Local Re-solve, on by default"] = \
    bpy.ops.nproto.subd_to_brep.get_rna_type().properties["reuse"].default is True
# Live Link: the body keeps following its SubD (construction history), trims on it applied again on top
_lvo = _sd_obj("SD_Live", lambda b_: _bmx.ops.create_cube(b_, size=2.0), levels=2)
_lvb, _lvs, _lvn = _sdb.convert(ctx, _lvo, live=True)[:3]
_lvh = features.history(_lvb)
_sdr["Live Link: the body records SubD to BREP with its SubD as input; nothing changed, nothing rebuilt"] = \
    _lvh is not None and _lvh["op"] == "subd_to_brep" and _lvh["inputs"][0][0] is _lvo \
    and features.rebuild(_lvb) is False
_lvt0 = _lvb[store.PROP_BREP]
_lvo.data.vertices[0].co.z -= 0.4
_lvo.data.update()
_lvr = features.rebuild(_lvb)
_lvg, _lvsh = _sd_seams(store.get_brep(_lvb))
_sdr[f"Live Link: a cage vertex moved (Object Mode): built again, still G1 ({_lvg:.4f} deg)"] = \
    _lvr is True and _lvb[store.PROP_BREP] != _lvt0 and _lvg < 0.01 and _lvsh == 0
for o_ in ctx.view_layer.objects:
    o_.select_set(o_ is _lvo)
ctx.view_layer.objects.active = _lvo
bpy.ops.object.mode_set(mode='EDIT')
_lvbm = _bmx.from_edit_mesh(_lvo.data)
_lvbm.verts.ensure_lookup_table()
_lvbm.verts[1].co.x += 0.3
_bmx.update_edit_mesh(_lvo.data)
_lvt1 = _lvb[store.PROP_BREP]
_lvr = features.rebuild(_lvb)
bpy.ops.object.mode_set(mode='OBJECT')
_sdr["Live Link: a cage vertex moved in Edit Mode: built again from the mesh being edited"] = \
    _lvr is True and _lvb[store.PROP_BREP] != _lvt1
_lvl = poly("SD_LiveCut_ln", [(-3, -3, 0.3), (3, -3, 0.3)])
bpy.ops.nproto.extrude(inputs=ins((_lvl, "S0")), distance=6.0, direction="Y")
_lvw = ctx.active_object
bpy.ops.nproto.trim(target=_lvb.name, cutters=json.dumps([[_lvw.name, "SURF"]]), ref_point=(0, 0, -0.7),
                    normal=(0, 0, 1), action="KEEP")
def _lvz(o_):
    zs_ = [(o_.matrix_world @ v_.co).z for v_ in o_.data.vertices]
    return max(zs_), min(zs_)
_lvz0 = _lvz(_lvb)
_lvlow = min(range(len(_lvo.data.vertices)), key=lambda i_: _lvo.data.vertices[i_].co.z)
_lvo.data.vertices[_lvlow].co.z -= 0.5
_lvo.data.update()
features.rebuild(_lvb)
_lvz1 = _lvz(_lvb)
_sdr[f"Live Link: trimmed by a plane at z 0.3, then a low cage vertex moved: built again with the trim on top "
     f"(top {_lvz1[0]:.4f}, bottom {_lvz0[1]:.3f} -> {_lvz1[1]:.3f})"] = \
    [e_["op"] for e_ in features.stack(_lvb)] == ["trim"] and abs(_lvz0[0] - 0.3) < 1e-4 and abs(_lvz1[0] - 0.3) < 1e-4 \
    and _lvz1[1] < _lvz0[1] - 0.05
_lvp = _sd_obj("SD_LivePlain", lambda b_: _bmx.ops.create_cube(b_, size=2.0))
_lvp.modifiers.remove(_lvp.modifiers[0])                           # a plain quad mesh, no Subdivision modifier
_lvpb = _sdb.convert(ctx, _lvp, live=True)[0]
_lvpt = _lvpb[store.PROP_BREP]
_lvp.data.vertices[0].co.z -= 0.4
_lvp.data.update()
_sdr["Live Link from a plain quad mesh (no Subdivision modifier): a cage vertex moved, built again"] = \
    features.rebuild(_lvpb) is True and _lvpb[store.PROP_BREP] != _lvpt
# SubD Wait: a live SubD body is built again only once its cage has been still a while (not mid-drag)
_lvpt = _lvpb[store.PROP_BREP]
_lvp.data.vertices[1].co.z += 0.3
_lvp.data.update()
ctx.view_layer.update()                                            # (the depsgraph handler notes the change)
_lvq = history._process()                                          # right after the change: it waits
_lvq_waited = _lvpb[store.PROP_BREP] == _lvpt and isinstance(_lvq, float) and _lvq > 0 and _lvpb.name in history._dirty
history._last_change[0] -= 10.0                                    # ... the cage has been still since
_lvq2 = history._process()
_sdr[f"SubD Wait: a cage change rebuilds the live body only after a pause (waited {_lvq or 0:.2f} s, then built)"] = \
    _lvq_waited and _lvpb[store.PROP_BREP] != _lvpt and _lvpb.name not in history._dirty \
    and abs(ctx.scene.nproto_settings.subd_delay - 0.3) < 1e-6
_sdr["the operator offers Live Link, on by default"] = \
    bpy.ops.nproto.subd_to_brep.get_rna_type().properties["live"].default is True
# the operator: the selected meshes, originals hidden, the new bodies selected; not for NURBS objects
for o_ in ctx.view_layer.objects:
    o_.select_set(o_ in (_ocube, _otor))
ctx.view_layer.objects.active = _ocube
_sdr["the operator offers Degree 6 (default) or 5 for G2"] = \
    [i_.identifier for i_ in bpy.ops.nproto.subd_to_brep.get_rna_type().properties["degree"].enum_items] == ["6", "5"] \
    and bpy.ops.nproto.subd_to_brep.get_rna_type().properties["degree"].default == "6"
_sdr["the operator's Boundary Smooth: Modifier (default), All or Keep Corners"] = \
    [i_.identifier for i_ in bpy.ops.nproto.subd_to_brep.get_rna_type().properties["boundary"].enum_items] \
    == ["MODIFIER", "ALL", "PRESERVE_CORNERS"] \
    and bpy.ops.nproto.subd_to_brep.get_rna_type().properties["boundary"].default == "MODIFIER"
_r = bpy.ops.nproto.subd_to_brep(angle=0.1)
_sel = list(ctx.selected_objects)
_sdr["the operator converts the selected SubD meshes, hides them and selects the new bodies"] = \
    _r == {'FINISHED'} and len(_sel) == 2 and all(store.is_nurbs(o_) for o_ in _sel) \
    and all(o_.hide_get() and o_.hide_render for o_ in (_ocube, _otor))
_sdr["the operator needs a mesh selected (not a NURBS body)"] = not bpy.ops.nproto.subd_to_brep.poll()
for k_, v_ in _sdr.items():
    check("SubD to BREP: " + k_, v_)

# -- Analysis meshes: zebra / draft / matcap / curvature map on their own mesh, built in numpy from the
#    control nets (surfeval, anamesh) -- exact normals and curvatures at every vertex, fast
from blender_nurbs import anamesh as _am, surfeval as _se, diagnostics as _dgA, square as _sqA
import time as _tmA
_cubeA = next(o_ for o_ in _sel if store.get_brep(o_).face_count() == 6)
_torA = next(o_ for o_ in _sel if store.get_brep(o_).face_count() == 48)
bpy.ops.nproto.add_primitive(kind="SPHERE", radius=2.0, degree="0"); _sphA = ctx.active_object
bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0, degree="0", caps=True); _cylA = ctx.active_object
bpy.ops.curve.primitive_bezier_circle_add(radius=1.0, location=(0, 0, 0)); _ccA = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); _ccA.select_set(True)
bpy.ops.nproto.planar(); _plA = ctx.active_object
def _diagA(o_):
    bb_ = store.get_brep(o_).bounding_box()
    return float(np.linalg.norm(np.array(bb_[1]) - np.array(bb_[0])))
# the evaluator is the kernel's surface: points and first derivatives to round-off, second derivatives
# as the finite differences give them
_e1, _e2 = 0.0, 0.0
for _o in (_cubeA, _sphA, _plA, _cylA):
    _dA = _diagA(_o)
    for _f in store.get_brep(_o).faces():
        _s = _f.surface(); _E = _se.Surf(_s)
        (_u0, _u1), (_v0, _v1) = _s.u_domain(), _s.v_domain()
        _rg = np.random.default_rng(7)
        _uu = np.r_[_u0, _u1, _rg.uniform(_u0, _u1, 10)]; _vv = np.r_[_v1, _v0, _rg.uniform(_v0, _v1, 10)]
        _D = _E.at(_uu, _vv, 2); _G = _E.grid(_uu, _vv, 1)
        for _i in range(len(_uu)):
            _pt, _su, _sv = (np.array(x_) for x_ in _s.derivatives(float(_uu[_i]), float(_vv[_i])))
            _sc = np.linalg.norm(_su) + np.linalg.norm(_sv) + 1e-9 * _dA
            _e1 = max(_e1, np.linalg.norm(_D[0, 0][_i] - _pt) / _dA, np.linalg.norm(_G[0, 0][_i, _i] - _pt) / _dA,
                      (np.linalg.norm(_D[1, 0][_i] - _su) + np.linalg.norm(_D[0, 1][_i] - _sv)) / _sc,
                      (np.linalg.norm(_G[1, 0][_i, _i] - _su) + np.linalg.norm(_G[0, 1][_i, _i] - _sv)) / _sc)
            if _i >= 2 and _o is _sphA:
                _, _, _suu, _suv, _svv = _sqA._second(_s, float(_uu[_i]), float(_vv[_i]))
                _e2 = max(_e2, (np.linalg.norm(_D[2, 0][_i] - _suu) + np.linalg.norm(_D[1, 1][_i] - _suv)
                                + np.linalg.norm(_D[0, 2][_i] - _svv)) / (np.linalg.norm(_suu) + np.linalg.norm(_svv)))
check("analysis mesh: the numpy evaluator is the kernel's surface (points / first derivatives to 1e-9, "
      "second derivatives as finite differences)", _e1 < 1e-9 and _e2 < 1e-4, f"first {_e1:.1e}, second {_e2:.1e}")
_fl = {n_: [w_ for _, w_, _ in _am._faces(store.get_brep(o_), None, _diagA(o_), None)]
       for n_, o_ in (("cube", _cubeA), ("torus", _torA), ("planar", _plA))}
check("analysis mesh: SubD faces are meshed as untrimmed (u, v) grids, a planar circle as a trimmed face",
      all(_fl["cube"]) and all(_fl["torus"]) and not any(_fl["planar"]), str({k_: sum(v_) for k_, v_ in _fl.items()}))
# area: the mesh covers each body (a flat face with a curved outline too, the planar circle)
_ar = {}
for _n, _o in (("cube", _cubeA), ("torus", _torA), ("sphere", _sphA), ("cylinder", _cylA), ("planar", _plA)):
    _zP, _zN, _zT = _dgA.zebra_mesh(_o, 1.0, 0.0005)
    _zP = np.asarray(_zP, float)
    _ma = 0.5 * np.linalg.norm(np.cross(_zP[_zT[:, 1]] - _zP[_zT[:, 0]], _zP[_zT[:, 2]] - _zP[_zT[:, 0]]), axis=1).sum()
    _ar[_n] = abs(_ma - store.get_brep(_o).area(1e-7)) / store.get_brep(_o).area(1e-7)
check("analysis mesh: covers the exact faces (area within 0.1 %, a planar circle's curved outline included)",
      max(_ar.values()) < 1e-3, " ".join(f"{k_} {v_:.1e}" for k_, v_ in _ar.items()))
# outward normals and convex curvature on a closed solid, whatever each face's parameter direction
_zP, _zN, _zT = _dgA.zebra_mesh(_cylA, 1.0, 0.0005)
_zP, _zN = np.asarray(_zP, float), np.asarray(_zN, float)
_out = np.sum(_zN * (_zP - _zP.mean(axis=0)), axis=1)
_cP, _cN, _cT, _cV = _dgA.face_curvature_mesh(_cylA, 5.0, 0.002)
_cN = np.asarray(_cN, float)
_lat, _cap = np.abs(_cN[:, 2]) < 0.05, np.abs(_cN[:, 2]) > 0.999
_sP, _sN, _sT, _sV = _dgA.face_curvature_mesh(_sphA, 5.0, 0.002)
check("analysis mesh: a capped cylinder's normals all point out; mean curvature +0.5 on its side (convex), "
      "0 on its caps; a sphere r=2: +0.5 and Gaussian 0.25 at every vertex, the poles too",
      _out.min() > 0 and _lat.any() and _cap.any() and np.abs(_cV["MEAN"][_lat] - 0.5).max() < 1e-3
      and np.abs(_cV["MEAN"][_cap]).max() < 1e-6 and np.abs(_sV["MEAN"] - 0.5).max() < 1e-3
      and np.abs(_sV["GAUSS"] - 0.25).max() < 1e-3,
      f"min n.(p - c) {_out.min():.3f}; side H {_cV['MEAN'][_lat].min():.4f}..{_cV['MEAN'][_lat].max():.4f}; "
      f"sphere H {_sV['MEAN'].min():.4f}..{_sV['MEAN'].max():.4f}")
# stripes: the normals turn by no more than the tolerance along any mesh edge, no folded triangles
_zP, _zN, _zT = _dgA.zebra_mesh(_cubeA, 1.0, 0.0005)
_zP, _zN = np.asarray(_zP, float), np.asarray(_zN, float)
_eang = np.degrees(np.arccos(np.clip(np.concatenate([np.sum(_zN[_zT[:, i]] * _zN[_zT[:, j]], axis=1)
                                                     for i, j in ((0, 1), (1, 2), (2, 0))]), -1, 1)))
_fn = np.cross(_zP[_zT[:, 1]] - _zP[_zT[:, 0]], _zP[_zT[:, 2]] - _zP[_zT[:, 0]])
_ar_ = np.linalg.norm(_fn, axis=1); _ok_ = _ar_ > 1e-3 * np.median(_ar_)
_cos = np.sum(_fn * (_zN[_zT[:, 0]] + _zN[_zT[:, 1]] + _zN[_zT[:, 2]]), axis=1)[_ok_] / (
    _ar_[_ok_] * np.linalg.norm((_zN[_zT[:, 0]] + _zN[_zT[:, 1]] + _zN[_zT[:, 2]])[_ok_], axis=1))
check("analysis mesh: SubD cube at 1 deg -- normals within 1.6 deg along every mesh edge (p99.9: a grid cell's "
      "diagonal, its sides within 1 deg each), none folded",
      float(np.percentile(_eang, 99.9)) < 1.6 and int(np.sum(_cos < 0.3)) == 0,
      f"p99.9 {np.percentile(_eang, 99.9):.2f} deg, max {_eang.max():.2f}, folded {int(np.sum(_cos < 0.3))}, {len(_zT)} tris")
# speed: well ahead of the kernel's tessellation at the same fineness; a second quality reuses the face set-up
_am._prep.clear()
_t0 = _tmA.perf_counter(); _zt1 = _dgA.zebra_mesh(_torA, 1.0, 0.0005); _t_new = _tmA.perf_counter() - _t0
_t0 = _tmA.perf_counter(); _zt2 = _dgA.zebra_mesh(_torA, 2.0, 0.0005); _t_again = _tmA.perf_counter() - _t0
_t0 = _tmA.perf_counter(); _kt = store._tessellate(store.get_brep(_torA), 0.0005, 1.0); _t_kern = _tmA.perf_counter() - _t0
check("analysis mesh: a 48-face SubD torus at 1 deg builds in well under a second, 5x+ faster than the kernel's "
      "mesh as fine; another quality reuses the per-face set-up (faster again, fewer triangles at 2 deg)",
      _t_new < 1.0 and _t_kern > 5 * _t_new and _t_again < _t_new and len(_zt2[2]) < len(_zt1[2])
      and _am._prep[_torA.session_uid][0] == _torA.get(store.PROP_REV, 0),
      f"new {_t_new * 1000:.0f} ms ({len(_zt1[2])} tris), again at 2 deg {_t_again * 1000:.0f} ms, "
      f"kernel {_t_kern * 1000:.0f} ms ({len(_kt[2])} tris)")
print(f"   zebra mesh, SubD torus: {_t_new * 1000:.0f} ms, kernel at the same angle {_t_kern * 1000:.0f} ms")
# quality presets: Display (four tessellation values, one retessellation) and Analysis (zebra / curvature)
_sA = ctx.scene.nproto_settings
_vA = ctx.window_manager.nproto_view
_calls = []
_orig_rt = store.retessellate
store.retessellate = lambda o_, c_=None: _calls.append(o_.name)
try:
    _q0 = _sA.display_quality
    _sA.display_quality = 'FAST'
    _q1 = (_sA.display_quality, round(_sA.angle_tolerance, 4), round(_sA.chord_tolerance, 6),
           round(_sA.edge_angle_tolerance, 4), round(_sA.max_edge_rel, 4), len(_calls), len(set(_calls)))
    _orig_rt(_cubeA, ctx); _tri_fast = len(_cubeA.data.polygons)
    _sA.angle_tolerance = 7.0
    _q2 = _sA.display_quality
    _sA.display_quality = 'CUSTOM'
    _q3 = round(_sA.angle_tolerance, 4)
    _sA.display_quality = 'MEDIUM'
    _q4 = (_sA.display_quality, round(_sA.angle_tolerance, 4), round(_sA.chord_tolerance, 6))
finally:
    store.retessellate = _orig_rt
    for k_ in ("chord_tolerance", "angle_tolerance", "edge_angle_tolerance", "max_edge_rel"):
        _sA.property_unset(k_)
_orig_rt(_cubeA, ctx); _tri_med = len(_cubeA.data.polygons)
_nn = sum(1 for o_ in ctx.scene.objects if store.is_nurbs(o_))
check("display quality: Medium on the defaults; Fast sets the four values with one retessellation of each body "
      "(fewer triangles); a hand-set value reads Custom, choosing Custom changes nothing; Medium restores the defaults",
      _q0 == 'MEDIUM' and _q1 == ('FAST', 10.0, 0.001, 3.0, 0.05, _nn, _nn) and _tri_fast < _tri_med
      and _q2 == 'CUSTOM' and _q3 == 7.0 and _q4 == ('MEDIUM', 5.0, 0.0005),
      f"{_q0} {_q1} (of {_nn}) tris fast {_tri_fast} medium {_tri_med} {_q2} {_q3} {_q4}")
try:
    _a0 = _vA.analysis_quality
    _vA.analysis_quality = 'FINE'
    _a1 = (_vA.analysis_quality, round(_vA.zebra_angle, 4), round(_vA.zebra_chord, 6), round(_vA.curv_angle, 4),
           round(_vA.curv_chord, 6))
    _vA.zebra_angle = 0.7
    _a2 = _vA.analysis_quality
    _vA.analysis_quality = 'FAST'
    _a3 = (_vA.analysis_quality, round(_vA.zebra_angle, 4), round(_vA.curv_angle, 4))
    _vA.show_zebra = True
    _pz = [e["prop"] for e in _draw(_ui.VIEW3D_PT_nproto_analyze_side) if "prop" in e]
finally:
    for k_ in ("zebra_angle", "zebra_chord", "curv_angle", "curv_chord", "show_zebra"):
        _vA.property_unset(k_)
_nf = len(_dgA.zebra_mesh(_cubeA, 2.0, 0.001)[2]), len(_dgA.zebra_mesh(_cubeA, 0.5, 0.0002)[2])
check("analysis quality: Medium on the defaults, Fine 0.5 / 2.5 deg, a hand-set value reads Custom, Fast 2 / 10 deg; "
      "the Analyze panel offers it (Fast meshes fewer triangles than Fine)",
      _a0 == 'MEDIUM' and _a1 == ('FINE', 0.5, 0.0002, 2.5, 0.001) and _a2 == 'CUSTOM'
      and _a3 == ('FAST', 2.0, 10.0) and "analysis_quality" in _pz and _nf[0] < _nf[1]
      and _vA.analysis_quality == 'MEDIUM', f"{_a0} {_a1} {_a2} {_a3} tris fast/fine {_nf}")

# --- Work planes: named frames saved with the scene; placed from the world, cursor, selection, 3 points
from blender_nurbs import workplanes as _wp
_fr = _wp.frame_from_points((1, 2, 3), (3, 2, 3), (1, 5, 7))
check("work plane: 3 points give an orthonormal frame (X to B, Y towards C, normal X x Y)",
      _fr is not None and np.allclose(_fr[1], (1, 0, 0)) and np.allclose(_fr[2], (0, 0.6, 0.8))
      and np.allclose(_fr[3], (0, -0.8, 0.6)) and _wp.frame_from_points((0, 0, 0), (1, 0, 0), (2, 0, 0)) is None,
      str(_fr))
_ff = _wp.face_frame((0, 0, 0), (0, 0, 2), [(0, 0, 0), (0, 1, 0), (3, 1, 0), (3, 0, 0)])
check("work plane: a face's X runs along its longest edge, its normal is the face's",
      np.allclose(abs(_ff[1][0]), 1) and np.allclose(_ff[3], (0, 0, 1)), str(_ff))
_scW = ctx.scene
_stW = _scW.nproto_workplanes
_stW.planes.clear(); _stW.active = -1
check("work plane: the World is active by default (active_frame None)", _wp.active_frame(_scW) is None)
for _src in ("WORLD_XY", "WORLD_XZ", "WORLD_YZ", "CURSOR"):
    bpy.ops.nproto.workplane_add(source=_src)
_names = [p_.name for p_ in _stW.planes]
_fz = _wp.frame_of(_stW.planes[1])
check("work plane: Top / Front / Right / Cursor added, the last one active; Front's normal is -Y",
      _names == ["Top", "Front", "Right", "Cursor"] and _stW.active == 3 and np.allclose(_fz[3], (0, -1, 0), atol=1e-6)
      and np.allclose(_fz[2], (0, 0, 1), atol=1e-6), f"{_names} {_stW.active}")
bpy.ops.nproto.workplane_add(source="WORLD_XY")
check("work plane: names stay unique (Top.001)", _stW.planes[-1].name == "Top.001")
# a rotated object: Selection in Object Mode takes its axes
bpy.ops.mesh.primitive_plane_add(size=2, location=(1, 2, 3), rotation=(0.3, -0.2, 0.7))
_pl = ctx.active_object
bpy.ops.nproto.workplane_add(source="SELECTION")
_fo = _wp.active_frame(_scW)
_Mo = np.array(_pl.matrix_world)
check("work plane: Selection (Object Mode) = the active object's origin and axes",
      _stW.planes[_stW.active].name == _pl.name and np.allclose(_fo[0], (1, 2, 3), atol=1e-6)
      and np.allclose(_fo[3], _Mo[:3, 2], atol=1e-6) and np.allclose(_fo[1], _Mo[:3, 0], atol=1e-6))
# Edit Mode: the selected face (centre, normal), then three vertices
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.nproto.workplane_add(source="SELECTION")
_fe = _wp.active_frame(_scW)
bpy.ops.object.mode_set(mode='OBJECT')
check("work plane: Selection (Edit Mode, one face) = the face's centre and normal",
      _stW.planes[_stW.active].name == "Face" and np.allclose(_fe[0], (1, 2, 3), atol=1e-5)
      and np.allclose(_fe[3], _Mo[:3, 2], atol=1e-5), str(_fe))
# turning it over / about its normal keeps the origin and stays right-handed
_i0 = _stW.active; _stW.index = _i0
_b0 = _wp.frame_of(_stW.planes[_i0])
bpy.ops.nproto.workplane_turn(action="FLIP"); _b1 = _wp.frame_of(_stW.planes[_i0])
bpy.ops.nproto.workplane_turn(action="ROT90"); _b2 = _wp.frame_of(_stW.planes[_i0])
check("work plane: Flip reverses the normal, Rotate 90 turns X onto Y, origin kept",
      np.allclose(_b1[3], -_b0[3], atol=1e-6) and np.allclose(_b2[1], _b1[2], atol=1e-6)
      and np.allclose(_b2[0], _b0[0]) and np.allclose(np.cross(_b2[1], _b2[2]), _b2[3], atol=1e-6))
# list bookkeeping: deleting / moving keeps the same plane active
_stW.active = 2; _stW.index = 0
bpy.ops.nproto.workplane_remove()
_a_rm = _stW.planes[_stW.active].name
_stW.index = _stW.active
bpy.ops.nproto.workplane_move(direction="UP")
_a_mv = (_stW.planes[_stW.active].name, _stW.active)
_stW.index = _stW.active
bpy.ops.nproto.workplane_remove()
check("work plane: delete / move keep the active plane; deleting it makes the World active",
      _a_rm == "Right" and _a_mv == ("Right", 0) and _stW.active == -1, f"{_a_rm} {_a_mv} {_stW.active}")
# saved with the .blend
_stW.active = 1
_keep = [(p_.name, tuple(round(c_, 6) for c_ in p_.origin), tuple(round(c_, 6) for c_ in p_.rotation)) for p_ in _stW.planes]
_wpf = os.path.join(tempfile.mkdtemp(), "wp.blend")
bpy.ops.wm.save_as_mainfile(filepath=_wpf, copy=True)
with bpy.data.libraries.load(_wpf) as (_src_, _dst_):
    _dst_.scenes = [_scW.name]
_sc2 = _dst_.scenes[0]
_got = [(p_.name, tuple(round(c_, 6) for c_ in p_.origin), tuple(round(c_, 6) for c_ in p_.rotation))
        for p_ in _sc2.nproto_workplanes.planes]
check("work plane: planes and the active one are saved in the .blend",
      _got == _keep and _sc2.nproto_workplanes.active == 1, f"{len(_got)} planes")
bpy.data.scenes.remove(_sc2)
_pwp = _draw(_wp.VIEW3D_PT_nproto_workplanes)
check("work plane: the panel draws (World, list, add menu, origin / rotation)",
      any(e.get("menu") == "NPROTO_MT_workplane_add" for e in _pwp) and any(e.get("prop") == "origin" for e in _pwp))
_stW.planes.clear(); _stW.active = -1; _stW.index = 0
bpy.data.objects.remove(_pl)

# --- Work plane grids: levels, the endless grid's square, Blender's floor hidden and put back
check("work plane grid: fixed spacing levels (0.25, major 4)", _wp.fixed_levels(0.25, 4, 4) == [0.25, 1.0, 4.0, 16.0])
check("work plane grid: patch sizes round up to 1 / 2 / 5 x 10^k",
      [_wp.patch_size(e_) for e_ in (0.7, 1.0, 3.0, 7.9)] == [1.0, 2.0, 5.0, 10.0])
check("work plane grid: a face patch covers the face from its origin",
      abs(_wp.covering_half(_wp.WORLD["XY"], [(3, -1, 0), (-0.5, 2, 0)]) - 3.0) < 1e-12)
_sqp = _wp.endless_square((2, 3, 5), (0, 0, -1), True, 0, 0, 1000.0, 100.0, 4.0, 1000.0)
_sqf = _wp.endless_square((2, 3, 5), (0, 0, -1), True, 0, 0, 1000.0, 0.01, 4.0, 1000.0)
_sqo = _wp.endless_square((0, 0, 5), (0, -0.6, -0.8), False, 3, 2, 0, 1.0, 4.0, 1000.0)
check("work plane grid: endless square around the eye (perspective), none when too far, the view's foot (ortho)",
      _sqp == (2.0, 3.0, 1000.0) and _sqf is None and np.allclose(_sqo[:2], (0, -3.75))
      and _wp.endless_square((0, 0, 5), (0, -1, 0), False, 3, 2, 0, 1.0, 4.0, 1000.0) is None, f"{_sqp} {_sqf} {_sqo}")
_ovs = [ov_ for _, ov_ in _wp._views()]
_before = [[getattr(ov_, p_) for p_ in _wp.FLOOR_PROPS] for ov_ in _ovs]
bpy.ops.nproto.workplane_add(source="WORLD_XZ")
_wpE = _stW.planes[_stW.active]
_patch_shows = all(ov_.show_floor == b_[0] for ov_, b_ in zip(_ovs, _before))
_wpE.grid = "ENDLESS"
_hid = all(not any(getattr(ov_, p_) for p_ in _wp.FLOOR_PROPS) for ov_ in _ovs) and _wp.FLOOR_STASH in _scW
bpy.ops.nproto.workplane_activate(index=-1)
_back = [[getattr(ov_, p_) for p_ in _wp.FLOOR_PROPS] for ov_ in _ovs]
bpy.ops.nproto.workplane_activate(index=_stW.planes.find(_wpE.name))
_hid2 = not any(ov_.show_floor for ov_ in _ovs)
_wpE.grid = "PATCH"
_back2 = [[getattr(ov_, p_) for p_ in _wp.FLOOR_PROPS] for ov_ in _ovs]
check("work plane grid: Endless hides Blender's floor in every 3D view; the World or a Patch puts it back as it was",
      len(_ovs) > 0 and _patch_shows and _hid and _back == _before and _hid2 and _back2 == _before
      and _wp.FLOOR_STASH not in _scW, f"{len(_ovs)} views")
_lv = _wp.plane_levels(ctx, _wpE, None)
_wpE.auto_spacing = False; _wpE.spacing = 0.05; _wpE.major = 5
check("work plane grid: Auto follows the scene's units (metric: um, mm, cm, 10 cm ...); fixed uses the plane's spacing",
      np.allclose(_lv[:4], [1e-6, 1e-3, 1e-2, 1e-1]) and np.allclose(_wp.plane_levels(ctx, _wpE, None)[:3], [0.05, 0.25, 1.25]), f"{_lv[:4]}")
_pwg = _draw(_wp.VIEW3D_PT_nproto_workplanes)
check("work plane grid: the panel offers grid kind, spacing and major lines",
      all(any(e.get("prop") == k_ for e in _pwg) for k_ in ("grid", "auto_spacing", "spacing", "major")))
_stW.planes.clear(); _stW.active = -1; _stW.index = 0

# --- Loft: a side of several edges end to end (chained sides, tangent chain picking)
from blender_nurbs import ops as _lc_ops, contcheck as _lc_cc
from blender_nurbs.kernel.classa import insert_knot as _lc_ins
_lcO = np.array([0.0, 300.0, 50.0])
_lcG = np.zeros((4, 4, 3))
for i_ in range(4):
    for j_ in range(4):
        _lcG[i_, j_] = _lcO + (i_ * 1.0, j_ * 1.2, 0.3 * np.sin(i_ + 0.7 * j_) + 0.15 * i_ * j_)
_lcK = [0, 0, 0, 0, 1, 1, 1, 1]
def _lc_surf(Gr):
    return ext.surface_from_cvs(3, 3, [tuple(map(float, q)) for q in Gr.reshape(-1, 3)], _lcK, _lcK)
_lcW = store.new_object(ctx, "LcWhole", _lc_surf(_lcG))
_lcR = []
for i_ in range(4):                          # split along v at 0.4, exactly
    P_, U_ = _lcG[i_], list(map(float, _lcK))
    for _ in range(3):
        P_, U_ = _lc_ins(np.asarray(P_, float), U_, 3, 0.4)
    _lcR.append(np.asarray(P_, float))
_lcR = np.array(_lcR)
_lcA = store.new_object(ctx, "LcHalfA", _lc_surf(_lcR[:, :4]))
_lcB = store.new_object(ctx, "LcHalfB", _lc_surf(_lcR[:, 3:]))
_lcT = store.new_object(ctx, "LcTarget", ext.surface_from_cvs(1, 1, [tuple(map(float, _lcO + q)) for q in
                        [(5, -0.5, 1.0), (5, 4.0, 1.0), (7, -0.5, 1.0), (7, 4.0, 1.0)]], [0, 0, 1, 1], [0, 0, 1, 1]))

def _lc_cont(src, loft_obj):
    best = None
    for e_ in ("U0", "U1", "V0", "V1"):
        try:
            r_ = _lc_cc.analyze({"a": (src.name, "U1", None), "b": (loft_obj.name, e_, None)}, 24)
        except Exception:
            continue
        if best is None or r_["gap"].max() < best["gap"].max():
            best = r_
    return best

_lc_res = []
for _cont in ("G0", "G1", "G2"):
    for _ord in ([(_lcA, "U1"), (_lcB, "U1")], [(_lcB, "U1"), (_lcA, "U1")]):
        _lo = store.new_object(ctx, "LcLoft", features.run("loft", _ord + [(_lcT, "U0")],
                                                            {"cont_a": _cont, "sides": [2, 1]}))
        for _h in (_lcA, _lcB):
            b_ = _lc_cont(_h, _lo)
            k_ = {"G0": 0, "G1": 1, "G2": 2}[_cont]
            _lc_res.append(b_ is not None and b_["gap"].max() < 1e-7 and (k_ < 1 or b_["ang"].max() < 1e-4)
                           and (k_ < 2 or b_["crel"].max() < 1e-4))
        bpy.data.objects.remove(_lo)
_lc_ref = features.run("loft", [(_lcW, "U1"), (_lcT, "U0")], {})
_lc_got = features.run("loft", [(_lcA, "U1"), (_lcB, "U1"), (_lcT, "U0")], {"sides": [2, 1]})
_lc_d = max(np.linalg.norm(np.array(_lc_ref.evaluate(u_, v_), float) - np.array(_lc_got.evaluate(u_, v_), float))
            for u_ in np.linspace(0, 1, 7) for v_ in np.linspace(0, 1, 7)) \
    if all(hasattr(g_, "evaluate") for g_ in (_lc_ref, _lc_got)) else -1
check("loft chained side: two half-edges as one side are G0 / G1 / G2 exact to both faces, either pick order; "
      "G0 is the whole edge's loft", all(_lc_res) and 0 <= _lc_d < 1e-9, f"{_lc_res} dev {_lc_d:.1e}")
check("loft chained side: without sides three inputs are still three sections",
      features.run("loft", [(_lcA, "U1"), (_lcB, "U1"), (_lcT, "U0")], {}) is not None)
try:
    features.chain_side([(_lcA, "U1"), (_lcT, "U0")]); _lc_ref_ok = False
except ValueError:
    _lc_ref_ok = True
check("loft chained side: edges that do not touch are refused as one side", _lc_ref_ok)

# a body of the two faces: its B-rep edges, found by tangent chain from one click
_lcobjs = {f"s{i_}": (store.brep_to_text(store.to_brep(store.get_surfaces(o_)[0])),
                      {"loc": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, 1.0], "angle": 0.0, "scale": [1.0, 1.0, 1.0]})
           for i_, o_ in enumerate((_lcA, _lcB))}
_, _lcbr = jobs.run("join", _lcobjs, ["joined"], order=list(_lcobjs), into="s0", tolerance=1e-7)
_lcJ = store.new_object(ctx, "LcBody", store.brep_from_text(_lcbr["joined"]))
_lcE = []
for i_, e_ in enumerate(store.get_brep(_lcJ).edges()):
    t0_, t1_ = e_.parameter_range()
    if len(e_.faces()) == 1 and all(abs(e_.point_at(t_)[0] - 3.0) < 1e-6 for t_ in (t0_, 0.5 * (t0_ + t1_), t1_)):
        _lcE.append(f"E{i_}")
_lcC = [c_ for c_ in _lc_ops._pick_candidates(ctx) if c_[0] in (_lcJ, _lcT)]
_lcby = {(c_[0].name, c_[1]): c_ for c_ in _lcC}
_lcCh = _lc_ops.tangent_chain(_lcby[(_lcJ.name, _lcE[-1])], _lcC, 2.0) if len(_lcE) == 2 else []
_lcSo = _lc_ops.group_into_sides([_lcby[(_lcJ.name, _lcE[0])], _lcby[(_lcJ.name, _lcE[1])], _lcby[(_lcT.name, "U0")]]) \
    if len(_lcE) == 2 else []
_lc_body = []
if len(_lcE) == 2:
    _lo = store.new_object(ctx, "LcLoftBody", features.run("loft", [(_lcJ, _lcE[1]), (_lcJ, _lcE[0]), (_lcT, "U0")],
                                                           {"cont_a": "G2", "sides": [2, 1]}))
    for _h in (_lcA, _lcB):
        b_ = _lc_cont(_h, _lo)
        _lc_body.append(b_ is not None and b_["gap"].max() < 1e-7 and b_["ang"].max() < 1e-4 and b_["crel"].max() < 1e-4)
    bpy.data.objects.remove(_lo)
check("loft chained side: a body's two edges -- one click chains both (tangent), they group as one side, "
      "the G2 loft is exact to both faces",
      {c_[1] for c_ in _lcCh} == set(_lcE) and len(_lcCh) == 2 and _lcSo == [0, 0, 1] and _lc_body == [True, True]
      and len(_lc_ops.tangent_chain(_lcby[(_lcT.name, "U0")], _lcC, 2.0)) == 1, f"{_lcE} {_lcSo} {_lc_body}")

# faces creased at the joint: exact along each face, blended only next to the joint, and reported
_lcRb = _lcR[:, 3:].copy()
_lcRb[:3, :, 2] += 0.4 * np.array([3, 2, 1])[:, None] / 3
_lcK2 = store.new_object(ctx, "LcTilted", _lc_surf(_lcRb))
_lcSx = features.side_exact([(_lcA, "U1"), (_lcK2, "U1")])
features.LOFT_NOTES.clear()
_lo = store.new_object(ctx, "LcLoftCrease", features.run("loft", [(_lcA, "U1"), (_lcK2, "U1"), (_lcT, "U0")],
                                                         {"cont_a": "G1", "sides": [2, 1]}))
_lc_cr = []
for _h in (_lcA, _lcK2):
    b_ = _lc_cont(_h, _lo)
    far_ = np.linalg.norm((b_["P"] - _lcSx["creases"][0][1])[:, :2], axis=1) > 0.05 * 3.6
    _lc_cr.append(b_["gap"].max() < 1e-7 and b_["ang"][far_].max() < 1e-4)
check("loft chained side: faces creased at the joint -- G1 exact along both away from it, the crease reported",
      len(_lcSx["creases"]) == 1 and _lc_cr == [True, True] and any("joint" in n_ for n_ in features.LOFT_NOTES),
      f"{_lc_cr} {features.LOFT_NOTES}")

# trimmed edges as one side (a spoon cut across two faces): the unit cross
# directions of trim edges lean at their ends; joined they kinked and the G1
# loft folded back over the faces
# a bowl (z = c (y - y0)^2 across, the low line at the halves' joint, as a spoon)
from math import comb as _comb
_bv = np.linspace(0, 1, 4)
_bM = np.array([[_comb(3, k_) * t_ ** k_ * (1 - t_) ** (3 - k_) for k_ in range(4)] for t_ in _bv])
_bz = np.linalg.solve(_bM, 0.05 * (3.6 * _bv - 1.44) ** 2)
_bG = np.array([[_lcO + (i_ * 1.0, j_ * 1.2, _bz[j_] + 0.03 * i_) for j_ in range(4)] for i_ in range(4)])
_bR = []
for i_ in range(4):
    P_, U_ = _bG[i_], list(map(float, _lcK))
    for _ in range(3):
        P_, U_ = _lc_ins(np.asarray(P_, float), U_, 3, 0.4)
    _bR.append(np.asarray(P_, float))
_bR = np.array(_bR)
_lcA = store.new_object(ctx, "LcBowlA", _lc_surf(_bR[:, :4]))
_lcB = store.new_object(ctx, "LcBowlB", _lc_surf(_bR[:, 3:]))
_lcW1, _lcW2 = _lcO + (2.3, -1.0, -2.0), _lcO + (2.7, 5.0, -2.0)
_lcWl = bpy.data.curves.new("LcWallLn", 'CURVE'); _lcWl.dimensions = '3D'
_sp = _lcWl.splines.new('POLY'); _sp.points.add(1)
_sp.points[0].co = (*_lcW1, 1.0); _sp.points[1].co = (*_lcW2, 1.0)
_lcWo = bpy.data.objects.new("LcWallLn", _lcWl); ctx.collection.objects.link(_lcWo)
bpy.ops.nproto.extrude(inputs=ins((_lcWo, "S0")), distance=6.0, direction="Z"); _lcWall = ctx.active_object
_lcWall.name = "LcWall"
_lcTr = []
for _h in (_lcA, _lcB):
    _hs = store.get_surfaces(_h)[0]
    _ref = tuple(float(x) for x in _hs.evaluate(0.3, 0.5))
    bpy.ops.object.select_all(action='DESELECT')
    _r = bpy.ops.nproto.trim(target=_h.name, cutters=json.dumps([[_lcWall.name, "SURF"]]), ref_point=_ref,
                             normal=(0, 0, 1), action="KEEP")
    _cut = None
    _wn = np.cross(_lcW2 - _lcW1, (0, 0, 1.0)); _wn /= np.linalg.norm(_wn)
    for i_, e_ in enumerate(store.get_brep(_h).edges()):
        t0_, t1_ = e_.parameter_range()
        if all(abs(float(np.dot(np.array(e_.point_at(t_), float) - _lcW1, _wn))) < 1e-4
               for t_ in np.linspace(t0_, t1_, 5)):
            _cut = f"E{i_}"
    _lcTr.append((_r, _cut))
_lc_trim_ok = all(r_ == {'FINISHED'} and c_ is not None for r_, c_ in _lcTr)
_lc_tr_res = "trim failed"
if _lc_trim_ok:
    _side = [(_lcA, _lcTr[0][1]), (_lcB, _lcTr[1][1])]
    _g = features.run("loft", _side + [(_lcT, "U0")], {"cont_a": "G1", "sides": [2, 1]})
    _lo = store.new_object(ctx, "LcLoftTrim", _g)
    _s = _g if hasattr(_g, "u_domain") else store.to_brep(_g).faces()[0].surface()
    (_u0, _u1), (_v0, _v1) = _s.u_domain(), _s.v_domain()
    _N = np.array([[np.cross(*_s.derivatives(_u0 + (_u1 - _u0) * a_, _v0 + (_v1 - _v0) * b_)[1:])
                    for b_ in np.linspace(0, 1, 61)] for a_ in np.linspace(0, 1, 31)])
    _N /= np.linalg.norm(_N, axis=2, keepdims=True)
    _turn = min(float(np.min(np.sum(_N[1:] * _N[:-1], axis=2))), float(np.min(np.sum(_N[:, 1:] * _N[:, :-1], axis=2))))
    _spread = float(np.degrees(np.arccos(np.clip(_turn, -1, 1))))      # largest turn between neighbouring samples
    _angs = []
    for _h, (_, _cut) in zip((_lcA, _lcB), _lcTr):
        e_ = store.get_brep(_h).edges()[int(_cut[1:])]; t0_, t1_ = e_.parameter_range()
        _mid = list(e_.point_at(0.5 * (t0_ + t1_)))
        best = None
        for ee_ in ("U0", "U1", "V0", "V1"):
            try:
                r_ = _lc_cc.analyze({"a": (_h.name, _cut, _mid), "b": (_lo.name, ee_, None)}, 24)
            except Exception:
                continue
            if best is None or r_["gap"].max() < best["gap"].max():
                best = r_
        _angs.append(float(np.median(best["ang"])) if best is not None else 99.0)
        _angs.append(float(best["ang"][2:-2].max()) if best is not None else 99.0)
    _lc_tr_res = (_spread, _angs)
check("loft chained side: two trimmed edges meeting mid-side (a cut spoon) -- the G1 loft does not fold "
      "and holds G1 along both cuts", _lc_trim_ok and _lc_tr_res[0] < 20 and max(_lc_tr_res[1]) < 0.05,
      f"{_lcTr} {_lc_tr_res}")

# Match Edges: at the corners of a side the loft carries on the source's own
# edges (a cut spoon's rims, curving in towards the cut): its outline leaves
# along them, with G2 in their curvature too; G1 / G2 to the faces still hold
_mO = _lcO + np.array([0.0, 20.0, 0.0])
_mf = [1.0, 0.95, 0.75, 0.55]                                   # the bowl narrows: its rims curve in
_mz = np.linalg.solve(_bM, 0.05 * (3.6 * _bv - 1.8) ** 2)
_mG = np.array([[_mO + (i_ * 1.0, 1.8 + _mf[i_] * (j_ * 1.2 - 1.8), _mz[j_] + 0.03 * i_) for j_ in range(4)]
                for i_ in range(4)])
_mR = []
for i_ in range(4):
    P_, U_ = _mG[i_], list(map(float, _lcK))
    for _ in range(3):
        P_, U_ = _lc_ins(np.asarray(P_, float), U_, 3, 0.5)
    _mR.append(np.asarray(P_, float))
_mR = np.array(_mR)
_mA = store.new_object(ctx, "LcSpoonA", _lc_surf(_mR[:, :4]))
_mB = store.new_object(ctx, "LcSpoonB", _lc_surf(_mR[:, 3:]))
_mT = store.new_object(ctx, "LcHandle", ext.surface_from_cvs(1, 1, [tuple(map(float, _mO + q)) for q in
                       [(5, 1.0, 0.3), (5, 2.6, 0.3), (7, 1.0, 0.3), (7, 2.6, 0.3)]], [0, 0, 1, 1], [0, 0, 1, 1]))
_mW1, _mW2 = _mO + (2.4, -1.0, -2.0), _mO + (2.6, 5.0, -2.0)
_mWl = bpy.data.curves.new("LcMWallLn", 'CURVE'); _mWl.dimensions = '3D'
_sp = _mWl.splines.new('POLY'); _sp.points.add(1)
_sp.points[0].co = (*_mW1, 1.0); _sp.points[1].co = (*_mW2, 1.0)
_mWo = bpy.data.objects.new("LcMWallLn", _mWl); ctx.collection.objects.link(_mWo)
bpy.ops.nproto.extrude(inputs=ins((_mWo, "S0")), distance=6.0, direction="Z"); _mWall = ctx.active_object
_mWall.name = "LcMWall"
_mn = np.cross(_mW2 - _mW1, (0, 0, 1.0)); _mn /= np.linalg.norm(_mn)
_mCut = []
for _h in (_mA, _mB):
    bpy.ops.object.select_all(action='DESELECT')
    _r = bpy.ops.nproto.trim(target=_h.name, cutters=json.dumps([[_mWall.name, "SURF"]]),
                             ref_point=tuple(float(x) for x in store.get_surfaces(_h)[0].evaluate(0.3, 0.5)),
                             normal=(0, 0, 1), action="KEEP")
    _c = None
    for i_, e_ in enumerate(store.get_brep(_h).edges()):
        t0_, t1_ = e_.parameter_range()
        if all(abs(float(np.dot(np.array(e_.point_at(t_), float) - _mW1, _mn))) < 1e-4 for t_ in np.linspace(t0_, t1_, 5)):
            _c = f"E{i_}"
    _mCut.append(_c if _r == {'FINISHED'} else None)

def _lc_frame(f, t, h):
    p_ = [np.array(f(t + k_ * h), float) for k_ in (0, 1, 2, 3)]
    d1_ = (-11 * p_[0] + 18 * p_[1] - 9 * p_[2] + 2 * p_[3]) / (6 * h)
    d2_ = (2 * p_[0] - 5 * p_[1] + 4 * p_[2] - p_[3]) / (h * h)
    T_ = d1_ / np.linalg.norm(d1_)
    return T_, (d2_ - (d2_ @ T_) * T_) / (d1_ @ d1_)
_lc_me = {}
if None not in _mCut:
    _mside = [(_mA, _mCut[0]), (_mB, _mCut[1])]
    _sdP = features.side_samples(_mside, 47).P
    for _cont in ("G1", "G2"):
        for _me in (False, True):
            _g = features.run("loft", _mside + [(_mT, "U0")], {"cont_a": _cont, "sides": [2, 1], "match_edges": _me})
            _s = _g if hasattr(_g, "u_domain") else store.to_brep(_g).faces()[0].surface()
            (_u0, _u1), (_v0, _v1) = _s.u_domain(), _s.v_domain()
            _row = []
            for _k, _c in enumerate((_sdP[0], _sdP[-1])):
                _o = features.chain_side(_mside)[0 if _k == 0 else -1][0]
                _rim = features.edge_into_corner(_o, _c, _sdP, np.array([1.0, 0, 0]), 1e-5)
                _hit = None
                for _vv in (_v0, _v1):
                    for _uu, _du in ((_u0, 1e-3 * (_u1 - _u0)), (_u1, -1e-3 * (_u1 - _u0))):
                        if np.linalg.norm(np.array(_s.evaluate(_uu, _vv)) - _c) < 1e-5:
                            _hit = _lc_frame(lambda t_: _s.evaluate(t_, _vv), _uu, _du)
                if _rim is None or _hit is None:
                    _row.append(None)
                    continue
                _row.append((round(float(np.degrees(np.arccos(min(1.0, abs(float(_hit[0] @ _rim[0])))))), 3),
                             round(float(np.linalg.norm(_hit[1] - _rim[1])) * 3.6, 4)))     # curvature gap x size
            _lo = store.new_object(ctx, "LcLoftMatch", _g)
            _worst = []
            for _h, _cut in zip((_mA, _mB), _mCut):
                e_ = store.get_brep(_h).edges()[int(_cut[1:])]; t0_, t1_ = e_.parameter_range()
                best = None
                for ee_ in ("U0", "U1", "V0", "V1"):
                    try:
                        r_ = _lc_cc.analyze({"a": (_h.name, _cut, list(e_.point_at(0.5 * (t0_ + t1_)))),
                                             "b": (_lo.name, ee_, None)}, 24)
                    except Exception:
                        continue
                    if best is None or r_["gap"].max() < best["gap"].max():
                        best = r_
                _worst.append((round(float(best["ang"][2:-2].max()), 4), round(float(best["crel"][2:-2].max()), 4)))
            bpy.data.objects.remove(_lo)
            _lc_me[(_cont, _me)] = (_row, _worst)
_ok_me = bool(_lc_me) and all(r_ is not None for v_ in _lc_me.values() for r_ in v_[0])
if _ok_me:
    _ok_me = (min(a_ for a_, _ in _lc_me[("G1", False)][0]) > 5.0               # square: the outline kinks at the rims
              and all(a_ < 0.5 for k_ in (("G1", True), ("G2", True)) for a_, _ in _lc_me[k_][0])
              and all(c_ < 0.05 for _, c_ in _lc_me[("G2", True)][0])         # G2: the rims' curvature carries on
              and all(a_ < 0.05 for v_ in _lc_me.values() for a_, _ in v_[1])
              and all(c_ < 0.01 for _, c_ in _lc_me[("G2", True)][1]))
check("loft Match Edges: the outline leaves a cut spoon along its rims (G2: in their curvature), "
      "G1 / G2 to both faces hold; square, it kinks there", _ok_me, f"{_mCut} {_lc_me}")

# the same loft at 24 and at 49 Samples (G2 held only at 49 once: the fit is
# checked between its samples now), and Align to Surface: along the faces' lines
def _lc_grid(g_):
    s_ = g_ if hasattr(g_, "u_domain") else store.to_brep(g_).faces()[0].surface()
    (a0_, a1_), (b0_, b1_) = s_.u_domain(), s_.v_domain()
    return np.array([s_.evaluate(a0_ + (a1_ - a0_) * x_, b0_ + (b1_ - b0_) * y_)
                     for x_ in np.linspace(0, 1, 9) for y_ in np.linspace(0, 1, 17)], float)
_lc_ss = {}
if None not in _mCut:
    for _mode in ("EDGES", "SURFACE", "FREE"):
        _gg = [features.run("loft", _mside + [(_mT, "U0")], {"cont_a": "G2", "sides": [2, 1], "edges_a": _mode,
                                                               "edges_b": "FREE", "samples": _n}) for _n in (24, 49)]
        _lo = store.new_object(ctx, "LcLoftMode", _gg[0])
        _wst = []
        for _h, _cut in zip((_mA, _mB), _mCut):
            e_ = store.get_brep(_h).edges()[int(_cut[1:])]; t0_, t1_ = e_.parameter_range()
            best = None
            for ee_ in ("U0", "U1", "V0", "V1"):
                try:
                    r_ = _lc_cc.analyze({"a": (_h.name, _cut, list(e_.point_at(0.5 * (t0_ + t1_)))),
                                         "b": (_lo.name, ee_, None)}, 24)
                except Exception:
                    continue
                if best is None or r_["gap"].max() < best["gap"].max():
                    best = r_
            _wst.append((round(float(best["ang"][2:-2].max()), 3), round(float(best["crel"][2:-2].max()), 4)))
        bpy.data.objects.remove(_lo)
        _lc_ss[_mode] = (float(np.abs(_lc_grid(_gg[0]) - _lc_grid(_gg[1])).max()), _wst, _lc_grid(_gg[0]))
_ok_ss = len(_lc_ss) == 3 and all(v_[0] < 2e-3 * 3.6 for v_ in _lc_ss.values()) \
    and all(a_ < 0.1 and c_ < 0.01 for v_ in _lc_ss.values() for a_, c_ in v_[1]) \
    and float(np.abs(_lc_ss["SURFACE"][2] - _lc_ss["FREE"][2]).max()) > 1e-3 \
    and float(np.abs(_lc_ss["SURFACE"][2] - _lc_ss["EDGES"][2]).max()) > 1e-3
check("loft edge modes: Align Edges / Align to Surface / Free -- three different lofts, each G2 to both faces "
      "and the same at 24 and 49 Samples", _ok_ss,
      f"{ {k_: (round(v_[0], 5), v_[1]) for k_, v_ in _lc_ss.items()} }")

# Bezier spans: a few spans of one degree, knots where continuity needs them --
# lighter than the least-squares net, symmetric on a symmetric side, the same
# at any Samples, held as the labels check it; Split into Faces: one per span
from blender_nurbs import contlabels as _lc_cl
from types import SimpleNamespace as _NS
_lc_sp = {}
if None not in _mCut:
    for _fit in ("SPANS", "DENSE"):
        for _n in (6, 49):
            _pp = {"cont_a": "G2", "cont_b": "G2", "sides": [2, 1], "edges_a": "EDGES", "edges_b": "FREE",
                   "samples": _n, "fit": _fit}
            _g = features.run("loft", _mside + [(_mT, "U0")], _pp)
            _d = ext.surface_cvs(_g)
            _G = np.array(_d["points"]).reshape(_d["count_u"], _d["count_v"], 3) - _mO
            _asym = float(np.abs(_G - _G[:, ::-1] * np.array([1, -1, 1]) - np.array([0, 3.6, 0])).max())
            _lo = store.new_object(ctx, "LcLoftSpan", _g)
            _hold = all(_lc_cl.measure(_lo, _h, _c, 2)["ok"] for _h, _c in zip((_mA, _mB), _mCut))
            bpy.data.objects.remove(_lo)
            _lc_sp[(_fit, _n)] = (_d["count_v"], round(_asym, 6), _hold, features.LAST_SPANS[0])
    _gf = features.run("loft", _mside + [(_mT, "U0")], {"cont_a": "G2", "cont_b": "G2", "sides": [2, 1],
                                                         "edges_a": "EDGES", "fit": "SPANS", "span_faces": True})
    _lc_sp["faces"] = (len(store.to_brep(_gf).faces()) if not hasattr(_gf, "faces") else len(_gf.faces()),
                       features.LAST_SPANS[0])
_ok_sp = bool(_lc_sp) and _lc_sp[("SPANS", 6)][:3] == _lc_sp[("SPANS", 49)][:3] \
    and _lc_sp[("SPANS", 6)][0] < _lc_sp[("DENSE", 49)][0] \
    and _lc_sp[("SPANS", 6)][2] and _lc_sp[("SPANS", 49)][2] and _lc_sp["faces"][0] == _lc_sp["faces"][1] > 1
check("loft Bezier spans: fewer CVs than the fitted net, the same at 6 and 49 Samples, G2 held as "
      "the labels check it; Split into Faces gives one face per span", _ok_sp, f"{_lc_sp}")

# Loft with rails (guide curves): a smooth surface cut into 3 x 3 patches with
# the middle one taken out -- left / right edges the profiles, bottom / top the
# rails. G1 / G2 along all four (the rails' solver) gives the middle back.
from blender_nurbs import contlabels as _rl_cl
_rO = np.array([40.0, 300.0, 0.0])
_rG = np.array([[_rO + (i_ * 1.0, j_ * 1.0, 0.25 * np.sin(1.1 * i_ + 0.4) * np.cos(0.8 * j_) + 0.05 * i_ * j_)
                 for j_ in range(4)] for i_ in range(4)])
def _rl_split(P_, ts_):
    A_, U_ = np.asarray(P_, float), list(map(float, _lcK))
    for t_ in ts_:
        for _ in range(3):
            A_, U_ = _lc_ins(A_, U_, 3, t_)
            A_ = np.asarray(A_, float)
    return [A_[3 * k_:3 * k_ + 4] for k_ in range(len(ts_) + 1)]
_rrows = [_rl_split(_rG[i_], [1 / 3, 2 / 3]) for i_ in range(4)]
_rpat = {}
for kv_ in range(3):
    _blk = np.array([_rrows[i_][kv_] for i_ in range(4)])
    _cols = [_rl_split(_blk[:, j_], [1 / 3, 2 / 3]) for j_ in range(4)]
    for ku_ in range(3):
        _rpat[(ku_, kv_)] = np.array([_cols[j_][ku_] for j_ in range(4)]).transpose(1, 0, 2)
_rL, _rR = store.new_object(ctx, "RlL", _lc_surf(_rpat[(0, 1)])), store.new_object(ctx, "RlR", _lc_surf(_rpat[(2, 1)]))
_rB, _rT = store.new_object(ctx, "RlB", _lc_surf(_rpat[(1, 0)])), store.new_object(ctx, "RlT", _lc_surf(_rpat[(1, 2)]))
_rC = _lc_surf(_rpat[(1, 1)])
_rins = [(_rL, "U1"), (_rR, "U0"), (_rB, "V1"), (_rT, "V0")]
_rres = {}
for _c in ("G1", "G2"):
    _g = features.run("loft", _rins, {"cont_a": _c, "cont_b": _c, "cont_ra": _c, "cont_rb": _c,
                                      "sides": [1, 1, 1, 1], "rails": 2})
    _lo = store.new_object(ctx, "RlLoft", _g)
    _s = store.get_brep(_lo).faces()[0].surface()
    _smk = _kl()
    def _rl_d(q_):
        w_ = _smk.surface_closest_point(_s, tuple(map(float, q_)))[1][0]
        return float(np.linalg.norm(np.array(_s.evaluate(float(w_.x), float(w_.y))) - q_))
    _dev = max(_rl_d(np.array(_rC.evaluate(a_, b_), float)) for a_ in np.linspace(0, 1, 7) for b_ in np.linspace(0, 1, 7))
    _ok = [_rl_cl.measure(_lo, o_, r_, int(_c[1]))["ok"] for o_, r_ in _rins]
    _rres[_c] = (round(_dev, 7), _ok, ext.surface_cvs(_s)["count_u"])
    bpy.data.objects.remove(_lo)
_g = features.run("loft", _rins[:3], {"cont_a": "G1", "cont_b": "G1", "sides": [1, 1, 1], "rails": 1})
_lo = store.new_object(ctx, "RlGuide", _g)
_guide = (_rl_cl.measure(_lo, _rB, "V1", 0)["ok"], [_rl_cl.measure(_lo, o_, r_, 1)["ok"] for o_, r_ in _rins[:2]])
bpy.data.objects.remove(_lo)
check("loft with rails: G1 / G2 along both profiles and both rails holds and gives the middle patch back; "
      "one rail is a guide the loft passes through",
      all(v_[0] < 1e-5 and all(v_[1]) for v_ in _rres.values()) and _guide[0] and all(_guide[1]), f"{_rres} {_guide}")
# the operator: rails in the history, labels on both profiles and both rails
for o_ in list(ctx.selected_objects):
    o_.select_set(False)
_r = bpy.ops.nproto.loft(inputs=ins(*_rins), sides="[1, 1, 1, 1]", rails=2, cont_a="G2", cont_b="G2",
                         cont_ra="G2", cont_rb="G2")
_lo = ctx.active_object
_hp = features.history(_lo)["params"]
from blender_nurbs import contlabels as _cl_mod
_labs = _cl_mod._compute_base(_lo)
check("loft with rails: the operator records the rails, and labels sit on both profiles and both rails",
      _r == {'FINISHED'} and _hp.get("rails") == 2 and _hp.get("cont_ra") == "G2"
      and sorted(L_["param"] for L_ in _labs) == ["cont_a", "cont_b", "cont_ra", "cont_rb"],
      f"{_r} {_hp.get('rails')} {[L_['param'] for L_ in _labs]}")
bpy.data.objects.remove(_lo)
# picking: R switches to rails; rails come after the profiles whatever the order
from types import SimpleNamespace as _NSr
_Lr = _lc_ops.NPROTO_OT_loft
_rc = _lc_ops._pick_candidates(ctx)
_rby = {(c_[0].name, c_[1]): c_ for c_ in _rc}
_f = _NSr(_side_of=[], _new_side=False, _picks=[], _cands_by_key=_rby, chain='SINGLE', chain_angle=2.0,
          _cands=_rc, _rail_groups=set(), _rail_mode=False, rails=0, sides="")
for nm_ in ("after_pick", "after_unpick", "pick_count", "_recolour", "tool_key", "pick_state", "expand_pick", "_label"):
    setattr(_f, nm_, getattr(_Lr, nm_).__get__(_f))
def _rclick(key_):
    grp_ = _f.expand_pick(_rby[key_])
    _f._picks += [(c_[0], c_[1]) for c_ in grp_]
    _f.after_pick(grp_, _NSr(shift=False))
_rclick(("RlL", "U1"))
_f.tool_key(None, _NSr(type='R', value='PRESS'))
_rclick(("RlB", "V1")); _rclick(("RlT", "V0"))
_f.tool_key(None, _NSr(type='R', value='PRESS'))
_cnt_before = _f.pick_count()
_rclick(("RlR", "U0"))
_so = list(_f._side_of)
_rails = sorted(_f._rail_groups)
_order = sorted(range(len(_so)), key=lambda i_: (_so[i_] in _f._rail_groups, _so[i_], i_))
check("loft picking: R picks rails (their own colour), back to profiles; the loft finishes at two profiles "
      "and lists the rails after them",
      _cnt_before == 1 and _f.pick_count() == 2 and len(_rails) == 2
      and [(_f._picks[i_][0].name, _f._picks[i_][1]) for i_ in _order] == [("RlL", "U1"), ("RlR", "U0"), ("RlB", "V1"), ("RlT", "V0")]
      and _lc_ops.draw.pick_overlay["colors"] == [0, 3, 3, 1], f"{_so} {_rails} {_lc_ops.draw.pick_overlay['colors']}")
# Ease Corners / Split into Faces with rails: one Bezier face per span both
# ways, the labels still hold (eased toward the rails); a chained side's zones
# sit on its first and last pieces, scaled to each piece's length
_ge = features.run("loft", _rins, {"cont_a": "G2", "cont_b": "G2", "cont_ra": "G2", "cont_rb": "G2",
                                   "sides": [1, 1, 1, 1], "rails": 2, "corner_ease": 0.15, "span_faces": True})
_se = _ge if hasattr(_ge, "faces") else store.to_brep(_ge)
_lo = store.new_object(ctx, "RlFaces", _ge)
_eok = [(lambda r_: r_["ok"] or r_.get("eased"))(_rl_cl.measure(_lo, o_, r_, 2)) for o_, r_ in _rins]
bpy.data.objects.remove(_lo)
_z = features.ease_zones([(_lcA, "U1"), (_lcB, "U1")], 0.1)
check("loft with rails: Ease Corners and Split into Faces -- one face per span, labels hold; a chained "
      "side's corner zones on its end pieces", len(_se.faces()) >= 1 and all(_eok)
      and sorted(_z) == sorted([(_lcA.name, "U1"), (_lcB.name, "U1")])
      and all(len(v_) == 1 and 0.1 < v_[0][1] < 0.3 for v_ in _z.values()),
      f"faces {len(_se.faces())} {_eok} {_z}")
# G1 profiles, G0 rail curves that leave the profiles' tangent planes at the
# corners: no surface is G1 there, so those ends are eased (orange labels)
# instead of chased with CVs -- the same net as when they meet in the plane
_rcv = []
for v_ in (0.0, 1.0):
    _pts = [np.array(_rC.evaluate(a_, v_), float) + (0, 0, 0.12 * np.sin(np.pi * a_)) for a_ in np.linspace(0, 1, 6)]
    _rcv.append(nurbs(f"RlRail{int(v_)}", [tuple(map(float, q_)) for q_ in _pts]))
_cins = _rins[:2] + [(c_, "S0") for c_ in _rcv]
_cp = {"cont_a": "G1", "cont_b": "G1", "sides": [1, 1, 1, 1], "rails": 2, "corner_ease": 0.15}
_cz = features.loft_rail_eases(_cins, _cp)
_gc = features.run("loft", _cins, _cp)
_lo = store.new_object(ctx, "RlTilt", _gc)
_cm = [_rl_cl.measure(_lo, o_, r_, 1, ease=_cz.get((o_.name, r_))) for o_, r_ in _rins[:2]]
_cn = ext.surface_cvs(store.get_brep(_lo).faces()[0].surface())["count_u"]
bpy.data.objects.remove(_lo)
_cz0 = features.loft_rail_eases(_rins, dict(_cp, cont_ra="G0", cont_rb="G0"))
check("loft with rails: G1 profiles whose rails leave their tangent planes are eased at those corners "
      "(none where the rails stay in them), labels hold there",
      sorted(_cz) == sorted([(o_.name, r_) for o_, r_ in _rins[:2]])
      and all(len(v_) == 2 and all("no surface is G1" in z_[2] for z_ in v_) for v_ in _cz.values())
      and not _cz0 and all(m_["ok"] or m_.get("eased") for m_ in _cm),
      f"{_cz} {_cz0} {[m_['text'][:60] for m_ in _cm]} CVs {_cn}")
for o_ in _rcv:
    bpy.data.objects.remove(o_)
# Patch, Quad Sided: the same four-sided hole gives one untrimmed surface
# (Loft with rails' solver), G2 all round, the middle patch back
from blender_nurbs import patch as _pq
for o_ in list(ctx.selected_objects):
    o_.select_set(False)
_r = bpy.ops.nproto.patch(inputs=ins(_rins[0], _rins[2], _rins[1], _rins[3]), default_cont="G2")  # 4 edges: Quad Sided
_po = ctx.active_object
_ps = store.get_brep(_po).faces()[0].surface()
_smq = _kl()
def _pq_d(q_):
    w_ = _smq.surface_closest_point(_ps, tuple(map(float, q_)))[1][0]
    return float(np.linalg.norm(np.array(_ps.evaluate(float(w_.x), float(w_.y))) - q_))
_pdev = max(_pq_d(np.array(_rC.evaluate(a_, b_), float)) for a_ in np.linspace(0, 1, 7) for b_ in np.linspace(0, 1, 7))
_plabs = _rl_cl.labels(_po)
_pph = features.history(_po)["params"]
check("patch: four edges start as Quad Sided -- one untrimmed surface inside them, G2 on every edge, the middle patch back",
      _r == {'FINISHED'} and store.is_natural(_po) and len(store.get_brep(_po).faces()) == 1 and _pdev < 1e-4
      and _pph.get("pt_quad") is True and _pq.info.get("quad") and len(_plabs) == 4
      and all(L_["check"] and L_["check"]["ok"] for L_ in _plabs),
      f"dev {_pdev:.1e} {[(L_['name'], (L_['check'] or {}).get('text', '')[:50]) for L_ in _plabs]}")
# the Feature panel switches it off and on again (a rebuild each time)
_pg = _po.nproto_feature
_fui.sync_from_history(_po)
_pg.pt_quad = False
_tr = not store.is_natural(_po) and features.history(_po)["params"]["pt_quad"] is False
_pg.pt_quad = True
check("patch Quad Sided: the Feature panel turns it off (trimmed fill) and on (untrimmed) again",
      features.history(_po)["params"]["pt_quad"] is True and store.is_natural(_po) and _tr)
# the Feature panel draws a Quad Sided patch (every property it shows exists there)
class _LayQ:
    def __getattr__(self, n_):
        def f_(*a_, **k_):
            if n_ in ("prop", "prop_enum") and len(a_) > 1 and isinstance(a_[1], str):
                getattr(a_[0], a_[1])
            return _LayQ()
        return f_
    def __setattr__(self, n_, v_):
        pass
try:
    _pq.draw_settings(_LayQ(), _pg, _pg.pt_edges, _pq.patch_info(_po), "pt_cont_all", obj=_po,
                      h=features.history(_po))
    _dq = ""
except Exception as exc_:
    _dq = repr(exc_)
check("patch Quad Sided: the Feature panel draws (Re-trim only in the operator's panel)", not _dq, _dq)
# Degree / Spans by hand (as Square had): that net exactly
_pg.pt_quad_degree = "5"
_pg.pt_quad_spans_u, _pg.pt_quad_spans_v = 3, 2
_pd = ext.surface_cvs(store.get_brep(_po).faces()[0].surface())
check("patch Quad Sided: Degree 5, Spans 3 x 2 set by hand give that net",
      sorted((_pd["count_u"], _pd["count_v"])) == [7, 8] and _pd["degree_u"] == 5, (_pd["count_u"], _pd["count_v"]))
bpy.data.objects.remove(_po)
# three edges (as Square's three-sided patch): one corner collapses to a point,
# still one untrimmed surface on them
def _rc_line(a_, b_, n_=6):
    return [tuple(map(float, np.array(_rC.evaluate(*a_), float) * (1 - t_) + np.array(_rC.evaluate(*b_), float) * t_))
            for t_ in np.linspace(0, 1, n_)]
_t3 = [nurbs("RlTri0", [tuple(map(float, _rC.evaluate(a_, 0.0))) for a_ in np.linspace(0, 1, 6)]),
       nurbs("RlTri1", _rc_line((1, 0), (0, 1))),
       nurbs("RlTri2", [tuple(map(float, _rC.evaluate(0.0, 1 - a_))) for a_ in np.linspace(0, 1, 6)])]
for o_ in list(ctx.selected_objects):
    o_.select_set(False)
_r = bpy.ops.nproto.patch(inputs=ins(*[(c_, "S0") for c_ in _t3]), default_cont="G0", pt_quad=True)
_po = ctx.active_object
_tl = _rl_cl.labels(_po)
check("patch Quad Sided on three edges: a three-sided untrimmed surface (one corner a point), edges held",
      _r == {'FINISHED'} and features.history(_po)["params"].get("pt_quad") is True and store.is_natural(_po)
      and len(store.get_brep(_po).faces()) == 1 and len(_tl) == 3 and all(L_["check"]["ok"] for L_ in _tl),
      [(L_["name"], L_["check"]["text"][:40]) for L_ in _tl])
bpy.data.objects.remove(_po)
for o_ in _t3:
    bpy.data.objects.remove(o_)
# a Free edge (as Square's): no condition there, the other edges still G2
_gf = features.run("patch", _rins, {"cont": ["G2", "G2", "G2", "FREE"], "pt_quad": True})
_lo = store.new_object(ctx, "RlFree", _gf)
_fm = [_rl_cl.measure(_lo, o_, r_, 2) for o_, r_ in _rins[:3]]
_fl = [L_ for L_ in _rl_cl.labels(_lo)]
bpy.data.objects.remove(_lo)
_lo = store.new_object(ctx, "RlFreeOp", _gf)
features.record(_lo, "patch", _rins, {"cont": ["G2", "G2", "G2", "FREE"], "pt_quad": True})
_fl = _rl_cl.labels(_lo)
bpy.data.objects.remove(_lo)
check("patch Quad Sided: a Free edge has no condition, the other three hold G2, its label offers Free",
      all(m_["ok"] for m_ in _fm) and any(L_["value"] == "FREE" and "FREE" in L_["values"] for L_ in _fl),
      [m_["text"][:50] for m_ in _fm])
# Re-trim Planar Targets on: Square's net (nothing planar to re-trim here), still G1 all round
_gr = features.run("patch", _rins, {"cont": ["G1"] * 4, "pt_quad": True, "pt_retrim": True})
_lo = store.new_object(ctx, "RlRetrim", _gr)
_rm = [_rl_cl.measure(_lo, o_, r_, 1) for o_, r_ in _rins]
bpy.data.objects.remove(_lo)
check("patch Quad Sided with Re-trim Planar Targets: Square's net, G1 on every edge", all(m_["ok"] for m_ in _rm),
      [m_["text"][:50] for m_ in _rm])
# Even Net: fair toward even CV spacing -- labels still hold, the solver's
# fairness reference is put back afterwards
from blender_nurbs import square as _sq_en
_ge = features.run("loft", _rins, {"cont_a": "G2", "cont_b": "G2", "cont_ra": "G1", "cont_rb": "G1",
                                   "sides": [1, 1, 1, 1], "rails": 2, "even_net": True})
_lo = store.new_object(ctx, "RlEven", _ge)
_em = [_rl_cl.measure(_lo, o_, r_, c_) for (o_, r_), c_ in zip(_rins, (2, 2, 1, 1))]
_eP = np.array(ext.surface_cvs(store.get_brep(_lo).faces()[0].surface())["points"], float)
bpy.data.objects.remove(_lo)
check("loft with rails: Even Net holds G2 / G1 and leaves the solver's fairness as it was",
      all(m_["ok"] for m_ in _em) and _sq_en.FAIR_REF == "x0" and np.isfinite(_eP).all(),
      [m_["text"][:50] for m_ in _em])
for o_ in (_rL, _rR, _rB, _rT):
    bpy.data.objects.remove(o_)

# the operator: sides kept in the history, a rebuild keeps them
_r = bpy.ops.nproto.loft(inputs=ins((_lcA, "U1"), (_lcB, "U1"), (_lcT, "U0")), sides="[2, 1]", cont_a="G1")
_lh = features.history(ctx.active_object)
check("loft chained side: the operator lofts with sides and the history keeps them",
      _r == {'FINISHED'} and _lh["params"].get("sides") == [2, 1] and _lc_ops.side_counts([0, 0, 1, 1, 1]) == [2, 3])
# Surface Align to a side of two edges (a G1 loft raised to G2): the touching chain is found (in order
# along the edge), kept in the step's history, and both pieces hold G2
from blender_nurbs import salign as _sa_ch
_hA = store.new_object(ctx, "LcAlA", _lc_surf(_lcR[:, :4]))      # the two halves of one smooth surface
_hB = store.new_object(ctx, "LcAlB", _lc_surf(_lcR[:, 3:]))
_lgo = store.new_object(ctx, "LcAlignMe", features.run("loft", [(_hA, "U1"), (_hB, "U1"), (_lcT, "U0")],
                                                       {"cont_a": "G1", "sides": [2, 1]}))
_cands = [c_ for c_ in _sa_ch._pick_candidates(ctx) if c_[0] in (_hA, _hB, _lcT)]
_che = {e_: _sa_ch.find_neighbour(_lgo, e_, _cands, chains=True) for e_ in ("U0", "U1", "V0", "V1")}
_che = {e_: t_ for e_, t_ in _che.items() if isinstance(t_, list)}
_ce = next(iter(_che), None)
_okc = False
if _ce:
    for o_ in list(ctx.selected_objects):
        o_.select_set(False)
    # picked by hand: the edge, then both pieces (out of order) as its target
    _rr = bpy.ops.nproto.surface_align(inputs=ins((_lgo, _ce), (_hB, "U1"), (_hA, "U1")), **{"cont_" + _ce: "G2"})
    _st = features.stack(_lgo)[-1]
    _mm = [_rl_cl.measure(_lgo, o_, r_, 2) for o_, r_ in _che[_ce]]
    _okc = _rr == {'FINISHED'} and _st["params"].get("targets") == [2] and len(_st["inputs"]) == 2 \
        and all(m_["ok"] for m_ in _mm)
check("surface align: an edge along two edges aligns G2 to the chain of both (history keeps it, both hold)",
      len(_che) == 1 and {n_ for n_, _ in [(o_.name, r_) for o_, r_ in _che[_ce]]} == {_hA.name, _hB.name} and _okc,
      f"{ {e_: [(o_.name, r_) for o_, r_ in t_] for e_, t_ in _che.items()} } "
      + ("" if not _ce else str([m_["text"][:60] for m_ in _mm])))
for o_ in (_lgo, _hA, _hB):
    bpy.data.objects.remove(o_)
for o_ in [o_ for o_ in bpy.data.objects if o_.name.startswith(("Lc", "Loft")) and o_.name not in ("Loft",)]:
    if o_.name.startswith("Lc") or o_ is ctx.active_object:
        bpy.data.objects.remove(o_)

# --- Look at a work plane: the view turns square onto it (X right, Y up), its centre moved onto the plane
_frV = _wp.frame_from_points((1, 2, 3), (2, 2, 3.5), (1, 3, 3))
_Rv, _cv = _wp.plan_view(_frV, (5, 5, 5))
_Rb, _ = _wp.plan_view(_frV, (5, 5, 5), "BOTTOM")
check("work plane view: plan rotation is the plane's axes (behind: Y and normal reversed), centre on the plane",
      np.allclose(_Rv, np.column_stack(_frV[1:])) and np.allclose(_Rb[:, 2], -_frV[3])
      and np.allclose(_Rb[:, 0], _frV[1]) and abs(float((_cv - _frV[0]) @ _frV[3])) < 1e-12
      and abs(np.linalg.det(_Rb) - 1) < 1e-12)
_v3 = next((a_ for w_ in bpy.context.window_manager.windows for a_ in w_.screen.areas if a_.type == 'VIEW_3D'), None)
_wpV = _wp.add_plane(ctx.scene, "Look", _frV)
_pv = ctx.preferences.view.smooth_view
ctx.preferences.view.smooth_view = 0
_r3 = _v3.spaces.active.region_3d if _v3 else None
if _r3 is not None:
    _r3.view_perspective = 'PERSP'
    with ctx.temp_override(area=_v3, region=next(r_ for r_ in _v3.regions if r_.type == 'WINDOW'), space_data=_v3.spaces.active):
        _rr = bpy.ops.nproto.workplane_view()
    _Rq = np.array(_r3.view_rotation.to_matrix(), float)
    _okv = _rr == {'FINISHED'} and np.allclose(_Rq, np.column_stack(_frV[1:]), atol=1e-6) \
        and abs(float((np.array(_r3.view_location) - _frV[0]) @ _frV[3])) < 1e-5 \
        and _r3.view_perspective == ('ORTHO' if ctx.preferences.inputs.use_auto_perspective else 'PERSP')
else:
    _okv = False
ctx.preferences.view.smooth_view = _pv
_stW.planes.clear(); _stW.active = -1; _stW.index = 0
check("work plane view: Look at turns a 3D view onto the active plane (orthographic with Auto Perspective)", _okv)


# --- Merge Patches: a 2 x 2 grid of patches cut from one cubic surface (one piece transposed,
# one flipped) merges back exactly; Reduce finds the original net again, Refit stays on it
from blender_nurbs import merge as _mg
from blender_nurbs.kernel.classa import insert_knot as _ik
_mP = np.array([[(i / 6, j / 5, 0.15 * math.sin(2.3 * i / 6 + 1.7 * j / 5) + 0.02 * ((i * 7 + j * 3) % 5 - 2))
                 for j in range(6)] for i in range(7)])
_mUu, _mUv = list(geom.clamped_uniform_knots(7, 3)), list(geom.clamped_uniform_knots(6, 3))

def _msplit(P, U, t, axis):
    A = P if axis == 0 else P.transpose(1, 0, 2)
    sh = A.shape
    A, U = A.reshape(sh[0], -1), list(U)
    for _ in range(3):
        A, U = _ik(A, U, 3, t)
    A = np.asarray(A)
    f = [i for i, x in enumerate(U) if abs(x - t) < 1e-12][0]
    L, R = A[:f].reshape(f, *sh[1:]), A[f - 1:].reshape(len(A) - f + 1, *sh[1:])
    UL, UR = U[:f + 3] + [t], [t] + U[f:]
    UL, UR = [(x - UL[0]) / (t - UL[0]) for x in UL], [(x - t) / (UR[-1] - t) for x in UR]
    return ((L, UL), (R, UR)) if axis == 0 else ((L.transpose(1, 0, 2), UL), (R.transpose(1, 0, 2), UR))

_mobjs = []
for (_N, _UN) in _msplit(_mP, _mUu, 0.5, 0):
    for (_A, _VA) in _msplit(_N, _mUv, 0.4, 1):
        _k_ = len(_mobjs)
        _UU, _VV = _UN, _VA
        if _k_ == 1:
            _A, _UU, _VV = _A.transpose(1, 0, 2), _VV, _UU
        if _k_ == 2:
            _A, _UU = _A[::-1], [1 - x for x in _UU[::-1]]
        _s = ext.surface_from_cvs(3, 3, [tuple(map(float, x)) for x in _A.reshape(-1, 3)],
                                  [float(x) for x in _UU], [float(x) for x in _VV])
        _mobjs.append(store.new_object(ctx, f"MergePiece{_k_}", _s))
bpy.ops.object.select_all(action='DESELECT')
for _o in _mobjs:
    _o.select_set(True)
ctx.view_layer.objects.active = _mobjs[0]
bpy.ops.nproto.join()
_MJ = ctx.active_object
_mtext = _MJ[store.PROP_BREP]
_mres = {m_: _mg.merge_text(_mtext, m_, 0, 3)[1] for m_ in ("EXACT", "REDUCE", "REFIT")}
r = bpy.ops.nproto.merge_patches(merge_mode="REDUCE")
check("merge patches: 2x2 grid merges exactly, Reduce gives the original 7x6 net back, Refit stays on it",
      _mres["EXACT"]["route"] == "grid" and _mres["EXACT"]["grid"] == (2, 2) and _mres["EXACT"]["deviation"] == 0.0
      and _mres["REDUCE"]["cvs"] == (7, 6) and _mres["REDUCE"]["deviation"] < 1e-9
      and _mres["REFIT"]["deviation"] < 1e-6, {m_: (v["cvs"], f"{v['deviation']:.1e}") for m_, v in _mres.items()})
check("merge patches operator: one untrimmed face with a merge step", r == {'FINISHED'}
      and store.get_brep(_MJ).face_count() == 1 and store.is_natural(_MJ))


# --- Merge Patches keeps the edges: a strip A | B | C cut from one bicubic; B + C merged and refitted
# with G2 edges keeps the tangent plane and curvature across the edge it shares with A, Free does not
_sP = np.array([[(i / 9 * 2, j / 5, 0.2 * math.sin(3.1 * i / 9 + 1.3 * j / 5) + 0.025 * ((i * 7 + j * 3) % 5 - 2))
                 for j in range(6)] for i in range(10)])
_sUu, _sUv = list(geom.clamped_uniform_knots(10, 3)), list(geom.clamped_uniform_knots(6, 3))
(_sA, _sUA), (_sBC, _sUBC) = _msplit(_sP, _sUu, 0.3, 0)
(_sB, _sUB), (_sC, _sUC) = _msplit(_sBC, _sUBC, 0.45, 0)
_smk = lambda N, U_: ext.surface_from_cvs(3, 3, [tuple(map(float, x)) for x in N.reshape(-1, 3)],
                                          [float(x) for x in U_], [float(x) for x in _sUv])
_sobjs = [store.new_object(ctx, "StripB", _smk(_sB, _sUB)), store.new_object(ctx, "StripC", _smk(_sC, _sUC))]
bpy.ops.object.select_all(action='DESELECT')
for _o in _sobjs:
    _o.select_set(True)
ctx.view_layer.objects.active = _sobjs[0]
bpy.ops.nproto.join()
_SJ = ctx.active_object
_SJ_text = _SJ[store.PROP_BREP]
_sAs = _smk(_sA, _sUA)

def _snbr(out_text):
    """Largest normal angle (deg) and relative normal-curvature jump between surface A's u=1 edge and the
    merged surface's matching side."""
    f = store.brep_from_text(out_text).faces()[0].surface()
    d = ext.surface_cvs(f)
    net = {"P": np.concatenate([np.array(d["points"]).reshape(d["count_u"], d["count_v"], 3),
                                np.ones((d["count_u"], d["count_v"], 1))], 2),
           "Uu": np.array(d["knots_u"]), "Uv": np.array(d["knots_v"]), "pu": d["degree_u"], "pv": d["degree_v"]}
    pm = np.array(_sAs.evaluate(1.0, 0.5))
    side = min(_mg.SIDES, key=lambda s_: np.linalg.norm(_mg._side_samples(net, s_, 3)[1] - pm))
    vs = np.linspace(0.05, 0.95, 13)
    au_ = side[0] == "u"
    c_ = [0.0 if side[1] == "0" else 1.0]
    args = (c_, vs) if au_ else (vs, c_)
    S_ = _mg.eval_d(net, *args).reshape(-1, 3)
    Su_, Sv_ = _mg.eval_d(net, *args, 1, 0).reshape(-1, 3), _mg.eval_d(net, *args, 0, 1).reshape(-1, 3)
    Scc = (_mg.eval_d(net, *args, 2, 0) if au_ else _mg.eval_d(net, *args, 0, 2)).reshape(-1, 3)
    Sc = Su_ if au_ else Sv_
    ang, dk, kr = 0.0, 0.0, 0.0
    for q, su_, sv_, sc, scc in zip(S_, Su_, Sv_, Sc, Scc):
        vv = min(np.linspace(0, 1, 801), key=lambda v: np.linalg.norm(np.array(_sAs.evaluate(1.0, v)) - q))
        _, au, av = [np.array(x) for x in _sAs.derivatives(1.0, vv)[:3]]
        h = 1e-5
        au2 = (au - np.array(_sAs.derivatives(1.0 - h, vv)[1])) / h
        na = np.cross(au, av); na /= np.linalg.norm(na)
        nn = np.cross(su_, sv_); nn /= np.linalg.norm(nn)
        ang = max(ang, math.degrees(math.acos(min(1.0, abs(float(na @ nn))))))
        ka, km = abs(au2 @ na) / (au @ au), abs(scc @ nn) / (sc @ sc)
        dk, kr = max(dk, abs(ka - km)), max(kr, ka)
    return ang, dk / kr

_sres = {}
for _c in ("G2", "FREE"):
    _t, _i = _mg.merge_text(_SJ[store.PROP_BREP], "REFIT", 0, 3, cont=_c, knots="UNIFORM")   # off the knot lines
    _sres[_c] = (_snbr(_t), _i)
_t, _i = _mg.merge_text(_SJ[store.PROP_BREP], "REDUCE", 0, 3, cont="G2")
_sres["REDUCE"] = (_snbr(_t), _i)
check("merge patches edges: Refit G2 keeps tangent plane and curvature at the neighbour, Free drifts",
      _sres["G2"][0][0] < 0.005 and _sres["G2"][0][1] < 0.01 and _sres["G2"][1]["edges_ok"]
      and (_sres["FREE"][0][0] > 0.02 or _sres["FREE"][0][1] > 0.03)
      and _sres["REDUCE"][0][0] < 1e-4 and _sres["REDUCE"][0][1] < 0.01,
      {k: (f"{v[0][0]:.4f} deg", f"{100 * v[0][1]:.2f}%", v[1]["cvs"]) for k, v in _sres.items()})
_t, _i = _mg.merge_text(_SJ[store.PROP_BREP], "REFIT", 0, 3, 1, 0, cont="G2")
check("merge patches: forced spans that cannot hold the edges stop and report it (no runaway spans)",
      _i["spans"][0] == 1 and _i["spans"][1] <= 12 and not _i["edges_ok"], _i["spans"])
_kres = {k_: _mg.merge_text(_SJ[store.PROP_BREP], "REFIT", 0, 3, cont="G2", knots=k_)[1] for k_ in ("UNIFORM", "ADAPTIVE")}
check("merge patches: Adaptive knots refit the strip with far fewer spans than Uniform (on the patches' knot lines)",
      _kres["ADAPTIVE"]["edges_ok"] and _kres["ADAPTIVE"]["deviation"] < 1e-6
      and _kres["ADAPTIVE"]["spans"][0] * 2 <= _kres["UNIFORM"]["spans"][0]
      and _kres["UNIFORM"]["edges_ok"], {k_: (v["spans"], f"{v['deviation']:.1e}") for k_, v in _kres.items()})
# a sampled grid (as the projected route gives) with a sharp ridge: edges checked on the grid's own samples
# (not rounded to them), and Adaptive puts the spans at the ridge
_gm = 61
_gt = np.linspace(0, 1, _gm)
_gU, _gV = np.meshgrid(_gt, _gt, indexing="ij")
_gG = np.stack([_gU * 2, _gV, 0.15 * np.tanh(8 * (_gU - 0.7)) + 0.05 * np.sin(3 * _gV)], -1)

def _gref(a, b, du=0, dv=0):
    if du or dv:
        return np.zeros((len(a), len(b), 3))
    return _gG[np.ix_(np.rint(np.asarray(a, float) * (_gm - 1)).astype(int),
                      np.rint(np.asarray(b, float) * (_gm - 1)).astype(int))]
_gerr = lambda nt: float(np.linalg.norm(_mg.eval_net(nt, _gt, _gt) - _gG, axis=2).max())
_gres = {k_: _mg.refit_net(_gref, 3, 0, 0, 2.2e-4, {s_: 0 for s_ in _mg.SIDES}, 2.2e-5, m=_gm, surf_err=_gerr,
                           knots=k_, grid_only=True) for k_ in ("UNIFORM", "ADAPTIVE")}
check("merge patches projected grid: edges pass on the grid samples, Adaptive needs fewer spans than Uniform",
      all(v[2] <= 2.2e-4 and _mg.edges_ok(v[3], 2.2e-5) for v in _gres.values())
      and _gres["ADAPTIVE"][1][0] < _gres["UNIFORM"][1][0], {k_: (v[1], f"{v[2]:.1e}") for k_, v in _gres.items()})
# a bicubic (3 x 2 spans) split in two by a wavy curve: the trimmed halves share one surface, so the
# projected grid is built in its parameters and Refit gets the bicubic back (an even 3D grid needed ~25 x 22)
_wP = [(i / 5 * 2, j / 4 + 30, 0.25 * math.sin(2.1 * i / 5 + 1.4 * j / 4) + 0.04 * ((i * 7 + j * 3) % 5 - 2))
       for i in range(6) for j in range(5)]
_wS = store.new_object(ctx, "WavySplit", ext.surface_from_cvs(
    3, 3, _wP, [float(x) for x in geom.clamped_uniform_knots(6, 3)], [float(x) for x in geom.clamped_uniform_knots(5, 3)]))
_wC = poly("WavySplit_cut", [(1.0 + 0.15 * math.sin(6 * y_), y_ + 30, 2.0) for y_ in np.linspace(-0.2, 1.2, 41)])
bpy.ops.object.select_all(action='DESELECT'); _wS.select_set(True); _wC.select_set(True); ctx.view_layer.objects.active = _wS
r = bpy.ops.nproto.split(view_dir=(0, 0, -1))
_wpcs = [o_ for o_ in bpy.data.objects if o_.name.startswith("WavySplit") and store.is_nurbs(o_) and not o_.hide_get()]
bpy.ops.object.select_all(action='DESELECT')
for _o in _wpcs:
    _o.select_set(True)
ctx.view_layer.objects.active = _wpcs[0]
bpy.ops.nproto.join()
_wres = {k_: _mg.merge_text(ctx.active_object[store.PROP_BREP], "REFIT", 0, 3, **kw_)[1]
         for k_, kw_ in (("G0", dict(cont="G0")), ("FREE", dict(cont="FREE", knots="UNIFORM")))}
check("merge patches projected: halves of a split bicubic refit on their surface's parameters in <= 4 x 4 spans",
      r == {'FINISHED'} and len(_wpcs) == 2
      and all(v["route"] == "projected" and v["grid_space"] == "surface" and v["edges_ok"]
              and v["deviation"] <= v["tolerance"] and max(v["spans"]) <= 4 for v in _wres.values()),
      {k_: (v["spans"], v.get("grid_space"), f"{v['deviation']:.1e}") for k_, v in _wres.items()})
# seams: two Bezier patches meeting with a kink, two meeting tangent with a curvature jump. The check
# names each seam; Refit keeps it sharp (a full / C1 knot there) and fits exactly, Smooth rounds it off
_zU4 = [0, 0, 0, 0, 1, 1, 1, 1]

def _zpatch(name, x0, zf):
    P_ = [(x0 + i / 3, j / 3, zf(i) + 0.05 * math.sin(j)) for i in range(4) for j in range(4)]
    return store.new_object(ctx, name, ext.surface_from_cvs(3, 3, P_, _zU4, _zU4))

def _zjoin(a_, b_):
    bpy.ops.object.select_all(action='DESELECT')
    a_.select_set(True)
    b_.select_set(True)
    ctx.view_layer.objects.active = a_
    bpy.ops.nproto.join()
    return ctx.active_object
_zcases = {"KINK": _zjoin(_zpatch("SeamKL", 0, lambda i: 0.3 * i / 3), _zpatch("SeamKR", 1, lambda i: 0.3 - 0.4 * i / 3)),
           "G1": _zjoin(_zpatch("SeamGL", 0, lambda i: 0.0), _zpatch("SeamGR", 1, lambda i: 0.3 * (0, 0, 1 / 3, 1)[i]))}
_zres = {}
for _zk, _zo in _zcases.items():
    _zT = _zo[store.PROP_BREP]
    _zres[_zk] = (_mg.merge_text(_zT, "EXACT", 0, 3)[1]["seams"],
                  _mg.merge_text(_zT, "REFIT", 0, 3, seams="KEEP")[1], _mg.merge_text(_zT, "REFIT", 0, 3, seams="SMOOTH")[1])
check("merge patches seams: a kink and a curvature jump are found, Refit keeps them sharp and exact, Smooth rounds them",
      [r_["kind"] for r_ in _zres["KINK"][0]] == ["KINK"] and _zres["KINK"][0][0]["angle"] > 30
      and [r_["kind"] for r_ in _zres["G1"][0]] == ["G1"] and _zres["G1"][0][0]["angle"] < 0.01
      and all(v[1]["deviation"] < 1e-9 and v[1]["spans"][0] == 2 and v[1]["seams_kept"] == 1 and v[1]["edges_ok"]
              for v in _zres.values())
      and all(v[2]["spans"][0] > 4 and v[2]["seams_kept"] == 0 for v in _zres.values())
      and _mres["EXACT"]["seams"] and all(r_["kind"] == "G2" for r_ in _mres["EXACT"]["seams"]),
      {k_: ([(r_["kind"], f"{r_['angle']:.1f}", f"{r_['curv']:.2f}") for r_ in v[0]], v[1]["spans"],
             f"{v[1]['deviation']:.1e}", v[2]["spans"], f"{v[2]['deviation']:.1e}") for k_, v in _zres.items()})
_zt, _zi = _mg.merge_text(_zcases["KINK"][store.PROP_BREP], "REFIT", 0, 3, 1, 0)
check("merge patches seams: spans set too few for a kink smooth it and say so",
      _zi["seams_kept"] == 0 and "too few" in _zi["note"], (_zi["spans"], _zi["note"]))
bpy.ops.object.select_all(action='DESELECT')
_SJ.select_set(True)
ctx.view_layer.objects.active = _SJ
r = _r_sj = bpy.ops.nproto.merge_patches(merge_mode="REFIT", merge_knots="UNIFORM")
_dmap_ok = False
try:
    from blender_nurbs import draw as _dr
    _dops = ctx.window_manager.operators       # (in the background the redo list may not hold it)
    _dpanel = bool(_dops) and _dops[-1].bl_idname in ("NPROTO_OT_merge_patches", "nproto.merge_patches")
    _dmap_ok = (_dr.merge_devmap_shown(ctx) is not None) == _dpanel and _mg.DEVMAP["obj"] == _SJ.name
except Exception as exc:
    _dmap_ok = repr(exc)
_dP, _dN, _dT, _dV = _mg.DEVMAP.get("data", ([], [], [], []))
_dt, _di = _mg.merge_text(_SJ_text, "REFIT", 0, 5, cont="FREE", knots="UNIFORM", devmap=True)
_dfc = store.brep_from_text(_SJ_text).faces()
_dk = np.random.default_rng(1).choice(len(_di["devmap"][0]), 60, replace=False)
_dkd = np.array([min(f_.closest_point(tuple(map(float, _di["devmap"][0][k_])))[1] for f_ in _dfc) for k_ in _dk])
check("merge patches deviation map: its distances agree with the kernel's closest points to the patches",
      float(np.abs(_dkd - np.abs(_di["devmap"][3][_dk])).max()) < 0.05 * _di["tolerance"]
      and float(np.abs(_di["devmap"][3]).max()) > 0.2 * _di["deviation"],
      (f"{float(np.abs(_dkd - np.abs(_di['devmap'][3][_dk])).max()):.1e}", f"{_di['deviation']:.1e}"))
check("merge patches deviation map: drawn while the panel is open, a mesh with one value per point, "
      "its largest value the reported deviation",
      _dmap_ok is True and len(_dT) > 0 and len(_dV) == len(_dP)
      and abs(float(np.abs(_dV).max()) - _mg.LAST_INFO["deviation"]) <= 0.2 * _mg.LAST_INFO["deviation"] + 1e-12,
      (_dmap_ok, len(_dP), float(np.abs(_dV).max()) if len(_dV) else None, _mg.LAST_INFO.get("deviation")))
# Reduce Degree: the 2 x 2 test cubic cut in two and raised to degree 5 comes back down to the cubic net
# (edges held at G2)
from blender_nurbs.geom import raise_array as _ra
_rdP = []
for (_N, _UN) in _msplit(_mP, _mUu, 0.5, 0):
    if sum(1 for x in _UN if x == _UN[0]) > 4:        # the right piece starts with an extra knot (a dead CV)
        _N, _UN = _N[1:], list(_UN)[1:]
    _q, _UU2, _A2 = _ra(3, list(_UN), _N.reshape(len(_N), -1), 5)
    _A2 = np.asarray(_A2).reshape(len(_A2), _N.shape[1], 3).transpose(1, 0, 2)
    _q, _VV2, _A3 = _ra(3, list(_mUv), _A2.reshape(len(_A2), -1), 5)
    _A3 = np.asarray(_A3).reshape(len(_A3), -1, 3).transpose(1, 0, 2)
    _rdP.append(store.new_object(ctx, f"RdPiece{len(_rdP)}", ext.surface_from_cvs(
        5, 5, [tuple(map(float, x)) for x in _A3.reshape(-1, 3)], [float(x) for x in _UU2], [float(x) for x in _VV2])))
_RDJ = _zjoin(*_rdP)
_rd0 = _mg.merge_text(_RDJ[store.PROP_BREP], "REDUCE", 0, 3, cont="G2")[1]
_rd1 = _mg.merge_text(_RDJ[store.PROP_BREP], "REDUCE", 0, 3, cont="G2", lower_degree=True)[1]
check("merge patches Reduce Degree: degree-5 patches of a cubic come back as the cubic net, edges held",
      _rd0["degree"] == (5, 5) and _rd1["degree"] == (3, 3) and _rd1["cvs"] == (7, 6) and _rd1["deviation"] < 1e-9
      and _rd1["edges_ok"], (_rd0["degree"], _rd0["cvs"], _rd1["degree"], _rd1["cvs"], f"{_rd1['deviation']:.1e}"))
# Keep Original: the merge goes to a new object, the joined patches stay (hidden) with their faces
bpy.ops.object.select_all(action='DESELECT')
_RDJ.select_set(True)
ctx.view_layer.objects.active = _RDJ
r = bpy.ops.nproto.merge_patches(merge_mode="REDUCE", merge_reduce_degree=True, merge_keep_original=True)
_rdn = ctx.view_layer.objects.active
check("merge patches Keep Original: a new merged object, the joined patches kept hidden and unchanged",
      r == {'FINISHED'} and _rdn != _RDJ and _rdn.name.endswith("Merged") and store.get_brep(_rdn).face_count() == 1
      and _RDJ.hide_get() and store.get_brep(_RDJ).face_count() == 2
      and features.stack(_rdn)[-1]["params"].get("merge_reduce_degree") is True,
      (r, _rdn.name, _RDJ.hide_get(), store.get_brep(_RDJ).face_count()))
# a joined feature that lost an input (deleted) still takes a history step: it starts from its last geometry
_lj = _zjoin(_zpatch("LostA", 5, lambda i: 0.0), _zpatch("LostB", 6, lambda i: 0.0))
_lost = [o_ for o_ in bpy.data.objects if o_.name.startswith("LostB")][0]
bpy.data.objects.remove(_lost)
_lj_hist = features.history(_lj) is None
bpy.ops.object.select_all(action='DESELECT')
_lj.select_set(True)
ctx.view_layer.objects.active = _lj
r = bpy.ops.nproto.merge_patches(merge_mode="REFIT")
check("history: a feature whose input was deleted still takes a step (Merge Patches on it works)",
      _lj_hist and r == {'FINISHED'} and store.get_brep(_lj).face_count() == 1 and features.PROP_BASE in _lj,
      (_lj_hist, r))
check("merge patches operator: Knots option reaches the history step", _r_sj == {'FINISHED'}
      and features.stack(_SJ)[-1]["op"] == "merge_patches"
      and features.stack(_SJ)[-1]["params"].get("merge_knots") == "UNIFORM")

# --- Set Origin / Apply Transform on NURBS objects (pivot.py): the B-rep follows the mesh, and things built
# from it (a G2 blend loft) and its own rebuilds stay where they were
from blender_nurbs import pivot
_pl1 = nurbs("PivLine1", [(0, 0, 0), (0, 1, 0)], order=2)
_pl2 = nurbs("PivLine2", [(1, 0, 1), (1, 1, 1)], order=2)
bpy.ops.nproto.extrude(inputs=ins((_pl1, "S0")), distance=-0.5, direction="Z")
_pe1 = ctx.active_object
bpy.ops.nproto.extrude(inputs=ins((_pl2, "S0")), distance=0.5, direction="Z")
_pe2 = ctx.active_object
def _wpts(o):
    M = np.array(geom.world_matrix(o))
    P = np.array([v.co[:] for v in o.data.vertices])
    return P @ M[:3, :3].T + M[:3, 3]
def _bpts(o):
    M = geom.world_matrix(o)
    return np.array([tuple(M @ Vector(e.point_at(t))) for e in store.get_brep(o).edges() for t in e.parameter_range()])
def _near(P, Q):
    return max(float(np.min(np.linalg.norm(Q - p, axis=1))) for p in P)
def _edge_of(o, z):       # the extrude's edge at height z (the one on its line)
    return min(range(len(store.get_brep(o).edges())), key=lambda i: abs(_bpts(o)[2 * i:2 * i + 2, 2].mean() - z))
r = bpy.ops.nproto.loft(inputs=ins((_pe1, f"E{_edge_of(_pe1, 0.0)}"), (_pe2, f"E{_edge_of(_pe2, 1.0)}")),
                        cont_a="G2", cont_b="G2")
_plo = ctx.active_object
_plo_w, _pe2_w, _pe2_b = _wpts(_plo), _wpts(_pe2), _bpts(_pe2)
bpy.ops.object.select_all(action='DESELECT')
_pe2.select_set(True); ctx.view_layer.objects.active = _pe2
bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
ctx.view_layer.update()
_moved = (_pe2.location - Vector((0, 0, 0))).length
check("set origin on a NURBS result: the object stays, its B-rep follows (placement stored)",
      _moved > 0.1 and _near(_bpts(_pe2), _pe2_b) < 1e-9 and features.PROP_PLACE in _pe2
      and _pe2.get(features.PROP_KEY) == features.build_key(_pe2), (_moved, _near(_bpts(_pe2), _pe2_b)))
features.rebuild(_pe2, ctx, force=True)
features.rebuild(_plo, ctx, force=True)
check("set origin: the result's rebuild and the blend built from it stay in place",
      _near(_bpts(_pe2), _pe2_b) < 1e-9 and _near(_wpts(_plo), _plo_w) < 1e-6 and _near(_wpts(_pe2), _pe2_w) < 1e-6,
      (_near(_bpts(_pe2), _pe2_b), _near(_wpts(_plo), _plo_w)))
_pe2.rotation_euler = (0.3, 0.2, 0.5); _pe2.scale = (1.5, 1.5, 1.5)
ctx.view_layer.update()
_rot_b = _bpts(_pe2)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
ctx.view_layer.update()
_ok_apply = _near(_bpts(_pe2), _rot_b) < 1e-6
features.rebuild(_pe2, ctx, force=True)
check("apply rotation / scale on a NURBS result: B-rep and rebuild stay where they were",
      _ok_apply and _near(_bpts(_pe2), _rot_b) < 1e-6 and tuple(_pe2.location) == (0, 0, 0),
      (_ok_apply, _near(_bpts(_pe2), _rot_b)))
# a plain NURBS object (no history), trimmed texts included
_pfree = store.new_object_text(ctx, "PivFree", store.get_brep(_pe1) and store.brep_to_text(store.get_brep(_pe1)))
_pfree[store.PROP_UNTRIM] = _pfree[store.PROP_BREP]
_pf_b = _bpts(_pfree)
bpy.ops.object.select_all(action='DESELECT')
_pfree.select_set(True); ctx.view_layer.objects.active = _pfree
ctx.scene.cursor.location = (3, 2, 1)
bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
ctx.view_layer.update()
_pf_u = store.brep_from_text(_pfree[store.PROP_UNTRIM])
check("set origin on a plain NURBS object: its B-rep and Untrim text follow",
      tuple(_pfree.location) == (3, 2, 1) and _near(_bpts(_pfree), _pf_b) < 1e-9
      and _near(np.array([tuple(geom.world_matrix(_pfree) @ Vector(e.point_at(e.parameter_range()[0])))
                          for e in _pf_u.edges()]), _pf_b) < 1e-9, tuple(_pfree.location))
ctx.scene.cursor.location = (0, 0, 0)
# a G move is not an origin change
_pfree.location = (4, 4, 4)
ctx.view_layer.update()
check("moving a NURBS object leaves its B-rep alone", not pivot.check(_pfree)
      and _near(_bpts(_pfree), _pf_b + np.array([1, 2, 3])) < 1e-9)

# Mirror (Ctrl+M, a negative scale) then Apply Scale: the surface keeps the side Blender showed (its B-rep is turned
# over with the mesh), and a G2 blend built from its edge keeps that edge (its ref follows the renumbering)
_pm1 = nurbs("PivMir1", [(5, 0, 0), (5, 1, 0)], order=2)
_pm2 = nurbs("PivMir2", [(6, 0, 1), (6, 1, 1)], order=2)
bpy.ops.nproto.extrude(inputs=ins((_pm1, "S0")), distance=-0.5, direction="Z"); _pme1 = ctx.active_object
bpy.ops.nproto.extrude(inputs=ins((_pm2, "S0")), distance=0.5, direction="Z"); _pme2 = ctx.active_object
bpy.ops.nproto.loft(inputs=ins((_pme1, f"E{_edge_of(_pme1, 0.0)}"), (_pme2, f"E{_edge_of(_pme2, 1.0)}")),
                    cont_a="G2", cont_b="G2")
_pml = ctx.active_object
def _shown(o):
    N = o.matrix_world.to_3x3().inverted().transposed()
    return (N @ sum((q.normal for q in o.data.polygons), Vector())).normalized()
def _brep_n(o):
    f_ = store.get_brep(o).faces()[0]; (a_, b_), (c_, d_) = f_.uv_domain()
    n_ = Vector(f_.outward_normal_at_uv((a_ + b_) / 2, (c_ + d_) / 2))
    return (geom.world_matrix(o).to_3x3().inverted().transposed() @ n_).normalized()
bpy.ops.object.select_all(action='DESELECT')
_pme2.select_set(True); ctx.view_layer.objects.active = _pme2
bpy.ops.transform.mirror(orient_type='LOCAL', constraint_axis=(True, False, False))
ctx.view_layer.update()
features.rebuild(_pml, ctx, force=True)
_pml_w, _pme2_w, _pme2_n = _wpts(_pml), _wpts(_pme2), _shown(_pme2)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
ctx.view_layer.update()
_pm_ok = (_shown(_pme2).dot(_pme2_n) > 0.999 and _brep_n(_pme2).dot(_pme2_n) > 0.999
          and _near(_wpts(_pme2), _pme2_w) < 1e-6)
features.rebuild(_pme2, ctx, force=True)
features.rebuild(_pml, ctx, force=True)
ctx.view_layer.update()
check("mirror + apply scale: same side shown and in the B-rep, rebuilds and the G2 blend stay in place",
      _pm_ok and _shown(_pme2).dot(_pme2_n) > 0.999 and _brep_n(_pme2).dot(_pme2_n) > 0.999
      and _near(_wpts(_pme2), _pme2_w) < 1e-6 and _near(_wpts(_pml), _pml_w) < 1e-6 and _near(_pml_w, _wpts(_pml)) < 1e-6,
      (_pm_ok, tuple(_shown(_pme2)), tuple(_brep_n(_pme2)), _near(_wpts(_pml), _pml_w), features.history(_pml)["refs"]))

# --- Move Pivot (pivottool.py): the origin goes to a point; the geometry, children and B-rep stay in the world.
# (The modal itself, its undo in Edit Mode and on a CV cage were checked in a GUI Blender with simulated input.)
from blender_nurbs import pivottool
# a turned, scaled mesh with a child
_mv_me = bpy.data.meshes.new("MvPivMesh")
_mv_me.from_pydata([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0.5)], [], [(0, 1, 2, 3)])
_mv = bpy.data.objects.new("MvPiv", _mv_me); ctx.collection.objects.link(_mv)
_mv.location = (2, 1, 0); _mv.rotation_euler = (0.2, 0.4, 0.9); _mv.scale = (1.5, 0.5, 2.0)
_mv_ch = bpy.data.objects.new("MvPivChild", None); ctx.collection.objects.link(_mv_ch)
_mv_ch.parent = _mv; _mv_ch.location = (0.3, 0.2, 0.1)
ctx.view_layer.update()
_mv_w, _mv_cw = _wpts(_mv), _mv_ch.matrix_world.copy()
_P = np.array([3.0, -1.0, 2.0])
_ok_m = pivottool.move_origin(_mv, _P)
ctx.view_layer.update()
check("move pivot, mesh: origin at the point, vertices and child stay in the world",
      _ok_m and np.linalg.norm(np.array(_mv.matrix_world.translation) - _P) < 1e-6 and _near(_wpts(_mv), _mv_w) < 1e-6
      and max(abs(a - b) for ra, rb in zip(_mv_ch.matrix_world, _mv_cw) for a, b in zip(ra, rb)) < 1e-6
      and tuple(_mv.scale) == (1.5, 0.5, 2.0), (tuple(_mv.matrix_world.translation), _near(_wpts(_mv), _mv_w)))
# a curve
_mvc = nurbs("MvPivCurve", [(0, 0, 0), (1, 1, 0), (2, 0, 1), (3, 1, 1)])
_mvc.rotation_euler = (0, 0, 0.7)
ctx.view_layer.update()
_cw = [tuple(_mvc.matrix_world @ p.co.xyz) for p in _mvc.data.splines[0].points]
pivottool.move_origin(_mvc, (1, 2, 3)); ctx.view_layer.update()
_cw2 = [tuple(_mvc.matrix_world @ p.co.xyz) for p in _mvc.data.splines[0].points]
check("move pivot, curve: its points stay", max(abs(a - b) for p, q in zip(_cw, _cw2) for a, b in zip(p, q)) < 1e-6
      and (Vector(_mvc.matrix_world.translation) - Vector((1, 2, 3))).length < 1e-6)
# a NURBS feature with a G2 blend built from its edge: everything stays through rebuilds
_mpl1 = nurbs("MvLine1", [(0, 5, 0), (0, 6, 0)], order=2)
_mpl2 = nurbs("MvLine2", [(1, 5, 1), (1, 6, 1)], order=2)
bpy.ops.nproto.extrude(inputs=ins((_mpl1, "S0")), distance=-0.5, direction="Z"); _mpe1 = ctx.active_object
bpy.ops.nproto.extrude(inputs=ins((_mpl2, "S0")), distance=0.5, direction="Z"); _mpe2 = ctx.active_object
bpy.ops.nproto.loft(inputs=ins((_mpe1, f"E{_edge_of(_mpe1, 0.0)}"), (_mpe2, f"E{_edge_of(_mpe2, 1.0)}")),
                    cont_a="G2", cont_b="G2")
_mplo = ctx.active_object
_plo_w, _pe2_w, _pe2_b = _wpts(_mplo), _wpts(_mpe2), _bpts(_mpe2)
_r = pivottool.move_origin(_mpe2, (1.0, 5.5, 1.25))
ctx.view_layer.update()
_k_ok = _mpe2.get(features.PROP_KEY) == features.build_key(_mpe2)
features.rebuild(_mpe2, ctx, force=True); features.rebuild(_mplo, ctx, force=True)
check("move pivot, NURBS feature: B-rep follows, its rebuild and the G2 blend built from it stay",
      _r and _k_ok and (Vector(_mpe2.location) - Vector((1.0, 5.5, 1.25))).length < 1e-6
      and _near(_bpts(_mpe2), _pe2_b) < 1e-9 and _near(_wpts(_mpe2), _pe2_w) < 1e-6 and _near(_wpts(_mplo), _plo_w) < 1e-6,
      (_r, _k_ok, _near(_bpts(_mpe2), _pe2_b), _near(_wpts(_mplo), _plo_w)))
# the operator's apply step on several objects, in mesh Edit Mode (it leaves and comes back)
bpy.ops.object.select_all(action='DESELECT')
_mv.select_set(True); ctx.view_layer.objects.active = _mv
bpy.ops.object.mode_set(mode='EDIT')
_mv_w = _wpts(_mv)
_moved = pivottool.apply_targets(ctx, [(_mv, np.array([0.0, 0.0, 0.0]))])
_back = _mv.mode == 'EDIT'
bpy.ops.object.mode_set(mode='OBJECT')
check("move pivot in Edit Mode: origin moved, still in Edit Mode, vertices stay",
      _moved == [_mv.name] and _back and Vector(_mv.matrix_world.translation).length < 1e-6
      and _near(_wpts(_mv), _mv_w) < 1e-6, (_moved, _back))
# a NURBS surface with its CV cage open: the surface's origin moves, Cancel still restores it exactly in place
bpy.ops.nproto.add_primitive(kind="PLANE"); _cs = ctx.active_object
_cs.location = (0, -6, 0); ctx.view_layer.update()
_cs_b = _bpts(_cs)
bpy.ops.nproto.edit_cvs()
_cage = ctx.active_object
_moved = pivottool.apply_targets(ctx, [(pivottool.owner(_cage), np.array([0.5, -5.5, 0.0]))])
_still = _cage.name in bpy.data.objects and _cage.mode == 'EDIT'
bpy.ops.nproto.cvs_cancel()
check("move pivot on a CV cage: the surface's origin moves, the cage stays open, Cancel keeps it in place",
      _moved == [_cs.name] and _still and (Vector(_cs.location) - Vector((0.5, -5.5, 0))).length < 1e-6
      and _near(_bpts(_cs), _cs_b) < 1e-9, (_moved, _still, tuple(_cs.location), _near(_bpts(_cs), _cs_b)))

# --- Nudge CVs (cvnudge.py): continuity guides at an edge that meets another surface, lock, put on guide
from blender_nurbs import cvnudge as _nu, match as _ma, salign as _sa, cvs as ncvs
import bmesh
def _nk(name, G, Uu, Uv, du=3, dv=3):
    pts = [tuple(map(float, G[i, j])) for i in range(G.shape[0]) for j in range(G.shape[1])]
    return store.new_object(ctx, name, ext.surface_from_cvs(du, dv, pts, list(Uu), list(Uv)))
def _nsplit(G, U, t, p=3):
    U = list(U)
    while U.count(t) < p:
        G, U = _sa.insert_knot(G, U, p, t)
    k = max(i for i, x in enumerate(U) if x == t) - p
    return (G[:k + 1], [x / t for x in U[:k + p + 1]] + [1.0]), (G[k:], [0.0] + [(x - t) / (1 - t) for x in U[k + 1:]])
# one surface cut at u = 0.5: NU_L | NU_S, exactly G2 across S's U0
_nn = 7
_nX, _nY = np.meshgrid(np.linspace(500, 504, _nn), np.linspace(0, 4, _nn), indexing="ij")
_nG = np.stack([_nX, _nY, 0.4 * np.sin(0.8 * (_nX - 500)) * np.cos(0.6 * _nY) + 0.15 * np.sin(_nY)], 2)
_nU = geom.clamped_uniform_knots(_nn, 3)
(_nL, _nUL), (_nR, _nUR) = _nsplit(_nG, _nU, 0.5)
nuL = _nk("NU_L", _nL, _nUL, _nU)
nuS = _nk("NU_S", _nR, _nUR, _nU)
# frames of a flat grid: N across, U / V along the hull lines
_fr = _nu.frames(np.stack(list(np.meshgrid(np.arange(4.0), np.arange(3.0), indexing="ij")) + [np.zeros((4, 3))], 2))
check("nudge CVs: frames of a flat grid (N = Z, U = X, V = Y)",
      np.allclose(_fr["N"], [0, 0, 1]) and np.allclose(_fr["U"], [1, 0, 0]) and np.allclose(_fr["V"], [0, 1, 0]))
bpy.ops.object.select_all(action='DESELECT'); nuS.select_set(True); ctx.view_layer.objects.active = nuS
bpy.ops.nproto.edit_cvs()
_ncage = ctx.active_object
_nnet = _nu.net_of_cage(_ncage)
_npart = _nu.find_partners(ctx, nuS, _nnet, (_ncage,))
_tol = _nu.tolerances(ctx)
_ng = _nu.guides(_nnet, _npart, 2)
check("nudge CVs: the edge on the other surface is found, on one layout, G2 allowed; the split is on every guide",
      list(_npart) == ["U0"] and _npart["U0"]["obj"] == nuL and _npart["U0"]["side"] is not None
      and _npart["U0"]["max"] == 2 and len(_ng) == 3 * _nn and all(_nu.holds(g, _tol) for g in _ng),
      ({e: (v["obj"].name, v["ref"], v["max"], v["why"]) for e, v in _npart.items()},
       max((g["err"] for g in _ng), default=None)))
_nix = _sa._grid_index("U0", *_nnet.G.shape[:2])
X0 = _nnet.G.reshape(-1, 3).copy()
_gd = _nu.Guided(_nnet, _npart)
_j = 3
_k1, _k2 = _nix(1, _j), _nix(2, _j)
_row2 = {_nix(2, j) for j in range(_nn)}
_fd = _nu.frames(_nnet.G)
_dir = lambda k: 0.05 * _fd["N"].reshape(-1, 3)[k]
# free: a row 1 CV pushed along N leaves its G1 line (and the row 2 CV's G2 plane)
X1 = _nu.nudge(X0, {_k1}, _dir)
_g1 = _nu.first_guides(_gd.guides(X1, 2))
check("nudge CVs: a free push of a row 1 CV breaks its G1 guide (and the G2 one of the CV after it)",
      not _nu.holds(_g1[_k1], _tol) and not _nu.holds(_g1[_k2], _tol), (_g1[_k1]["err"], _g1[_k2]["err"]))
# locked: a push that leans along the hull and lifts it: the CV slides along its line, row 2 is refitted
_held = {k for k, g in _nu.first_guides(_gd.guides(X0, 2)).items() if _nu.holds(g, _tol)}
_push = lambda k: 0.2 * _fd["U"].reshape(-1, 3)[k] + 0.05 * _fd["N"].reshape(-1, 3)[k]
X2 = _nu.nudge(X0, {_k1}, _push, _gd, 2, True, _held)
_g2 = _nu.first_guides(_gd.guides(X2, 2))
_moved = np.linalg.norm(X2 - X0, axis=1)
_others = [k for k in range(len(X0)) if k != _k1 and k not in _row2]
check("nudge CVs: locked, the row 1 CV slides along its G1 line, row 2 follows onto its G2 planes, "
      "nothing else moves",
      _moved[_k1] > 1e-3 and _g2[_k1]["err"] < 1e-9 and max(_g2[k]["err"] for k in _row2) < 1e-6
      and _moved[_k2] > 0 and _moved[_others].max() == 0.0,
      (_moved[_k1], _g2[_k1]["err"], max(_g2[k]["err"] for k in _row2)))
# P: a CV put back onto its guide holds it; a whole row put on its guides = its least-squares best
X3 = _nu.put_on(_gd, X1, [_k1], 2)
X4 = _nu.put_on(_gd, X1 + 0.0, sorted(_row2 | {_k1}), 2)
_g4 = _nu.first_guides(_gd.guides(X4, 2))
check("nudge CVs: put on guide (P) brings the row 1 CV back onto its line; with row 2 every CV ends on its plane",
      _nu.first_guides(_gd.guides(X3, 2))[_k1]["err"] < 1e-9 and max(_g4[k]["err"] for k in _row2) < 1e-6)
# the surface itself: a locked nudge written into the cage keeps the edge G2 as measured
_M = np.array(_ncage.matrix_world, float)
bm = bmesh.from_edit_mesh(_ncage.data); bm.verts.ensure_lookup_table()
for v, p in zip(bm.verts, (X2 - _M[:3, 3]) @ np.linalg.inv(_M[:3, :3]).T):
    v.co = p
bmesh.update_edit_mesh(_ncage.data); ctx.view_layer.update(); ncvs._process()
_nm = _ma.measure(nuS, "U0", nuL, "U1")
check("nudge CVs: after the locked nudge the edge measures G0 / G1 / G2 (Continuity Check)",
      _nm["gap"] < 1e-6 and _nm["angle"] < _tol["G1"] and _nm["curv"] < _tol["G2"], _nm)
# a target on a different layout along the edge: no guides, said why
bpy.ops.nproto.cvs_cancel()
_nTv, _nTU = _sa.insert_knot(_nL.transpose(1, 0, 2), list(_nU), 3, 0.3)
nuT = _nk("NU_T", _nTv.transpose(1, 0, 2), _nUL, _nTU)      # NU_L's shape with one more knot along the edge
nuL.hide_set(True)
bpy.ops.object.select_all(action='DESELECT'); nuS.select_set(True); ctx.view_layer.objects.active = nuS
bpy.ops.nproto.edit_cvs()
_ncage = ctx.active_object
_npart = _nu.find_partners(ctx, nuS, _nu.net_of_cage(_ncage), (_ncage,))
check("nudge CVs: a neighbour on another layout along the edge gives no guides and says so",
      list(_npart) == ["U0"] and _npart["U0"]["obj"] == nuT and _npart["U0"]["side"] is None
      and "layouts" in (_npart["U0"]["why"] or "") and _nu.guides(_nu.net_of_cage(_ncage), _npart, 2) == [],
      {e: (v["obj"].name, v["why"]) for e, v in _npart.items()})
bpy.ops.nproto.cvs_cancel()
nuL.hide_set(False)

# --- CV Tweak (cvtweak.py): CV edits on a surface with history become a step that replays them
from blender_nurbs import cvtweak as _tw
def _cvs_of(o):
    return np.array(ext.surface_cvs(store.get_surfaces(o)[0])["points"], float)
def _cage_to(cage, Xw):
    M_ = np.array(cage.matrix_world, float)
    bm_ = bmesh.from_edit_mesh(cage.data); bm_.verts.ensure_lookup_table()
    for v, p in zip(bm_.verts, (Xw - M_[:3, 3]) @ np.linalg.inv(M_[:3, :3]).T):
        v.co = p
    bmesh.update_edit_mesh(cage.data); ctx.view_layer.update(); ncvs._process()
_tf = _tw.local_frames(np.stack(list(np.meshgrid(np.arange(4.0), np.arange(3.0), indexing="ij")) + [np.zeros((4, 3))], 2))
_tX = np.random.default_rng(3).normal(size=(12, 3))
_tY = _tX.copy(); _tY[[2, 7]] += [[0.1, -0.2, 0.3], [0.0, 0.05, 0.0]]
check("CV tweak: frames are orthonormal (U, across, normal); offsets replay exactly on the same CVs",
      np.allclose(_tf[0], [1, 0, 0]) and np.allclose(_tf[1], [0, 1, 0]) and np.allclose(_tf[2], [0, 0, 1])
      and [m[0] for m in _tw.offsets(_tX, _tY, (4, 3, 3))] == [2, 7]
      and np.abs(_tw.moved(_tX, _tw.offsets(_tX, _tY, (4, 3, 3)), (4, 3, 3)) - _tY).max() < 1e-12)
# a tweak on an extrusion: the curve turns, the pushed CV's offset turns with the surface
_twc = nurbs("TW_crv", [(600, 700, 0), (601, 700.5, 0), (602, 700, 0), (603, 700.4, 0)])
bpy.ops.nproto.extrude(inputs=ins((_twc, "S0")), distance=2.0, direction="Z")
_twE = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); _twE.select_set(True); ctx.view_layer.objects.active = _twE
bpy.ops.nproto.edit_cvs(); _tcg = ctx.active_object
_tA = np.array(geom.world_matrix(_twE), float)[:3, :3]
_tW0 = _cvs_of(_twE) @ _tA.T + np.array(geom.world_matrix(_twE), float)[:3, 3]
_tk = len(_tW0) // 2
_tN = _tw.local_frames(_tW0.reshape(*[ext.surface_cvs(store.get_surfaces(_twE)[0])[c] for c in ("count_u", "count_v")], 3))[2]
_tW1 = _tW0.copy(); _tW1[_tk] += 0.3 * _tN[_tk]
_cage_to(_tcg, _tW1)
r1 = bpy.ops.nproto.cvs_done()
_tst = features.stack(_twE)
_tL0 = _cvs_of(_twE)
_twc.rotation_euler.z = math.radians(30.0); ctx.view_layer.update()
features.rebuild(_twE, ctx)
_tL1 = _cvs_of(_twE)
_tst_raw = json.loads(_twE[features.PROP_STACK]); _tst_raw[-1]["params"]["_off"] = True
_twE[features.PROP_STACK] = json.dumps(_tst_raw); features.rebuild(_twE, ctx, force=True)
_tL1off = _cvs_of(_twE)
_tst_raw[-1]["params"].pop("_off"); _twE[features.PROP_STACK] = json.dumps(_tst_raw); features.rebuild(_twE, ctx, force=True)
_tR = np.array(_Mx.Rotation(math.radians(30.0), 3, "Z"), float)
_tdk = (_tL1[_tk] - _tL1off[_tk]) @ _tA.T
_td0 = _tL0[_tk] @ _tA.T + np.array(geom.world_matrix(_twE), float)[:3, 3] - _tW0[_tk]     # the push as kept (32-bit cage)
_trest = np.delete(np.abs(_tL1 - _tL1off).max(axis=1), _tk).max()
check("CV tweak: Done on an extrusion adds a CV Tweak step (history kept); the curve turns 30 deg, the surface "
      "follows and the pushed CV's offset turns with it",
      r1 == {'FINISHED'} and features.history(_twE)["op"] == "extrude" and [e["op"] for e in _tst] == ["cv_tweak"]
      and np.abs(_tL1 - _tL0).max() > 0.1 and abs(np.linalg.norm(_td0) - 0.3) < 1e-4
      and np.linalg.norm(_tdk - _tR @ _td0) < 1e-4 and np.linalg.norm(_tdk - _td0) > 0.1 and _trest < 1e-12,
      (np.linalg.norm(_td0), np.linalg.norm(_tdk - _tR @ _td0), _trest))
# (1e-4: the curve's 32-bit world matrix at x ~ 600 rounds the rebuilt CVs by ~1e-5, so the frames lean that much)
# a locked nudge at an edge on another surface: the CVs on their guides stay on them when that surface changes
_tG = np.stack(list(np.meshgrid(np.linspace(600, 604, _nn), np.linspace(800, 804, _nn), indexing="ij"))
               + [0.4 * np.sin(0.8 * np.linspace(0, 4, _nn))[:, None] * np.cos(0.6 * np.linspace(0, 4, _nn))[None, :]], 2)
(_tLG, _tUL), (_tRG, _tUR) = _nsplit(_tG, _nU, 0.5)
twL = _nk("TW_L", _tLG, _tUL, _nU)
twS = _nk("TW_S", _tRG, _tUR, _nU)
features.push_step(twS, "flip_normal", [], {"_off": True})         # some history (a muted step) to keep
bpy.ops.object.select_all(action='DESELECT'); twS.select_set(True); ctx.view_layer.objects.active = twS
bpy.ops.nproto.edit_cvs(); _tcg = ctx.active_object
_tnet = _nu.net_of_cage(_tcg)
_tgd = _nu.Guided(_tnet, _nu.find_partners(ctx, twS, _tnet, (_tcg,)))
_tX0 = _tnet.G.reshape(-1, 3).copy()
_tfd = _nu.frames(_tnet.G)
_tix = _sa._grid_index("U0", *_tnet.G.shape[:2])
_tk1 = _tix(1, 3)
_theld = {k for k, g in _nu.first_guides(_tgd.guides(_tX0, 2)).items() if _nu.holds(g, _tol)}
_tX1 = _nu.nudge(_tX0, {_tk1}, lambda k: 0.2 * _tfd["U"].reshape(-1, 3)[k] + 0.05 * _tfd["N"].reshape(-1, 3)[k],
                 _tgd, 2, True, _theld)
_cage_to(_tcg, _tX1)
bpy.ops.nproto.cvs_done()
_tlast = features.stack(twS)[-1]
_tholds = _tlast["params"]["holds"]
check("CV tweak: a locked nudge records the edge's neighbour as an input and the moved CVs on their guides",
      _tlast["op"] == "cv_tweak" and [(o.name, r_) for o, r_ in _tlast["inputs"]] == [("TW_L", "U1")]
      and len(_tholds) == 1 and _tholds[0]["edge"] == "U0" and _tk1 in _tholds[0]["ks"] and len(_tholds[0]["ks"]) > 1,
      (_tholds, [(o.name, r_) for o, r_ in _tlast["inputs"]]))
_tdL = ext.surface_cvs(store.get_surfaces(twL)[0])
_tPL = np.array(_tdL["points"], float).reshape(_tdL["count_u"], _tdL["count_v"], 3)
_tPL[-2, :, 2] -= 0.15; _tPL[-3, :, 2] += 0.1                     # the neighbour's tangent rows move: its G1 lines turn
store.set_geometry(twL, ext.surface_from_cvs(3, 3, [tuple(q) for q in _tPL.reshape(-1, 3)], _tdL["knots_u"],
                                             _tdL["knots_v"]), ctx)
ctx.view_layer.update(); nhist._process()
_tS = _cvs_of(twS) @ np.array(geom.world_matrix(twS), float)[:3, :3].T + np.array(geom.world_matrix(twS), float)[:3, 3]
_tnS = _ma.Net(_tS.reshape(_tnet.G.shape), _tnet.knots, _tnet.deg)
_tg = _nu.first_guides(_nu.guides(_tnS, {"U0": _nu.partner_of(_tnS, "U0", twL, "U1")}, 2))
_tother1 = max(_tg[_tix(1, j)]["err"] for j in range(_nn) if _tix(1, j) != _tk1)
check("CV tweak: the neighbour changes, the surface rebuilds and the tweaked CVs are back on its new guides "
      "(the untweaked ones come from the snapshot and don't)",
      _tg[_tk1]["err"] < 1e-9 and max(_tg[k]["err"] for k in _tholds[0]["ks"] if k != _tk1) < 1e-6 and _tother1 > 1.0,
      (_tg[_tk1]["err"], max(_tg[k]["err"] for k in _tholds[0]["ks"] if k != _tk1), _tother1))
# a step before it that changes the CV layout: the tweak says so
features._write_stack(twS, [{"op": "rebuild_surface", "inputs": [],
                             "params": {"degree_u": 3, "degree_v": 3, "spans_u": 6, "spans_v": 6}}]
                      + [{"op": e["op"], "inputs": e["inputs"], "params": e["params"]} for e in features.stack(twS)])
try:
    features.rebuild(twS, ctx, force=True); _tmsg = ""
except Exception as exc:
    _tmsg = str(exc)
check("CV tweak: a CV layout changed before it stops the rebuild with a clear message", "CV layout changed" in _tmsg,
      _tmsg)

# Surface fillet round a cylinder through a tilted plane (the angle between them
# goes 59 -> 121 deg round the crossing): a chord keeps the rails that far apart
# all round (the ball radius changes), and a G2 blend is a closed ring, G2 to both
from blender_nurbs.ops_sweep import continuity_report as _crt
def _tilt_pair(tag, y):
    ln = poly(tag + "_pl", [(-1.2, y - 1.2, -0.72), (1.2, y - 1.2, 0.72)])
    bpy.ops.nproto.extrude(inputs=ins((ln, "S0")), distance=2.4, direction="Y")
    pl = ctx.active_object; pl.name = tag + "_plane"
    bpy.ops.curve.primitive_nurbs_circle_add(radius=0.3, location=(0, y, 0))
    ci = ctx.active_object
    bpy.ops.nproto.extrude(inputs=ins((ci, "S0")), distance=1.0, direction="NORMAL", symmetric=True)
    cy = ctx.active_object; cy.name = tag + "_cyl"
    return cy, pl
def _rails_apart(o):
    s_ = store.get_surfaces(o)[0]; (u0, u1), (v0, v1) = s_.u_domain(), s_.v_domain()
    du = [np.linalg.norm(np.array(s_.evaluate(u, v0)) - np.array(s_.evaluate(u, v1))) for u in np.linspace(u0, u1, 97)]
    dv = [np.linalg.norm(np.array(s_.evaluate(u0, v)) - np.array(s_.evaluate(u1, v))) for v in np.linspace(v0, v1, 97)]
    return du if np.mean(du) > np.mean(dv) else dv
_tc = {}
for _tag, _shape, _rail in (("TCc", "CIRCULAR", "CHORD"), ("TCb", "BLEND", "CHORD"), ("TCr", "BLEND", "RADIUS")):
    _cy, _pl = _tilt_pair(_tag, 1100 + 10 * len(_tc))
    _r = bpy.ops.nproto.fillet(surf_a=_cy.name, surf_b=_pl.name, radius=0.1, shape=_shape, rail_type=_rail,
                               continuity="2", trim_bases=True)
    _f = ctx.active_object
    _d = _rails_apart(_f)
    _c = [_crt(_f, o_) or _crt(o_, _f) for o_ in (_cy, _pl)]
    _tc[_tag] = (_r, _f, min(_d), max(_d), _c, [e["op"] for e in features.stack(_cy)], [e["op"] for e in features.stack(_pl)])
_r, _f, _lo, _hi, _c, _s1, _s2 = _tc["TCc"]
check("surface fillet by chord round a cylinder through a tilted plane: rails 0.1 apart all round, tangent to both, "
      "both trimmed", _r == {'FINISHED'} and abs(_lo - 0.1) < 1e-4 and abs(_hi - 0.1) < 1e-4 and all(_c)
      and max(c["angle"] for c in _c) < 0.05 and "fillet_trim" in _s1 and "fillet_trim" in _s2,
      f"chord {_lo:.6f}..{_hi:.6f} {[c and round(c['angle'], 4) for c in _c]} {_s1} {_s2}")
for _tag, _what in (("TCb", "chord"), ("TCr", "rolling ball")):
    _r, _f, _lo, _hi, _c, _s1, _s2 = _tc[_tag]
    check(f"surface fillet G2 blend ({_what}) round a cylinder through a tilted plane: a closed ring, G2 to both",
          _r == {'FINISHED'} and all(_c) and max(c["gap"] for c in _c) < 1e-4 and max(c["angle"] for c in _c) < 0.05
          and max(c["curv_diff"] for c in _c) < 0.02 and (_what != "chord" or abs(_lo - 0.1) < 1e-4 and abs(_hi - 0.1) < 1e-4),
          f"apart {_lo:.5f}..{_hi:.5f} {[c and (round(c['gap'], 6), round(c['angle'], 4), round(c['curv_diff'], 4)) for c in _c]}")
_f = _tc["TCb"][1]
features.rebuild(_f, ctx, force=True)
_c = [_crt(_f, o_) or _crt(o_, _f) for o_ in (bpy.data.objects["TCb_cyl"], bpy.data.objects["TCb_plane"])]
check("surface fillet G2 blend by chord: a history rebuild keeps it G2 and the chord",
      all(_c) and max(c["curv_diff"] for c in _c) < 0.02 and abs(max(_rails_apart(_f)) - 0.1) < 1e-4
      and features.history(_f)["params"].get("blend") == 2 and features.history(_f)["params"].get("chord"),
      f"{[c and round(c['curv_diff'], 4) for c in _c]}")

# the kernel returns a crossing in pieces where it passes an edge (here the joint of a
# two-face sheet, in the user's scene a split region's edge): the pieces are one ring
_y2 = 1140
_h1 = poly("TJ1_ln", [(-1.2, _y2 - 1.2, -0.72), (0.0, _y2 - 1.2, 0.0)])
bpy.ops.nproto.extrude(inputs=ins((_h1, "S0")), distance=2.4, direction="Y"); _hs1 = ctx.active_object
_h2 = poly("TJ2_ln", [(0.0, _y2 - 1.2, 0.0), (1.2, _y2 - 1.2, 0.72)])
bpy.ops.nproto.extrude(inputs=ins((_h2, "S0")), distance=2.4, direction="Y"); _hs2 = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); _hs1.select_set(True); _hs2.select_set(True)
ctx.view_layer.objects.active = _hs1
bpy.ops.nproto.join(); _hj = ctx.active_object; _hj.name = "TJ_sheet"
bpy.ops.curve.primitive_nurbs_circle_add(radius=0.3, location=(0, _y2, 0))
_hc = ctx.active_object
bpy.ops.nproto.extrude(inputs=ins((_hc, "S0")), distance=1.0, direction="NORMAL", symmetric=True)
_hcy = ctx.active_object
_r = bpy.ops.nproto.fillet(surf_a=_hcy.name, surf_b=_hj.name, radius=0.1, shape="CIRCULAR", rail_type="CHORD",
                           trim_bases=True)
_f = ctx.active_object
_d = _rails_apart(_f)
_c = [_crt(_f, o_) or _crt(o_, _f) for o_ in (_hcy, _hj)]
check("surface fillet by chord across a joined sheet's joint (the crossing in two pieces): one closed ring, "
      "rails 0.1 apart all round, tangent to both",
      _r == {'FINISHED'} and store.get_brep(_hj).face_count() == 2 and abs(min(_d) - 0.1) < 1e-4
      and abs(max(_d) - 0.1) < 1e-4 and all(_c) and max(c["angle"] for c in _c) < 0.05
      and np.linalg.norm(np.array(store.get_surfaces(_f)[0].evaluate(0.0, 0.5))
                         - np.array(store.get_surfaces(_f)[0].evaluate(1.0, 0.5))) < 1e-9,
      f"faces {store.get_brep(_hj).face_count()} chord {min(_d):.6f}..{max(_d):.6f} {[c and round(c['angle'], 4) for c in _c]}")

# Edge fillet by chord along an edge whose angle changes (a twisted wall on a floor,
# about 70 -> 110 deg): exact sections keep the rails the chord apart all along
_fl = poly("ECf_ln", [(0, 1180, 0), (0, 1184, 0)])
bpy.ops.nproto.extrude(inputs=ins((_fl, "S0")), distance=2.0, direction="X"); _ecf = ctx.active_object
_w0 = poly("ECw0", [(2, 1180, 0), (2, 1184, 0)]); _w1 = poly("ECw1", [(2.6, 1180, 1.6), (1.4, 1184, 1.6)])
bpy.ops.nproto.loft(inputs=ins((_w0, "S0"), (_w1, "S0"))); _ecw = ctx.active_object
bpy.ops.object.select_all(action='DESELECT'); _ecf.select_set(True); _ecw.select_set(True)
ctx.view_layer.objects.active = _ecf
bpy.ops.nproto.join(); _ecj = ctx.active_object
_n_shared = len(store.shared_edges(_ecj))
_r = fillet_all(_ecj, shape="CIRCULAR", radius=0.3, rail_type="CHORD")
_eb = store.get_brep(_ecj); _M = np.array(geom.world_matrix(_ecj))
def _iso_chords(f):
    s_ = f.surface(); (u0, v0), (u1, v1) = f.uv_domain()
    du = [np.linalg.norm(np.array(s_.evaluate(u0, v)) - np.array(s_.evaluate(u1, v))) for v in np.linspace(v0, v1, 41)]
    dv = [np.linalg.norm(np.array(s_.evaluate(u, v0)) - np.array(s_.evaluate(u, v1))) for u in np.linspace(u0, u1, 41)]
    return min((du, dv), key=lambda d: abs(np.mean(d) - 0.3))
_fd = min((_iso_chords(f) for f in _eb.faces()), key=lambda d: abs(np.mean(d) - 0.3))
_sm = _kl()
_angs = []
for e_ in _eb.edges():
    fs_ = e_.faces()
    if len(fs_) != 2:
        continue
    t0_, t1_ = e_.parameter_range()
    for t_ in np.linspace(t0_, t1_, 21)[1:-1]:
        p_ = np.array(e_.point_at(t_)); nn_ = []
        for f_ in fs_:
            uv_ = _sm.surface_closest_point(f_.surface(), tuple(p_))[1][0]
            nn_.append(np.array(f_.outward_normal_at_uv(uv_.x, uv_.y)))
        _angs.append(math.degrees(math.acos(min(1.0, abs(nn_[0] @ nn_[1]) / np.linalg.norm(nn_[0]) / np.linalg.norm(nn_[1])))))
check("edge fillet by chord on a twisted wall (angle 70 -> 110 deg along the edge): rails 0.3 apart all along, "
      "tangent to both faces, one sheet",
      _n_shared == 1 and _r == {'FINISHED'} and abs(min(_fd) - 0.3) < 1e-4 and abs(max(_fd) - 0.3) < 1e-4
      and max(_angs) < 0.05 and _eb.face_count() == 3,
      f"chord {min(_fd):.6f}..{max(_fd):.6f} angle {max(_angs):.4f} faces {_eb.face_count()}")

# --- Detach at Isoparm (isodetach.py): exact knot-insertion cuts of untrimmed surfaces
from blender_nurbs import isodetach as _iso

def _iso_err(src, pieces, cu, cv):
    """Largest distance between each piece and the source over the sub-domain it covers."""
    def ev(s_, fu, fv):
        (u0, u1), (v0, v1) = s_.u_domain(), s_.v_domain()
        return np.array(s_.evaluate(u0 + (u1 - u0) * fu, v0 + (v1 - v0) * fv))
    doms = _iso.piece_domains(cu, cv, store._periodic(src))
    err = 0.0
    for pc, (a, b, c, d) in zip(pieces, doms):
        for x in np.linspace(0, 1, 7):
            for y in np.linspace(0, 1, 7):
                fu, fv = a + (b - a) * x, c + (d - c) * y
                err = max(err, np.linalg.norm(ev(pc, x, y) - ev(src, fu - (fu > 1), fv - (fv > 1))))
    return len(pieces) == len(doms), err

_hist0 = ctx.scene.nproto_settings.construction_history
ctx.scene.nproto_settings.construction_history = False
_rng = np.random.default_rng(7)                         # cubic x quadratic, non-uniform knots, rational
_iso_P = [(i + 0.2 * _rng.standard_normal(), j + 0.2 * _rng.standard_normal(), 0.5 * _rng.standard_normal())
          for i in range(6) for j in range(5)]
_iso_src = ext.surface_from_cvs(3, 2, _iso_P, [0, 0, 0, 0, 0.15, 0.6, 1, 1, 1, 1], [0, 0, 0, 0.3, 0.45, 1, 1, 1],
                                [float(0.6 + _rng.random()) for _ in range(30)])
_n_ok, _e = _iso_err(_iso_src, _iso.detach(_iso_src, [0.37, 0.6], [0.2, 0.81]), [0.37, 0.6], [0.2, 0.81])
check("detach at isoparm: a rational non-uniform surface cut at u 0.37 / 0.6 (on a knot) and v 0.2 / 0.81 "
      "is 9 exact pieces", _n_ok and _e < 1e-12, f"err {_e:.1e}")
bpy.ops.nproto.add_primitive(kind="CYLINDER"); _iso_cyl = ctx.active_object
_iso_cs = store.get_surfaces(_iso_cyl)[0]
_n_ok, _e = _iso_err(_iso_cs, _iso.detach(_iso_cs, [0.1, 0.6], [0.5]), [0.1, 0.6], [0.5])
check("detach at isoparm: a rational closed cylinder cut at u 0.1 / 0.6 and v 0.5 is 4 exact pieces "
      "(the piece across the seam joined)", _n_ok and _e < 1e-12, f"err {_e:.1e}")
_one = _iso.detach(_iso_cs, [0.3], [])
_n_ok, _e = _iso_err(_iso_cs, _one, [0.3], [])
_ed = np.linalg.norm(np.array(_one[0].evaluate(0.0, 0.5)) - np.array(_iso_cs.evaluate(
    _iso_cs.u_domain()[0] + 0.3 * (_iso_cs.u_domain()[1] - _iso_cs.u_domain()[0]), sum(_iso_cs.v_domain()) / 2)))
check("detach at isoparm: one cut on a closed direction opens it there (one piece, its seam at the cut)",
      _n_ok and len(_one) == 1 and _e < 1e-12 and _ed < 1e-12, f"err {_e:.1e} seam {_ed:.1e}")
_iso_n0 = len(bpy.data.objects)
r = bpy.ops.nproto.detach_iso(target=_iso_cyl.name, cuts_u="[0.0, 0.5]")
_made = list(ctx.selected_objects)
check("detach at isoparm (no history): a cylinder cut at the seam and at 0.5 is two untrimmed halves, "
      "the first one in the source object",
      r == {'FINISHED'} and len(_made) == 2 and _iso_cyl in _made and all(store.is_natural(o) for o in _made)
      and len(bpy.data.objects) == _iso_n0 + 1
      and abs(sum(store.get_brep(o).area(1e-7) for o in _made) - store.to_brep(_iso_cs).area(1e-7)) < 1e-5)

# trimmed surfaces: the trim clipped to each piece, the cut lines exact isoparms of the piece's surface
from blender_nurbs.kernel import trimext as _te
def _iso_trim_err(src_brep, out, cu, cv):
    """(area error, largest distance of a cut edge off the source / off its cut line (fraction))."""
    s_ = src_brep.faces()[0].surface()
    (a0, a1), (b0, b1) = s_.u_domain(), s_.v_domain()
    tot, off = 0.0, 0.0
    for g, _ in out:
        if g is None:
            continue
        pb = g if isinstance(g, _kl().Brep) else store.to_brep(_iso.copy(g))
        tot += pb.area(1e-7)
        for e in pb.edges():
            t0, t1 = e.parameter_range()
            P = [np.array(e.point_at(float(t))) for t in np.linspace(t0, t1, 7)]
            F = [((u - a0) / (a1 - a0), (v - b0) / (b1 - b0)) for u, v in (_te._uv(_kl(), s_, p) for p in P)]
            d = max(min([_iso._circ(fu - x) for x in cu] + [_iso._circ(fv - x) for x in cv] + [1.0]) for fu, fv in F)
            if d < 1e-6:               # a cut edge: on the source, on its line
                off = max(off, d, max(np.linalg.norm(np.array(s_.evaluate(*_te._uv(_kl(), s_, p))) - p) for p in P))
    return abs(tot - src_brep.area(1e-7)), off
def _iso_kinds(out):
    return "".join("-" if g is None else ("T" if isinstance(g, _kl().Brep) else "N") for g, _ in out)
def _iso_holed_plane():
    bpy.ops.nproto.add_primitive(kind="PLANE"); o = ctx.active_object
    bpy.ops.curve.primitive_bezier_circle_add(radius=0.3, location=(0.2, 0.1, 1)); c_ = ctx.active_object
    bpy.ops.nproto.trim(target=o.name, cutters=ins((c_, "S0")), ref_point=(0.2, 0.1, 0), normal=(0, 0, 1),
                        view_dir=(0, 0, -1), action="DELETE")
    bpy.data.objects.remove(c_)
    return o
_iso_hp = _iso_holed_plane()
_out = _iso.detach_trimmed(store.get_brep(_iso_hp), [0.6, 0.9], [])
_ae, _off = _iso_trim_err(store.get_brep(_iso_hp), _out, [0.6, 0.9], [])
check("detach at isoparm (trimmed): a plane with a hole cut through the hole and beside it -- two trimmed "
      "pieces and one the trim does not cut into, untrimmed", _iso_kinds(_out) == "TTN" and _ae < 1e-5 and _off < 1e-9,
      f"{_iso_kinds(_out)} area err {_ae:.1e} cut edges off {_off:.1e}")
_iso_hs, _iso_hf = store.get_surfaces(_iso_hp)[0], store.get_brep(_iso_hp).faces()[0]
check("detach at isoparm (trimmed): the picked isoparm is drawn where it runs on the face (broken by the hole)",
      len(_iso.iso_runs(_iso_hs, _iso_hf, "U", 0.6)) == 2 and len(_iso.iso_runs(_iso_hs, _iso_hf, "U", 0.2)) == 1)


def _iso_snap_on(s_, pts, d):
    """Largest distance from a snap point to the surface along its isoparm (the
    isoparm passes through the edge middle; for a CV, 0 by construction)."""
    (u0, u1), (v0, v1) = s_.u_domain(), s_.v_domain()
    foot = store._make_foot(s_)
    off = 0.0
    for f, p, k in pts:
        if k != "mid":
            continue
        u, v = foot(p)
        q = s_.evaluate(u0 + (u1 - u0) * f, v) if d == "U" else s_.evaluate(u, v0 + (v1 - v0) * f)
        off = max(off, float(np.linalg.norm(np.array(q) - p)))
    return off


bpy.ops.nproto.add_primitive(kind="PLANE"); _iso_sp = ctx.active_object
_iso_ss = store.get_surfaces(_iso_sp)[0]
_iso_su = _iso.snap_points(_iso_ss, None, None, "U")
_iso_sc = _iso._k()[1].surface_cvs(_iso_ss)
_iso_gre = _iso._greville_fracs(_iso_ss, "U")
check("detach at isoparm snaps: an untrimmed plane's U cuts snap to the middles of its two u sides (0.5) and to "
      "every CV at its row's Greville fraction, the end rows left out (the surface's edge)",
      sorted(round(f, 9) for f, _, k in _iso_su if k == "mid") == [0.5, 0.5]
      and sum(k == "cv" for _, _, k in _iso_su) == (int(_iso_sc["count_u"]) - 2) * int(_iso_sc["count_v"])
      and {round(f, 9) for f, _, k in _iso_su if k == "cv"} == {round(g, 9) for g in _iso_gre[1:-1]}
      and not any(k == "cv" for _, _, k in _iso.snap_points(_iso_ss, None, None, "U", cvs=False)),
      str(sorted({round(f, 4) for f, _, _ in _iso_su})))
_iso_hb = store.get_brep(_iso_hp)
_iso_hm = {d: _iso.snap_points(_iso_hs, _iso_hb, _iso_hf, d) for d in "UV"}
check("detach at isoparm snaps: on a plane with a hole every edge middle (the hole's too) lies on the isoparm "
      "it snaps to", all(_iso_snap_on(_iso_hs, _iso_hm[d], d) < 1e-9 for d in "UV")
      and all(sum(k == "mid" for _, _, k in _iso_hm[d]) == 3 for d in "UV"),
      str({d: [round(f, 4) for f, _, k in _iso_hm[d] if k == "mid"] for d in "UV"}))
bpy.ops.nproto.add_primitive(kind="PLANE"); _iso_kp = ctx.active_object
bpy.ops.curve.primitive_bezier_circle_add(radius=0.3, location=(0.2, 0.1, 1)); _c_ = ctx.active_object
bpy.ops.nproto.trim(target=_iso_kp.name, cutters=ins((_c_, "S0")), ref_point=(0.2, 0.1, 0), normal=(0, 0, 1),
                    view_dir=(0, 0, -1), action="KEEP")
bpy.data.objects.remove(_c_)
_iso_kb = store.get_brep(_iso_kp)
_iso_kf = {round(f, 4) for f, _, k in _iso.snap_points(store.get_surfaces(_iso_kp)[0], _iso_kb, _iso_kb.faces()[0], "U")
           if k == "cv"}
check("detach at isoparm snaps: CV rows whose isoparm the trim cuts away (a small disc kept from a plane) do not "
      "snap", bool(_iso_kf) and all(0.44 < f < 0.76 for f in _iso_kf), str(sorted(_iso_kf)))
bpy.ops.nproto.add_primitive(kind="CYLINDER"); _iso_cy = ctx.active_object
_iso_cs = store.get_surfaces(_iso_cy)[0]
_iso_cu = _iso.snap_points(_iso_cs, None if store.is_natural(_iso_cy) else store.get_brep(_iso_cy),
                           None if store.is_natural(_iso_cy) else store.get_brep(_iso_cy).faces()[0], "U")
check("detach at isoparm snaps: on a closed direction (a cylinder's u) the seam's CV row snaps (a cut there "
      "opens the seam) and every fraction is in [0, 1)",
      any(f == 0.0 and k == "cv" for f, _, k in _iso_cu) and all(0.0 <= f < 1.0 for f, _, _ in _iso_cu),
      str(sorted({round(f, 4) for f, _, _ in _iso_cu})))
for _o in (_iso_sp, _iso_kp, _iso_cy):
    bpy.data.objects.remove(_o)
bpy.ops.nproto.add_primitive(kind="DISK"); _iso_disk = ctx.active_object
_out = _iso.detach_trimmed(store.get_brep(_iso_disk), [0.5], [0.5])
_ae, _off = _iso_trim_err(store.get_brep(_iso_disk), _out, [0.5], [0.5])
check("detach at isoparm (trimmed): a disk cut at u 0.5 and v 0.5 is four exact quarters",
      _iso_kinds(_out) == "TTTT" and _ae < 1e-6 and _off < 1e-9
      and all(abs(g.area(1e-7) - math.pi / 4) < 1e-5 for g, _ in _out), f"area err {_ae:.1e} off {_off:.1e}")
bpy.ops.nproto.add_primitive(kind="PLANE"); _iso_star = ctx.active_object     # a five-pointed star trim
_cu = bpy.data.curves.new("IsoStar", 'CURVE'); _cu.dimensions = '3D'
_sp = _cu.splines.new('BEZIER'); _sp.bezier_points.add(9); _sp.use_cyclic_u = True
for _k_, _bp in enumerate(_sp.bezier_points):
    _r_, _a_ = (0.9 if _k_ % 2 == 0 else 0.35), 2 * math.pi * _k_ / 10
    _bp.co = (_r_ * math.cos(_a_), _r_ * math.sin(_a_), 1.0); _bp.handle_left_type = _bp.handle_right_type = 'AUTO'
_st = bpy.data.objects.new("IsoStar", _cu); ctx.collection.objects.link(_st)
bpy.ops.nproto.trim(target=_iso_star.name, cutters=ins((_st, "S0")), ref_point=(0, 0, 0), normal=(0, 0, 1),
                    view_dir=(0, 0, -1), action="KEEP")
_ss = store.get_surfaces(_iso_star)[0]
_fv = (_te._uv(_kl(), _ss, (0.0, 0.45, 0.0))[1] - _ss.v_domain()[0]) / (_ss.v_domain()[1] - _ss.v_domain()[0])
_out = _iso.detach_trimmed(store.get_brep(_iso_star), [], [_fv])      # the line y = 0.45: above two inner corners
_ae, _off = _iso_trim_err(store.get_brep(_iso_star), _out, [], [_fv])
check("detach at isoparm (trimmed): a star cut across two of its points -- the piece beyond the cut is the two "
      "point tips (two faces)", _iso_kinds(_out) == "TT" and _out[1][0].face_count() == 2 and _ae < 1e-5 and _off < 1e-9,
      f"faces {[g.face_count() for g, _ in _out]} area err {_ae:.1e} off {_off:.1e}")
_vp = nurbs("IsoVaseProf", [(1.2, 0, 0), (1.5, 0, 0.8), (0.8, 0, 1.8), (1.1, 0, 3.0)])
_vp.location = (90, 0, 0); ctx.view_layer.update()
bpy.ops.nproto.revolve(inputs=ins((_vp, "S0")), axis="Z", pivot="OBJECT", angle=360); _iso_vase = ctx.active_object
bpy.ops.curve.primitive_nurbs_circle_add(radius=0.4, location=(90, 0, 1.5), rotation=(math.pi / 2, 0, 0))
_vc = ctx.active_object
bpy.ops.nproto.trim(target=_iso_vase.name, cutters=ins((_vc, "S0")), ref_point=(90, 1.25, 1.5), normal=(0, 1, 0),
                    view_dir=(0, -1, 0), action="DELETE")
_vb = store.get_brep(_iso_vase)
_res = []
for _cu_, _cv_ in (([0.25], []), ([0.1, 0.6], []), ([0.3, 0.8], [0.45]), ([0.0], [0.5])):
    _out = _iso.detach_trimmed(_vb, _cu_, _cv_)
    _ae, _off = _iso_trim_err(_vb, _out, _cu_, _cv_)
    _res.append((_iso_kinds(_out), _ae < 1e-5 and _off < 1e-9))
check("detach at isoparm (trimmed): a closed revolve with holes -- one cut through the holes moves the seam "
      "(one piece), two cuts, cuts both ways, a cut at the seam", [k for k, _ in _res] == ["T", "TT", "TTTT", "TT"]
      and all(g for _, g in _res), str(_res))
bpy.ops.nproto.add_primitive(kind="CYLINDER"); _iso_cy = ctx.active_object
_iso_cy.location = (95, 0, 0); ctx.view_layer.update()
bpy.ops.curve.primitive_bezier_circle_add(radius=0.3, location=(95, -3, 0), rotation=(math.pi / 2, 0, 0))
_c2 = ctx.active_object
bpy.ops.nproto.trim(target=_iso_cy.name, cutters=ins((_c2, "S0")), ref_point=(95, -1, 0), normal=(0, -1, 0),
                    view_dir=(0, 1, 0), action="DELETE")
_A = store.get_brep(_iso_cy).area(1e-7)
r = bpy.ops.nproto.detach_iso(target=_iso_cy.name, cuts_u="[0.25, 0.75]")
_made = list(ctx.selected_objects)
_ok = r == {'FINISHED'} and len(_made) == 2 and not any(store.is_natural(o) for o in _made)
_ok = _ok and abs(sum(store.get_brep(o).area(1e-7) for o in _made) - _A) < 1e-5
for o in _made:
    bpy.ops.object.select_all(action='DESELECT'); o.select_set(True); ctx.view_layer.objects.active = o
    bpy.ops.nproto.untrim()
check("detach at isoparm (no history): a tube with a hole across its seam cut in two; Untrim gives each half "
      "its own half tube", _ok and all(store.is_natural(o) and abs(store.get_brep(o).area(1e-7) - 2 * math.pi) < 1e-4
                                    for o in _made),
      ", ".join(f"{store.get_brep(o).area(1e-7):.5f}" for o in _made))
bpy.ops.nproto.add_primitive(kind="BOX"); _iso_box = ctx.active_object
r = bpy.ops.nproto.detach_iso(target=_iso_box.name, cuts_u="[0.5]")
check("detach at isoparm refuses a joined surface (several faces)", r == {'CANCELLED'})

ctx.scene.nproto_settings.construction_history = True
bpy.ops.nproto.add_primitive(kind="PLANE"); _iso_pl = ctx.active_object
r = bpy.ops.nproto.detach_iso(target=_iso_pl.name, cuts_u="[0.5]", cuts_v="[0.25]")
_made = sorted(ctx.selected_objects, key=lambda o: o.name)
_hist_ok = all(features.history(o) is not None and features.history(o)["op"] == "iso_detach" for o in _made)
check("detach at isoparm (history): 4 pieces follow the source, which is kept hidden",
      r == {'FINISHED'} and len(_made) == 4 and _hist_ok and _iso_pl.hide_get() and _iso_pl not in _made)
_s0 = store.get_surfaces(_iso_pl)[0]
_d = ext.surface_cvs(_s0)
_P = np.array(_d["points"], float).reshape(_d["count_u"], _d["count_v"], 3)
_P[..., 2] += 0.3 * np.sin(_P[..., 0] * 2.0)            # bend the source
store.set_geometry(_iso_pl, ext.surface_from_cvs(_d["degree_u"], _d["degree_v"], [tuple(x) for x in _P.reshape(-1, 3)],
                                                 list(_d["knots_u"]), list(_d["knots_v"]), None), ctx)
for o in _made:
    features.rebuild(o, ctx, force=True)
_s1 = store.get_surfaces(_iso_pl)[0]
_n_ok, _e = _iso_err(_s1, [store.get_surfaces(o)[0] for o in sorted(_made, key=lambda o: features.history(o)["params"]["piece"])],
                     [0.5], [0.25])
check("detach at isoparm (history): pieces rebuild exactly from the edited source", _n_ok and _e < 1e-12, f"err {_e:.1e}")
_iso_hp2 = _iso_holed_plane()
_A = store.get_brep(_iso_hp2).area(1e-7)
r = bpy.ops.nproto.detach_iso(target=_iso_hp2.name, cuts_u="[0.6]", cuts_v="[0.55]")
_made = sorted(ctx.selected_objects, key=lambda o: features.history(o)["params"]["piece"])
_a0 = [store.get_brep(o).area(1e-7) for o in _made]
for o in _made:
    features.rebuild(o, ctx, force=True)
_a1 = [store.get_brep(o).area(1e-7) for o in _made]
_carrier = _iso.detach(store.get_surfaces(_iso_hp2)[0], [0.6], [0.55], clamp_all=True)[0]
_ca = store.to_brep(_carrier).area(1e-7)
o = _made[0]
bpy.ops.object.select_all(action='DESELECT'); o.select_set(True); ctx.view_layer.objects.active = o
r2 = bpy.ops.nproto.untrim()
check("detach at isoparm (history): a trimmed plane cut both ways is 4 trimmed pieces that rebuild the same; "
      "Untrim on one gives its own quarter of the surface",
      r == {'FINISHED'} and len(_made) == 4 and _iso_hp2.hide_get() and abs(sum(_a0) - _A) < 1e-5
      and max(abs(x - y) for x, y in zip(_a0, _a1)) < 1e-9 and r2 == {'FINISHED'} and store.is_natural(o)
      and abs(store.get_brep(o).area(1e-7) - _ca) < 1e-7, f"{_a0} -> {_a1}; untrim {store.get_brep(o).area(1e-7):.5f} vs {_ca:.5f}")
ctx.scene.nproto_settings.construction_history = _hist0

print("ALL OK" if ok else "SOME FAILED")
