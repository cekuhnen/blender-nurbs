"""Unroll Surface and Flow Along Surface:
    blender -b --factory-startup --python tests/test_unroll_flow.py"""
import json, math, os, sys, time
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, flow, geom, isodetach, store, surfeval, surfmap, unroll

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
sm, ext = store._k()

def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

def select(objs, active):
    for o in ctx.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = active

def poly(name, pts, cyclic=True):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.use_cyclic_u = cyclic
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def wbrep(o): return flow.world_brep(o)
def area(o): return wbrep(o).area(1e-7)

def faces_of(o):
    return [surfeval.Surf(f.surface()) for f in wbrep(o).faces()]

def dist_to(o, P):
    """Largest distance of points P to the object's face surfaces."""
    best = np.full(len(P), np.inf)
    for S in faces_of(o):
        uv = S.feet(P)
        best = np.minimum(best, np.linalg.norm(P - S.at(uv[:, 0], uv[:, 1], 0)[0, 0], axis=1))
    return float(best.max())

def curve_points(o, per=64):
    from blender_nurbs import curvejoin, curvetrim
    M = np.array(geom.world_matrix(o), float)
    out = []
    for sp in o.data.splines:
        P = curvetrim.sample_spans(curvejoin.transform_spans(curvejoin.spline_spans(sp), M), per)[0]
        out.append(P)
    return out

def length(P): return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())

def zspan(o):
    P = np.vstack([surfmap.edge_samples(e, 9) for e in wbrep(o).edges()])
    return float(np.ptp(P[:, 2]))

def run(op, **kw):
    try:
        return op(**kw)
    except RuntimeError as exc:
        print("   ", exc)
        return {'CANCELLED'}

for hist in (True, False):
    ctx.scene.nproto_settings.construction_history = hist
    tag = "history" if hist else "no history"
    clear()

    # -- a cone frustum, exact (rational) -------------------------------------
    bpy.ops.nproto.add_primitive(kind="CONE", radius=1.0, radius2=0.4, height=2.0)
    cone = ctx.active_object
    cone.location = (0.3, -0.2, 0.5)
    select([cone], cone)
    t0 = time.perf_counter()
    r = run(bpy.ops.nproto.unroll, place="BESIDE")
    dt = time.perf_counter() - t0
    flat = ctx.active_object
    check(f"[{tag}] unroll cone", r == {'FINISHED'} and flat is not cone and unroll.PROP_UNROLL in flat,
          f"{dt * 1000:.0f} ms")
    check(f"[{tag}] cone flat: same area", abs(area(flat) - area(cone)) < 1e-5 * area(cone),
          f"{area(flat):.7f} vs {area(cone):.7f}")
    check(f"[{tag}] cone flat: flat", zspan(flat) < 1e-9, f"z span {zspan(flat):.1e}")
    # the sector: the slant length is the width between the arcs
    slant = math.hypot(2.0, 0.6)
    P = np.vstack([surfmap.edge_samples(e, 33) for e in wbrep(flat).edges()])
    c = None
    check(f"[{tag}] cone flat: 4 edges", wbrep(flat).edge_count() == 4, f"{wbrep(flat).edge_count()}")

    # a circle drawn on the sheet folds back onto the cone, same length
    S = faces_of(flat)[0]
    (u0, u1), (v0, v1) = S.u_domain, S.v_domain
    mid = S.at([0.5 * (u0 + u1)], [0.5 * (v0 + v1)], 0)[0, 0][0]
    rad = 0.25
    ring = [tuple(mid + rad * np.array([math.cos(a), math.sin(a), 0.0])) for a in np.linspace(0, 2 * math.pi, 49)[:-1]]
    hole = poly("Hole", ring)
    L0 = 48 * 2 * rad * math.sin(math.pi / 48)          # the 48-gon's perimeter
    select([hole, flat, cone], cone)
    r = run(bpy.ops.nproto.flow, base=flat.name, target=cone.name, objects=json.dumps([hole.name]))
    out = ctx.active_object
    check(f"[{tag}] fold back a circle", r == {'FINISHED'} and out.type == 'CURVE' and out is not hole)
    if out.type == 'CURVE':
        Q = curve_points(out)[0]
        check(f"[{tag}] folded circle lies on the cone", dist_to(cone, Q) < 1e-5, f"{dist_to(cone, Q):.1e}")
        check(f"[{tag}] folded circle keeps its length", abs(length(Q) - L0) < 1e-4 * L0,
              f"{length(Q):.6f} vs {L0:.6f}")
        check(f"[{tag}] folded circle is closed", np.linalg.norm(Q[0] - Q[-1]) < 1e-6)
    # the default roles from the selection (invoke: target active, base the unrolled sheet)
    select([hole, flat, cone], cone)
    objs, base, target = flow.guess_roles(ctx)
    check(f"[{tag}] roles from the selection", base is flat and target is cone and objs == [hole],
          f"{[o.name for o in objs]} {base and base.name} {target and target.name}")

    # a trimmed disk lying on the sheet folds back as a surface: on the cone, same area
    disk = store.new_object(ctx, "Disk", sm.create_planar_faces([sm.create_curve_interp_points(ring + [ring[0]])]))
    A0 = area(disk)
    r = run(bpy.ops.nproto.flow, base=flat.name, target=cone.name, objects=json.dumps([disk.name]))
    out = ctx.active_object
    check(f"[{tag}] fold back a disk", r == {'FINISHED'} and store.is_nurbs(out) and out is not disk)
    if store.is_nurbs(out):
        Q = np.vstack([surfmap.edge_samples(e, 33) for e in wbrep(out).edges()])
        check(f"[{tag}] folded disk on the cone", dist_to(cone, Q) < 1e-5, f"{dist_to(cone, Q):.1e}")
        check(f"[{tag}] folded disk keeps its area", abs(area(out) - A0) < 1e-4 * A0, f"{area(out):.6f} vs {A0:.6f}")

    # a curve on the cone comes along: in the sheet's plane, on the sheet, same length
    cs = store.get_surfaces(cone)[0]
    M = np.array(geom.world_matrix(cone), float)
    on = [tuple(M[:3, :3] @ np.array(cs.evaluate(0.1 + 0.5 * t, 0.3 + 0.4 * t)) + M[:3, 3])
          for t in np.linspace(0, 1, 400)]
    wire = poly("Wire", on, cyclic=False)
    Lw = length(np.vstack(curve_points(wire)))
    select([cone, wire], cone)
    r = run(bpy.ops.nproto.unroll, place="BESIDE")
    fw = [o for o in ctx.selected_objects if o.type == 'CURVE']
    sheet2 = [o for o in ctx.selected_objects if store.is_nurbs(o)]
    if r == {'FINISHED'} and fw and sheet2:
        Q = np.vstack(curve_points(fw[0]))
        check(f"[{tag}] curve along: on the sheet", dist_to(sheet2[0], Q) < 1e-5, f"{dist_to(sheet2[0], Q):.1e}")
        check(f"[{tag}] curve along: same length", abs(length(Q) - Lw) < 1e-4 * Lw, f"{length(Q):.6f} vs {Lw:.6f}")
    else:
        check(f"[{tag}] curve along", False, f"{r} {len(fw)}")

    if hist:
        # the sheet follows its cone
        h = features.history(cone)
        prm = dict(h["params"]); prm["radius"] = 1.3
        raw = json.loads(cone[features.PROP_HIST]); raw["params"] = prm
        cone[features.PROP_HIST] = json.dumps(raw)
        features.rebuild(cone, ctx, force=True)
        features.rebuild(flat, ctx, force=True)
        check(f"[{tag}] sheet follows the cone", abs(area(flat) - area(cone)) < 1e-5 * area(cone),
              f"{area(flat):.6f} vs {area(cone):.6f}")
        check(f"[{tag}] sheet keeps its face table", unroll.PROP_UNROLL in flat
              and json.loads(flat[unroll.PROP_UNROLL])["source"] == cone.name)

    # -- a box: one cross-shaped sheet, folds at the bend lines ----------------
    clear()
    bpy.ops.nproto.add_primitive(kind="BOX", width=2.0, depth=1.0, height=0.5)
    box = ctx.active_object
    select([box], box)
    r = run(bpy.ops.nproto.unroll, place="BESIDE")
    sheets = [o for o in ctx.selected_objects if store.is_nurbs(o)]
    check(f"[{tag}] box: one sheet of 6 faces", r == {'FINISHED'} and len(sheets) == 1
          and wbrep(sheets[0]).face_count() == 6, f"{[wbrep(o).face_count() for o in sheets]}")
    sheet = sheets[0]
    check(f"[{tag}] box sheet area", abs(area(sheet) - 7.0) < 1e-6, f"{area(sheet):.7f}")
    check(f"[{tag}] box sheet joined along 5 bends", wbrep(sheet).edge_count() == 19, f"{wbrep(sheet).edge_count()}")
    tab = json.loads(sheet[unroll.PROP_UNROLL])["faces"]
    check(f"[{tag}] box face table", sorted(k for k, _ in tab) == list(range(6)), f"{tab}")
    # a curve round a vertical edge that the cross cuts open: it comes apart in two
    bw = poly("Round", [(0.5, -0.5, 0.0), (1.0, -0.5, 0.0), (1.0, 0.0, 0.0)], cyclic=False)
    select([box, bw], box)
    r = run(bpy.ops.nproto.unroll, place="BESIDE")
    fw = [o for o in ctx.selected_objects if o.type == 'CURVE']
    check(f"[{tag}] box: a curve over a cut comes apart", r == {'FINISHED'} and fw and len(fw[0].data.splines) == 2,
          f"{[len(o.data.splines) for o in fw]}")
    for o in [o for o in bpy.data.objects if o.name.startswith(box.name + "_flat") and o is not sheet] + fw:
        bpy.data.objects.remove(o)
    # a line across the whole sheet folds over the box, staying on it
    lo, hi = unroll._box(wbrep(sheet))
    y = 0.5 * (lo[1] + hi[1])
    z = float(surfmap.edge_samples(wbrep(sheet).edges()[0], 3)[0, 2])
    line = poly("Line", [(lo[0] + 1e-3, y, z), (hi[0] - 1e-3, y, z)], cyclic=False)
    L0 = hi[0] - lo[0] - 2e-3
    r = run(bpy.ops.nproto.flow, base=sheet.name, target=box.name, objects=json.dumps([line.name]))
    out = ctx.active_object
    if r == {'FINISHED'} and out.type == 'CURVE':
        Q = np.vstack(curve_points(out))
        check(f"[{tag}] line folded onto the box", dist_to(box, Q) < 1e-5, f"{dist_to(box, Q):.1e}")
        check(f"[{tag}] folded line keeps its length", abs(length(Q) - L0) < 1e-5 * L0, f"{length(Q):.6f} vs {L0:.6f}")
        check(f"[{tag}] folded line: one piece per face", len(out.data.splines) == 1, f"{len(out.data.splines)} splines")
    else:
        check(f"[{tag}] line folded onto the box", False)

    # -- a pipe cut obliquely: the classic sine pattern -------------------------
    clear()
    bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0)
    cyl = ctx.active_object
    s = store.get_surfaces(cyl)[0]
    vc = lambda u: 0.5 + 0.2 * math.cos(2 * math.pi * u)
    ip = lambda pts: sm.create_curve_interp_points([tuple(s.evaluate(*q)) for q in pts])
    curves = [ip([(u, 0.0) for u in np.linspace(0, 1, 65)]),
              sm.create_line_segment(tuple(s.evaluate(1, 0)), tuple(s.evaluate(1, vc(1)))),
              ip([(u, vc(u)) for u in np.linspace(1, 0, 65)]),
              sm.create_line_segment(tuple(s.evaluate(0, vc(0))), tuple(s.evaluate(0, 0)))]
    cut = store.new_object(ctx, "PipeCut", ext.trim_surface_with_3d_curves_tol(isodetach.copy(s), [4], curves, 1e-6, 0.01))
    select([cut], cut)
    r = run(bpy.ops.nproto.unroll, place="BESIDE")
    flat = ctx.active_object
    lo, hi = unroll._box(wbrep(flat))
    E = [surfmap.edge_samples(e, 65) for e in wbrep(flat).edges()]
    L = sorted(length(P) for P in E)
    check(f"[{tag}] pipe cut unrolls", r == {'FINISHED'} and wbrep(flat).face_count() == 1)
    check(f"[{tag}] pipe cut: 2 pi wide", any(abs(l - 2 * math.pi) < 1e-5 for l in L), f"{L}")
    check(f"[{tag}] pipe cut: same area", abs(area(flat) - area(cut)) < 1e-5 * area(cut))

    # -- a cone with a round hole (an inner loop across nothing but the face) ---
    clear()
    bpy.ops.nproto.add_primitive(kind="CONE", radius=1.0, radius2=0.4, height=2.0)
    cone = ctx.active_object
    cone.rotation_euler = (0.3, 0.0, 0.7)
    cone.scale = (1.5, 1.5, 1.5)
    s = store.get_surfaces(cone)[0]
    ip = lambda pts: sm.create_curve_interp_points([tuple(s.evaluate(*q)) for q in pts])
    outer = [ip([(u, 0.0) for u in np.linspace(0, 1, 65)]),
             sm.create_line_segment(tuple(s.evaluate(1, 0)), tuple(s.evaluate(1, 1))),
             ip([(u, 1.0) for u in np.linspace(1, 0, 65)]),
             sm.create_line_segment(tuple(s.evaluate(0, 1)), tuple(s.evaluate(0, 0)))]
    inner = [ip([(0.5 + 0.08 * math.cos(t), 0.5 - 0.15 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 65)])]
    holed = store.new_object(ctx, "Holed", ext.trim_surface_with_3d_curves_tol(isodetach.copy(s), [4, 1], outer + inner,
                                                                              1e-6, 0.01))
    holed.matrix_basis = cone.matrix_basis.copy()
    select([holed], holed)
    r = run(bpy.ops.nproto.unroll, place="BESIDE")
    flat = ctx.active_object
    loops = [lp for lp in wbrep(flat).faces()[0].boundary_loops() if lp] if r == {'FINISHED'} else []
    check(f"[{tag}] holed cone: unrolled with its hole", r == {'FINISHED'} and len(loops) == 2, f"{len(loops)} loops")
    check(f"[{tag}] holed cone: same area (moved, turned, scaled)", abs(area(flat) - area(holed)) < 1e-5 * area(holed),
          f"{area(flat):.6f} vs {area(holed):.6f}")
    check(f"[{tag}] holed cone: flat", zspan(flat) < 1e-9, f"{zspan(flat):.1e}")

    # -- general flow: a curve on a plane bent around a cylinder ----------------
    clear()
    bpy.ops.nproto.add_primitive(kind="PLANE", width=2 * math.pi, depth=2.0)
    plane = ctx.active_object
    bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=1.0, height=2.0)
    cyl = ctx.active_object
    cyl.location = (5, 0, 0)
    zig = poly("Zig", [(-2.5, -0.5, 0), (-1.0, 0.5, 0), (0.5, -0.5, 0), (2.0, 0.5, 0)], cyclic=False)
    select([zig, plane, cyl], cyl)
    objs, base, target = flow.guess_roles(ctx)
    check(f"[{tag}] roles: plane is the base", base is plane and target is cyl, f"{base and base.name}")
    r = run(bpy.ops.nproto.flow, base=plane.name, target=cyl.name, objects=json.dumps([zig.name]))
    out = ctx.active_object
    if r == {'FINISHED'} and out.type == 'CURVE':
        Q = np.vstack(curve_points(out))
        tol = flow.CURVE_TOL * 8.0
        check(f"[{tag}] zigzag on the cylinder", dist_to(cyl, Q) < 1.5 * tol, f"{dist_to(cyl, Q):.1e}")
        check(f"[{tag}] zigzag: a light spline", len(out.data.splines[0].points) < 200,
              f"{len(out.data.splines[0].points)} CVs")
    else:
        check(f"[{tag}] zigzag on the cylinder", False)
    # rigid: a small square moved and turned onto the cylinder, not bent
    sq = poly("Sq", [(-0.1, -0.1, 0.05), (0.1, -0.1, 0.05), (0.1, 0.1, 0.05), (-0.1, 0.1, 0.05)])
    r = run(bpy.ops.nproto.flow, base=plane.name, target=cyl.name, objects=json.dumps([sq.name]), rigid=True)
    out = ctx.active_object
    if r == {'FINISHED'} and out.type == 'CURVE':
        Q = np.vstack(curve_points(out, 4))
        P = np.vstack(curve_points(sq, 4))
        d0 = np.linalg.norm(P[:, None] - P[None], axis=2)
        d1 = np.linalg.norm(Q[:, None] - Q[None], axis=2)
        check(f"[{tag}] rigid keeps the shape", np.abs(d0 - d1).max() < 1e-6, f"{np.abs(d0 - d1).max():.1e}")
    else:
        check(f"[{tag}] rigid keeps the shape", False)

    # -- not developable: reported, relaxed --------------------------------------
    clear()
    bpy.ops.nproto.add_primitive(kind="SPHERE", radius=1.0)
    sph = ctx.active_object
    select([sph], sph)
    r = run(bpy.ops.nproto.unroll)
    check(f"[{tag}] sphere: a pattern anyway", r == {'FINISHED'} and ctx.active_object is not sph)

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
