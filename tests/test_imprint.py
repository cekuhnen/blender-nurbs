"""Project with Output: Edge (Imprint), and Intersect with Output: Edge:
    blender -b --factory-startup --python tests/test_imprint.py"""
import json, math, os, sys
import bpy
from mathutils import Matrix, Vector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, store, subsel

ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

def poly(name, pts, cyclic=False):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for p, c in zip(sp.points, pts): p.co = (*c, 1.0)
    sp.use_cyclic_u = cyclic
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def circle(name, c, r, n=4):
    """An exact NURBS circle (rational, 9 CVs) in the plane z = c[2]."""
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('NURBS')
    w = math.sqrt(0.5)
    pts = [(1, 0, 1), (1, 1, w), (0, 1, 1), (-1, 1, w), (-1, 0, 1), (-1, -1, w), (0, -1, 1), (1, -1, w)]
    sp.points.add(len(pts) - 1)
    for p, (x, y, wt) in zip(sp.points, pts):
        p.co = (c[0] + r * x, c[1] + r * y, c[2], wt)
    sp.use_cyclic_u = True
    sp.order_u = 3
    sp.use_endpoint_u = False
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

def ins(*pairs): return json.dumps([[o.name, r] for o, r in pairs])
def brep(o): return store.get_brep(o)
def vol(o): return abs(brep(o).volume(1e-6))
def solid(o): return bool(brep(o).is_manifold_solid())
def nfaces(o): return brep(o).face_count()
def faces_at(o, z, tol=1e-6):
    """Faces of o lying in the plane z (world, o unmoved)."""
    out = []
    for f in brep(o).faces():
        p = f.interior_point()
        p = (p.x, p.y, p.z) if hasattr(p, "x") else tuple(p)
        if abs(p[2] - z) <= tol and f.surface().is_planar():
            out.append(f)
    return out

def select(objs, active):
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = active

def box(name="Box", h=2.0):
    R = poly(name + "_R", [(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)], cyclic=True)
    bpy.ops.nproto.extrude(inputs=ins((R, "S0")), distance=h, direction="Z", body="NEW")
    o = ctx.active_object; o.name = name
    return o

def project(curves, target, **kw):
    select(curves + [target], target)
    return bpy.ops.nproto.project_curve(output="EDGE", **kw)

for hist in (True, False):
    ctx.scene.nproto_settings.construction_history = hist
    tag = "[history]" if hist else "[no history]"

    # ---- a rectangle sketched on the top of a box: the top is divided, nothing else
    clear()
    B = box()
    V0 = vol(B)
    K = poly("K", [(1, 1, 2), (3, 1, 2), (3, 2, 2), (1, 2, 2)], cyclic=True)
    r = project([K], B)
    top = faces_at(B, 2.0)
    check(f"{tag} rectangle on the top: the top divided into two faces", len(top) == 2 and nfaces(B) == 7,
          f"{len(top)} top faces, {nfaces(B)} faces")
    check(f"{tag} still one closed solid, same volume", solid(B) and abs(vol(B) - V0) < 1e-6, f"{vol(B)}")
    areas = sorted(round(f.area(1e-6), 6) for f in top)
    check(f"{tag} the inner face is the rectangle (2 x 1)", areas == [2.0, 10.0], str(areas))
    if hist:
        st = features.stack(B)
        check("[history] the imprint is a step on the box", [e["op"] for e in st] == ["imprint"], str(st))
        # move the sketch: the edges follow
        K.location = (0.5, 0.5, 0.0)
        features.rebuild(B, ctx, force=True)
        top = faces_at(B, 2.0)
        inner = min(top, key=lambda f: f.area(1e-6)) if top else None
        c = inner.centroid() if inner else None
        c = (c.x, c.y, c.z) if hasattr(c, "x") else c
        check("[history] the sketch moved: the imprinted face moved with it",
              len(top) == 2 and c is not None and abs(c[0] - 2.5) < 1e-4 and abs(c[1] - 2.0) < 1e-4,
              str(c))
        K.location = (0, 0, 0)
        features.rebuild(B, ctx, force=True)

    # ---- Push / Pull the imprinted face: a boss
    inner = min(faces_at(B, 2.0), key=lambda f: f.area(1e-6))
    ip = inner.interior_point()
    ip = (ip.x, ip.y, ip.z) if hasattr(ip, "x") else tuple(ip)
    try:
        r = bpy.ops.nproto.push_pull(faces=json.dumps([[B.name, list(ip)]]), distance=0.5)
    except RuntimeError as exc:
        r = {str(exc)}
    check(f"{tag} Push / Pull the imprinted face: a boss on the box", r == {'FINISHED'} and solid(B)
          and abs(vol(B) - (V0 + 2.0 * 0.5)) < 1e-4, f"{r} {vol(B)}")

    # ---- a line drawn above the box, across it: only the top takes it (Nearest)
    clear()
    B = box()
    L = poly("L", [(1, -1, 3), (1, 4, 3)])
    project([L], B, direction="Z")
    check(f"{tag} line above, Nearest: only the top divided (7 faces)", nfaces(B) == 7 and len(faces_at(B, 2.0)) == 2,
          f"{nfaces(B)} faces")
    walls = [f for f in brep(B).faces() if abs((lambda p: p.y if hasattr(p, 'y') else p[1])(f.interior_point())) < 1e-6]
    check(f"{tag} the front wall's top edge is split where the line meets it",
          walls and sum(len(lp) for lp in walls[0].boundary_loops()) == 5)
    clear()
    B = box()
    L = poly("L", [(1, -1, 3), (1, 4, 3)])
    project([L], B, direction="Z", scope="THROUGH")
    check(f"{tag} line above, Through: top, bottom and both walls divided (10 faces)", nfaces(B) == 10,
          f"{nfaces(B)} faces")

    # ---- an open line ending inside the top: an edge that ends in the face
    clear()
    B = box()
    E0 = len(brep(B).edges())
    L = poly("L", [(1, 1, 2), (3, 1, 2)])
    project([L], B)
    check(f"{tag} open line inside the top: one more edge, the faces as they were",
          len(brep(B).edges()) == E0 + 1 and nfaces(B) == 6 and solid(B), f"{len(brep(B).edges())} edges")
    clear()
    B = box()
    L = poly("L", [(1, 1, 2), (3, 1, 2)])
    project([L], B, extend=True)
    check(f"{tag} the same line with Extend to Edges: the top divided", nfaces(B) == 7, f"{nfaces(B)} faces")

    # ---- Picked Faces: a square above the box, onto the bottom only
    clear()
    B = box()
    Q = poly("Q", [(1.5, 1, 3), (2.5, 1, 3), (2.5, 2, 3), (1.5, 2, 3)], cyclic=True)
    select([Q, B], B)
    subsel.set_picks(B, "FACE", [i for i, f in enumerate(brep(B).faces()) if f in faces_at(B, 0.0)])
    project([Q], B, direction="Z", scope="FACES")
    check(f"{tag} picked face (bottom): the square lands there, not on the top",
          len(faces_at(B, 0.0)) == 2 and len(faces_at(B, 2.0)) == 1, f"{nfaces(B)} faces")
    small = [round(f.area(1e-7), 6) for f in faces_at(B, 0.0)]
    check(f"{tag} the bottom's new face is the square (1 x 1)", min(small or [0]) == 1.0, str(small))

    # ---- a sketch with a stray one-point spline (a click Draw Line left behind): left out
    for out in ("CURVE", "EDGE"):
        clear()
        B = box()
        L = poly("L", [(1, -1, 3), (1, 4, 3)])
        sp = L.data.splines.new('NURBS')
        sp.points[0].co = (3, 1, 3, 1)
        select([L, B], B)
        try:
            r = bpy.ops.nproto.project_curve(output=out, direction="Z")
        except RuntimeError as exc:
            r = {str(exc).strip().splitlines()[-1]}
        check(f"{tag} Output {out}: a one-point spline in the sketch is skipped", r == {'FINISHED'}
              and (ctx.active_object.type == 'CURVE' if out == "CURVE" else nfaces(B) == 7), str(r))

    # ---- Output Curve still makes a curve and leaves the box alone
    clear()
    B = box()
    C = circle("C", (2, 1.5, 3), 0.8)
    select([C, B], B)
    bpy.ops.nproto.project_curve(output="CURVE")
    made = ctx.active_object
    check(f"{tag} Output Curve: a new curve, the box unchanged", made is not None and made.type == 'CURVE'
          and nfaces(B) == 6)

    # ---- a groove outline on a cylinder (Shapr3D's example): the front only. Away
    # from the tube's seam (an outline across the seam comes out as two faces:
    # the kernel's trims do not cross a closed surface's seam)
    clear()
    bpy.ops.nproto.add_primitive(kind="CYLINDER")
    Y = ctx.active_object
    srf = brep(Y).faces()[0].surface()
    (u0, u1), (v0, v1) = srf.u_domain(), srf.v_domain()
    sp = Vector(tuple(srf.evaluate(u0, 0.5 * (v0 + v1))))
    bb = [Vector(c) for c in Y.bound_box]
    zmid = 0.5 * (min(v.z for v in bb) + max(v.z for v in bb))
    rad = max(v.x for v in bb)
    out = -Vector((sp.x, sp.y, 0)).normalized()          # facing away from the seam
    side = out.cross(Vector((0, 0, 1)))
    c0 = out * 3 * rad + Vector((0, 0, zmid))
    h = 0.2 * rad
    G = poly("G", [tuple(c0 + side * x + Vector((0, 0, z))) for x, z in ((-h, -h), (h, -h), (h, h), (-h, h))],
             cyclic=True)
    n0 = nfaces(Y)
    project([G], Y, direction="VIEW", view_dir=tuple(-out))
    front = [f for f in brep(Y).faces() if f.area(1e-6) < 0.5 * rad * rad]
    check(f"{tag} cylinder: the rectangle lands on the front only (one face more)",
          nfaces(Y) == n0 + 1 and len(front) == 1, f"{n0} -> {nfaces(Y)}, small faces {len(front)}")

    # ---- the same rectangle right across the tube's seam: still one face (the
    # two pieces either side of the seam trimmed again as one, seam moved away)
    clear()
    bpy.ops.nproto.add_primitive(kind="CYLINDER")
    Y = ctx.active_object
    out = Vector((sp.x, sp.y, 0)).normalized()           # facing the seam
    side = out.cross(Vector((0, 0, 1)))
    c0 = out * 3 * rad + Vector((0, 0, zmid))
    G = poly("G", [tuple(c0 + side * x + Vector((0, 0, z))) for x, z in ((-h, -h), (h, -h), (h, h), (-h, h))],
             cyclic=True)
    A0 = brep(Y).area(1e-6)
    project([G], Y, direction="VIEW", view_dir=tuple(-out))
    small = [f for f in brep(Y).faces() if f.area(1e-6) < 0.5 * rad * rad]
    check(f"{tag} cylinder, across the seam: one face for the rectangle, not two",
          nfaces(Y) == n0 + 1 and len(small) == 1 and abs(brep(Y).area(1e-6) - A0) < 1e-4 * A0,
          f"{n0} -> {nfaces(Y)}, small faces {[round(f.area(1e-6), 4) for f in small]}")
    if hist:
        G.location = (0, 0, 0.1 * rad)
        features.rebuild(Y, ctx, force=True)
        small = [f for f in brep(Y).faces() if f.area(1e-6) < 0.5 * rad * rad]
        check("[history] across the seam: the sketch moved, still one face", len(small) == 1,
              str([round(f.area(1e-6), 4) for f in small]))

    # ---- Intersect with Output Edge: a plane through a box, each takes the other's edges
    clear()
    B = box()
    bpy.ops.nproto.add_primitive(kind="PLANE")
    P = ctx.active_object
    P.location = (2, 1.5, 1.0)
    P.scale = (10, 10, 1)
    bpy.context.view_layer.update()
    nP = nfaces(P)
    select([B, P], B)
    r = bpy.ops.nproto.intersect(output="EDGE", onto="BOTH")
    check(f"{tag} Intersect Edge: the four walls divided at the plane", nfaces(B) == 10 and solid(B),
          f"{nfaces(B)} faces")
    check(f"{tag} Intersect Edge: the plane divided along the box", nfaces(P) == nP + 1, f"{nP} -> {nfaces(P)}")
    if hist:
        check("[history] both took an imprint step", [e["op"] for e in features.stack(B)] == ["imprint"]
              and [e["op"] for e in features.stack(P)] == ["imprint"])
        P.location = (2, 1.5, 1.5)
        bpy.context.view_layer.update()
        features.rebuild(B, ctx, force=True)
        zs = sorted({round((lambda p: p.z if hasattr(p, "z") else p[2])(v.point()), 4)
                     for v in brep(B).vertices()}) if hasattr(brep(B), "vertices") else None
        check("[history] the plane moved up: the box's imprint followed", zs is None or 1.5 in zs, str(zs))

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
