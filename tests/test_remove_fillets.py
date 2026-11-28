"""Remove Fillets: a solid's fillets taken away, the faces round them meeting in
sharp edges again:
    blender -b --factory-startup --python tests/test_remove_fillets.py"""
import json, math, os, sys, tempfile, time
import bpy
import numpy as np
from mathutils import Matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, removefillets, steps, store, subsel
from blender_nurbs.kernel import load

sm = load()
ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
TMP = tempfile.mkdtemp(prefix="nproto_rf_")

def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

def vol(o): return abs(store.get_brep(o).volume(1e-6))
def nfaces(o): return store.get_brep(o).face_count()

def step_hl(o, k):
    """Faces the history panel outlines in blue with step k of o picked."""
    from blender_nurbs import subsel as _ss
    ctx.view_layer.objects.active = o
    bpy.ops.nproto.tree_step(name=o.name, root=o.name, index=k)
    out = sorted(_ss.step_faces(o))
    o.nproto_step.focus = False
    return out

def near_face(o, q):
    fs = store.get_brep(o).faces()
    return min(range(len(fs)), key=lambda j: fs[j].closest_point(tuple(map(float, q)))[1])

def face_info(o, js):
    fs = store.get_brep(o).faces()
    return [(j, [round(float(x), 3) for x in fs[j].centroid()], round(fs[j].area(), 3)) for j in js]
def solid(o): return bool(store.get_brep(o).is_manifold_solid())

def body(name, b, matrix=None):
    p = os.path.join(TMP, name + ".sm")
    b.write_to_file(p, ascii=True)
    with open(p) as f:
        return store.new_object_text(ctx, name, f.read(), matrix)

def edges_at(b, pts):
    return [min(b.edges(), key=lambda e: e.closest_point(p)[2]) for p in pts]

def box():
    return sm.create_box((0, 0, 0), 2, 1.5, 1)

def select(*objs):
    for o in ctx.selected_objects: o.select_set(False)
    for o in objs: o.select_set(True)
    ctx.view_layer.objects.active = objs[-1]

def faces(*items): return json.dumps([[o.name, list(map(float, p))] for o, p in items])

def run(**kw):
    try:
        return bpy.ops.nproto.remove_fillets(**kw)
    except RuntimeError as exc:
        print("   ", exc)
        return {'CANCELLED'}

TOP_BOTTOM_8 = [(1, 0, 1), (2, 0.75, 1), (1, 1.5, 1), (0, 0.75, 1), (1, 0, 0), (2, 0.75, 0), (1, 1.5, 0), (0, 0.75, 0)]

check("kernel has fillet removal", removefillets.kernel_has_it())

for hist in (True, False):
    ctx.scene.nproto_settings.construction_history = hist
    tag = "history" if hist else "no history"

    # every edge of a box rounded (12 fillets, 8 corners): one click, a sharp box
    clear()
    b = box(); sm.circular_fillet(b, 0.2)
    B = body("Rounded", b, Matrix.Translation((3, -1, 0.5)))
    found = removefillets.find(B)
    check(f"[{tag}] rounded box: 12 fillets, 8 corners, one chain", len(found["fillets"]) == 12
          and len(found["corners"]) == 8 and len(found["chains"]) == 1
          and all(abs(f["radius"] - 0.2) < 1e-6 for f in found["fillets"]),
          f"{len(found['fillets'])} {len(found['corners'])} {len(found['chains'])}")
    select(B)
    t0 = time.perf_counter()
    r = run(all_fillets=True)
    dt = time.perf_counter() - t0
    check(f"[{tag}] Remove All Fillets: a sharp box", r == {'FINISHED'} and nfaces(B) == 6 and solid(B)
          and abs(vol(B) - 3.0) < 1e-7, f"faces {nfaces(B)} vol {vol(B):.7f} {dt * 1000:.0f} ms")
    V = np.array([v.point() for v in store.get_brep(B).vertices()], float)
    check(f"[{tag}] the box is where it was (local space kept)", np.allclose(V.min(0), (0, 0, 0), atol=1e-7)
          and np.allclose(V.max(0), (2, 1.5, 1), atol=1e-7), f"{V.min(0)} {V.max(0)}")
    if hist:
        st = features.stack(B)
        check("history: one remove_fillets step", len(st) == 1 and st[0]["op"] == "remove_fillets"
              and steps.LABEL["remove_fillets"](st[0]["params"]) == "Remove Fillets · all", str(st)[:200])
        hl = step_hl(B, 0)
        check("history: picked Remove Fillets step highlights the faces that took over (all six)",
              hl == list(range(6)), f"{face_info(B, hl)}")
        B.location.x = 10; ctx.view_layer.update()
        features.rebuild(B, ctx, force=True)
        check("history: moved body rebuilds sharp", nfaces(B) == 6 and abs(vol(B) - 3.0) < 1e-7,
              f"faces {nfaces(B)} vol {vol(B):.6f}")
        raw = json.loads(B[features.PROP_STACK]); raw[0]["params"]["max_radius"] = 0.1
        B[features.PROP_STACK] = json.dumps(raw)
        try:
            features.rebuild(B, ctx, force=True)
        except Exception as exc:
            print("   ", exc)
        check("history: Max Radius 0.1 (below the fillets) leaves the rounded box", nfaces(B) == 26,
              f"faces {nfaces(B)}")

    # the four vertical fillets picked: they go, the top and bottom rounds are put back
    clear()
    b = box(); sm.circular_fillet(b, 0.2)
    B = body("Pick", b)
    ref = box(); sm.fillet_edges(ref, edges_at(ref, TOP_BOTTOM_8), radius=0.2)
    vert = [f["point"] for f in removefillets.find(B)["fillets"] if 0.3 < f["probe"][2] < 0.7]
    select(B)
    r = run(faces=faces(*[(B, p) for p in vert]))
    check(f"[{tag}] 4 vertical fillets picked: the 8 others put back", r == {'FINISHED'} and len(vert) == 4
          and nfaces(B) == ref.face_count() and abs(vol(B) - abs(ref.volume(1e-6))) < 1e-7,
          f"faces {nfaces(B)} want {ref.face_count()} vol {vol(B):.7f} want {abs(ref.volume(1e-6)):.7f}")
    if hist:
        st = features.stack(B)
        check("history: picked step label", steps.LABEL["remove_fillets"](st[-1]["params"])
              == "Remove Fillets · 4 picked", steps.LABEL["remove_fillets"](st[-1]["params"]))
        features.rebuild(B, ctx, force=True)
        check("history: picked step rebuilds the same", nfaces(B) == ref.face_count()
              and abs(vol(B) - abs(ref.volume(1e-6))) < 1e-7, f"faces {nfaces(B)}")

    # Keep Others off: the whole chain goes
    clear()
    b = box(); sm.circular_fillet(b, 0.2)
    B = body("Chain", b)
    p = [f["point"] for f in removefillets.find(B)["fillets"]][:1]
    select(B)
    r = run(faces=faces((B, p[0])), keep_others=False)
    check(f"[{tag}] Keep Others off: one pick takes the whole chain", r == {'FINISHED'} and nfaces(B) == 6
          and abs(vol(B) - 3.0) < 1e-7, f"faces {nfaces(B)}")

    # one of two separate fillets, picked in Face mode
    clear()
    b = box(); sm.fillet_edges(b, edges_at(b, [(1, 0, 1), (1, 1.5, 0)]), radius=0.2)
    B = body("Two", b)
    fl = removefillets.find(B)["fillets"]
    from blender_nurbs.pushpull import _face_index
    subsel.set_picks(B, "FACE", [_face_index(B, fl[0]["point"])])
    select(B)
    r = run()
    one = box(); sm.fillet_edges(one, edges_at(one, [(1, 0, 1)]), radius=0.2)
    check(f"[{tag}] Face-mode pick: one of two fillets removed", r == {'FINISHED'} and nfaces(B) == 7
          and abs(vol(B) - abs(one.volume(1e-6))) < 1e-7 and not subsel.picks(B, "FACE"), f"faces {nfaces(B)}")

# the add-on's own Edge Fillet (circular, Soft) under history, then Remove All Fillets on top
def poly(name, pts):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY'); sp.points.add(len(pts) - 1)
    for q, c in zip(sp.points, pts): q.co = (*c, 1.0)
    sp.use_cyclic_u = True
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o); return o

ctx.scene.nproto_settings.construction_history = True
for shape in ("CIRCULAR", "SOFT"):
    clear()
    R = poly("R", [(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)])
    bpy.ops.nproto.extrude(inputs=json.dumps([[R.name, "S0"]]), distance=2, direction="Z", body="NEW")
    E = ctx.active_object
    r = bpy.ops.nproto.edge_fillet(inputs=json.dumps([[E.name, f"E{i}"] for i in store.shared_edges(E)]),
                                   shape=shape, radius=0.25)
    filleted = nfaces(E)
    select(E)
    r2 = run(all_fillets=True)
    ops_ = [e["op"] for e in features.stack(E)]
    check(f"Edge Fillet {shape} (history) then Remove All Fillets: the block again",
          r == {'FINISHED'} and r2 == {'FINISHED'} and filleted > 6 and nfaces(E) == 6
          and abs(vol(E) - 24.0) < 1e-6 and ops_[-1] == "remove_fillets",
          f"filleted {filleted} faces -> {nfaces(E)}, vol {vol(E):.6f}, steps {ops_}")

# no history from here
ctx.scene.nproto_settings.construction_history = False

# a hole's rim and a boss's root: ring fillets on flat seats
clear()
a = box(); c = sm.create_cylinder((1, 0.75, -0.5), 0.3, 2)
b = sm.boolean(a, c, operation=sm.BooleanOp.DIFFERENCE)
sm.fillet_edges(b, edges_at(b, [(1.3, 0.75, 1)]), radius=0.1)
H = body("Hole", b)
select(H)
r = run(all_fillets=True)
want = 3.0 - math.pi * 0.09
check("hole rim fillet removed: a sharp hole", r == {'FINISHED'} and nfaces(H) == 7 and abs(vol(H) - want) < 1e-6,
      f"faces {nfaces(H)} vol {vol(H):.6f} want {want:.6f}")
clear()
a = sm.create_box((-1, -1, 0), 2, 2, 0.3); c = sm.create_cylinder((0, 0, 0.1), 0.4, 0.7)
b = sm.boolean(a, c, operation=sm.BooleanOp.UNION)
sm.fillet_edges(b, edges_at(b, [(0.4, 0, 0.3)]), radius=0.1)
Bo = body("Boss", b)
select(Bo)
r = run(all_fillets=True)
want = 1.2 + math.pi * 0.16 * 0.5
check("boss root fillet removed", r == {'FINISHED'} and nfaces(Bo) == 8 and abs(vol(Bo) - want) < 1e-6,
      f"faces {nfaces(Bo)} vol {vol(Bo):.6f} want {want:.6f}")

# a concave edge (an L) and a G2 blend
clear()
a = sm.create_box((0, 0, 0), 2, 1, 0.5); c = sm.create_box((0, 0, 0), 0.5, 1, 2)
b = sm.boolean(a, c, operation=sm.BooleanOp.UNION)
sm.fillet_edges(b, edges_at(b, [(0.5, 0.5, 0.5)]), radius=0.2)
L = body("L", b)
b = box(); sm.fillet_edges(b, edges_at(b, [(1, 0, 1)]), radius=0.2, xsect_type=2, continuity=2)
G = body("G2", b)
select(L, G)
r = run(all_fillets=True)
check("concave L fillet and G2 blend removed (two bodies at once)", r == {'FINISHED'} and nfaces(L) == 8
      and abs(vol(L) - 1.75) < 1e-7 and nfaces(G) == 6 and abs(vol(G) - 3.0) < 1e-7,
      f"L {nfaces(L)} {vol(L):.6f} G {nfaces(G)} {vol(G):.6f}")

# a slot (flat sides tangent to half-round ends) with its top rim rounded: the rim goes, the ends stay
clear()
a = sm.create_box((0, 0, 0), 2, 1, 0.5)
c1 = sm.create_cylinder((0, 0.5, 0), 0.5, 0.5); c2 = sm.create_cylinder((2, 0.5, 0), 0.5, 0.5)
slot = sm.boolean(sm.boolean(a, c1, operation=sm.BooleanOp.UNION), c2, operation=sm.BooleanOp.UNION)
v_slot = abs(slot.copy().volume(1e-6))
b = slot.copy()
b = sm.boolean(sm.boolean(sm.create_box((0, 0, 0), 2, 1, 0.5), sm.create_cylinder((0, 0.5, 0), 0.5, 0.5),
               operation=sm.BooleanOp.UNION), sm.create_cylinder((2, 0.5, 0), 0.5, 0.5), operation=sm.BooleanOp.UNION)
sm.fillet_edges(b, [e for e in b.edges() if abs(e.point_at(sum(e.parameter_range()) / 2)[2] - 0.5) < 1e-6], radius=0.1)
S = body("Slot", b)
found = removefillets.find(S)
check("slot: its half-round ends are no fillets", len(found["fillets"]) == 5 and len(found["skipped"]) >= 2,
      f"{len(found['fillets'])} fillets, skipped {[s['why'][:24] for s in found['skipped']]}")
select(S)
r = run(all_fillets=True)
check("slot: the rim's rounds removed, the ends kept", r == {'FINISHED'} and nfaces(S) == 7
      and abs(vol(S) - v_slot) < 1e-6, f"faces {nfaces(S)} vol {vol(S):.6f} want {v_slot:.6f}")

# nothing to remove: a plain slot (only tangent faces), a sphere -- nothing changes
clear()
S = body("PlainSlot", slot)
text = S[store.PROP_BREP]
select(S)
r = run(all_fillets=True)
check("plain slot: nothing removed, body unchanged", r == {'CANCELLED'} and S[store.PROP_BREP] == text)

# a revolved puck (top, side, a 26.6 deg bevel, bottom), its three rims filleted by
# chord: the fillets were 150-CV skins the kernel's defeature could not take off
# ("not a valid brep"), and the bevel | bottom one alone came back unchanged
# (the faces' default extension too short at that shallow angle)
def puck():
    clear()
    cu = bpy.data.curves.new("Puck", 'CURVE'); cu.dimensions = '3D'
    PUCK = [(0, 0, 0.508), (0.2032, 0, 0.508), (0.2032, 0, 0.4953), (0.1778, 0, 0.4826), (0, 0, 0.4826)]
    sp = cu.splines.new('POLY'); sp.points.add(len(PUCK) - 1)
    for q, c in zip(sp.points, PUCK):
        q.co = (*c, 1.0)
    pr = bpy.data.objects.new("Puck", cu); ctx.collection.objects.link(pr)
    bpy.ops.nproto.revolve(inputs=json.dumps([[pr.name, "S0"]]), angle=360.0, axis="Z", pivot="WORLD", body="NEW")
    V = ctx.active_object
    b = store.get_brep(V)
    rims = [(0.2032, 0.508), (0.2032, 0.4953), (0.1778, 0.4826)]
    idx = [min(range(b.edge_count()), key=lambda i: (lambda e, q: abs(q[2] - z) + abs(math.hypot(q[0], q[1]) - rr)
                                                      + (0 if len(e.faces()) == 2 else 1e3))(
               b.edges()[i], b.edges()[i].point_at(0.5 * sum(b.edges()[i].parameter_range())))) for rr, z in rims]
    bpy.ops.nproto.edge_fillet(inputs=json.dumps([[V.name, f"E{i}"] for i in idx]), shape="CIRCULAR",
                               rail_type="CHORD", radius=0.00635)
    return V

V = puck()
fl = removefillets.find(V)["fillets"]
cvs = [len(sm.blender_ext.surface_cvs(store.get_brep(V).faces()[f["face"]].surface())["points"]) for f in fl]
check("puck: chord fillets on its rims are exact rings", nfaces(V) == 7 and len(fl) == 3 and max(cvs) <= 27,
      f"faces {nfaces(V)} fillets {len(fl)} CVs {cvs}")
select(V)
r = run(all_fillets=True)
h, R0, R1 = 0.0127, 0.2032, 0.1778
want = math.pi * R0 * R0 * h + math.pi * h / 3 * (R0 * R0 + R0 * R1 + R1 * R1)
check("puck: all chord fillets removed, the sharp profile back", r == {'FINISHED'} and nfaces(V) == 4 and solid(V)
      and abs(vol(V) - want) < 1e-5 * want, f"faces {nfaces(V)} vol {vol(V):.9f} want {want:.9f}")
V = puck()
select(V)
p = min(removefillets.find(V)["fillets"], key=lambda f: abs(f["point"][2] - 0.4829))["point"]
r = run(faces=faces((V, p)))
check("puck: the bevel | bottom fillet alone removed", r == {'FINISHED'} and nfaces(V) == 6 and solid(V),
      f"faces {nfaces(V)}")
# then the others one by one: after a removal the kernel's interior_point() of a
# fillet lay microns off its rail, so the new sharp corner passed within the
# "fillet gone" tolerance of it and every good result was turned down
for z, n in ((0.4953, 5), (0.508, 4)):
    p = min(removefillets.find(V)["fillets"], key=lambda f: abs(f["point"][2] - z))["point"]
    r = run(faces=faces((V, p)))
    check(f"puck: next fillet removed one by one ({n} faces)", r == {'FINISHED'} and nfaces(V) == n and solid(V),
          f"faces {nfaces(V)}")
check("puck: one by one ends at the sharp profile", abs(vol(V) - want) < 1e-5 * want,
      f"vol {vol(V):.9f} want {want:.9f}")

# a free-form skin: every face tangent to its neighbours is no fillet
clear()
P = body("Ball", sm.create_sphere((0, 0, 0), 1.0))
check("sphere: no fillets", not removefillets.find(P)["fillets"])

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
