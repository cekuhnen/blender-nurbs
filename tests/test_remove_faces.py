"""Remove Faces: faces of a solid taken away and the gap healed by extending
the faces round it, so the body stays closed:
    blender -b --factory-startup --python tests/test_remove_faces.py"""
import json, os, sys, tempfile, time
import bpy
from mathutils import Matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import blender_nurbs
blender_nurbs.register()
from blender_nurbs import features, steps, store, subsel
from blender_nurbs.kernel import load
from blender_nurbs.kernel import healfaces as hf

sm = load()
ok = True
def check(name, cond, extra=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name, extra)
    ok &= bool(cond)

ctx = bpy.context
TMP = tempfile.mkdtemp(prefix="nproto_rface_")

def clear():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)

def vol_b(b): return abs(b.volume(1e-6))
def vol(o): return vol_b(store.get_brep(o))
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

def select(*objs):
    for o in ctx.selected_objects: o.select_set(False)
    for o in objs: o.select_set(True)
    ctx.view_layer.objects.active = objs[-1]

def run(**kw):
    try:
        return bpy.ops.nproto.remove_faces(**kw)
    except RuntimeError as exc:
        print("   ", exc)
        return {'CANCELLED'}

def box(): return sm.create_box((0, 0, 0), 2, 1.5, 1)
def through_hole(): return sm.boolean_difference(box(), sm.create_cylinder((1, 0.75, -0.5), 0.3, 2))
def blind_hole(): return sm.boolean_difference(box(), sm.create_cylinder((1, 0.75, 0.5), 0.3, 1))
def boss(): return sm.boolean_union(box(), sm.create_cylinder((1, 0.75, 1), 0.3, 0.5))
def pocket(): return sm.boolean_difference(box(), sm.create_box((0.5, 0.4, 0.6), 1, 0.7, 0.6))

def edge_at(b, p): return min(b.edges(), key=lambda e: e.closest_point(p)[2])

# --- kernel (healfaces) ---------------------------------------------------------------
check("kernel has face defeaturing", hf.has_kernel(sm))

def healed(name, b, pts, want_faces, want_vol=3.0, extra=None):
    t0 = time.perf_counter()
    out, rep = hf.remove(sm, b, pts)
    dt = (time.perf_counter() - t0) * 1000
    good = out is not None and out.face_count() == want_faces and out.is_manifold_solid() \
        and (want_vol is None or abs(vol_b(out) - want_vol) < 1e-6)
    if extra:
        good = good and all(rep.get(k) == v for k, v in extra.items())
    check(f"{name}: {b.face_count()} -> {want_faces} faces", good,
          f"got {out and out.face_count()} vol {out and round(vol_b(out), 7)} {dt:.0f} ms {rep}")
    return out, rep

healed("through hole: its wall", through_hole(), [(1.3, 0.75, 0.5)], 6)
healed("blind hole: wall and floor", blind_hole(), [(1.3, 0.75, 0.8), (1, 0.75, 0.5)], 6)
healed("blind hole: the wall alone takes its floor along", blind_hole(), [(1.3, 0.75, 0.8)], 6, extra={"added": 1})
healed("boss: wall and top", boss(), [(1.3, 0.75, 1.2), (1, 0.75, 1.5)], 6)
healed("boss: the wall alone takes its top along", boss(), [(1.3, 0.75, 1.2)], 6, extra={"added": 1})
healed("pocket: floor and four walls", pocket(), [(1, 0.75, 0.6), (0.5, 0.75, 0.8), (1.5, 0.75, 0.8),
                                                  (1, 0.4, 0.8), (1, 1.1, 0.8)], 6)
two = sm.boolean_union(sm.boolean_difference(box(), sm.create_cylinder((0.5, 0.75, -0.5), 0.2, 2)),
                       sm.create_cylinder((1.5, 0.75, 1), 0.2, 0.3))
healed("a hole and a boss in one go (two features)", two, [(0.7, 0.75, 0.5), (1.7, 0.75, 1.1), (1.5, 0.75, 1.3)], 6,
       extra={"groups": 2})
ch = box(); sm.fillet_edges(ch, [edge_at(ch, (1, 0, 1))], radius=0.2, xsect_type=int(sm.FilletXSect.LINEAR))
healed("a chamfer face: the sharp edge again", ch, [(1, 0.06, 0.94)], 6)
st = box(); sm.fillet_edges(st, [edge_at(st, (1, 0, 1))], radius=0.03, xsect_type=int(sm.FilletXSect.LINEAR))
healed("a narrow strip (0.03): the sharp edge again", st, [(1, 0.01, 0.99)], 6)
rb = box(); sm.circular_fillet(rb, 0.2)
t0 = time.perf_counter()
out, rep = hf.remove(sm, rb, [(1, 0.06, 0.94)])
dt = time.perf_counter() - t0
check("one rounded edge of a fully rounded box: by Remove Fillets' rules, quickly (no kernel hang)",
      out is not None and out.is_manifold_solid() and out.face_count() < rb.face_count() and rep["fillets"] == 1
      and dt < 10, f"{rb.face_count()} -> {out and out.face_count()} {dt:.2f}s {rep}")
out, rep = hf.remove(sm, box(), [(1, 0.75, 1)])
check("a box's top: can't heal (its walls never meet), reported", out is None and rep["failed"]
      and "could not be extended" in rep["failed"][0]["why"], str(rep))
out, rep = hf.remove(sm, through_hole(), [(9, 9, 9)])
check("a pick far from every face: missed", out is None and rep["missed"] == [0], str(rep))

# --- the operator ---------------------------------------------------------------------
for hist in (True, False):
    ctx.scene.nproto_settings.construction_history = hist
    tag = "history" if hist else "no history"
    clear()
    B = body("Plate", through_hole(), Matrix.Translation((3, -1, 0.5)))
    select(B)
    from blender_nurbs.pushpull import _face_index
    subsel.set_picks(B, "FACE", [_face_index(B, (1.3, 0.75, 0.5))])
    r = run()
    check(f"[{tag}] Face mode pick: the hole is gone, the plate whole", r == {'FINISHED'} and nfaces(B) == 6
          and solid(B) and abs(vol(B) - 3.0) < 1e-7 and not subsel.picks(B, "FACE"), f"{r} faces {nfaces(B)}")
    if hist:
        st_ = features.stack(B)
        check("history: one remove_faces step", [e["op"] for e in st_] == ["remove_faces"]
              and steps.LABEL["remove_faces"](st_[0]["params"]) == "Remove Faces · 1 healed", str(st_)[:200])
        hl = step_hl(B, 0)
        want = sorted({near_face(B, (1, 0.75, 1)), near_face(B, (1, 0.75, 0))})
        check("history: picked Remove Faces step highlights the healed top and bottom", hl == want,
              f"{face_info(B, hl)} want {want}")
        B.location.x = 10; ctx.view_layer.update()
        features.rebuild(B, ctx, force=True)
        check("history: moved body rebuilds healed", nfaces(B) == 6 and abs(vol(B) - 3.0) < 1e-7, f"faces {nfaces(B)}")
    else:
        check("no history: no steps recorded", not features.stack(B))
    clear()
    B = body("Box", box())
    select(B)
    r = run(faces=json.dumps([[B.name, [1, 0.75, 1]]]))
    check(f"[{tag}] a box's top: cancelled, the body unchanged", r == {'CANCELLED'} and nfaces(B) == 6, f"{r}")
    r = run()
    check(f"[{tag}] nothing picked: cancelled", r == {'CANCELLED'})

print("ALL OK" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
