"""Builds a scene to try the Emboss tool in, headless:
    blender -b --factory-startup --python tools/emboss_scene.py -- "<out.blend>"

Stations along X, a label on the ground in front of each. 'Try' stations have
the inputs laid out to run Emboss on (select the lettering, then the body,
Surface Edit > Solids > Emboss); 'Done' ones have the result made with
history (Height, Overlap, Mode, Emboss / Deboss editable in the Feature panel;
the inputs are in the NURBS History collection). Lettering is Blender text
converted to curves (Bezier outlines with their holes). Prints a REPORT line."""
import math, os, sys, time
import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
out = sys.argv[sys.argv.index("--") + 1] if "--" in sys.argv else os.path.join(ROOT, "Blend Files", "Emboss demo.blend")
import blender_nurbs; blender_nurbs.register()
from blender_nurbs import emboss, flow, store, surfeval, surfmap, unroll
sm = store._k()[0]
ctx = bpy.context
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
ctx.scene.nproto_settings.construction_history = True
report = {}
ROW_A, ROW_B = 4.5, -4.5   # y of the two rows (labels 3.2 in front of each station)


def select(objs, active):
    for o in list(ctx.view_layer.objects):
        if o is not None:
            o.select_set(False)
    for o in objs:
        o.select_set(True)
    ctx.view_layer.objects.active = active


def label(text, x, y, size=0.26):
    cu = bpy.data.curves.new("Label", 'FONT'); cu.body = text; cu.size = size; cu.align_x = 'CENTER'
    o = bpy.data.objects.new("Label: " + text.split("\n")[0], cu)
    o.location = (x, y, 0.0)
    ctx.collection.objects.link(o)
    return o


def lettering(name, text, loc, size, rot=(0.0, 0.0, 0.0), mirror=False):
    """Blender text converted to curves (one curve object, a spline per outline / hole)."""
    cu = bpy.data.curves.new(name, 'FONT'); cu.body = text; cu.size = size
    cu.align_x = 'CENTER'; cu.align_y = 'CENTER'
    cu.space_character = 1.15            # kerned letters touch (W A): Emboss wants outlines apart
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o)
    o.location, o.rotation_euler = loc, rot
    if mirror:
        o.scale.x = -1.0
    select([o], o)
    bpy.ops.object.convert(target='CURVE')
    o = ctx.active_object
    o.name = name
    o.data.dimensions = '3D'
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    return o


def star(name, c, r_out, r_in, n=5):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'
    sp = cu.splines.new('POLY')
    pts = [(c[0] + (r_out if k % 2 == 0 else r_in) * math.cos(math.pi / 2 + k * math.pi / n),
            c[1] + (r_out if k % 2 == 0 else r_in) * math.sin(math.pi / 2 + k * math.pi / n), c[2]) for k in range(2 * n)]
    sp.points.add(len(pts) - 1)
    for p, q in zip(sp.points, pts):
        p.co = (*q, 1.0)
    sp.use_cyclic_u = True
    o = bpy.data.objects.new(name, cu); ctx.collection.objects.link(o)
    return o


def dome_body(name, x, y=0.0):
    """A doubly curved housing: the sheet z = 0.5 (1 - (x/2)^2) + 0.4 (1 - y^2)
    over 4 x 2, thickened 0.3 down into a solid."""
    secs = [sm.create_curve_interp_points([(float(u), float(v), float(0.5 * (1 - (u / 2) ** 2) + 0.4 * (1 - v * v)))
                                           for u in np.linspace(-2, 2, 9)]) for v in np.linspace(-1, 1, 5)]
    sheet = store.new_object(ctx, name + " Sheet", sm.create_skin_surface(secs))
    sheet.location = (x, y, 0.0)
    ctx.view_layer.update()
    select([sheet], sheet)
    bpy.ops.nproto.thicken(distance=0.3)
    b = ctx.active_object
    b.name = name
    return b


def vol(o):
    return abs(flow.world_brep(o).volume(1e-7))


def run_emboss(tag, objs, body, **kw):
    print(f"STATION {tag} ...", flush=True)
    V0 = vol(body)
    select(objs + [body], body)
    t0 = time.perf_counter()
    r = bpy.ops.nproto.emboss(**kw)
    dt = time.perf_counter() - t0
    res = ctx.active_object
    good = r == {'FINISHED'} and res is not body and store.is_nurbs(res)
    report[tag] = {"ok": good, "solid": good and bool(flow.world_brep(res).is_manifold_solid()),
                   "dV": round(vol(res) - V0, 6) if good else None, "s": round(dt, 1),
                   "mode": dict(emboss.LAST_INFO).get("wrap", kw.get("mode"))}
    print(f"STATION {tag}: {report[tag]}", flush=True)
    return res


def sheet_with(body, face_index):
    """(sheet, face index on it) of Unroll's sheets that holds the body's
    face; the other pieces (caps, sides) are deleted."""
    import json
    found = None
    sheets = [o for o in ctx.selected_objects if unroll.PROP_UNROLL in o]
    for o in sheets:
        for i, (src, _) in enumerate(json.loads(o[unroll.PROP_UNROLL])["faces"]):
            if src == face_index and found is None:
                found = (o, i)
    if found is None:
        raise RuntimeError("no sheet holds that face")
    for o in sheets:
        if o is not found[0]:
            bpy.data.objects.remove(o)
    return found


def spot_and_turn(sheet, fi, body, toward=(0.0, -1.0, 0.0), along=(1.0, 0.0, 0.0)):
    """(centre, turn about Z, mirror) for lettering drawn on the sheet's face
    fi: the spot that lands most toward `toward` once folded back (the
    camera's side, or the top), the lettering reading left to right along
    `along` there, mirrored if the sheet is seen from the body's inside."""
    f = flow.world_brep(sheet).faces()[fi]
    P = np.vstack([surfmap.edge_samples(e, 9) for e in f.edges()])
    lo, hi = P.min(axis=0), P.max(axis=0)
    fmap, _ = emboss.wrap_map(sheet, body, None, {})
    g = np.linspace(0.25, 0.75, 11)
    C = np.array([[lo[0] + a * (hi[0] - lo[0]), lo[1] + b * (hi[1] - lo[1]), lo[2]] for a in g for b in g])
    S = surfeval.Surf(f.surface())
    C = C[[f.classify_uv(*map(float, S.feet(c[None])[0])) == "inside" for c in C]]
    c = C[int(np.argmax(fmap.map(C) @ np.asarray(toward, float)))]
    e = 1e-3 * float(np.linalg.norm(hi - lo))
    Q = fmap.map(np.array([c, c + [e, 0, 0], c + [0, e, 0]]))
    _, _, nout = emboss._hit_face(flow.world_brep(body), Q[0])
    mirror = float(np.cross(Q[1] - Q[0], Q[2] - Q[0]) @ nout) < 0.0
    m = -1.0 if mirror else 1.0
    best = None
    for k in range(4):
        th = k * math.pi / 2
        d = m * np.array([math.cos(th), math.sin(th), 0.0])            # where the text's x runs on the sheet
        q = fmap.map(np.array([c, c + e * d]))
        score = float((q[1] - q[0]) @ np.asarray(along, float))
        if best is None or score > best[0]:
            best = (score, th)
    return c, (0.0, 0.0, best[1]), mirror


# 1 -- Project, try ------------------------------------------------------------------------
x, y = -3.5, ROW_A
d1 = dome_body("Dome (try)", x, y)
t1 = lettering("Lettering (try)", "NURBS", (x, y, 2.2), 0.6)
label("PROJECT  (try)\nSelect the lettering, then the dome,\nSurface Edit > Solids > Emboss.\nIt is thrown straight down (the\nplane's normal): walls stay vertical", x, y - 3.2)

# 2 -- Project, done -----------------------------------------------------------------------
x, y = 3.5, ROW_A
d2 = dome_body("Dome", x, y)
t2 = lettering("Lettering", "NURBS", (x, y, 2.2), 0.6)
run_emboss("project emboss", [t2], d2, height=0.05, overlap=0.015)
label("PROJECT  (done)\nHeight 0.05 out of the double-curved\ntop. Feature panel: change Height,\nOverlap, or Deboss", x, y - 3.2)

# 3 -- Deboss onto a box's side, projected -------------------------------------------------
x, y = 10.5, ROW_A
bpy.ops.nproto.add_primitive(kind="BOX", width=3.0, depth=1.2, height=1.2)
box = ctx.active_object
box.name = "Block"
box.location = (x, y, 0.6)
ctx.view_layer.update()
lo, hi = unroll._box(flow.world_brep(box))
tb = lettering("Lettering (front)", "LOGO", (0.5 * (lo[0] + hi[0]), lo[1] - 1.5, 0.5 * (lo[2] + hi[2])), 0.55,
               rot=(math.pi / 2, 0.0, 0.0))
run_emboss("project deboss", [tb], box, operation="DEBOSS", height=0.06, overlap=0.015)
label("DEBOSS  (done)\nSunk 0.06 into the front face,\nthrown along -Y from a plane\nin front of the block", x, y - 3.2)

# 4 / 5 -- Wrap around a cylinder: unroll it, letter the flat sheet ---------------------------
for (tag, title), x in zip((("try", "WRAP  (try)\nUnroll made the sheet. Select the\nlettering, the sheet, then the\ncylinder: Emboss wraps it round,\nits size kept exactly"),
                            ("done", "WRAP  (done)\nThe lettering drawn on the unrolled\nsheet, folded back onto the\ncylinder at its true size")),
                           (-17.0, -9.0)):
    y = ROW_B
    bpy.ops.nproto.add_primitive(kind="CYLINDER", radius=0.7, height=2.0, caps=True)
    cyl = ctx.active_object
    cyl.name = f"Bottle ({tag})"
    cyl.location = (x, y, 0.0)
    ctx.view_layer.update()
    bf = flow.world_brep(cyl).faces()
    side = max(range(len(bf)), key=lambda i: bf[i].area())
    # the seam to the back (+Y): the sheet's middle, where the lettering goes, then folds to the front
    Ss = surfeval.Surf(bf[side].surface())
    q = Ss.at([Ss.u_domain[0]], [0.5 * sum(Ss.v_domain)], 0)[0, 0][0] - np.array([x, y, 0.0])
    cyl.rotation_euler.z = math.pi / 2 - math.atan2(q[1], q[0])
    ctx.view_layer.update()
    select([cyl], cyl)
    bpy.ops.nproto.unroll(place="BESIDE")
    sheet, fi = sheet_with(cyl, side)
    sheet.name = f"Bottle sheet ({tag})"
    c, rot, mir = spot_and_turn(sheet, fi, cyl)
    tw = lettering(f"Lettering ({tag} wrap)", "WRAP", (c[0], c[1], c[2]), 0.5, rot=rot, mirror=mir)
    report[f"wrap {tag}: sheet seen from inside"] = mir
    if tag == "done":
        run_emboss("wrap cylinder", [tw, sheet], cyl, mode="WRAP", height=0.05, overlap=0.015)
    label(title, x + 2.0, y - 3.2)

# 6 -- Wrap onto a double-curved top: the dome unrolled ------------------------------------
x, y = 0.0, ROW_B
d6 = dome_body("Dome (wrap)", x, y)
bf = flow.world_brep(d6).faces()
top = max(range(len(bf)), key=lambda i: bf[i].centroid()[2])
select([d6], d6)
bpy.ops.nproto.unroll(place="BESIDE")
sheet6, fi = sheet_with(d6, top)
sheet6.name = "Dome sheet (wrap)"
c, rot6, mir6 = spot_and_turn(sheet6, fi, d6, toward=(0.0, 0.0, 1.0))
t6 = lettering("Lettering (dome wrap)", "NURBS", (c[0], c[1], c[2]), 0.5, rot=rot6, mirror=mir6)
report["dome wrap: sheet seen from inside"] = mir6
run_emboss("wrap dome", [t6, sheet6], d6, mode="WRAP", height=0.05, overlap=0.015)
label("WRAP, DOUBLE CURVED  (done)\nThe dome unrolled (relaxed: it is not\ndevelopable), lettered on its top\nface, folded back", x + 2.0, y - 3.2)

# 7 -- Place a badge on a sphere ------------------------------------------------------------
x, y = -10.5, ROW_A
bpy.ops.nproto.add_primitive(kind="SPHERE", radius=1.0)
sph = ctx.active_object
sph.name = "Ball"
sph.location = (x, y, 1.0)
ctx.view_layer.update()
st = star("Badge", (x + 0.35, y - 0.35, 3.0), 0.28, 0.12)
run_emboss("place sphere", [st], sph, mode="PLACE", height=0.05, overlap=0.05)
label("PLACE  (done)\nA star moved and turned onto the\nball as a whole: no bending (Overlap\ncovers the sag under it)", x, y - 3.2)

# view: a three-quarter perspective over both rows ---------------------------------------------
for scr in bpy.data.screens:
    for a in scr.areas:
        if a.type == 'VIEW_3D':
            for sp in a.spaces:
                if sp.type == 'VIEW_3D':
                    r3 = sp.region_3d
                    from mathutils import Euler
                    r3.view_rotation = Euler((math.radians(55), 0.0, math.radians(12))).to_quaternion()
                    r3.view_perspective = 'PERSP'
                    r3.view_location = (-1.0, 0.0, 0.5)
                    r3.view_distance = 34.0
                    sp.clip_end = 1000.0
bpy.ops.wm.save_as_mainfile(filepath=out)
print("REPORT", report)
