"""Face CVs with simulated input in a real window: Face mode, a click on the
box top, Tab, CVs clicked and moved with G Z <number> Enter, Tab = Done:
    blender -b --factory-startup --python tests/test_face_cvs_gui.py
The background run writes an empty .blend, starts a GUI Blender with
--enable-event-simulate on it (a window opens for ~20 s), and reads back what
that run found. Screenshots go next to the results (the path is printed)."""
import json, os, subprocess, sys, tempfile

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI = "--face-cvs-gui" in sys.argv


def background():
    tmp = tempfile.mkdtemp(prefix="nproto_face_cvs_gui_")
    blend, out = os.path.join(tmp, "empty.blend"), os.path.join(tmp, "results.json")
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    cmd = [bpy.app.binary_path, "--factory-startup", "--enable-event-simulate", "--window-geometry", "0", "0",
           "1400", "900", blend, "--python", os.path.abspath(__file__), "--", "--face-cvs-gui", out]
    try:
        subprocess.run(cmd, timeout=240, capture_output=True)
    except subprocess.TimeoutExpired:
        print("FAIL the GUI run did not finish in 240 s")
        os._exit(1)
    if not os.path.isfile(out):
        print("FAIL the GUI run wrote no results")
        os._exit(1)
    res = json.load(open(out))
    ok = True
    for name, (good, extra) in res["checks"].items():
        print(("PASS " if good else "FAIL ") + name, extra)
        ok &= bool(good)
    if res.get("error"):
        print("FAIL the GUI run raised\n" + res["error"])
        ok = False
    print("screenshots in", tmp)
    print("ALL OK" if ok else "SOME FAILED")
    sys.stdout.flush()
    os._exit(0 if ok else 1)


def gui():
    import traceback
    import numpy as np
    from mathutils import Euler, Vector
    from bpy_extras import view3d_utils
    sys.path.insert(0, ROOT)
    import blender_nurbs
    blender_nurbs.register()
    from blender_nurbs import cvs, facecvs, features, store, subsel

    out = sys.argv[sys.argv.index("--face-cvs-gui") + 1]
    shots = os.path.dirname(out)
    checks = {}
    res = {"checks": checks}

    def check(name, cond, extra=""):
        checks[name] = (bool(cond), str(extra))

    ctx = bpy.context
    win = ctx.window_manager.windows[0]
    area = next(a for a in win.screen.areas if a.type == 'VIEW_3D')
    region = next(r for r in area.regions if r.type == 'WINDOW')
    over = lambda: ctx.temp_override(window=win, area=area, region=region)

    def wpx(P):
        q = view3d_utils.location_3d_to_region_2d(region, area.spaces.active.region_3d, Vector(P))
        return int(round(region.x + q[0])), int(round(region.y + q[1]))

    at = [0, 0]

    def move(p):
        for k in (3, 2, 1, 0):
            win.event_simulate('MOUSEMOVE', 'NOTHING', x=p[0] + k, y=p[1] + k)
        at[:] = p

    def key(t, uni=None, **mods):
        kw = {"unicode": uni} if uni else {}
        win.event_simulate(t, 'PRESS', x=at[0], y=at[1], **kw, **mods)
        win.event_simulate(t, 'RELEASE', x=at[0], y=at[1], **mods)

    def click(**mods):
        win.event_simulate('LEFTMOUSE', 'PRESS', x=at[0], y=at[1], **mods)
        win.event_simulate('LEFTMOUSE', 'RELEASE', x=at[0], y=at[1], **mods)

    def typed(s):
        names = {".": 'PERIOD', "-": 'MINUS'}
        digits = ['ZERO', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE', 'SIX', 'SEVEN', 'EIGHT', 'NINE']
        for ch in s:
            key(names.get(ch) or digits[int(ch)], ch)

    def shot(name):
        area.tag_redraw()
        with over():
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        with ctx.temp_override(window=win, screen=win.screen):
            bpy.ops.screen.screenshot(filepath=os.path.join(shots, name + ".png"))

    brep = lambda o: store.get_brep(o)

    def cage_world(cage):
        M = cage.matrix_world
        return [M @ Vector(q) for q in cvs.cage_points(cage)]       # (Edit Mode: the bmesh, not cage.data)

    def selected(cage):
        import bmesh
        bm = bmesh.from_edit_mesh(cage.data)
        return [v.index for v in bm.verts if v.select]

    def steps():
        with over():
            bpy.ops.nproto.add_primitive(kind='BOX', width=4.0, depth=3.0, height=2.0)
        box = ctx.view_layer.objects.active
        box.name = "Box"
        lo, hi = (np.array(v, float) for v in brep(box).bounding_box())
        c = 0.5 * (lo + hi)
        r3 = area.spaces.active.region_3d
        r3.view_perspective = 'PERSP'
        r3.view_rotation = Euler((1.05, 0.0, 0.6)).to_quaternion()
        r3.view_location = Vector(c)
        r3.view_distance = 11.0
        yield 0.5
        with over():
            bpy.ops.ed.undo_push(message="setup")
        # Face mode, click the top
        ctx.window_manager.nproto_view.sel_mode = {"FACE"}
        yield 0.5
        top = (c[0], c[1], hi[2])
        move(wpx(top))
        yield 0.3
        click()
        yield 0.4
        fs = brep(box).faces()
        itop = min(range(len(fs)), key=lambda k: fs[k].closest_point(tuple(map(float, top)))[1])
        check("Face mode: a click picks the top", subsel.picks(box, "FACE") == [itop], subsel.picks(box, "FACE"))
        shot("1_top_picked")
        # Tab: the face cage
        key('TAB')
        yield 0.8
        cage = ctx.view_layer.objects.active
        check("Tab on the picked top opens a face cage in Edit Mode",
              facecvs.is_face_cage(cage) and cage.mode == 'EDIT', f"{cage.name if cage else None} "
              f"{cage.mode if cage else ''}")
        if not facecvs.is_face_cage(cage):
            return
        try:
            for a in win.screen.areas:
                if a.type == 'VIEW_3D':
                    a.spaces.active.show_region_ui = True
            ui = next(r for r in area.regions if r.type == 'UI')
            ui.active_panel_category = "NURBS Proto"
        except Exception as exc:
            res["sidebar"] = str(exc)
        yield 0.5
        shot("2_face_cage")
        # the U field in the panel before a move: a fresh even grid (through a timer)
        nv0 = json.loads(cage[cvs.CAGE_META])["nv"]
        ctx.window_manager.nproto_face_cvs.nu = 7
        yield 0.8
        m7 = json.loads(cage[cvs.CAGE_META])
        check("U = 7 in the panel before a move: the cage gets 7 CVs along U at once",
              m7["nu"] == 7 and len(cvs.cage_points(cage)) == 7 * nv0 and cage.mode == 'EDIT',
              f"{m7['nu']} x {m7['nv']}, {len(cvs.cage_points(cage))} CVs, mode {cage.mode}")
        meta = json.loads(cage[cvs.CAGE_META])
        nu, nv = meta["nu"], meta["nv"]
        W = cage_world(cage)
        # an inside CV: click it, G Z 1 Enter
        k_in = (nu // 2) * nv + nv // 2
        move(wpx(W[k_in]))
        yield 0.2
        click()
        yield 0.3
        check("a click selects the inside CV", selected(cage) == [k_in], selected(cage))
        key('G'); key('Z'); typed("1"); key('RET')
        yield 1.2
        zmax = max(np.array(e.point_at(t), float)[2] for e in brep(box).edges()
                   for t in np.linspace(*e.parameter_range(), 5))
        P = np.array(cvs.cage_points(cage))
        check("G Z 1 on the inside CV: the top bulges, the box stays closed with 6 faces",
              brep(box).is_manifold_solid() and brep(box).face_count() == 6 and P[k_in][2] > hi[2] + 0.9
              and not cage.get(facecvs.CAGE_ERROR),
              f"CV z {P[k_in][2]:.3f}, err {cage.get(facecvs.CAGE_ERROR)}")
        shot("3_bulge")
        # two CVs on an edge row (not the corners): click, shift-click, G Z -0.8 Enter
        W = cage_world(cage)
        ka, kb = 2 * nv, 3 * nv                       # row v = 0, u = 2 and 3
        move(wpx(W[ka])); yield 0.1
        click(); yield 0.2
        move(wpx(W[kb])); yield 0.1
        click(shift=True); yield 0.2
        sel = selected(cage)
        check("click + shift-click select two edge CVs", sorted(sel) == sorted([ka, kb]), sel)
        key('G'); key('Z'); typed("-0.8"); key('RET')
        yield 1.2
        y_wall = float(np.array(cage_world(cage)[ka])[1])
        pts = [np.array(e.point_at(t), float) for e in brep(box).edges() for t in np.linspace(*e.parameter_range(), 9)]
        on_wall = [q for q in pts if abs(q[1] - y_wall) < 1e-6 and lo[0] + 0.5 < q[0] < hi[0] - 0.5 and q[2] > c[2]]
        dip = min((q[2] for q in on_wall), default=hi[2])
        check("G Z -0.8 on two edge CVs: the wall under them is re-trimmed lower",
              brep(box).is_manifold_solid() and dip < hi[2] - 0.3 and not cage.get(facecvs.CAGE_ERROR),
              f"wall top lowest {dip:.3f} (top at {hi[2]:.3f}), err {cage.get(facecvs.CAGE_ERROR)}")
        shot("4_edge_down")
        # Tab: Done
        key('TAB')
        yield 1.0
        st = features.stack(box) if box.name in bpy.data.objects else []
        check("Tab again: Done, the cage gone", cvs.cage_for(box) is None and ctx.view_layer.objects.active == box,
              ctx.view_layer.objects.active.name if ctx.view_layer.objects.active else None)
        check("history on: a Face CVs step", [e["op"] for e in st][-1:] == ["face_cvs"] or
              not features.history_enabled(ctx), [e["op"] for e in st])
        ctx.window_manager.nproto_view.sel_mode = {"BODY"}
        yield 0.5
        shot("5_done")

    g = steps()

    def tick():
        try:
            return next(g)
        except StopIteration:
            pass
        except Exception:
            res["error"] = traceback.format_exc()
        json.dump(res, open(out, "w"), indent=1, default=str)
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(tick, first_interval=1.0)


if GUI:
    gui()
else:
    background()
