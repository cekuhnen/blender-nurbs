"""Remove Edges' Fit Tolerance with simulated input in a real window: Edge
mode, click the edge where a slot's flat side runs into its round end, X
(kept: the faces are tangent but on different surfaces, the report names the
Fit Tolerance, the operator stays open for F9), then the redo with Fit
Tolerance 0.001 (the side and the end are one refitted face, still a solid):
    blender -b --factory-startup --python tests/test_fit_tolerance_gui.py
The background run writes an empty .blend, starts a GUI Blender with
--enable-event-simulate on it (a window opens for ~10 s), and reads back what
that run found. Screenshots go next to the results (the path is printed)."""
import json, os, subprocess, sys, tempfile

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI = "--fit-gui" in sys.argv


def background():
    tmp = tempfile.mkdtemp(prefix="nproto_fit_gui_")
    blend, out = os.path.join(tmp, "empty.blend"), os.path.join(tmp, "results.json")
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    cmd = [bpy.app.binary_path, "--factory-startup", "--enable-event-simulate", "--window-geometry", "0", "0",
           "1400", "900", blend, "--python", os.path.abspath(__file__), "--", "--fit-gui", out]
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
    from mathutils import Euler, Vector
    from bpy_extras import view3d_utils
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import blender_nurbs
    blender_nurbs.register()
    from blender_nurbs import store, subsel
    from blender_nurbs.kernel import load

    sm = load()
    out = sys.argv[sys.argv.index("--fit-gui") + 1]
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

    def key(t, **mod):
        win.event_simulate(t, 'PRESS', x=at[0], y=at[1], **mod)
        win.event_simulate(t, 'RELEASE', x=at[0], y=at[1], **mod)

    def click(**mod):
        win.event_simulate('LEFTMOUSE', 'PRESS', x=at[0], y=at[1], **mod)
        win.event_simulate('LEFTMOUSE', 'RELEASE', x=at[0], y=at[1], **mod)

    def shot(name):
        area.tag_redraw()
        with over():
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        with ctx.temp_override(window=win, screen=win.screen):
            bpy.ops.screen.screenshot(filepath=os.path.join(shots, name + ".png"))

    nf = lambda o: store.get_brep(o).face_count()
    vol = lambda o: abs(store.get_brep(o).volume(1e-6))

    def steps():
        b = sm.boolean_union(sm.create_box((0, -0.5, 0), 1, 1, 0.4), sm.create_cylinder((1, 0, 0), 0.5, 0.4))
        n0, v0 = b.face_count(), abs(b.volume(1e-6))
        p = os.path.join(shots, "slot.sm")
        b.write_to_file(p, ascii=True)
        o = store.new_object_text(ctx, "Slot", open(p).read())
        for x in ctx.selected_objects:
            x.select_set(False)
        o.select_set(True)
        ctx.view_layer.objects.active = o
        r3 = area.spaces.active.region_3d
        r3.view_perspective = 'PERSP'
        r3.view_rotation = Euler((1.0, 0.0, 2.6)).to_quaternion()
        r3.view_location = Vector((0.8, 0.0, 0.2))
        r3.view_distance = 3.5
        yield 0.5
        with over():
            bpy.ops.ed.undo_push(message="setup")
        ctx.window_manager.nproto_view.sel_mode = {"EDGE"}
        yield 0.6
        move(wpx((1.0, 0.5, 0.2)))                 # the flat side's edge onto the round end
        yield 0.3
        click()
        yield 0.3
        check("click: the tangent edge picked", len(subsel.picks(o, "EDGE")) == 1, subsel.picks(o, "EDGE"))
        area.tag_redraw()
        yield 0.5
        shot("1_edge_picked")
        key('X')
        yield 1.5
        last = ctx.window_manager.operators[-1] if ctx.window_manager.operators else None
        check("X: kept (tangent, different surfaces), the body unchanged, Remove Edges open for F9", nf(o) == n0
              and last is not None and last.bl_idname == "NPROTO_OT_remove_edges_key",
              f"faces {nf(o)} last {last and last.bl_idname}")
        last.properties.fit_tolerance = 0.001          # what F9's panel does
        with over():
            bpy.ops.ed.undo_redo()
        yield 2.0
        o = bpy.data.objects.get("Slot")
        b2 = store.get_brep(o) if o else None
        check("redo with Fit Tolerance 0.001: the side and the end one face, a solid of the same volume",
              b2 is not None and b2.face_count() == n0 - 1 and b2.is_manifold_solid()
              and abs(abs(b2.volume(1e-6)) - v0) < 1e-4,
              f"faces {b2 and b2.face_count()} vol {b2 and abs(b2.volume(1e-6))} {v0}")
        ctx.window_manager.nproto_view.sel_mode = {"BODY"}
        area.tag_redraw()
        yield 0.5
        shot("2_refitted")

    gen = steps()

    def tick():
        try:
            return next(gen)
        except StopIteration:
            pass
        except Exception:
            res["error"] = traceback.format_exc()
        with open(out, "w") as f:
            json.dump(res, f)
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(tick, first_interval=1.0)


if GUI:
    gui()
else:
    background()
