"""Remove Faces with simulated input in a real window: Face mode, click the
wall of a plate's hole, Shift+X (the hole heals, the plate is whole), Ctrl+Z,
then plain X on the wall (Delete Faces: the body opens):
    blender -b --factory-startup --python tests/test_remove_faces_gui.py
The background run writes an empty .blend, starts a GUI Blender with
--enable-event-simulate on it (a window opens for ~10 s), and reads back what
that run found. Screenshots go next to the results (the path is printed)."""
import json, os, subprocess, sys, tempfile

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI = "--rf2-gui" in sys.argv


def background():
    tmp = tempfile.mkdtemp(prefix="nproto_rface_gui_")
    blend, out = os.path.join(tmp, "empty.blend"), os.path.join(tmp, "results.json")
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    cmd = [bpy.app.binary_path, "--factory-startup", "--enable-event-simulate", "--window-geometry", "0", "0",
           "1400", "900", blend, "--python", os.path.abspath(__file__), "--", "--rf2-gui", out]
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
    out = sys.argv[sys.argv.index("--rf2-gui") + 1]
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
        b = sm.boolean_difference(sm.create_box((0, 0, 0), 2, 1.5, 1), sm.create_cylinder((1, 0.75, -0.5), 0.3, 2))
        p = os.path.join(shots, "plate.sm")
        b.write_to_file(p, ascii=True)
        o = store.new_object_text(ctx, "Plate", open(p).read())
        for x in ctx.selected_objects:
            x.select_set(False)
        o.select_set(True)
        ctx.view_layer.objects.active = o
        r3 = area.spaces.active.region_3d
        r3.view_perspective = 'PERSP'
        r3.view_rotation = Euler((0.75, 0.0, 0.5)).to_quaternion()      # from the front right, above
        r3.view_location = Vector((1.0, 0.75, 0.5))
        r3.view_distance = 5.0
        yield 0.5
        with over():
            bpy.ops.ed.undo_push(message="setup")
        ctx.window_manager.nproto_view.sel_mode = {"FACE"}
        yield 0.6
        check("Face mode: the picker runs", subsel.RUN["on"], subsel.RUN)
        wall = (1.0, 1.05, 0.5)                   # the far side of the hole's wall, seen through its top
        move(wpx(wall))
        yield 0.3
        click()
        yield 0.3
        picks = subsel.picks(o, "FACE")
        F = store.get_brep(o).faces()
        check("click: the hole's wall picked", len(picks) == 1 and F[picks[0]].closest_point(wall)[1] < 1e-3,
              picks)
        area.tag_redraw()
        yield 0.5
        shot("1_picked")
        key('X', shift=True)
        yield 1.5
        check("Shift+X: the hole heals (6 faces, the full plate volume)", nf(o) == 6 and abs(vol(o) - 3.0) < 1e-7
              and not subsel.picks(o, "FACE"), f"faces {nf(o)} vol {vol(o):.6f}")
        area.tag_redraw()
        yield 0.5
        shot("2_healed")
        with over():
            bpy.ops.ed.undo()
        yield 0.6
        o = bpy.data.objects.get("Plate")
        check("Ctrl+Z: the hole is back", o is not None and nf(o) == 7, nf(o) if o else None)
        move(wpx(wall))
        yield 0.3
        click()
        yield 0.3
        key('X')
        yield 1.0
        o = bpy.data.objects.get("Plate")
        check("plain X: Delete Faces (the wall goes, the body opens)", o is not None and nf(o) == 6
              and not store.get_brep(o).is_manifold_solid(), nf(o) if o else None)
        ctx.window_manager.nproto_view.sel_mode = {"BODY"}

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
