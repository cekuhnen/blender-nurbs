"""Remove Edges with simulated input in a real window: Edge mode, click the
diagonal of a triangulated box's top, Shift+click a box edge, X (the diagonal
goes, the needed box edge stays picked), Ctrl+Z, then Remove Redundant Edges:
    blender -b --factory-startup --python tests/test_remove_edges_gui.py
The background run writes an empty .blend, starts a GUI Blender with
--enable-event-simulate on it (a window opens for ~10 s), and reads back what
that run found. Screenshots go next to the results (the path is printed)."""
import json, os, subprocess, sys, tempfile

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI = "--re-gui" in sys.argv


def background():
    tmp = tempfile.mkdtemp(prefix="nproto_re_gui_")
    blend, out = os.path.join(tmp, "empty.blend"), os.path.join(tmp, "results.json")
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    cmd = [bpy.app.binary_path, "--factory-startup", "--enable-event-simulate", "--window-geometry", "0", "0",
           "1400", "900", blend, "--python", os.path.abspath(__file__), "--", "--re-gui", out]
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
    out = sys.argv[sys.argv.index("--re-gui") + 1]
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

    def tribox():
        # the regression test's triangulated box (12 flat triangles)
        import importlib.util
        spec = importlib.util.spec_from_file_location("_re", os.path.join(ROOT, "tests", "test_remove_edges.py"))
        src = open(spec.origin).read()
        start, end = src.index("def tribox():"), src.index("# --- kernel (mergefaces)")
        ns = {"sm": sm, "ext": sm.blender_ext, "np": __import__("numpy")}
        exec(src[start:end], ns)
        return ns["tribox"]()

    def steps():
        p = os.path.join(shots, "tribox.sm")
        tribox().write_to_file(p, ascii=True)
        o = store.new_object_text(ctx, "Tri", open(p).read())
        for x in ctx.selected_objects:
            x.select_set(False)
        o.select_set(True)
        ctx.view_layer.objects.active = o
        r3 = area.spaces.active.region_3d
        r3.view_perspective = 'PERSP'
        r3.view_rotation = Euler((0.95, 0.0, 0.5)).to_quaternion()      # from the front right, above
        r3.view_location = Vector((1.0, 0.75, 0.5))
        r3.view_distance = 6.0
        yield 0.5
        with over():
            bpy.ops.ed.undo_push(message="setup")
        ctx.window_manager.nproto_view.sel_mode = {"EDGE"}
        yield 0.6
        check("Edge mode: the picker runs", subsel.RUN["on"], subsel.RUN)
        move(wpx((1.0, 0.75, 1.0)))                  # the top's diagonal
        yield 0.3
        click()
        yield 0.3
        move(wpx((1.0, 0.0, 1.0)))                   # the top front box edge
        yield 0.3
        click(shift=True)
        yield 0.3
        picks = subsel.picks(o, "EDGE")
        check("click + Shift+click: two edges picked", len(picks) == 2, picks)
        area.tag_redraw()
        yield 0.5
        shot("1_picked")
        move(wpx((1.0, 0.4, 1.0)))
        yield 0.2
        key('X')
        yield 1.5
        check("X: the diagonal goes (11 faces), the box edge stays picked", nf(o) == 11
              and abs(vol(o) - 3.0) < 1e-7 and len(subsel.picks(o, "EDGE")) == 1,
              f"faces {nf(o)} picks {subsel.picks(o, 'EDGE')} objects {[x.name for x in bpy.data.objects]}")
        area.tag_redraw()
        yield 0.5
        shot("2_after_x")
        with over():
            bpy.ops.ed.undo()
        yield 0.6
        o = bpy.data.objects.get("Tri")
        check("Ctrl+Z: the diagonal is back", o is not None and nf(o) == 12, nf(o) if o else None)
        with over():
            bpy.ops.nproto.remove_edges('EXEC_DEFAULT', all_edges=True)
        yield 1.5
        o = bpy.data.objects.get("Tri")
        check("Remove Redundant Edges: the plain box", o is not None and nf(o) == 6 and abs(vol(o) - 3.0) < 1e-7,
              nf(o) if o else None)
        ctx.window_manager.nproto_view.sel_mode = {"BODY"}
        area.tag_redraw()
        yield 0.5
        shot("3_all")

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
