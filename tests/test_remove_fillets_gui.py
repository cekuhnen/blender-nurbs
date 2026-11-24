"""Remove Fillets with simulated input in a real window (every fillet lit at
the start, A for none / all, a click keeps a fillet, hover says what a face
is, Enter, undo):
    blender -b --factory-startup --python tests/test_remove_fillets_gui.py
The background run writes an empty .blend, starts a GUI Blender with
--enable-event-simulate on it (a window opens for ~15 s), and reads back what
that run found. Screenshots go next to the results (the path is printed)."""
import json, math, os, subprocess, sys, tempfile

import bpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI = "--rf-gui" in sys.argv


def background():
    tmp = tempfile.mkdtemp(prefix="nproto_rf_gui_")
    blend, out = os.path.join(tmp, "empty.blend"), os.path.join(tmp, "results.json")
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    bpy.ops.wm.save_as_mainfile(filepath=blend)
    cmd = [bpy.app.binary_path, "--factory-startup", "--enable-event-simulate", "--window-geometry", "0", "0",
           "1400", "900", blend, "--python", os.path.abspath(__file__), "--", "--rf-gui", out]
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
    from blender_nurbs import draw, removefillets, store
    from blender_nurbs.kernel import load

    sm = load()
    out = sys.argv[sys.argv.index("--rf-gui") + 1]
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

    def key(t):
        win.event_simulate(t, 'PRESS', x=at[0], y=at[1])
        win.event_simulate(t, 'RELEASE', x=at[0], y=at[1])

    def click():
        win.event_simulate('LEFTMOUSE', 'PRESS', x=at[0], y=at[1])
        win.event_simulate('LEFTMOUSE', 'RELEASE', x=at[0], y=at[1])

    def shot(name):
        area.tag_redraw()
        with over():
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        with ctx.temp_override(window=win, screen=win.screen):
            bpy.ops.screen.screenshot(filepath=os.path.join(shots, name + ".png"))

    modal = lambda: [op.bl_idname for op in win.modal_operators]
    running = lambda: "NPROTO_OT_remove_fillets" in modal()
    regs = lambda: draw.pick_overlay.get("regions") or []
    red = lambda: sum(1 for r in regs() if not r[2])
    nf = lambda o: store.get_brep(o).face_count()
    vol = lambda o: abs(store.get_brep(o).volume(1e-6))

    def steps():
        b = sm.create_box((0, 0, 0), 2, 1.5, 1)
        sm.circular_fillet(b, 0.2)
        p = os.path.join(shots, "rounded.sm")
        b.write_to_file(p, ascii=True)
        o = store.new_object_text(ctx, "Rounded", open(p).read())
        for x in ctx.selected_objects:
            x.select_set(False)
        o.select_set(True)
        ctx.view_layer.objects.active = o
        r3 = area.spaces.active.region_3d
        r3.view_perspective = 'PERSP'
        r3.view_rotation = Euler((1.1, 0.0, 0.62)).to_quaternion()      # from front right, above
        r3.view_location = Vector((1.0, 0.75, 0.5))
        r3.view_distance = 6.0
        yield 0.5
        with over():
            bpy.ops.ed.undo_push(message="setup")
        found = removefillets.find(o)
        # the fillet along the top front edge (y = 0, z = 1): a point in its middle
        f_top = min(found["fillets"], key=lambda f: np.linalg.norm(np.array(f["probe"]) - (1.0, 0.1, 0.9)))
        F = store.get_brep(o).faces()[f_top["face"]]
        mid = np.array(F.closest_point(tuple(f_top["probe"]))[0], float)
        top = (1.0, 0.75, 1.0)

        with over():
            bpy.ops.nproto.remove_fillets('INVOKE_DEFAULT')
        yield 0.4
        check("the tool starts with every fillet (and corner) lit red", running() and len(regs()) == 20
              and red() == 20, f"running {running()} regions {len(regs())} red {red()}")
        area.tag_redraw()
        yield 0.4
        area.tag_redraw()
        yield 0.8
        shot("1_all_lit")
        move(wpx(mid))
        yield 0.4
        check("hover lights the face under the mouse", draw.pick_overlay.get("preview") is not None)
        move(wpx(top))
        yield 0.2
        click()                           # the flat top: not a fillet, nothing changes
        yield 0.2
        check("a click on the flat top changes nothing", red() == 20 and running(), red())
        key('A')
        yield 0.2
        check("A: none to remove", red() == 0, red())
        key('A')
        yield 0.2
        check("A again: all", red() == 20, red())
        move(wpx(mid))
        yield 0.3
        click()
        yield 0.3
        check("a click on a fillet keeps it (dims)", red() == 19 and len(regs()) == 20, f"red {red()}")
        area.tag_redraw()
        yield 0.4
        area.tag_redraw()
        yield 0.8
        shot("2_one_kept")
        key('RET')
        yield 1.0
        one = sm.create_box((0, 0, 0), 2, 1.5, 1)
        e = min(one.edges(), key=lambda x: x.closest_point((1.0, 0.0, 1.0))[2])
        sm.fillet_edges(one, [e], radius=0.2)
        check("Enter: 11 removed, the kept one put back", not running() and nf(o) == one.face_count()
              and abs(vol(o) - abs(one.volume(1e-6))) < 1e-7 and not regs(),
              f"faces {nf(o)} want {one.face_count()} vol {vol(o):.7f} want {abs(one.volume(1e-6)):.7f}")
        area.tag_redraw()
        yield 0.4
        shot("3_result")
        with over():
            bpy.ops.ed.undo()
        yield 0.5
        o = bpy.data.objects.get("Rounded")
        check("Ctrl+Z brings the fillets back", o is not None and nf(o) == 26, nf(o) if o else None)
        # Escape leaves the body as it is
        with over():
            bpy.ops.nproto.remove_fillets('INVOKE_DEFAULT')
        yield 0.3
        key('ESC')
        yield 0.3
        check("Esc: cancelled, nothing lit, body unchanged", not running() and not regs() and nf(o) == 26,
              f"running {running()} regions {len(regs())} faces {nf(o)}")

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
