"""Run crash-prone kernel jobs (trim, fillet) in kernel/worker.py, in a
separate Python process, and bring the results back into Blender."""

import json
import os
import shutil
import subprocess
import sys
import tempfile

from . import geom, store

WORKER = os.path.join(os.path.dirname(__file__), "kernel", "worker.py")


class JobError(RuntimeError):
    pass


def xform(obj):
    loc, rot, scale = geom.world_matrix(obj).decompose()
    axis, angle = rot.to_axis_angle()
    return {"loc": list(loc), "axis": list(axis), "angle": float(angle) * 57.29577951308232,
            "scale": list(scale)}


def curve_json(kcurve):
    from .kernel import ext
    d = ext.curve_cvs(kcurve)
    return {"degree": d["degree"], "points": [list(p) for p in d["points"]],
            "knots": list(d["knots"]), "weights": list(d["weights"])}


class _Server:
    """One worker kept running (worker.py --serve): a job then costs no Python start-up or kernel load. A
    crash or a time-out ends it like a one-off worker's would; the next job starts a new one."""
    MARK = b"@@nproto-result@@ "
    MAX_JOBS = 500                        # then a fresh one (whatever the kernel keeps between jobs)

    def __init__(self):
        import queue
        import threading
        self.proc = subprocess.Popen([sys.executable, "-I", WORKER, "--serve"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.lines = queue.Queue()
        self.jobs = 0

        def pump(out, q):
            for line in iter(out.readline, b""):
                q.put(line)
            q.put(None)                   # the worker is gone
        threading.Thread(target=pump, args=(self.proc.stdout, self.lines), daemon=True).start()

    def ask(self, jpath, timeout):
        """The worker's result line for one job (None: it died during the job)."""
        import queue
        import time
        self.jobs += 1
        self.proc.stdin.write((jpath + "\n").encode())
        self.proc.stdin.flush()
        end = time.monotonic() + timeout
        while True:
            left = end - time.monotonic()
            if left <= 0:
                raise subprocess.TimeoutExpired("worker", timeout)
            try:
                line = self.lines.get(timeout=left)
            except queue.Empty:
                raise subprocess.TimeoutExpired("worker", timeout)
            if line is None:
                return None
            if line.startswith(self.MARK):
                return line[len(self.MARK):].decode()

    def stop(self):
        try:
            self.proc.kill()
            self.proc.wait(timeout=2)
        except Exception:
            pass


_server = [None]


def shutdown():
    """End the kept worker (the add-on is switched off; Blender quits)."""
    if _server[0] is not None:
        _server[0].stop()
        _server[0] = None


import atexit  # noqa: E402
atexit.register(shutdown)


def _ask_server(op, jpath, timeout):
    s = _server[0]
    if s is None or s.proc.poll() is not None or s.jobs >= _Server.MAX_JOBS:
        shutdown()
        s = _server[0] = _Server()
    try:
        line = s.ask(jpath, timeout)
    except subprocess.TimeoutExpired:
        shutdown()
        raise JobError(f"kernel {op} timed out after {timeout:.0f}s")
    except OSError:                       # (the pipe broke: it had died before this job)
        shutdown()
        line = None
    if line is None:
        shutdown()
        raise JobError(f"kernel crashed during {op}; Blender is fine -- try other inputs or settings")
    return json.loads(line)


def run(op, objects, out_keys, timeout=120.0, persistent=False, **params):
    """objects: {key: blender object (NURBS)}. Returns (result dict, {key: Brep}). persistent: in the kept
    worker (no start-up per job; SubD to BREP's live rebuilds) instead of a new process."""
    tmp = tempfile.mkdtemp(prefix="nproto_job_")
    try:
        job = {"op": op, "breps": {}, "xforms": {}, "out": {}}
        for key, obj in objects.items():
            path = os.path.join(tmp, f"in_{key}.sm")
            if isinstance(obj, tuple):          # (brep text, xform dict): an intermediate result
                text, xf = obj
            else:
                text, xf = obj[store.PROP_BREP], xform(obj)
            with open(path, "w", newline="\n") as f:      # the same bytes on Windows (no \r added)
                f.write(text)
            job["breps"][key] = path
            job["xforms"][key] = xf
        for key in out_keys:
            job["out"][key] = os.path.join(tmp, f"out_{key}.sm")
        job["out_dir"] = tmp
        job.update(params)
        jpath = os.path.join(tmp, "job.json")
        with open(jpath, "w") as f:
            json.dump(job, f)
        if persistent:
            res = _ask_server(op, jpath, timeout)
        else:
            try:
                proc = subprocess.run([sys.executable, "-I", WORKER, jpath], capture_output=True,
                                      text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                raise JobError(f"kernel {op} timed out after {timeout:.0f}s")
            lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
            if proc.returncode != 0 or not lines:
                raise JobError(f"kernel crashed during {op} (exit {proc.returncode}); "
                               "Blender is fine -- try other inputs or settings")
            res = json.loads(lines[-1])
        if not res.get("ok"):
            raise JobError(res.get("error", "kernel error"))
        breps = {}
        for name in sorted(os.listdir(tmp)):  # declared outputs + any out_*.sm a job wrote
            if name.startswith("out_") and name.endswith(".sm"):
                with open(os.path.join(tmp, name)) as f:
                    breps[name[4:-3]] = f.read()
        return res["result"], breps
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
