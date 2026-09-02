"""Persistent headless Isaac Sim REPL daemon.

Keeps ONE Isaac Sim app loaded and pumps its update loop, while listening on a TCP socket.
Clients send a blob of Python; the daemon `exec`s it against a persistent namespace that
already holds the live stage + sim + common modules, captures stdout/stderr, and returns the
result as JSON. This replaces the slow "write script -> 2-min boot -> render -> read" loop
with an interactive REPL that stays warm across calls.

  Start:   <xpurt python> sims/scripts/isaac_repl_daemon.py --headless [--port 8765] [--usd <path>]
  Drive:   <any python>   sims/scripts/irepl.py --code 'print(stage.GetPrimAtPath("/World"))'
           or             sims/scripts/irepl.py --file snippet.py

Protocol (both directions): 4-byte big-endian length prefix + UTF-8 JSON.
  request  = {"code": "<python>", "timeout": <opt s>}  |  {"cmd": "ping"|"shutdown"}
  response = {"ok": bool, "stdout": str, "stderr": str, "error": str|null, "result": <repr or null>}

The persistent namespace G preloads: simulation_app, sim, stage, sim_utils, torch (as torch),
numpy (as np), and helper render(eye, tgt, path, w=1280, h=720). Assign to G by doing
`globals()['x'] = ...` inside sent code, or just define names normally (exec writes into G).
"""
import argparse, io, json, os, socket, struct, sys, traceback, contextlib

freshscheduler_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, freshscheduler_root)
for p in ("isaaclab", "isaaclab_assets", "isaaclab_rl", "isaaclab_contrib"):
    sys.path.insert(0, f"/scratch2/dima/IsaacLab/source/{p}")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--usd", type=str, default=None, help="optional USD to load into /World at boot")
parser.add_argument("--dt", type=float, default=1 / 60)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app = AppLauncher(args_cli)
simulation_app = app.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg

# ---- boot a persistent sim context ----
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=args_cli.dt, device=args_cli.device))
# create the inspection camera BEFORE the boot reset so the reset initializes the sensor
# (a Camera created after reset lacks _ALL_INDICES and can't be posed).
_cam = {"obj": Camera(CameraCfg(
    prim_path="/World/ReplCam", update_period=0.0, height=720, width=1280, data_types=["rgb"],
    spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.05, 800))))}


def _ensure_cam():
    return _cam["obj"]


def render(eye, tgt, path, w=1280, h=720, settle=20, dome=None):
    """Reposition the repl camera, converge the RTX render, and save a PNG. Returns path."""
    cam = _ensure_cam()
    if dome is not None:
        from pxr import UsdLux
        pr = sim.stage.GetPrimAtPath("/World/ReplDome")
        if pr.IsValid():
            UsdLux.LightAPI(pr).GetIntensityAttr().Set(float(dome))
        else:
            d = sim_utils.DomeLightCfg(intensity=float(dome), color=(0.9, 0.9, 0.95))
            d.func("/World/ReplDome", d)
    cam.set_world_poses_from_view(torch.tensor([eye], device=sim.device),
                                  torch.tensor([tgt], device=sim.device))
    for _ in range(settle):
        sim.step(); cam.update(dt=sim.get_physics_dt())
    rgb = cam.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    import imageio.v2 as imageio
    imageio.imwrite(path, rgb)
    return path


def load_usd(usd_path, prim="/World/Loaded"):
    cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
    cfg.func(prim, cfg)
    return prim


# persistent exec namespace
G = {"__name__": "__repl__", "simulation_app": simulation_app, "sim": sim, "sim_utils": sim_utils,
     "torch": torch, "np": np, "Camera": Camera, "CameraCfg": CameraCfg,
     "render": render, "load_usd": load_usd}

if args_cli.usd:
    load_usd(args_cli.usd, "/World/Boot")
sim.reset()
for _ in range(60):
    sim.step()
G["stage"] = sim.stage  # after reset the stage is populated

# ---- socket server ----
srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", args_cli.port))
srv.listen(4)
srv.setblocking(False)
print(f"[repl] listening on 127.0.0.1:{args_cli.port} (device={args_cli.device})", flush=True)


def _recvall(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _read_msg(conn):
    hdr = _recvall(conn, 4)
    if hdr is None:
        return None
    (ln,) = struct.unpack(">I", hdr)
    body = _recvall(conn, ln)
    return json.loads(body.decode("utf-8")) if body is not None else None


def _send_msg(conn, obj):
    payload = json.dumps(obj).encode("utf-8")
    conn.sendall(struct.pack(">I", len(payload)) + payload)


def handle(req):
    cmd = req.get("cmd")
    if cmd == "ping":
        return {"ok": True, "stdout": "pong", "stderr": "", "error": None, "result": None}
    if cmd == "shutdown":
        return {"ok": True, "stdout": "bye", "stderr": "", "error": None, "result": "__shutdown__"}
    code = req.get("code", "")
    out, err = io.StringIO(), io.StringIO()
    result, error = None, None
    G["stage"] = sim.stage
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            # compile as exec; if it's a single expression, also capture its value
            try:
                compiled = compile(code, "<repl>", "eval")
                result = repr(eval(compiled, G))
            except SyntaxError:
                exec(compile(code, "<repl>", "exec"), G)
    except Exception:
        error = traceback.format_exc()
    return {"ok": error is None, "stdout": out.getvalue(), "stderr": err.getvalue(),
            "error": error, "result": result}


running = True
_pump = 0
while running and simulation_app.is_running():
    # keep the app + renderer alive
    simulation_app.update()
    _pump += 1
    # poll for a client without blocking the update loop
    try:
        import select
        r, _, _ = select.select([srv], [], [], 0.02)
        if r:
            conn, _addr = srv.accept()
            conn.setblocking(True)
            try:
                req = _read_msg(conn)
                if req is not None:
                    resp = handle(req)
                    _send_msg(conn, resp)
                    if resp.get("result") == "__shutdown__":
                        running = False
            finally:
                conn.close()
    except BlockingIOError:
        pass
    except Exception as e:
        print(f"[repl] server error: {e}", flush=True)

print("[repl] shutting down", flush=True)
srv.close()
simulation_app.close()
