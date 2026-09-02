"""Client for the persistent Isaac REPL daemon (isaac_repl_daemon.py).

  python sims/scripts/irepl.py --code 'print(len(list(stage.Traverse())))'
  python sims/scripts/irepl.py --file snippet.py
  echo 'print("hi")' | python sims/scripts/irepl.py -
  python sims/scripts/irepl.py --ping
  python sims/scripts/irepl.py --shutdown

Prints the daemon's stdout/stderr/result; exits non-zero if the sent code raised.
This client is dependency-free (stdlib only) so it runs under ANY python.
"""
import argparse, json, socket, struct, sys


def send(port, obj):
    s = socket.create_connection(("127.0.0.1", port), timeout=600)
    payload = json.dumps(obj).encode("utf-8")
    s.sendall(struct.pack(">I", len(payload)) + payload)

    def recvall(n):
        buf = b""
        while len(buf) < n:
            c = s.recv(n - len(buf))
            if not c:
                return None
            buf += c
        return buf

    hdr = recvall(4)
    (ln,) = struct.unpack(">I", hdr)
    body = recvall(ln)
    s.close()
    return json.loads(body.decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--code", type=str, default=None)
    ap.add_argument("--file", type=str, default=None)
    ap.add_argument("--ping", action="store_true")
    ap.add_argument("--shutdown", action="store_true")
    ap.add_argument("pos", nargs="?", default=None, help="'-' to read code from stdin")
    a = ap.parse_args()

    if a.ping:
        req = {"cmd": "ping"}
    elif a.shutdown:
        req = {"cmd": "shutdown"}
    else:
        if a.file:
            code = open(a.file).read()
        elif a.code is not None:
            code = a.code
        elif a.pos == "-":
            code = sys.stdin.read()
        else:
            ap.error("need --code / --file / '-' / --ping / --shutdown")
        req = {"code": code}

    resp = send(a.port, req)
    if resp.get("stdout"):
        sys.stdout.write(resp["stdout"])
        if not resp["stdout"].endswith("\n"):
            sys.stdout.write("\n")
    if resp.get("result") not in (None, "__shutdown__"):
        print(f"=> {resp['result']}")
    if resp.get("error"):
        sys.stderr.write(resp["error"])
        sys.exit(1)
    if resp.get("stderr"):
        sys.stderr.write(resp["stderr"])


if __name__ == "__main__":
    main()
