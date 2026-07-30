from __future__ import annotations
import json
import os
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.moves import Move, Plan, apply_plan, build_plan, duplicate_organized, load_student
from core.taxonomy import Taxonomy

PORT = 18765
CKPT = ROOT / "artifacts" / "student_model_ckpt"


def _pids_on_port(port: int) -> list[int]:
    try:
        out = subprocess.check_output(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    pids = []
    for line in out.splitlines():
        line = line.strip()
        if line.isdigit():
            pid = int(line)
            if pid != os.getpid():
                pids.append(pid)
    return pids


def _free_port(port: int) -> None:
    pids = _pids_on_port(port)
    if not pids:
        return
    print(f"port {port} busy (pid {', '.join(map(str, pids))}) — stopping leftover server…")
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 3.0
    while time.time() < deadline and _pids_on_port(port):
        time.sleep(0.1)
    for pid in _pids_on_port(port):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(0.1)


def _plan_to_dict(plan: Plan) -> dict:
    return {
        "root": plan.root,
        "dirs": list(plan.dirs),
        "moves": [
            {
                "src": m.src,
                "dst": m.dst,
                "category": m.category,
                "confidence": m.confidence,
                "tier": m.tier,
            }
            for m in plan.moves
        ],
    }


def _plan_from_dict(data: dict) -> Plan:
    plan = Plan(root=str(data.get("root") or ""))
    plan.dirs = list(data.get("dirs") or [])
    for m in data.get("moves") or []:
        plan.moves.append(
            Move(
                src=str(m["src"]),
                dst=str(m["dst"]),
                category=str(m.get("category") or ""),
                confidence=float(m.get("confidence") or 0),
                tier=str(m.get("tier") or "student"),
            )
        )
    return plan


def _plan_lines(plan: Plan, limit: int = 40) -> list[str]:
    out = [f"plan for {plan.root}", f"  {len(plan.moves)} moves, {len(plan.dirs)} folders"]
    for m in plan.moves[:limit]:
        name = Path(m.src).name
        try:
            rel = str(Path(m.dst).relative_to(plan.root))
        except ValueError:
            rel = m.dst
        out.append(f"  [{m.tier:9s}] {m.confidence*100:5.1f}%  {name}  ->  {rel}")
    if len(plan.moves) > limit:
        out.append(f"  ... +{len(plan.moves) - limit} more")
    return out


def _read_json(handler: BaseHTTPRequestHandler) -> dict:
    n = int(handler.headers.get("Content-Length", 0) or 0)
    raw = handler.rfile.read(n) if n else b"{}"
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


class Handler(BaseHTTPRequestHandler):
    student = None
    taxonomy_classes: list[str] = []

    def log_message(self, fmt, *args):
        print(f"[serve] {args[0]}")

    def do_GET(self):
        if self.path == "/health":
            self._json(
                200,
                {
                    "ok": True,
                    "loaded": Handler.student is not None,
                    "taxonomy": Handler.taxonomy_classes,
                    "port": PORT,
                },
            )
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/plan":
            self._handle_plan()
        elif self.path == "/organize":
            self._handle_organize()
        else:
            self._json(404, {"error": "not found"})

    def _handle_plan(self):
        body = _read_json(self)
        folder = Path(body.get("folder", "")).expanduser().resolve()
        threshold = float(body.get("threshold", 0.5))
        if not folder.is_dir():
            self._json(400, {"error": f"not a folder: {folder}"})
            return
        plan = build_plan(str(folder), Handler.student, conf_threshold=threshold)
        payload = _plan_to_dict(plan)
        payload["ok"] = True
        payload["taxonomy"] = Handler.taxonomy_classes
        self._json(200, payload)

    def _handle_organize(self):
        body = _read_json(self)
        folder = Path(body.get("folder", "")).expanduser().resolve()
        mode = body.get("mode", "dry")
        threshold = float(body.get("threshold", 0.5))
        if not folder.is_dir():
            self._json(400, {"error": f"not a folder: {folder}"})
            return

        lines: list[str] = []
        if mode == "dupe":
            work = duplicate_organized(folder)
            lines.append(f"copied to {work}")
            plan = build_plan(str(work), Handler.student, conf_threshold=threshold)
            lines.extend(_plan_lines(plan))
            n = apply_plan(plan)
            lines.append(f"done — {n} moves in {work}")
            self._json(
                200,
                {
                    "ok": True,
                    "mode": mode,
                    "folder": str(work),
                    "moves": n,
                    "log": lines,
                    "plan": _plan_to_dict(plan),
                },
            )
            return

        plan = build_plan(str(folder), Handler.student, conf_threshold=threshold)
        lines.extend(_plan_lines(plan))
        n_moved = 0
        if mode == "apply":
            n_moved = apply_plan(plan)
            lines.append(f"moved {n_moved} files")
        else:
            lines.append("dry-run")
        self._json(
            200,
            {
                "ok": True,
                "mode": mode,
                "folder": str(folder),
                "moves": len(plan.moves),
                "moved": n_moved,
                "log": lines,
                "plan": _plan_to_dict(plan),
            },
        )

    def _json(self, code, data):
        raw = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main():
    if not CKPT.exists():
        sys.exit(f"missing checkpoint: {CKPT}")
    _free_port(PORT)
    print("loading model...")
    Handler.student = load_student(str(CKPT), quiet=True)
    Handler.taxonomy_classes = list(Taxonomy().classes)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as exc:
        if getattr(exc, "errno", None) == 48 or "Address already in use" in str(exc):
            _free_port(PORT)
            try:
                server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
            except OSError:
                sys.exit(
                    f"port {PORT} still in use. Run:  lsof -iTCP:{PORT} -sTCP:LISTEN\n"
                    f"then:  kill <pid>"
                )
        else:
            raise
    print(f"ready on http://127.0.0.1:{PORT}")
    print("keep this window open!")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        server.shutdown()


if __name__ == "__main__":
    main()
