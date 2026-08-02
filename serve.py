from __future__ import annotations
import json
import socket
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

PORT = 18765
CKPT = ROOT / "artifacts" / "student_model_ckpt"


class LocalThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        # SO_REUSEADDR permits multiple listeners on the same port on Windows.
        # Exclusive binding guarantees there can be only one model server/copy.
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def _existing_server_health() -> dict | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None


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
    load_state = "starting"
    load_error: str | None = None

    def log_message(self, fmt, *args):
        print(f"[serve] {args[0]}")

    def do_GET(self):
        if self.path == "/health":
            self._json(
                200,
                {
                    "ok": True,
                    "loaded": Handler.student is not None,
                    "state": Handler.load_state,
                    "error": Handler.load_error,
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
        if not self._model_ready():
            return
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
        if not self._model_ready():
            return
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

    def _model_ready(self) -> bool:
        if Handler.student is not None:
            return True
        if Handler.load_error:
            self._json(500, {"ok": False, "state": "error", "error": Handler.load_error})
        else:
            self._json(503, {"ok": False, "state": Handler.load_state, "loading": True})
        return False

    def _json(self, code, data):
        raw = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _load_model() -> None:
    global Move, Plan, apply_plan, build_plan, duplicate_organized, load_student

    try:
        Handler.load_state = "importing"
        from core.moves import Move, Plan, apply_plan, build_plan, duplicate_organized, load_student
        from core.taxonomy import Taxonomy

        Handler.load_state = "loading_weights"
        student = load_student(str(CKPT), quiet=True)
        Handler.taxonomy_classes = list(Taxonomy().classes)
        Handler.student = student
        Handler.load_state = "ready"
        print("model loaded; server ready", flush=True)
    except Exception as exc:
        Handler.load_error = f"{type(exc).__name__}: {exc}"
        Handler.load_state = "error"
        print(f"model failed to load: {Handler.load_error}", file=sys.stderr, flush=True)


def main():
    if not CKPT.exists():
        sys.exit(f"missing checkpoint: {CKPT}")
    try:
        server = LocalThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as exc:
        health = _existing_server_health()
        if health is not None:
            state = health.get("state") or ("ready" if health.get("loaded") else "loading")
            print(f"server already running on http://127.0.0.1:{PORT} ({state})")
            return
        sys.exit(f"could not bind http://127.0.0.1:{PORT}: {exc}")

    print(f"listening on http://127.0.0.1:{PORT}; loading model...", flush=True)
    print("keep this window open!")
    loader = threading.Thread(target=_load_model, name="model-loader", daemon=True)
    loader.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
