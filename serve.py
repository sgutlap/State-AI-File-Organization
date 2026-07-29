import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.moves import apply_plan, build_plan, duplicate_organized, load_student

PORT = 18765
CKPT = ROOT / "artifacts" / "student_model_ckpt"


class Handler(BaseHTTPRequestHandler):
    student = None

    def log_message(self, fmt, *args):
        print(f"[serve] {args[0]}")

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "loaded": Handler.student is not None})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/organize":
            self._json(404, {"error": "not found"})
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or b"{}"))
        folder = Path(body.get("folder", "")).expanduser().resolve()
        mode = body.get("mode", "dry")
        threshold = float(body.get("threshold", 0.5))
        if not folder.is_dir():
            self._json(400, {"error": f"not a folder: {folder}"})
            return

        lines = []
        if mode == "dupe":
            work = duplicate_organized(folder)
            lines.append(f"copied to {work}")
            plan = build_plan(str(work), Handler.student, conf_threshold=threshold)
            lines.extend(_plan_lines(plan))
            n = apply_plan(plan)
            lines.append(f"done — {n} moves in {work}")
            self._json(200, {"ok": True, "mode": mode, "folder": str(work), "moves": n, "log": lines})
            return

        plan = build_plan(str(folder), Handler.student, conf_threshold=threshold)
        lines.extend(_plan_lines(plan))
        if mode == "apply":
            n = apply_plan(plan)
            lines.append(f"moved {n} files")
        else:
            lines.append("dry-run")
        self._json(200, {"ok": True, "mode": mode, "folder": str(folder), "moves": len(plan.moves), "log": lines})

    def _json(self, code, data):
        raw = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def _plan_lines(plan, limit=40):
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


def main():
    if not CKPT.exists():
        sys.exit(f"missing checkpoint: {CKPT}")
    print("loading model...")
    Handler.student = load_student(str(CKPT))
    print(f"ready on http://127.0.0.1:{PORT}")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
        server.shutdown()


if __name__ == "__main__":
    main()
