from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "artifacts" / "urtc_eval_package"
PROMPT_PATH = PKG / "AGENT_PACKET_PROMPT.md"


def load_tasks(packet: Path) -> list[dict[str, Any]]:
    path = packet / "tasks.jsonl" if packet.is_dir() else packet
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_done(out_path: Path) -> dict[str, dict[str, Any]]:
    if not out_path.is_file():
        return {}
    done: dict[str, dict[str, Any]] = {}
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tid = str(row.get("task_id") or "")
        if tid:
            done[tid] = row
    return done


def append_row(out_path: Path, row: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    dec = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            candidates.append(obj)
        elif isinstance(obj, list):
            candidates.extend(x for x in obj if isinstance(x, dict))
    except json.JSONDecodeError:
        pass
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            candidates.append(obj)
    # Prefer objects that look like our prediction contract
    for obj in reversed(candidates):
        if any(k in obj for k in ("task_id", "ranked_folder_ids", "abstained", "raw_pred_folder_id")):
            return obj
    if candidates:
        return candidates[-1]
    raise ValueError(f"no JSON object in model output: {text[:240]!r}")

def task_prompt(system: str, task: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        "Classify this single task. Return ONE JSON object only (not array).\n"
        f"{json.dumps(task, ensure_ascii=False)}\n"
    )


def is_rate_limit_error(msg: str) -> bool:
    m = msg.lower()
    return any(
        s in m
        for s in (
            "rate limit",
            "ratelimit",
            "429",
            "too many requests",
            "resource_exhausted",
            "quota",
            "overloaded",
            "capacity",
            "timeout",
            "temporar",
        )
    )


def ollama_chat(
    host: str,
    model: str,
    prompt: str,
    timeout_s: int,
    *,
    num_predict: int = 512,
    num_ctx: int = 4096,
    think: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    """Chat completion tuned for short structured JSON classification.

    Slow-path audit (URTC bakeoff):
    - qwen3.x defaults to long chain-of-thought (often 3k–10k tok out) unless think=false.
    - Unbounded num_predict lets gemma/qwen emit multi-hundred-token explanations.
    - Large num_ctx (model default 32k on qwen3.5) bloates KV cache for ~1k-token packet prompts.
    """
    url = host.rstrip("/") + "/api/chat"
    opts: dict[str, Any] = {
        "temperature": 0.1,
        "num_predict": int(num_predict),
        "num_ctx": int(num_ctx),
    }
    body: dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": opts,
        "keep_alive": "60m",
        # Force structured generation: stops free-form "thinking aloud" that burns num_predict
        # before a parseable object appears (gemma4 exhibited this under caps).
        "format": "json",
    }
    # Qwen3 / Qwen3.5 / Qwen3.6: disable internal thinking channel unless explicitly asked for.
    ml = model.lower()
    if think is None:
        think = False if ("qwen3" in ml or "deepseek-r1" in ml) else None
    if think is not None:
        body["think"] = bool(think)
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    msg = payload.get("message") or {}
    content = msg.get("content") or payload.get("response") or ""
    # If a thinking model still returns empty content, fall back to thinking text for JSON extract.
    if not str(content).strip() and msg.get("thinking"):
        content = str(msg.get("thinking"))
    meta = {
        "tokens_in": (payload.get("prompt_eval_count") or None),
        "tokens_out": (payload.get("eval_count") or None),
        "ollama_num_predict": int(num_predict),
        "ollama_num_ctx": int(num_ctx),
        "ollama_think": body.get("think"),
    }
    return str(content), meta


def run_codex(model: str, effort: str, prompt: str, timeout_s: int) -> tuple[str, dict[str, Any]]:
    """Returns (text, usage_meta). Uses --json to capture token usage when present."""
    cmd = [
        "codex",
        "exec",
        "--json",
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{effort}"',
        "-s",
        "read-only",
        "--skip-git-repo-check",
        "--ephemeral",
        prompt,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    if r.returncode != 0 and not (r.stdout or "").strip():
        raise RuntimeError(f"codex exit {r.returncode}: {out[:500]}")
    texts: list[str] = []
    usage: dict[str, Any] = {}
    for line in (r.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "item.completed":
            item = ev.get("item") or {}
            if item.get("type") == "agent_message" and item.get("text"):
                texts.append(str(item["text"]))
        if ev.get("type") == "turn.completed":
            u = ev.get("usage") or {}
            if u:
                usage = {
                    "tokens_in": u.get("input_tokens"),
                    "tokens_out": u.get("output_tokens"),
                    "tokens_cached_in": u.get("cached_input_tokens"),
                    "tokens_reasoning_out": u.get("reasoning_output_tokens"),
                }
    text = texts[-1] if texts else out
    return text, usage


def run_agy(model: str, prompt: str, timeout_s: int) -> tuple[str, dict[str, Any]]:
    # Working form: -p PROMPT then --model (Gemini 3.1 Pro low confirmed).
    # agy does not expose reliable token accounting in print mode yet.
    cmd = [
        "agy",
        "-p",
        prompt,
        "--model",
        model,
        "--print-timeout",
        f"{max(60, timeout_s)}s",
        "--dangerously-skip-permissions",
        "--output-format",
        "text",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 30)
    out = (r.stdout or "").strip() or (r.stderr or "").strip()
    if r.returncode != 0 and not out:
        raise RuntimeError(f"agy exit {r.returncode}")
    return out, {}


def normalize_pred(task: dict[str, Any], obj: dict[str, Any], model: str, wall_ms: int, extra: dict[str, Any]) -> dict[str, Any]:
    abstained = bool(obj.get("abstained") or obj.get("abstain"))
    ranked = obj.get("ranked_folder_ids") or obj.get("ranked") or []
    if isinstance(ranked, str):
        ranked = [ranked]
    ranked = [str(x) for x in ranked]
    # keep only known candidates
    valid = {c["id"] for c in task.get("candidates") or []}
    ranked = [x for x in ranked if x in valid]
    if not ranked and not abstained:
        # free-form top
        for key in ("raw_pred_folder_id", "folder_id", "label"):
            if obj.get(key) and str(obj[key]) in valid:
                ranked = [str(obj[key])]
                break
    if not ranked and not abstained:
        abstained = True
    row = {
        "task_id": task["task_id"],
        "ranked_folder_ids": [] if abstained else ranked,
        "abstained": abstained,
        "model": model,
        "wall_ms": wall_ms,
        "tokens_in": extra.get("tokens_in"),
        "tokens_out": extra.get("tokens_out"),
        "est_cost_usd": extra.get("est_cost_usd"),
        "n_tool_events": extra.get("n_tool_events"),
        "n_turns": extra.get("n_turns"),
        "error": obj.get("error"),
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backend", required=True, choices=["ollama", "codex", "agy"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--packet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--limit", type=int, default=0, help="optional max tasks (0=all)")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--timeout-s", type=int, default=180)
    ap.add_argument("--sleep-s", type=float, default=0.4, help="pace between tasks")
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--reasoning-effort", default="medium", help="codex only")
    ap.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ap.add_argument(
        "--ollama-num-predict",
        type=int,
        default=int(os.environ.get("OLLAMA_NUM_PREDICT", "512")),
        help="cap generation tokens (JSON classification; prevents multi-k CoT blowups)",
    )
    ap.add_argument(
        "--ollama-num-ctx",
        type=int,
        default=int(os.environ.get("OLLAMA_NUM_CTX", "4096")),
        help="context window; packet prompts are ~1k tok — avoid model default 32k KV",
    )
    ap.add_argument(
        "--ollama-think",
        choices=["auto", "true", "false"],
        default=os.environ.get("OLLAMA_THINK", "auto"),
        help="qwen3 thinking channel; auto=off for qwen3*/deepseek-r1",
    )
    ap.add_argument("--stop-on-error", action="store_true")
    args = ap.parse_args()

    system = PROMPT_PATH.read_text(encoding="utf-8") if PROMPT_PATH.is_file() else ""
    tasks = load_tasks(args.packet)
    if args.offset:
        tasks = tasks[args.offset :]
    if args.limit:
        tasks = tasks[: args.limit]

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    responses = out_dir / "responses.jsonl"
    done = load_done(responses)
    started = datetime.now(timezone.utc).isoformat()
    n_ok = n_err = n_skip = 0

    for i, task in enumerate(tasks):
        tid = str(task["task_id"])
        if tid in done:
            n_skip += 1
            continue
        prompt = task_prompt(system, task)
        last_err = ""
        for attempt in range(1, args.max_retries + 1):
            t0 = time.perf_counter()
            try:
                extra: dict[str, Any] = {}
                if args.backend == "ollama":
                    think_arg: bool | None
                    if args.ollama_think == "auto":
                        think_arg = None
                    else:
                        think_arg = args.ollama_think == "true"
                    text, meta = ollama_chat(
                        args.ollama_host,
                        args.model,
                        prompt,
                        args.timeout_s,
                        num_predict=args.ollama_num_predict,
                        num_ctx=args.ollama_num_ctx,
                        think=think_arg,
                    )
                    extra.update(meta)
                elif args.backend == "codex":
                    text, meta = run_codex(args.model, args.reasoning_effort, prompt, args.timeout_s)
                    extra.update(meta)
                else:
                    text, meta = run_agy(args.model, prompt, args.timeout_s)
                    extra.update(meta)
                wall = int((time.perf_counter() - t0) * 1000)
                obj = extract_json_object(text)
                row = normalize_pred(task, obj, args.model, wall, extra)
                append_row(responses, row)
                done[tid] = row
                n_ok += 1
                print(f"[{i+1}/{len(tasks)}] ok {tid} wall_ms={wall}", flush=True)
                break
            except Exception as e:  # noqa: BLE001 — resume-safe runner
                last_err = str(e)
                wait = min(120.0, (2 ** (attempt - 1)) * 2.0)
                rate = is_rate_limit_error(last_err)
                print(
                    f"[{i+1}/{len(tasks)}] err attempt {attempt}/{args.max_retries} "
                    f"{'RATE' if rate else 'err'}: {last_err[:180]}",
                    flush=True,
                )
                if attempt >= args.max_retries:
                    row = {
                        "task_id": tid,
                        "ranked_folder_ids": [],
                        "abstained": True,
                        "model": args.model,
                        "wall_ms": int((time.perf_counter() - t0) * 1000),
                        "error": last_err[:500],
                    }
                    append_row(responses, row)
                    n_err += 1
                    if args.stop_on_error:
                        raise
                else:
                    time.sleep(wait if rate else min(10.0, wait))
                    continue
        time.sleep(args.sleep_s)

    meta = {
        "backend": args.backend,
        "model": args.model,
        "packet": str(args.packet),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "n_tasks": len(tasks),
        "n_ok": n_ok,
        "n_err": n_err,
        "n_skipped_resume": n_skip,
        "n_written": len(done),
        "responses": str(responses),
        "protocol": "v2_observable_agent_packet_v1",
        "rate_limit_policy": f"exponential backoff, max_retries={args.max_retries}",
        "ollama_host": args.ollama_host if args.backend == "ollama" else None,
        "ollama_num_predict": args.ollama_num_predict if args.backend == "ollama" else None,
        "ollama_num_ctx": args.ollama_num_ctx if args.backend == "ollama" else None,
        "ollama_think": args.ollama_think if args.backend == "ollama" else None,
        "reasoning_effort": args.reasoning_effort if args.backend == "codex" else None,
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
