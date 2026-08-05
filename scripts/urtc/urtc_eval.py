from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "artifacts" / "urtc_eval_package"
CONFIG_PATH = PKG / "config.json"
sys.path.insert(0, str(ROOT))

from scripts.urtc_metrics import (  # noqa: E402
    canonicalize_predictions,
    diagnostic_metrics,
    leaderboard_row,
    pareto_points,
    per_source_accuracy,
    read_jsonl,
    systems_rollup,
    write_json,
    write_jsonl,
)
from scripts.preflight_t3_eval import preflight as preflight_t3  # noqa: E402

CKPT_DEFAULT = "artifacts/pilot_data_r3/virtual_abstain_then_r3_cuda.pt"
CAL_DEFAULT = "artifacts/final_owner_study/virtual_dual_calibration.json"
CKPT_SHA_DEFAULT = "a6bf3d657b057d797e4a89bbda832ef1f5f34f7d87f6126b3dbe070f63d92641"
CAL_SHA_DEFAULT = "d1a07d79f610f6e3b04d22c20a6417d73fb10bbb81d881a7d5c764f66e68e22a"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> Dict[str, Any]:
    if CONFIG_PATH.is_file():
        return _load_json(CONFIG_PATH)
    return {}


def v2_policy(track: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Explicit dual-encoder scoring config per track (paper-equivalent by default)."""
    base = {
        "checkpoint": cfg.get("checkpoint", CKPT_DEFAULT),
        "calibration": cfg.get("calibration", CAL_DEFAULT),
        "checkpoint_sha256": cfg.get("checkpoint_sha256", CKPT_SHA_DEFAULT),
        "calibration_sha256": cfg.get("calibration_sha256", CAL_SHA_DEFAULT),
        "example_extension_fallback": bool(cfg.get("example_extension_fallback", True)),
        "split": "test",
    }
    track_policy = ((cfg.get("tracks") or {}).get(track.upper()) or {}).get("v2_policy") or {}
    base.update({k: v for k, v in track_policy.items() if v is not None})
    return base


def resolve_artifact(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def verify_artifact_hash(path: Path, expected: Optional[str], *, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"missing {label}: {path}")
    if not expected:
        return
    actual = _sha256_file(path)
    if actual != expected:
        raise SystemExit(
            f"{label} hash mismatch for {path}\n  expected {expected}\n  actual   {actual}"
        )


def track_paths(track: str) -> Dict[str, Any]:
    t = track.upper()
    cfg = load_config().get("tracks", {}).get(t, {})
    allow_cloud = bool(cfg.get("allow_cloud_agents", False))
    if t == "T1":
        study = ROOT / "artifacts" / "final_owner_study"
        return {
            "manifest": study / "locked_test_manifest.jsonl",
            "metrics_frozen": study / "locked_test_metrics.json",
            "workspaces": study / "locked_test",
            "ready": True,
            # Owner consent enables cloud/local agents on T1; still no T3 preflight gate.
            "agent_ready": bool(cfg.get("owner_consent_cloud_agents") or allow_cloud),
            "allow_cloud": allow_cloud,
            "allow_ollama": bool(cfg.get("allow_ollama", False)),
            "allow_products": bool(cfg.get("allow_products", False)),
            "cfg": cfg,
        }
    if t == "T2":
        return {
            "manifest": PKG / "T2_SYNTH_OPEN_TAX" / "manifests" / "locked_test_all.jsonl",
            "workspaces": PKG / "T2_SYNTH_OPEN_TAX" / "workspaces",
            "ready": True,
            "agent_ready": allow_cloud or bool(cfg.get("allow_ollama", False)),
            "allow_cloud": allow_cloud,
            "allow_ollama": bool(cfg.get("allow_ollama", False)),
            "allow_products": bool(cfg.get("allow_products", False)),
            "cfg": cfg,
        }
    if t == "T3":
        status_path = PKG / "T3_OTAB_AGENT" / "STATUS.json"
        status = _load_json(status_path) if status_path.is_file() else {"status": "MISSING"}
        score_ready = bool(status.get("ready_for_score_v2") or status.get("ready_for_freeze"))
        agent_ready = bool(status.get("ready_for_agent_bakeoff"))
        return {
            "status_path": status_path,
            "status": status,
            "manifest": PKG / "T3_OTAB_AGENT" / "manifests" / "otab_all.jsonl",
            "workspaces": PKG / "T3_OTAB_AGENT" / "workspaces",
            "normalized": PKG / "T3_OTAB_AGENT" / "OTAB-6-v3.normalized.json",
            "t3_root": PKG / "T3_OTAB_AGENT",
            "ready": score_ready,
            "agent_ready": agent_ready,
            "allow_cloud": allow_cloud,
            "allow_ollama": bool(cfg.get("allow_ollama", True)),
            "allow_products": bool(cfg.get("allow_products", True)),
            "cfg": cfg,
        }
    raise SystemExit(f"unknown track {track}")


def resolve_backends(args: argparse.Namespace, cfg: Dict[str, Any]) -> List[str]:
    if getattr(args, "profile", None):
        prof = (cfg.get("default_profiles") or {}).get(args.profile)
        if not prof:
            raise SystemExit(f"unknown profile {args.profile}")
        return list(prof.get("backends") or [])
    if getattr(args, "backends", None):
        return [b.strip() for b in args.backends.split(",") if b.strip()]
    return ["stateai_v2"]


def resolve_tracks(args: argparse.Namespace, cfg: Dict[str, Any]) -> List[str]:
    if getattr(args, "profile", None):
        prof = (cfg.get("default_profiles") or {}).get(args.profile)
        if not prof:
            raise SystemExit(f"unknown profile {args.profile}")
        return [t.upper() for t in prof.get("tracks") or []]
    if getattr(args, "tracks", None):
        return [t.strip().upper() for t in args.tracks.split(",") if t.strip()]
    return ["T3"]


def cmd_status(_: argparse.Namespace) -> int:
    cfg = load_config()
    print("URTC eval package:", PKG)
    print("config:", CONFIG_PATH if CONFIG_PATH.is_file() else "(missing)")
    for t in ("T1", "T2", "T3"):
        p = track_paths(t)
        man = Path(p["manifest"])
        n = sum(1 for line in man.open() if line.strip()) if man.is_file() else 0
        print(
            f"  {t}: manifest_ok={man.is_file()} n_tasks≈{n} "
            f"score={p.get('ready')} agents={p.get('agent_ready')} cloud={p.get('allow_cloud')}"
        )
        if t == "T3" and p.get("status"):
            st = p["status"]
            print(f"       status={st.get('status')} version_id={st.get('version_id')}")
    policy = v2_policy("T1", cfg)
    print("v2_policy:", json.dumps({
        "checkpoint": policy["checkpoint"],
        "calibration": policy["calibration"],
        "example_extension_fallback": policy["example_extension_fallback"],
    }))
    print("profiles:", ", ".join(sorted((cfg.get("default_profiles") or {}).keys())))
    consent = cfg.get("consent") or {}
    if consent:
        print("consent.T1_owner_cloud_and_local_agents:", consent.get("T1_owner_cloud_and_local_agents"))
    hosts = cfg.get("hosts") or {}
    for name, meta in sorted(hosts.items()):
        print(f"  host {name}: ready={meta.get('ready')} role={meta.get('role')}")
    return 0


def cmd_verify_dataset(_: argparse.Namespace) -> int:
    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tracks": {},
        "pass": True,
    }
    for t in ("T1", "T2", "T3"):
        p = track_paths(t)
        man = Path(p["manifest"])
        issues = []
        if not man.is_file():
            issues.append("missing_manifest")
            report["pass"] = False
            report["tracks"][t] = {"ok": False, "issues": issues}
            continue
        tasks = read_jsonl(man)
        n_abs = sum(1 for x in tasks if x.get("abstain"))
        sources = sorted({x.get("source_group_id") for x in tasks})
        ws = p.get("workspaces")
        if ws and Path(ws).is_dir() and t != "T1":
            for s in sources:
                inbox = Path(ws) / str(s) / "inbox"
                if not inbox.is_dir():
                    issues.append(f"missing_inbox:{s}")
        if t == "T3":
            pf = preflight_t3(Path(p["t3_root"]))
            report["tracks"][t] = {
                "ok": bool(pf["pass"]) and len(issues) == 0,
                "n_tasks": len(tasks),
                "n_abstain": n_abs,
                "n_sources": len(sources),
                "manifest": str(man),
                "issues": issues + ([f"preflight:{f['check']}" for f in pf["failures"]] if not pf["pass"] else []),
                "preflight_pass": pf["pass"],
                "version_id": (p.get("status") or {}).get("version_id"),
            }
            if not report["tracks"][t]["ok"]:
                report["pass"] = False
            continue
        ok = len(issues) == 0
        if not ok:
            report["pass"] = False
        report["tracks"][t] = {
            "ok": ok,
            "n_tasks": len(tasks),
            "n_abstain": n_abs,
            "n_sources": len(sources),
            "manifest": str(man),
            "issues": issues,
        }
    out = PKG / "audits" / "DATASET_LINK_VERIFY.json"
    write_json(out, report)
    print(json.dumps(report, indent=2))
    print("wrote", out)
    return 0 if report["pass"] else 2


def cmd_preflight_t3(args: argparse.Namespace) -> int:
    t3 = track_paths("T3")["t3_root"]
    report = preflight_t3(
        t3,
        near_duplicate_threshold=args.near_duplicate_threshold,
        max_near_duplicate_pairs=args.max_near_duplicate_pairs,
        agent_view_root=args.agent_view_root,
        for_agents=bool(args.agent_view_root or args.for_agents),
    )
    out = PKG / "audits" / "T3_PREFLIGHT_LATEST.json"
    write_json(out, report)
    print(json.dumps({
        "pass": report["pass"],
        "failure_count": report.get("failure_count"),
        "failures": [{"check": f["check"], "count": f.get("count")} for f in report["failures"]],
        "out": str(out),
    }, indent=2))
    return 0 if report["pass"] else 2


def _backend_hosts_ready(meta: Dict[str, Any], hosts_cfg: Dict[str, Any]) -> bool:
    """True if any listed host is ready, or hosts are unrestricted / 'any'."""
    host_ids = list(meta.get("hosts") or [])
    if not host_ids or "any" in host_ids:
        return True
    return any(bool((hosts_cfg.get(h) or {}).get("ready")) for h in host_ids)


def build_run_plan(
    tracks: Sequence[str],
    backends: Sequence[str],
    cfg: Dict[str, Any],
    *,
    execute: bool,
) -> Dict[str, Any]:
    be_cfg = cfg.get("backends") or {}
    hosts_cfg = cfg.get("hosts") or {}
    plan_runs = []
    refused = []
    for track in tracks:
        tp = track_paths(track)
        for backend in backends:
            meta = be_cfg.get(backend) or {"family": "unknown", "tracks": ["T3"], "enabled": True}
            if meta.get("enabled") is False:
                refused.append({"track": track, "backend": backend, "reason": "backend_disabled"})
                continue
            allowed_tracks = [x.upper() for x in meta.get("tracks") or []]
            if allowed_tracks and track.upper() not in allowed_tracks:
                refused.append({"track": track, "backend": backend, "reason": "backend_track_deny"})
                continue
            if meta.get("external_api") and not tp.get("allow_cloud"):
                refused.append({"track": track, "backend": backend, "reason": "cloud_denied_on_track"})
                continue
            if (
                meta.get("external_api")
                and track.upper() == "T3"
                and not tp.get("agent_ready")
                and execute
            ):
                refused.append(
                    {"track": track, "backend": backend, "reason": "t3_agents_require_author_verify"}
                )
                continue
            if backend.startswith("ollama") and not tp.get("allow_ollama"):
                refused.append({"track": track, "backend": backend, "reason": "ollama_denied_on_track"})
                continue
            if str(backend).startswith("product:") and not tp.get("allow_products"):
                refused.append({"track": track, "backend": backend, "reason": "products_denied_on_track"})
                continue
            if not _backend_hosts_ready(meta, hosts_cfg):
                refused.append(
                    {
                        "track": track,
                        "backend": backend,
                        "reason": "host_not_ready",
                        "hosts": meta.get("hosts") or [],
                    }
                )
                continue
            suite_dirs = []
            ws = tp.get("workspaces")
            if ws and Path(ws).is_dir():
                suite_dirs = sorted([p.name for p in Path(ws).iterdir() if p.is_dir()])
            is_product = str(backend).startswith("product:")
            packet_protocol = not is_product and backend != "stateai_v2"
            if is_product:
                protocol = meta.get("protocol") or "native_file_dry_run"
                agent_prompt = str(PKG / "T3_OTAB_AGENT" / "AGENT_PROMPT.md")
                prepare_cmd = (
                    ".venv/bin/python scripts/urtc_eval.py prepare-agent-view --out <sandbox>"
                    if track.upper() == "T3"
                    else "native-file gold-free sandbox (manual: taxonomy.json + inbox/* only)"
                )
                runner_status = "plan_only_adapter_missing"
            elif backend == "stateai_v2":
                protocol = "manifest_semantic_score_v2"
                agent_prompt = None
                prepare_cmd = None
                runner_status = "executable_score_v2"
            else:
                protocol = meta.get("protocol") or "v2_observable_task_packet"
                agent_prompt = str(PKG / "AGENT_PACKET_PROMPT.md")
                prepare_cmd = (
                    f".venv/bin/python scripts/prepare_eval_packet.py --track {track.upper()} "
                    "--out <packet-dir>"
                )
                runner_status = "plan_only_manual_packet_collect"
            plan_runs.append(
                {
                    "track": track.upper(),
                    "backend": backend,
                    "family": meta.get("family"),
                    "external_api": bool(meta.get("external_api")),
                    "local_zero_cost": bool(meta.get("local_zero_cost")),
                    "manifest": str(tp["manifest"]),
                    "workspaces": str(ws) if ws else None,
                    "suites": suite_dirs,
                    "resources": meta.get("resources") or [],
                    "who_runs": meta.get("who_runs") or "any",
                    "runner": meta.get("runner"),
                    "runner_status": runner_status,
                    "hosts": meta.get("hosts") or [],
                    "execute": execute,
                    "v2_policy": v2_policy(track, cfg) if backend == "stateai_v2" else None,
                    "agent_prompt": agent_prompt,
                    "prepare_cmd": prepare_cmd,
                    "protocol": protocol,
                    "input_channel": "native_file" if is_product else (
                        "manifest_content_sample" if backend == "stateai_v2" else "v2_packet"
                    ),
                    "strict_vs_v2": (not is_product) and packet_protocol,
                }
            )
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "execute": execute,
        "n_runs": len(plan_runs),
        "runs": plan_runs,
        "refused": refused,
        "metrics_schema": "artifacts/urtc_eval_package/schemas/METRICS.md",
        "prediction_contract": "task_id + ranked_folder_ids/abstained (+ free-form raw, canonicalized before score)",
        "agent_fairness_note": (
            "Strict CLI/Ollama comparisons use prepare_eval_packet.py (V2-observable strings). "
            "Products use native-file sandbox (separate table). Never point agents at gold labels/manifests."
        ),
        "safety": cfg.get("safety") or {},
        "output_root": str(ROOT / cfg.get("output_root", PKG / "runs")),
    }


def cmd_plan(args: argparse.Namespace) -> int:
    cfg = load_config()
    tracks = resolve_tracks(args, cfg)
    backends = resolve_backends(args, cfg)
    plan = build_run_plan(tracks, backends, cfg, execute=False)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = PKG / "runs" / "plans" / f"plan_{stamp}.json"
    write_json(out, plan)
    write_json(PKG / "runs" / "plans" / "plan_latest.json", plan)
    print(json.dumps({k: plan[k] for k in ("timestamp", "execute", "n_runs", "refused")}, indent=2))
    for r in plan["runs"]:
        print(f"  + {r['track']:3} | {r['backend']:22} | suites={len(r.get('suites') or [])}")
    for r in plan["refused"]:
        print(f"  - REFUSE {r['track']} {r['backend']}: {r['reason']}")
    print("wrote", out)
    return 0


def _run(cmd: List[str]) -> int:
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


def _run_measured(cmd: List[str]) -> tuple[int, Dict[str, Any]]:
    """Run one local backend and measure its whole predictor process."""
    print("+", " ".join(cmd))
    start = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=str(ROOT))
    peak_rss = 0
    try:
        import psutil
        process = psutil.Process(proc.pid)
        while proc.poll() is None:
            for candidate in [process, *process.children(recursive=True)]:
                try:
                    peak_rss = max(peak_rss, candidate.memory_info().rss)
                except psutil.Error:
                    pass
            time.sleep(0.05)
    except ImportError:
        proc.wait()
    rc = proc.wait()
    return rc, {
        "wall_ms_sum": round((time.perf_counter() - start) * 1000, 3),
        "peak_rss_mb_max": round(peak_rss / (1024 * 1024), 3) if peak_rss else None,
        "telemetry_scope": "whole_predictor_process",
        "telemetry_source": "harness_psutil" if peak_rss else "harness_wall_only",
    }


def _t3_preflight_gate(*, for_agents: bool = False, agent_view: Path | None = None) -> bool:
    report = preflight_t3(
        for_agents=for_agents,
        agent_view_root=agent_view,
    )
    out = PKG / "audits" / "T3_PREFLIGHT_LATEST.json"
    write_json(out, report)
    if report["pass"]:
        return True
    summary = ", ".join(f"{item['check']}={item.get('count', 0)}" for item in report["failures"])
    print(f"REFUSE T3 scoring: preflight failed ({summary}).", file=sys.stderr)
    print("Run: .venv/bin/python scripts/preflight_t3_eval.py", file=sys.stderr)
    print("Report:", out, file=sys.stderr)
    return False


def _backend_meta(backend: str) -> Dict[str, Any]:
    return (load_config().get("backends") or {}).get(backend) or {}


def _apply_local_zero_cost(backend: str, systems: Dict[str, Any]) -> Dict[str, Any]:
    """Ollama/local/product backends count as $0 for Pareto and tables."""
    meta = _backend_meta(backend)
    zero = bool(meta.get("local_zero_cost")) or backend.startswith("ollama") or backend.startswith(
        "product:"
    )
    if zero:
        systems["est_cost_usd_sum"] = float(
            (load_config().get("metrics") or {}).get("local_cost_usd_default", 0.0)
        )
        systems["billing_note"] = "local_zero_usd"
    return systems


def _score_with_canonical(
    *,
    track: str,
    backend: str,
    manifest: Path,
    preds_path: Path,
    out_dir: Path,
    run_systems: Optional[Dict[str, Any]] = None,
) -> int:
    """Canonicalize free-form labels then evaluate quality metrics."""
    tasks = read_jsonl(manifest)
    raw_preds = read_jsonl(preds_path)
    canon = canonicalize_predictions(tasks, raw_preds)
    canon_path = out_dir / "predictions.canonical.jsonl"
    write_jsonl(canon_path, canon)
    quality = out_dir / "quality.json"
    py = sys.executable
    rc = _run(
        [
            py,
            "scripts/evaluate_taxonomy_predictions.py",
            str(manifest),
            str(canon_path),
            "--out",
            str(quality),
        ]
    )
    if rc != 0:
        return rc
    sys_roll = systems_rollup(canon)
    if run_systems:
        sys_roll.update(run_systems)
    sys_roll = _apply_local_zero_cost(backend, sys_roll)
    sys_roll["n_files"] = len(tasks)
    sys_roll["n_predictions"] = len(canon)
    if sys_roll.get("wall_ms_sum") is not None:
        sys_roll["wall_ms_mean"] = float(sys_roll["wall_ms_sum"]) / max(1, len(canon))
    per_src = per_source_accuracy(tasks, canon)
    diagnostics = diagnostic_metrics(tasks, canon)
    q = _load_json(quality) if quality.is_file() else {}
    deep = {
        "backend": backend,
        "track": track.upper(),
        "quality": q,
        "systems": sys_roll,
        "per_source": per_src,
        "diagnostics": diagnostics,
        "canonicalization": {
            "raw_predictions": str(preds_path),
            "canonical_predictions": str(canon_path),
            "n_raw": len(raw_preds),
            "n_canonical": len(canon),
        },
        "leaderboard_row": leaderboard_row(
            backend=backend, track=track.upper(), quality=q, systems=sys_roll
        ),
    }
    write_json(out_dir / "systems.json", sys_roll)
    write_json(out_dir / "per_source.json", per_src)
    write_json(out_dir / "diagnostics.json", diagnostics)
    write_json(out_dir / "deep_report.json", deep)
    lb_path = PKG / "runs" / "leaderboard_latest.json"
    lb = _load_json(lb_path) if lb_path.is_file() else {"rows": []}
    lb.setdefault("rows", []).append(deep["leaderboard_row"])
    lb["pareto"] = pareto_points(lb["rows"])
    lb["updated"] = datetime.now(timezone.utc).isoformat()
    write_json(lb_path, lb)
    print(json.dumps(deep["leaderboard_row"], indent=2))
    print("wrote", out_dir)
    return 0


def cmd_score_v2(args: argparse.Namespace) -> int:
    cfg = load_config()
    paths = track_paths(args.track)
    track = args.track.upper()
    if track == "T3":
        if not paths["ready"]:
            print("T3 structure not ready", file=sys.stderr)
            return 2
        if not _t3_preflight_gate():
            return 2
    if track == "T1" and not args.force:
        frozen = paths.get("metrics_frozen")
        if frozen and Path(frozen).is_file():
            print("T1 frozen metrics (pass --force to recompute under explicit policy):\n")
            print(Path(frozen).read_text()[:1200])
            return 0

    policy = v2_policy(track, cfg)
    if args.checkpoint:
        policy["checkpoint"] = args.checkpoint
    if args.calibration is not None:
        policy["calibration"] = args.calibration
    if args.no_example_extension_fallback:
        policy["example_extension_fallback"] = False
    if args.example_extension_fallback:
        policy["example_extension_fallback"] = True

    ckpt = resolve_artifact(policy["checkpoint"])
    cal = resolve_artifact(policy["calibration"]) if policy.get("calibration") else None
    verify_artifact_hash(ckpt, policy.get("checkpoint_sha256"), label="checkpoint")
    if cal is not None and policy.get("calibration"):
        verify_artifact_hash(cal, policy.get("calibration_sha256"), label="calibration")

    manifest = Path(paths["manifest"])
    if not manifest.is_file():
        print("missing manifest", manifest, file=sys.stderr)
        return 2

    out_dir = PKG / "runs" / track / "stateai_v2" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "v2_policy.json", {
        **policy,
        "checkpoint_resolved": str(ckpt),
        "calibration_resolved": str(cal) if cal else None,
        "checkpoint_sha256_actual": _sha256_file(ckpt),
        "calibration_sha256_actual": _sha256_file(cal) if cal and cal.is_file() else None,
    })

    preds = out_dir / "predictions.jsonl"
    cmd = [
        sys.executable,
        "scripts/predict_dual_taxonomy.py",
        str(manifest),
        "--checkpoint",
        str(ckpt),
        "--split",
        str(policy.get("split") or "test"),
        "--out",
        str(preds),
    ]
    if cal is not None and policy.get("calibration"):
        cmd.extend(["--calibration", str(cal)])
    if policy.get("example_extension_fallback"):
        cmd.append("--example-extension-fallback")
    rc, measured = _run_measured(cmd)
    if rc != 0:
        return rc
    return _score_with_canonical(
        track=track,
        backend="stateai_v2",
        manifest=manifest,
        preds_path=preds,
        out_dir=out_dir,
        run_systems={
            **measured,
            "model": "stateai_v2",
            "external_api": False,
            "runtime": {"python": sys.version, "platform": platform.platform()},
        },
    )


def cmd_score_preds(args: argparse.Namespace) -> int:
    paths = track_paths(args.track)
    if args.track.upper() == "T3" and not _t3_preflight_gate():
        return 2
    manifest = Path(paths["manifest"])
    preds_path = Path(args.predictions)
    if not manifest.is_file() or not preds_path.is_file():
        print("need valid --track manifest and --predictions", file=sys.stderr)
        return 2
    out_dir = PKG / "runs" / args.track.upper() / args.backend / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    # preserve raw
    shutil.copy2(preds_path, out_dir / "predictions.raw.jsonl")
    run_systems = _load_json(Path(args.run_metadata)) if args.run_metadata else None
    if run_systems is not None and not isinstance(run_systems, dict):
        print("--run-metadata must contain one JSON object", file=sys.stderr)
        return 2
    return _score_with_canonical(
        track=args.track.upper(),
        backend=args.backend,
        manifest=manifest,
        preds_path=preds_path,
        out_dir=out_dir,
        run_systems=run_systems,
    )


def cmd_prepare_agent_view(args: argparse.Namespace) -> int:
    """Copy gold-free raw files for native-file product adapters only.

    Claude/Codex/agy/Ollama strict comparisons must use prepare_eval_packet.py,
    which exposes exactly the V2 semantic inputs rather than raw files.
    """
    if not _t3_preflight_gate(for_agents=False):
        return 2
    t3 = track_paths("T3")
    src_root = Path(t3["workspaces"])
    dest = Path(args.out) if args.out else PKG / "runs" / "agent_views" / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    if dest.exists():
        print(f"REFUSE: agent-view destination already exists: {dest}", file=sys.stderr)
        return 2
    dest.mkdir(parents=True)
    for ws in sorted(p for p in src_root.iterdir() if p.is_dir()):
        d = dest / ws.name
        (d / "inbox").mkdir(parents=True)
        shutil.copy2(ws / "taxonomy.json", d / "taxonomy.json")
        for f in (ws / "inbox").iterdir():
            if f.is_file():
                shutil.copy2(f, d / "inbox" / f.name)
    # freeze prompt next to view
    prompt = PKG / "T3_OTAB_AGENT" / "AGENT_PROMPT.md"
    if prompt.is_file():
        shutil.copy2(prompt, dest / "AGENT_PROMPT.md")
    # preflight agent view
    if not _t3_preflight_gate(for_agents=True, agent_view=dest):
        return 2
    print(json.dumps({"agent_view": str(dest), "prompt": str(prompt), "ok": True}, indent=2))
    return 0


def cmd_agents(args: argparse.Namespace) -> int:
    cfg = load_config()
    backends = resolve_backends(args, cfg)
    tracks = resolve_tracks(args, cfg) if (getattr(args, "tracks", None) or getattr(args, "profile", None)) else ["T1", "T3"]
    plan = build_run_plan(tracks, backends, cfg, execute=bool(args.execute))
    if args.execute:
        for track in tracks:
            tp = track_paths(track)
            if track.upper() == "T3":
                if not tp.get("agent_ready"):
                    print(
                        "REFUSE execute on T3: set STATUS ready_for_agent_bakeoff=true after verification.",
                        file=sys.stderr,
                    )
                    write_json(PKG / "runs" / "agents" / "plan_refused_execute.json", plan)
                    return 2
                if not _t3_preflight_gate(for_agents=False):
                    return 2
            elif track.upper() == "T1" and not tp.get("agent_ready"):
                print("REFUSE execute on T1: owner consent / allow_cloud_agents not set.", file=sys.stderr)
                return 2
        print(
            "Execute mode is gated. Sandbox copies only; run CLIs offline with fixed prompt; "
            "emit predictions.jsonl then score-preds. Automated CLI launch not enabled in this revision.",
            file=sys.stderr,
        )
        write_json(PKG / "runs" / "agents" / "plan_execute_scaffold.json", plan)
        return 2

    adapters = []
    for track in tracks:
        tp = track_paths(track)
        for backend in backends:
            adapters.append(
                {
                    "track": track.upper(),
                    "backend": backend,
                    "prompt": str(PKG / "AGENT_PACKET_PROMPT.md"),
                    "prompt_mode": "v2_observable_task_packet",
                    "manifest": str(tp["manifest"]),
                    "workspaces": str(tp.get("workspaces")),
                    "input_policy": {
                        "allowed": ["tasks.jsonl", "packet_meta.json", "AGENT_PACKET_PROMPT.md"],
                        "forbidden": ["raw workspace files", "labels.jsonl", "manifests/*", "STATUS.json"],
                        "prepare_cmd": (
                            f".venv/bin/python scripts/prepare_eval_packet.py --track {track.upper()} "
                            "--out <packet-dir>"
                        ),
                        "collect_cmd": (
                            ".venv/bin/python scripts/collect_eval_predictions.py "
                            "--packet <packet-dir> --responses <responses.jsonl> --out <predictions.jsonl>"
                        ),
                    },
                    "output_predictions": f"runs/{track.upper()}/{backend}/<stamp>/predictions.jsonl",
                    "required_fields": [
                        "task_id",
                        "ranked_folder_ids|raw_pred_folder_id",
                        "abstained",
                        "wall_ms",
                        "model",
                    ],
                    "telemetry": (cfg.get("metrics") or {}).get("systems") or [],
                    "local_zero_cost": bool(_backend_meta(backend).get("local_zero_cost")),
                    "score_cmd": (
                        f".venv/bin/python scripts/urtc_eval.py score-preds --track {track.upper()} "
                        f"--backend {backend} --predictions <predictions.jsonl>"
                    ),
                }
            )
    plan["adapters"] = adapters
    plan["note"] = "plan-only: no competitors invoked; config.consent allows T1 agents"
    out = PKG / "runs" / "agents" / "plan_latest.json"
    write_json(out, plan)
    print(json.dumps({"n_runs": plan["n_runs"], "refused": plan["refused"], "path": str(out)}, indent=2))
    for a in adapters:
        print(f"  adapter {a['track']} | {a['backend']}: {a['output_predictions']}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("status", help="Package + track readiness")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("verify-dataset", help="Link integrity for T1/T2/T3 (includes T3 preflight)")
    s.set_defaults(func=cmd_verify_dataset)

    s = sub.add_parser("preflight-t3", help="Fail-closed T3 leak/hash/dup gate")
    s.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    s.add_argument("--max-near-duplicate-pairs", type=int, default=0)
    s.add_argument("--for-agents", action="store_true")
    s.add_argument("--agent-view-root", type=Path)
    s.set_defaults(func=cmd_preflight_t3)

    s = sub.add_parser("plan", help="Build toggleable run plan (no execution)")
    s.add_argument("--profile", default=None)
    s.add_argument("--tracks", default=None)
    s.add_argument("--backends", default=None)
    s.set_defaults(func=cmd_plan)

    s = sub.add_parser("score-v2", help="Predict+score dual encoder with explicit paper policy")
    s.add_argument("--track", required=True, choices=["T1", "T2", "T3", "t1", "t2", "t3"])
    s.add_argument("--checkpoint", default=None)
    s.add_argument("--calibration", default=None)
    s.add_argument("--example-extension-fallback", action="store_true", default=False)
    s.add_argument("--no-example-extension-fallback", action="store_true", default=False)
    s.add_argument("--force", action="store_true", help="T1: recompute instead of frozen metrics")
    s.set_defaults(func=cmd_score_v2)

    s = sub.add_parser("score-preds", help="Canonicalize + score predictions JSONL")
    s.add_argument("--track", required=True, choices=["T1", "T2", "T3", "t1", "t2", "t3"])
    s.add_argument("--backend", required=True)
    s.add_argument("--predictions", required=True)
    s.add_argument("--run-metadata", default=None, help="optional aggregate telemetry JSON for this backend run")
    s.set_defaults(func=cmd_score_preds)

    s = sub.add_parser(
        "prepare-agent-view",
        help="Gold-free raw T3 view for native-file product adapters (not strict agent comparisons)",
    )
    s.add_argument("--out", default=None, help="destination root (default runs/agent_views/<stamp>)")
    s.set_defaults(func=cmd_prepare_agent_view)

    s = sub.add_parser("agents", help="Multi-track competitor plan (execute gated; default T1+T3)")
    s.add_argument("--backends", default="stateai_v2,codex,claude,agy,ollama:qwen2.5:7b,extension_rules")
    s.add_argument("--tracks", default=None, help="comma T1,T2,T3 (default T1,T3 when no profile)")
    s.add_argument("--profile", default=None)
    s.add_argument("--plan-only", action="store_true", default=True)
    s.add_argument("--execute", action="store_true")
    s.set_defaults(func=cmd_agents)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
