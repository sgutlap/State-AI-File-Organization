"""Safety invariants for filesystem mutation and transaction logs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.agent.action_planner import MovePlan
from core.agent.executor import PlanExecutor
from core.agent.state_space import ActionType, FileAction


def _move(src: Path, dst: Path) -> FileAction:
    return FileAction(
        action_type=ActionType.MOVE,
        source_path=str(src),
        target_path=str(dst),
        confidence=1.0,
        target_category="test",
        reason="test",
    )


class ExecutorSafetyTests(unittest.TestCase):
    def test_execute_rejects_target_outside_root(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "workspace"
            root.mkdir()
            source = root / "inside.txt"
            source.write_text("safe", encoding="utf-8")
            outside = Path(td) / "outside.txt"
            plan = MovePlan(root_dir=str(root), actions=[_move(source, outside)])
            result = PlanExecutor(log_dir=str(root / "logs")).execute(
                plan, dry_run=False, confirm=False
            )
            self.assertEqual(result["status"], "invalid_plan")
            self.assertTrue(source.exists())
            self.assertFalse(outside.exists())


    def test_execute_reports_missing_source_as_partial_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = MovePlan(root_dir=str(root), actions=[_move(root / "gone.txt", root / "x" / "gone.txt")])
            result = PlanExecutor(log_dir=str(root / "logs")).execute(
                plan, dry_run=False, confirm=False
            )
            self.assertEqual(result["status"], "partial_failure")
            self.assertEqual(result["errors"], [f"source missing: {root / 'gone.txt'}"])


    def test_v2_rollback_rejects_tampered_escape_record(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "workspace"
            root.mkdir()
            target = root / "moved.txt"
            target.write_text("x", encoding="utf-8")
            log = root / "transaction.json"
            log.write_text(json.dumps({"version": 2, "root_dir": str(root), "transactions": [{
                "action_type": "MOVE", "original_source": str(Path(td) / "outside.txt"),
                "moved_target": str(target),
            }]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes"):
                PlanExecutor(log_dir=str(root / "logs")).rollback(str(log))
