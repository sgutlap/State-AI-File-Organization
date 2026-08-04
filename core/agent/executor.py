"""Execute move plans: dry-run default, contain paths, confirm taxonomy, rollback logs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import shutil
from typing import Callable, Dict, List, Any, Optional, Set

from rich.console import Console
from rich.table import Table

from core.agent.action_planner import MovePlan
from core.agent.state_space import FileAction, ActionType
from core.taxonomy.spec import has_unapproved_edits

console = Console()

# Common taxonomy ids offered as shortcuts when choosing a category.
_COMMON_CATEGORIES = [
    "documents/research",
    "documents/financial",
    "code/projects",
    "data/datasets",
    "media/images",
    "media/audio_video",
    "archives",
    "misc/uncategorized",
]


class PlanExecutor:
    """Executes move plans safely with confirmation and rollback logs."""

    def __init__(self, log_dir: Optional[str] = None, quarantine_dir: str = "_duplicates"):
        self.log_dir = Path(log_dir) if log_dir else Path.home() / ".core" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir = quarantine_dir

    @staticmethod
    def _inside(root: Path, candidate: Path) -> bool:
        """True only when candidate resolves inside root, including symlink checks."""
        try:
            candidate.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False

    def _validate_plan_paths(self, plan: MovePlan) -> List[str]:
        """Reject paths outside the workspace before any filesystem mutation."""
        root = Path(plan.root_dir).resolve()
        errors: List[str] = []
        if not root.is_dir():
            return [f"plan root is not a directory: {root}"]
        for directory in plan.directories_to_create:
            if not self._inside(root, Path(directory)):
                errors.append(f"directory escapes plan root: {directory}")
        for action in plan.actions:
            if action.action_type in (ActionType.SKIP, ActionType.ASK_USER, ActionType.DEDUPE_KEEP):
                continue
            if action.source_path and action.action_type not in (ActionType.MKDIR, ActionType.CREATE_DIR):
                if not self._inside(root, Path(action.source_path)):
                    errors.append(f"source escapes plan root: {action.source_path}")
            if action.target_path and not self._inside(root, Path(action.target_path)):
                errors.append(f"target escapes plan root: {action.target_path}")
        return errors

    @staticmethod
    def _ask_items(plan: MovePlan) -> List[FileAction]:
        ask = list(getattr(plan, "ask_user", None) or [])
        if ask:
            return ask
        return [a for a in plan.actions if a.action_type == ActionType.ASK_USER]

    def print_dry_run(self, plan: MovePlan) -> None:
        """Renders dry-run table of planned operations."""
        console.print(f"\n[bold cyan]=== State-AI Organize Plan Preview ===[/bold cyan]")
        console.print(f"Target Directory: [bold]{plan.root_dir}[/bold]")
        console.print(f"Directories to Create: [green]{len(plan.directories_to_create)}[/green]")
        console.print(f"Actions: [yellow]{len(plan.actions)}[/yellow]")
        console.print(f"Files Skipped: [dim]{len(plan.skipped_files)}[/dim]")
        ask_n = len(self._ask_items(plan))
        if ask_n:
            console.print(f"Ask User: [magenta]{ask_n}[/magenta]")
        proposals = list(getattr(plan, "taxonomy_proposals", []) or [])
        if proposals:
            pending = sum(1 for proposal in proposals if proposal.requires_confirmation and not proposal.approved)
            console.print(f"Taxonomy proposals: [magenta]{len(proposals)}[/magenta] (pending approval: {pending})")
        console.print()

        actionable = [
            a for a in plan.actions
            if a.action_type not in (ActionType.ASK_USER, ActionType.SKIP, ActionType.DEDUPE_KEEP)
        ]
        if actionable:
            table = Table(title="Planned File Operations", show_header=True, header_style="bold magenta")
            table.add_column("Type", style="bold")
            table.add_column("Source File", style="dim", overflow="fold")
            table.add_column("Target Category", style="cyan")
            table.add_column("Confidence", style="bold green", justify="right")
            table.add_column("Target Path", overflow="fold")

            for act in actionable[:40]:
                src_name = Path(act.source_path).name if act.source_path else "(mkdir)"
                try:
                    rel_dst = (
                        str(Path(act.target_path).relative_to(plan.root_dir))
                        if act.target_path and act.target_path.startswith(plan.root_dir)
                        else act.target_path
                    )
                except ValueError:
                    rel_dst = act.target_path
                table.add_row(
                    act.action_type.value,
                    src_name,
                    act.target_category or "-",
                    f"{act.confidence*100:.1f}%",
                    str(rel_dst),
                )

            console.print(table)
            if len(actionable) > 40:
                console.print(f"[dim]... and {len(actionable) - 40} more actions.[/dim]\n")

        ask_user = self._ask_items(plan)
        if ask_user:
            console.print("[bold magenta]Needs user input (ASK_USER):[/bold magenta]")
            for i, a in enumerate(ask_user, 1):
                name = Path(a.source_path).name if a.source_path else "(unknown)"
                hint = f"  suggested={a.target_category}" if a.target_category else ""
                console.print(f"  {i}. {name}: {a.reason}{hint}")
            console.print(
                "[dim]On --apply (without --yes): you will be prompted for each "
                "(skip / choose category / keep path). "
                "With --yes: ASK files are left unmoved (SKIP).[/dim]\n"
            )

    def resolve_ask_user(
        self,
        plan: MovePlan,
        *,
        auto_skip: bool = False,
        input_fn: Optional[Callable[[str], str]] = None,
    ) -> MovePlan:
        """
        Resolve ASK_USER actions before apply.

        auto_skip=True (--yes): convert each ASK_USER → SKIP (file left unmoved).
        Otherwise prompt stdin for each item.
        """
        ask_items = self._ask_items(plan)
        if not ask_items:
            return plan

        console.print("\n[bold magenta]=== ASK_USER — decide before applying ===[/bold magenta]")
        for i, a in enumerate(ask_items, 1):
            name = Path(a.source_path).name if a.source_path else "(unknown)"
            console.print(f"  {i}. {name}")
            console.print(f"     [dim]{a.reason}[/dim]")
            if a.target_category:
                console.print(f"     suggested category: [cyan]{a.target_category}[/cyan]")

        if auto_skip:
            console.print(
                "[yellow]--yes / auto_skip: leaving ASK_USER files unmoved (SKIP).[/yellow]"
            )
            replacements = {
                a.source_path: self._as_skip(a, "ASK_USER skipped (--yes); left unmoved")
                for a in ask_items
                if a.source_path
            }
            return self._replace_ask_actions(plan, replacements)

        read = input_fn or (lambda prompt: console.input(prompt))
        replacements: Dict[str, FileAction] = {}

        for idx, act in enumerate(ask_items, 1):
            name = Path(act.source_path).name if act.source_path else "(unknown)"
            console.print(f"\n[bold]ASK {idx}/{len(ask_items)}:[/bold] {name}")
            console.print(f"[dim]{act.reason}[/dim]")
            if act.target_category:
                console.print(f"Suggested: [cyan]{act.target_category}[/cyan]")
            console.print(
                "  [cyan](s)[/cyan]kip — leave unmoved\n"
                "  [cyan](c)[/cyan]hoose category — type a folder id\n"
                "  [cyan](k)[/cyan]eep path — leave at current location\n"
                + (
                    "  [cyan](m)[/cyan]ove suggested — use suggested category\n"
                    if act.target_category
                    else ""
                )
            )
            while True:
                ans = read("Choice [s/c/k" + ("/m" if act.target_category else "") + "]: ").strip().lower()
                if ans in ("s", "skip"):
                    replacements[act.source_path] = self._as_skip(
                        act, "User skipped ASK_USER; left unmoved"
                    )
                    break
                if ans in ("k", "keep", "keep path", "keep_path"):
                    replacements[act.source_path] = self._as_skip(
                        act, "User kept current path; left unmoved"
                    )
                    break
                if ans in ("m", "move", "suggested") and act.target_category:
                    replacements[act.source_path] = self._as_move(
                        act, plan.root_dir, act.target_category, "User accepted suggested category"
                    )
                    break
                if ans in ("c", "choose", "category"):
                    cat = self._prompt_category(read, suggested=act.target_category)
                    if cat:
                        replacements[act.source_path] = self._as_move(
                            act, plan.root_dir, cat, f"User chose category '{cat}'"
                        )
                        break
                    console.print("[yellow]No category entered; try again.[/yellow]")
                    continue
                console.print("[yellow]Enter s, c, k" + (", or m" if act.target_category else "") + ".[/yellow]")

        return self._replace_ask_actions(plan, replacements)

    @staticmethod
    def _as_skip(act: FileAction, reason: str) -> FileAction:
        return FileAction(
            action_type=ActionType.SKIP,
            source_path=act.source_path,
            target_path="",
            confidence=act.confidence,
            target_category=act.target_category or "",
            reason=reason,
            content_hash=act.content_hash,
            related_paths=list(act.related_paths),
        )

    @staticmethod
    def _as_move(act: FileAction, root_dir: str, category: str, reason: str) -> FileAction:
        name = Path(act.source_path).name
        target = str(Path(root_dir) / category / name)
        return FileAction(
            action_type=ActionType.MOVE,
            source_path=act.source_path,
            target_path=target,
            confidence=1.0,
            target_category=category,
            reason=reason,
            content_hash=act.content_hash,
            related_paths=list(act.related_paths),
        )

    @staticmethod
    def _prompt_category(read: Callable[[str], str], suggested: str = "") -> str:
        console.print("Common categories:")
        for i, cat in enumerate(_COMMON_CATEGORIES, 1):
            mark = " ← suggested" if cat == suggested else ""
            console.print(f"  {i}) {cat}{mark}")
        raw = read("Category id or number: ").strip()
        if not raw:
            return suggested or ""
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(_COMMON_CATEGORIES):
                return _COMMON_CATEGORIES[idx - 1]
        return raw

    @staticmethod
    def _replace_ask_actions(plan: MovePlan, replacements: Dict[str, FileAction]) -> MovePlan:
        if not replacements:
            return plan
        new_actions: List[FileAction] = []
        for a in plan.actions:
            if a.action_type == ActionType.ASK_USER and a.source_path in replacements:
                new_actions.append(replacements[a.source_path])
            else:
                new_actions.append(a)
        plan.actions = new_actions
        plan.ask_user = []
        # Ensure MOVE targets create their parent dirs
        dirs = set(plan.directories_to_create)
        for a in plan.actions:
            if a.action_type == ActionType.MOVE and a.target_path:
                dirs.add(str(Path(a.target_path).parent))
        plan.directories_to_create = sorted(dirs)
        return plan

    def execute(
        self,
        plan: MovePlan,
        dry_run: bool = True,
        confirm: bool = True,
        auto_skip_ask: bool = False,
        input_fn: Optional[Callable[[str], str]] = None,
    ) -> Dict[str, Any]:
        proposals = list(getattr(plan, "taxonomy_proposals", []) or [])
        if has_unapproved_edits(proposals):
            return {
                "status": "taxonomy_confirmation_required",
                "moves_executed": 0,
                "actions_executed": 0,
                "pending_taxonomy_proposal_ids": [
                    proposal.proposal_id
                    for proposal in proposals
                    if proposal.requires_confirmation and not proposal.approved
                ],
            }
        path_errors = self._validate_plan_paths(plan)
        if path_errors:
            for error in path_errors:
                console.print(f"[red]Unsafe plan rejected: {error}[/red]")
            return {
                "status": "invalid_plan",
                "moves_executed": 0,
                "actions_executed": 0,
                "errors": path_errors,
            }
        self.print_dry_run(plan)

        if dry_run:
            console.print("[bold yellow]Dry-run mode active. No physical files were changed.[/bold yellow]")
            return {"status": "dry_run", "moves_executed": 0, "actions_executed": 0}

        # Resolve ASK_USER before the overall confirm / apply
        if self._ask_items(plan):
            plan = self.resolve_ask_user(plan, auto_skip=auto_skip_ask, input_fn=input_fn)
            remaining_ask = self._ask_items(plan)
            if remaining_ask:
                # Should not happen after resolve, but safety
                console.print("[red]Unresolved ASK_USER items remain; aborting apply.[/red]")
                return {"status": "cancelled", "moves_executed": 0, "actions_executed": 0, "reason": "unresolved_ask"}

        if confirm:
            answer = console.input("\n[bold red]Execute this organize plan? (y/N): [/bold red]").strip().lower()
            if answer not in ["y", "yes"]:
                console.print("[yellow]Operation cancelled by user.[/yellow]")
                return {"status": "cancelled", "moves_executed": 0, "actions_executed": 0}

        for d in plan.directories_to_create:
            Path(d).mkdir(parents=True, exist_ok=True)

        executed_transactions = []
        failures: List[str] = []
        success_count = 0
        move_count = 0
        ask_skipped = sum(
            1
            for a in plan.actions
            if a.action_type == ActionType.SKIP
            and (
                "ASK_USER" in (a.reason or "")
                or "left unmoved" in (a.reason or "")
                or "kept current path" in (a.reason or "")
            )
        )

        # Stable order: MKDIR → DEDUPE_KEEP (noop) → RENAME/MOVE/DEDUPE_REMOVE; skip ASK/SKIP
        order = {
            ActionType.MKDIR: 0,
            ActionType.CREATE_DIR: 0,
            ActionType.DEDUPE_KEEP: 1,
            ActionType.RENAME: 2,
            ActionType.MOVE: 2,
            ActionType.BATCH_MOVE: 2,
            ActionType.DEDUPE_REMOVE: 3,
        }
        actions = sorted(
            plan.actions,
            key=lambda a: order.get(a.action_type, 9),
        )

        for act in actions:
            if act.action_type in (ActionType.ASK_USER, ActionType.SKIP, ActionType.DEDUPE_KEEP):
                continue
            if act.action_type in (ActionType.MKDIR, ActionType.CREATE_DIR):
                Path(act.target_path).mkdir(parents=True, exist_ok=True)
                executed_transactions.append({
                    "action_type": act.action_type.value,
                    "original_source": "",
                    "moved_target": act.target_path,
                    "timestamp": datetime.utcnow().isoformat(),
                })
                success_count += 1
                continue

            src = Path(act.source_path)
            dst = Path(act.target_path)
            if not src.exists():
                failures.append(f"source missing: {src}")
                console.print(f"[red]Source missing: {src}[/red]")
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)
            final_dst = self._collision_free(src, dst)

            try:
                shutil.move(str(src), str(final_dst))
                executed_transactions.append({
                    "action_type": act.action_type.value,
                    "original_source": str(src),
                    "moved_target": str(final_dst),
                    "timestamp": datetime.utcnow().isoformat(),
                })
                success_count += 1
                if act.action_type in (ActionType.MOVE, ActionType.BATCH_MOVE, ActionType.RENAME, ActionType.DEDUPE_REMOVE):
                    move_count += 1
            except Exception as e:
                message = f"Error on {act.action_type.value} {src.name}: {e}"
                failures.append(message)
                console.print(f"[red]{message}[/red]")

        log_file = self.log_dir / f"transaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "version": 2,
                    "root_dir": str(Path(plan.root_dir).resolve()),
                    "transactions": executed_transactions,
                },
                f,
                indent=2,
            )

        console.print(f"[bold green]Successfully applied {success_count} actions ({move_count} file moves/renames).[/bold green]")
        if ask_skipped:
            console.print(f"[dim]ASK_USER left unmoved: {ask_skipped}[/dim]")

        pruned = self._prune_empty_dirs(Path(plan.root_dir))
        if pruned:
            console.print(f"[dim]Pruned {pruned} empty directories left behind by moves.[/dim]")

        console.print(f"Transaction log written to: [dim]{log_file}[/dim]")

        return {
            "status": "partial_failure" if failures else "success",
            "moves_executed": move_count,
            "actions_executed": success_count,
            "ask_skipped": ask_skipped,
            "empty_dirs_pruned": pruned,
            "log_file": str(log_file),
            "errors": failures,
        }

    def _prune_empty_dirs(self, root: Path) -> int:
        """Remove empty directories left after moves (bottom-up). Skip system/hidden shells."""
        if not root.is_dir():
            return 0
        skip = {".git", ".Trashes", ".fseventsd", ".hidden", "__MACOSX", self.quarantine_dir}
        removed = 0
        dirs = sorted(
            (p for p in root.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        )
        for d in dirs:
            if d.name in skip:
                continue
            try:
                next(d.iterdir())
            except StopIteration:
                try:
                    d.rmdir()
                    removed += 1
                except OSError:
                    pass
            except OSError:
                pass
        return removed

    def rollback(self, transaction_log_file: str, root_dir: Optional[str] = None) -> int:
        """Rolls back a previously executed move plan using its transaction log."""
        log_path = Path(transaction_log_file)
        if not log_path.exists():
            raise FileNotFoundError(f"Transaction log file not found: {transaction_log_file}")

        with open(log_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if isinstance(payload, dict):
            records = payload.get("transactions") or []
            logged_root = payload.get("root_dir")
        else:
            # Old logs have no trustworthy root. Require an explicit workspace.
            records = payload
            logged_root = root_dir
        if not logged_root:
            raise ValueError("rollback requires a version-2 transaction log or explicit root_dir")
        root = Path(logged_root).resolve()

        restored_count = 0
        for rec in reversed(records):
            if rec.get("action_type") in ("MKDIR", "CREATE_DIR"):
                continue
            moved_target = Path(rec["moved_target"])
            original_source = Path(rec["original_source"])
            if not original_source.as_posix() or original_source.as_posix() == ".":
                continue

            if not self._inside(root, moved_target) or not self._inside(root, original_source):
                raise ValueError("rollback record escapes its transaction root")
            if moved_target.exists():
                original_source.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(moved_target), str(original_source))
                restored_count += 1

        console.print(f"[bold green]Rollback complete: {restored_count} files restored to original locations.[/bold green]")
        return restored_count

    @staticmethod
    def _collision_free(src: Path, dst: Path) -> Path:
        final_dst = dst
        counter = 1
        while final_dst.exists() and final_dst.resolve() != src.resolve():
            final_dst = dst.parent / f"{dst.stem}_{counter}{dst.suffix}"
            counter += 1
        return final_dst
