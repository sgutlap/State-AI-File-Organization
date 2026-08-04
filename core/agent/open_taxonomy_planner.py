"""Plan open-taxonomy moves from ranking scorer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from core.agent.action_planner import MovePlan
from core.agent.state_space import ActionType, FileAction
from core.extractors.file_scanner import FileState
from core.taxonomy.discovery import TaxonomyDiscoverer
from core.taxonomy.spec import TaxonomySpec


class OpenTaxonomyPlanner:
    def __init__(self, scorer: Any, taxonomy: TaxonomySpec):
        self.scorer = scorer
        self.taxonomy = taxonomy

    def plan(
        self,
        root_dir: str,
        file_states: List[FileState],
        *,
        discoverer: Optional[TaxonomyDiscoverer] = None,
    ) -> MovePlan:
        root = Path(root_dir).resolve()
        plan = MovePlan(root_dir=str(root))
        needed, unresolved = set(), []
        rank_many = getattr(self.scorer, "rank_many", None)
        choices = (
            rank_many(file_states, self.taxonomy)
            if callable(rank_many)
            else [self.scorer.rank(state, self.taxonomy) for state in file_states]
        )
        if len(choices) != len(file_states):
            raise ValueError("taxonomy scorer returned a different number of choices than file states")
        for state, choice in zip(file_states, choices):
            if choice.abstained or not choice.folder_id:
                unresolved.append(state)
                plan.ask_user.append(
                    FileAction(
                        action_type=ActionType.ASK_USER,
                        source_path=state.absolute_path,
                        target_path="",
                        confidence=round(choice.score, 4),
                        target_category="",
                        reason=(
                            f"Open-taxonomy {choice.source} scorer abstained "
                            f"(score={choice.score:.3f}, margin={choice.margin:.3f}); "
                            "leave unmoved or choose a folder."
                        ),
                    )
                )
                continue
            target_dir = root / choice.folder_id
            target_path = target_dir / Path(state.absolute_path).name
            if Path(state.absolute_path).parent.resolve() == target_dir.resolve():
                plan.skipped_files.append(state)
                continue
            needed.add(str(target_dir))
            plan.actions.append(
                FileAction(
                    action_type=ActionType.MOVE,
                    source_path=state.absolute_path,
                    target_path=str(target_path),
                    confidence=round(choice.score, 4),
                    target_category=choice.folder_id,
                    reason=(
                        f"Taxonomy-conditioned {choice.source} scorer selected "
                        f"'{choice.folder_id}' (score={choice.score:.3f}, margin={choice.margin:.3f})."
                    ),
                )
            )
        plan.directories_to_create = sorted(needed)
        if discoverer is not None and unresolved:
            plan.taxonomy_proposals = discoverer.discover(unresolved, self.taxonomy)
        return plan
