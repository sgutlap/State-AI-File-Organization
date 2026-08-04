"""Move plan structure for open-taxonomy execution.

V2 retains only the plan dataclass. Fixed-taxonomy ActionPlanner lives in
Baselines/old/v1 and is not part of the open-taxonomy runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from core.agent.state_space import ActionType, FileAction
from core.extractors.file_scanner import FileState
from core.taxonomy.spec import TaxonomyProposal


@dataclass
class MovePlan:
    root_dir: str
    actions: List[FileAction] = field(default_factory=list)
    directories_to_create: List[str] = field(default_factory=list)
    skipped_files: List[FileState] = field(default_factory=list)
    ask_user: List[FileAction] = field(default_factory=list)
    taxonomy_proposals: List[TaxonomyProposal] = field(default_factory=list)

    def summary(self) -> Dict[str, int]:
        return {
            "total_moves": len([a for a in self.actions if a.action_type == ActionType.MOVE]),
            "total_actions": len(self.actions),
            "directories_to_create": len(self.directories_to_create),
            "skipped_files": len(self.skipped_files),
            "ask_user": len(self.ask_user),
        }

    @classmethod
    def from_organize_plan(cls, organize_plan) -> "MovePlan":
        return cls(
            root_dir=organize_plan.root_dir,
            actions=list(organize_plan.actions),
            directories_to_create=list(organize_plan.directories_to_create),
            skipped_files=list(organize_plan.skipped_files),
            ask_user=list(getattr(organize_plan, "ask_user", []) or []),
            taxonomy_proposals=list(getattr(organize_plan, "taxonomy_proposals", []) or []),
        )
