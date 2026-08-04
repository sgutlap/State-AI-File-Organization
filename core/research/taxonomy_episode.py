"""Collection-level contracts for open-taxonomy induction experiments.

An episode is a complete workspace dump, its editable taxonomy, and only the
destination decisions used for supervised development.  It prevents reducing
taxonomy induction to unrelated file-folder pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from core.research.taxonomy_dataset import DatasetSplit, TaxonomyRankingTask
from core.taxonomy.spec import TaxonomySpec


@dataclass(frozen=True)
class TaxonomyEpisode:
    episode_id: str
    source_group_id: str
    split: DatasetSplit
    taxonomy: TaxonomySpec
    tasks: tuple[TaxonomyRankingTask, ...]

    def __post_init__(self) -> None:
        if len(self.tasks) < 2:
            raise ValueError("an induction episode needs at least two files")
        if {task.source_group_id for task in self.tasks} != {self.source_group_id}:
            raise ValueError("episode tasks must have one source group")
        if {task.split for task in self.tasks} != {self.split}:
            raise ValueError("episode tasks must have one split")
        if any(task.taxonomy.to_dict() != self.taxonomy.to_dict() for task in self.tasks):
            raise ValueError("episode tasks must have one taxonomy")
        file_ids = [task.file_id for task in self.tasks]
        if len(file_ids) != len(set(file_ids)):
            raise ValueError("episode tasks must have unique file ids")

    @property
    def non_abstain_tasks(self) -> tuple[TaxonomyRankingTask, ...]:
        return tuple(task for task in self.tasks if not task.abstain)

    @property
    def oracle_assignments(self) -> Mapping[str, str]:
        return {task.file_id: task.acceptable_folder_ids[0] for task in self.non_abstain_tasks}


def episodes_from_tasks(tasks: Sequence[TaxonomyRankingTask]) -> list[TaxonomyEpisode]:
    """Group tasks by source and taxonomy; fail if an episode mixes taxonomies."""
    groups: dict[tuple[str, str, DatasetSplit], list[TaxonomyRankingTask]] = {}
    for task in tasks:
        groups.setdefault((task.source_group_id, task.taxonomy_id, task.split), []).append(task)
    episodes = []
    for (source_group_id, taxonomy_id, split), group in sorted(groups.items()):
        if len(group) < 2:
            continue
        taxonomy = group[0].taxonomy
        episodes.append(TaxonomyEpisode(
            episode_id=f"{source_group_id}:{taxonomy_id}",
            source_group_id=source_group_id,
            split=split,
            taxonomy=taxonomy,
            tasks=tuple(group),
        ))
    return episodes
