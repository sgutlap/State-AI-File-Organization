"""Action enums + FileAction for plans."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionType(str, Enum):
    MOVE = "MOVE"
    RENAME = "RENAME"
    MKDIR = "MKDIR"
    DEDUPE_KEEP = "DEDUPE_KEEP"
    DEDUPE_REMOVE = "DEDUPE_REMOVE"
    SKIP = "SKIP"
    ASK_USER = "ASK_USER"
    CREATE_DIR = "CREATE_DIR"
    BATCH_MOVE = "BATCH_MOVE"


@dataclass
class FileAction:
    action_type: ActionType
    source_path: str
    target_path: str
    confidence: float
    target_category: str
    reason: str
    content_hash: Optional[str] = None
    related_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "source_path": self.source_path,
            "target_path": self.target_path,
            "confidence": self.confidence,
            "target_category": self.target_category,
            "reason": self.reason,
            "content_hash": self.content_hash,
            "related_paths": list(self.related_paths),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileAction":
        return cls(
            action_type=ActionType(data["action_type"]),
            source_path=data["source_path"],
            target_path=data.get("target_path", ""),
            confidence=float(data.get("confidence", 0.0)),
            target_category=data.get("target_category", ""),
            reason=data.get("reason", ""),
            content_hash=data.get("content_hash"),
            related_paths=list(data.get("related_paths") or []),
        )
