"""
File Scanner Module.
Scans workspace directories and constructs structured FileState representations.
"""

from dataclasses import dataclass, field
from fnmatch import fnmatch
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional

from core.config import ScanConfig
from core.extractors.content_extractor import ContentExtractor, FileMetadata


@dataclass
class FileState:
    file_id: str                          # MD5 hash of path
    absolute_path: str
    relative_path: str
    metadata: FileMetadata
    content_sample: str
    current_class: Optional[str] = None   # Current assigned class if known
    target_class: Optional[str] = None    # Predicted or synthetic target class
    teacher_probs: Optional[List[float]] = None # KD teacher soft labels

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "absolute_path": self.absolute_path,
            "relative_path": self.relative_path,
            "metadata": self.metadata.to_dict(),
            "content_sample": self.content_sample,
            "current_class": self.current_class,
            "target_class": self.target_class,
            "teacher_probs": self.teacher_probs,
        }

    def format_state_string(self) -> str:
        """Formats the file state into a concise text representation for LLM prompt ingestion."""
        meta = self.metadata
        return (
            f"File: {meta.filename}\n"
            f"Path: {self.relative_path}\n"
            f"Extension: {meta.extension} | Size: {meta.size_bytes} bytes | Age: {meta.age_days} days\n"
            f"MIME: {meta.mime_type} | Binary: {meta.is_binary}\n"
            f"Content Sample:\n---\n{self.content_sample}\n---"
        )


class FileScanner:
    """Recursively scans workspace directories for candidate files."""

    def __init__(self, config: Optional[ScanConfig] = None):
        self.config = config or ScanConfig()
        self.extractor = ContentExtractor(
            max_bytes=self.config.max_content_bytes,
            max_chars=self.config.max_text_chars,
            image_ocr=self.config.image_ocr,
        )

    def scan_directory(self, target_dir: str) -> List[FileState]:
        root_path = Path(target_dir).resolve()
        if not root_path.exists() or not root_path.is_dir():
            raise ValueError(f"Target directory does not exist or is not a directory: {target_dir}")

        file_states: List[FileState] = []

        for p in root_path.rglob("*"):
            if not p.is_file():
                continue

            # Ignore hidden files / ignore patterns
            rel_parts = p.relative_to(root_path).parts
            if self.config.ignore_hidden and any(part.startswith(".") for part in rel_parts):
                continue

            if any(
                fnmatch(part, pattern)
                for part in rel_parts
                for pattern in self.config.ignore_patterns
            ):
                continue

            # Generate stable ID
            file_id = hashlib.md5(str(p).encode("utf-8")).hexdigest()[:12]
            rel_path = str(p.relative_to(root_path))

            metadata = self.extractor.extract_metadata(p, root_path=root_path)
            content_sample = self.extractor.extract_sample(p, metadata) if self.config.include_content_samples else ""

            state = FileState(
                file_id=file_id,
                absolute_path=str(p),
                relative_path=rel_path,
                metadata=metadata,
                content_sample=content_sample,
            )
            file_states.append(state)

        return file_states
