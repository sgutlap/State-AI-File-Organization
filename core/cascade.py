from __future__ import annotations

from dataclasses import dataclass

from core.model import Student
from core.scan import FileState

EXT_RULES = {
    ".png": "media/images",
    ".jpg": "media/images",
    ".jpeg": "media/images",
    ".gif": "media/images",
    ".webp": "media/images",
    ".svg": "media/images",
    ".fig": "media/images",
    ".psd": "media/images",
    ".ai": "media/images",
    ".mp4": "media/audio_video",
    ".mov": "media/audio_video",
    ".mkv": "media/audio_video",
    ".mp3": "media/audio_video",
    ".wav": "media/audio_video",
    ".zip": "archives",
    ".rar": "archives",
    ".7z": "archives",
    ".tgz": "archives",
    ".py": "code/projects",
    ".ipynb": "code/projects",
    ".js": "code/projects",
    ".ts": "code/projects",
    ".tsx": "code/projects",
    ".jsx": "code/projects",
    ".rs": "code/projects",
    ".go": "code/projects",
    ".java": "code/projects",
    ".cpp": "code/projects",
    ".c": "code/projects",
    ".h": "code/projects",
    ".csv": "data/datasets",
    ".parquet": "data/datasets",
    ".tsv": "data/datasets",
}

AMBIGUOUS = ("untitled", "temp", "download", "file", "document", "empty")
TEXTISH = {".py", ".js", ".ts", ".json", ".csv", ".tsv", ".md", ".txt", ".html", ".css", ".sh"}


@dataclass
class Decision:
    category: str
    confidence: float
    tier: str


class Cascade:
    def __init__(self, student: Student, threshold: float = 0.65):
        self.student = student
        self.threshold = threshold

    def decide(self, state: FileState) -> Decision:
        ext = (state.metadata.extension or "").lower()
        if ext in EXT_RULES:
            name = state.metadata.filename.lower()
            sample = (state.content_sample or "").strip()
            stub = sample.startswith(("[Binary", "[PDF", "[XLSX"))
            use_heuristic = True
            if any(x in name for x in AMBIGUOUS):
                use_heuristic = False
            elif (
                sample
                and sample != "[Empty File]"
                and not stub
                and state.metadata.size_bytes > 0
                and ext in TEXTISH
            ):
                use_heuristic = False
            if use_heuristic:
                return Decision(EXT_RULES[ext], 0.95, "heuristic")

        cat, conf = self.student.predict(state)
        return Decision(cat, float(conf), "student")
