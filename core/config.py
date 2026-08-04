"""Scan knobs for FileScanner."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ScanConfig:
    max_content_bytes: int = 4096
    max_text_chars: int = 1000
    include_content_samples: bool = True
    image_ocr: bool = False  # local opt-in only
    ignore_hidden: bool = True
    ignore_patterns: List[str] = field(
        default_factory=lambda: [
            ".git", "__pycache__", ".venv", "node_modules", ".DS_Store", ".idea", ".vscode",
            "build", "dist", ".env", ".env.*", "*.pem", "*.key", "*.secret", "id_rsa*",
            "*password*", "*credential*", "*api_key*", "*apikey*", "*access_token*",
            "*auth_token*", "*private_key*", "*wallet*",
        ]
    )
