"""
Content Extractor Module.
Extracts structured file metadata (tabular features) and representative content samples (text features).
"""

from dataclasses import dataclass
import mimetypes
import os
import struct
import subprocess
import tarfile
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import time
from typing import Dict, Any, Optional


@dataclass
class FileMetadata:
    filename: str
    extension: str
    size_bytes: int
    modified_time_iso: str
    age_days: float
    depth: int
    mime_type: str
    is_binary: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "modified_time_iso": self.modified_time_iso,
            "age_days": self.age_days,
            "depth": self.depth,
            "mime_type": self.mime_type,
            "is_binary": self.is_binary,
        }


class ContentExtractor:
    """Extracts text content snippets and tabular features from files."""

    def __init__(self, max_bytes: int = 4096, max_chars: int = 1000, image_ocr: bool = False):
        self.max_bytes = max_bytes
        self.max_chars = max_chars
        self.image_ocr = image_ocr

    def _extract_image_ocr(self, file_path: Path) -> str:
        """Return bounded local OCR text; failure keeps normal opaque handling."""
        if not self.image_ocr:
            return ""
        try:
            result = subprocess.run(
                ["tesseract", str(file_path), "stdout", "--psm", "6"],
                capture_output=True, text=True, timeout=10,
            )
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return ""
        text = "\n".join(line.strip() for line in result.stdout.splitlines() if line.strip())
        if result.returncode or len(text) < 12:
            return ""
        # Keep the representation modality-neutral: downstream semantic models
        # should learn from visible words, not from a hidden image/PDF type cue.
        return f"[OCR Text]\n{text[:self.max_chars]}"

    def _extract_pdf_ocr(self, file_path: Path) -> str:
        """OCR page one of a PDF when text extraction is unavailable.

        This is local-only and opt-in with ``image_ocr``.  It intentionally
        limits work to one moderate-resolution page so a folder scan cannot
        turn into an unbounded document-processing job.
        """
        if not self.image_ocr:
            return ""
        try:
            with tempfile.TemporaryDirectory(prefix="state_ai_pdf_") as directory:
                output = Path(directory) / "page"
                rendered = subprocess.run(
                    ["pdftoppm", "-f", "1", "-l", "1", "-r", "150", "-png", "-singlefile", str(file_path), str(output)],
                    capture_output=True, text=True, timeout=15,
                )
                image = output.with_suffix(".png")
                if rendered.returncode or not image.is_file():
                    return ""
                return self._extract_image_ocr(image)
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            return ""

    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract bounded embedded PDF text without shelling out when available."""
        try:
            with open(file_path, "rb") as stream:
                if not stream.read(5).startswith(b"%PDF-"):
                    return ""
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            chunks = []
            for page in reader.pages:
                chunks.append(page.extract_text() or "")
                if sum(len(chunk) for chunk in chunks) >= self.max_chars:
                    break
            text = "\n".join(chunk.strip() for chunk in chunks if chunk.strip()).strip()
        except (ImportError, OSError, ValueError, KeyError):
            return ""
        if not text:
            return ""
        return f"[Extracted Text]\n{text[:self.max_chars]}"

    def extract_metadata(self, file_path: Path, root_path: Optional[Path] = None) -> FileMetadata:
        file_path = Path(file_path)
        stat = file_path.stat()
        
        filename = file_path.name
        extension = file_path.suffix.lower()
        size_bytes = stat.st_size
        mod_time = stat.st_mtime
        mod_time_iso = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(mod_time))
        age_days = round((time.time() - mod_time) / (24 * 3600), 2)

        if root_path:
            try:
                rel_path = file_path.relative_to(root_path)
                depth = len(rel_path.parts) - 1
            except ValueError:
                depth = len(file_path.parts)
        else:
            depth = len(file_path.parts)

        mime, _ = mimetypes.guess_type(str(file_path))
        mime_type = mime or "application/octet-stream"

        is_binary = self._is_binary_file(file_path, extension, mime_type)

        return FileMetadata(
            filename=filename,
            extension=extension,
            size_bytes=size_bytes,
            modified_time_iso=mod_time_iso,
            age_days=age_days,
            depth=depth,
            mime_type=mime_type,
            is_binary=is_binary,
        )

    def extract_sample(self, file_path: Path, metadata: FileMetadata) -> str:
        """Extracts a clean text sample from the file."""
        if metadata.is_binary:
            try:
                mime = metadata.mime_type

                # Office formats are ZIP containers. Extract their structured
                # text before falling back to an opaque binary preview.
                if metadata.extension in {".docx", ".xlsx", ".pptx"}:
                    extracted = self._extract_ooxml(file_path, metadata.extension)
                    if extracted:
                        return extracted
                
                # 1. PDFs
                if mime == "application/pdf" or metadata.extension == ".pdf":
                    extracted = self._extract_pdf_text(file_path)
                    if extracted:
                        return extracted
                    try:
                        result = subprocess.run(
                            ["pdftotext", str(file_path), "-"],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            text = result.stdout.strip()
                            if len(text) > self.max_chars:
                                text = text[: self.max_chars] + "..."
                            return f"[Extracted Text]\n{text}"
                    except (subprocess.SubprocessError, FileNotFoundError, OSError):
                        pass
                    ocr = self._extract_pdf_ocr(file_path)
                    if ocr:
                        return ocr

                # 2. ZIP/TAR archives
                if metadata.extension == ".zip" or mime == "application/zip":
                    try:
                        with zipfile.ZipFile(file_path, "r") as zf:
                            infos = zf.infolist()[:20]
                            lines = [f"{i.filename} ({i.file_size} bytes)" for i in infos]
                            content = "\n".join(lines)
                            if len(zf.infolist()) > 20:
                                content += "\n... and more"
                            return f"[ZIP Archive Contents]\n{content}"
                    except zipfile.BadZipFile:
                        pass
                elif metadata.extension in {".tar", ".gz", ".tgz", ".bz2", ".xz"} or "tar" in mime:
                    try:
                        with tarfile.open(file_path, "r") as tf:
                            members = tf.getmembers()
                            lines = [f"{m.name} ({m.size} bytes)" for m in members[:20]]
                            content = "\n".join(lines)
                            if len(members) > 20:
                                content += "\n... and more"
                            return f"[TAR Archive Contents]\n{content}"
                    except tarfile.TarError:
                        pass

                # 3. Images (JPEG/PNG dimensions)
                if mime.startswith("image/"):
                    ocr = self._extract_image_ocr(file_path)
                    if ocr:
                        return ocr
                    try:
                        with open(file_path, "rb") as f:
                            data = f.read(24)
                            if data.startswith(b"\x89PNG\r\n\x1a\n"):
                                w, h = struct.unpack(">LL", data[16:24])
                                return f"[PNG Image, Size: {metadata.size_bytes} bytes, Dimensions: {w}x{h}]"
                            elif data.startswith(b"\xff\xd8"):
                                f.seek(0)
                                b = f.read(4096)
                                i = 2
                                while i < len(b) - 8:
                                    if b[i] == 0xFF:
                                        marker = b[i+1]
                                        if 0xC0 <= marker <= 0xC3:
                                            h, w = struct.unpack(">HH", b[i+5:i+9])
                                            return f"[JPEG Image, Size: {metadata.size_bytes} bytes, Dimensions: {w}x{h}]"
                                        else:
                                            length = struct.unpack(">H", b[i+2:i+4])[0]
                                            i += 2 + length
                                    else:
                                        break
                    except Exception:
                        pass

                # 4. Fallback: hex dump
                try:
                    with open(file_path, "rb") as f:
                        head = f.read(64)
                    hex_dump = " ".join(f"{b:02x}" for b in head)
                    return f"[Binary File: {metadata.mime_type}, Size: {metadata.size_bytes} bytes, Extension: {metadata.extension}]\nHex dump: {hex_dump}"
                except Exception as e:
                    return f"[Binary File Error: {str(e)}]"
            except Exception as e:
                return f"[Binary File Error: {str(e)}]"

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(self.max_bytes)

            # Clean and truncate text
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            clean_text = "\n".join(lines[:30]) # Top 30 non-empty lines
            if len(clean_text) > self.max_chars:
                clean_text = clean_text[: self.max_chars] + "..."

            return clean_text if clean_text else "[Empty File]"
        except Exception as e:
            return f"[Extraction Error: {str(e)}]"

    def _is_binary_file(self, file_path: Path, extension: str, mime_type: str) -> bool:
        text_extensions = {
            ".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".tsv",
            ".html", ".css", ".js", ".ts", ".c", ".cpp", ".h", ".hpp", ".java",
            ".sh", ".tex", ".bib", ".rs", ".go", ".sql", ".xml", ".ini", ".log"
        }
        if extension in text_extensions:
            return False
        
        container_extensions = {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp"}
        if extension in container_extensions:
            return True
        binary_prefixes = ("image/", "video/", "audio/", "application/zip", "application/x-rar", "application/pdf")
        if any(mime_type.startswith(p) for p in binary_prefixes):
            return True
        if mime_type.startswith("application/"):
            return True
        try:
            with open(file_path, "rb") as f:
                return b"\x00" in f.read(1024)
        except OSError:
            return True

    def _extract_ooxml(self, file_path: Path, extension: str) -> str:
        """Extract bounded readable text from Word, Excel, and PowerPoint ZIPs."""
        try:
            with zipfile.ZipFile(file_path) as archive:
                names = archive.namelist()
                if extension == ".docx":
                    candidates = ["word/document.xml"]
                elif extension == ".pptx":
                    candidates = sorted(name for name in names if name.startswith("ppt/slides/slide") and name.endswith(".xml"))
                else:
                    candidates = [name for name in ("xl/sharedStrings.xml",) if name in names]
                    candidates.extend(sorted(name for name in names if name.startswith("xl/worksheets/") and name.endswith(".xml")))
                # Some small exporters use a readable content file instead of
                # complete Office XML; preserve that local semantic content.
                candidates.extend(name for name in names if name.lower().endswith(".txt"))
                paragraphs = []
                for name in dict.fromkeys(candidates):
                    raw = archive.read(name)
                    try:
                        root = ET.fromstring(raw)
                        text = " ".join(
                            (node.text or "").strip() for node in root.iter()
                            if node.tag.endswith("}t") and (node.text or "").strip()
                        )
                    except ET.ParseError:
                        text = raw.decode("utf-8", errors="ignore").strip()
                    if text:
                        paragraphs.append(text)
                    if len("\n".join(paragraphs)) >= self.max_chars:
                        break
            sample = "\n".join(paragraphs)
            if len(sample) > self.max_chars:
                sample = sample[: self.max_chars] + "..."
            return f"[Office Content]\n{sample}" if sample else ""
        except (ET.ParseError, KeyError, OSError, zipfile.BadZipFile):
            return ""
