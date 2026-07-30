from __future__ import annotations

import logging
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from core.scan import FileState
from core.taxonomy import Taxonomy

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("transformers").setLevel(logging.ERROR)

BASE_MODEL = "distilbert-base-uncased"


def _hf(loader, name, **kw):
    """Prefer local files; fall back to hub once if needed."""
    try:
        return loader(name, local_files_only=True, **kw)
    except Exception:
        return loader(name, local_files_only=False, **kw)


def _load_tokenizer(base_model: str, ckpt: Path | None):
    if ckpt is not None and (ckpt / "tokenizer.json").exists():
        try:
            return AutoTokenizer.from_pretrained(str(ckpt), local_files_only=True)
        except Exception:
            pass
    return _hf(AutoTokenizer.from_pretrained, base_model)


def _load_config(base_model: str, ckpt: Path | None):
    if ckpt is not None and (ckpt / "config.json").exists():
        try:
            return AutoConfig.from_pretrained(str(ckpt), local_files_only=True)
        except Exception:
            pass
    return _hf(AutoConfig.from_pretrained, base_model)


class TabularEncoder(nn.Module):
    def __init__(self, num_extensions=100, embed_dim=64):
        super().__init__()
        self.ext_embedding = nn.Embedding(num_extensions, 32, padding_idx=0)
        self.continuous_mlp = nn.Sequential(
            nn.Linear(4, 32), nn.LayerNorm(32), nn.ReLU(), nn.Linear(32, 32),
        )
        self.fusion_projection = nn.Sequential(
            nn.Linear(64, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU(),
        )

    def forward(self, ext_indices, continuous_feats):
        ext = self.ext_embedding(ext_indices)
        cont = self.continuous_mlp(continuous_feats)
        return self.fusion_projection(torch.cat([ext, cont], dim=-1))


class StudentNet(nn.Module):
    def __init__(self, config, num_classes, tabular_embed_dim=64, dropout=0.1):
        super().__init__()
        # from_config = architecture only, no hub download (weights come from student_model.pt)
        self.transformer = AutoModel.from_config(config)
        self.tabular_encoder = TabularEncoder(embed_dim=tabular_embed_dim)
        hidden = config.hidden_size + tabular_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def _text(self, input_ids, attention_mask):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return (
            out.pooler_output
            if getattr(out, "pooler_output", None) is not None
            else out.last_hidden_state[:, 0, :]
        )

    def fused(self, input_ids, attention_mask, ext_indices, continuous_feats):
        text = self._text(input_ids, attention_mask)
        tab = self.tabular_encoder(ext_indices, continuous_feats)
        return torch.cat([text, tab], dim=-1)

    def forward(self, input_ids, attention_mask, ext_indices, continuous_feats):
        return self.classifier(
            self.fused(input_ids, attention_mask, ext_indices, continuous_feats)
        )


COMMON_EXTENSIONS = [
    "<PAD>", ".pdf", ".tex", ".bib", ".txt", ".md", ".csv", ".xlsx", ".json", ".jsonl",
    ".parquet", ".py", ".ipynb", ".sh", ".js", ".ts", ".html", ".css", ".cpp", ".c", ".h",
    ".java", ".rs", ".go", ".png", ".jpg", ".jpeg", ".svg", ".gif", ".mp4", ".mov", ".mp3",
    ".wav", ".zip", ".tar.gz", ".tgz", ".rar", ".7z", ".tmp", ".log", ".bak",
]


class Student:
    def __init__(self, taxonomy=None, base_model=BASE_MODEL, ckpt: str | Path | None = None):
        self.taxonomy = taxonomy or Taxonomy()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.label2idx = self.taxonomy.label_to_idx()
        self.idx2label = self.taxonomy.idx_to_label()
        self.ext2idx = {e: i for i, e in enumerate(COMMON_EXTENSIONS)}
        ckpt_path = Path(ckpt) if ckpt else None
        try:
            cfg = _load_config(base_model, ckpt_path)
            self.tokenizer = _load_tokenizer(base_model, ckpt_path)
            self.model = StudentNet(cfg, self.taxonomy.num_classes)
        except Exception as exc:
            raise RuntimeError(
                "Could not build DistilBERT student.\n"
                "Fix: ensure artifacts/student_model_ckpt/ has config.json + tokenizer files,\n"
                "or run once online so Hugging Face can cache distilbert-base-uncased.\n"
                f"Original error: {exc}"
            ) from exc
        self.model.to(self.device)
        if ckpt_path and (ckpt_path / "student_model.pt").exists():
            self.load(str(ckpt_path))

    def preprocess(self, states: list[FileState]) -> dict:
        texts = [
            f"Path: {s.relative_path}\nFile: {s.metadata.filename}\n"
            f"Extension: {s.metadata.extension}\nContent: {s.content_sample}"
            for s in states
        ]
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
        ext_idx, cont, targets = [], [], []
        for s in states:
            ext_idx.append(self.ext2idx.get(s.metadata.extension.lower(), 0))
            cont.append([
                math.log1p(max(0, s.metadata.size_bytes)) / 20.0,
                math.log1p(max(0, s.metadata.age_days)) / 10.0,
                min(1.0, s.metadata.depth / 10.0),
                1.0 if s.metadata.is_binary else 0.0,
            ])
            if s.target_class and s.target_class in self.label2idx:
                targets.append(self.label2idx[s.target_class])
            else:
                targets.append(-1)
        return {
            "input_ids": enc["input_ids"].to(self.device),
            "attention_mask": enc["attention_mask"].to(self.device),
            "ext_indices": torch.tensor(ext_idx, dtype=torch.long, device=self.device),
            "continuous_feats": torch.tensor(cont, dtype=torch.float32, device=self.device),
            "targets": torch.tensor(targets, dtype=torch.long, device=self.device),
        }

    def predict_probs(self, state: FileState):
        """Softmax over the locked KD taxonomy head (fixed num_classes)."""
        self.model.eval()
        with torch.no_grad():
            batch = self.preprocess([state])
            logits = self.model(
                batch["input_ids"], batch["attention_mask"],
                batch["ext_indices"], batch["continuous_feats"],
            )
            probs = F.softmax(logits, dim=-1)[0].cpu()
        return {self.idx2label[i]: float(probs[i]) for i in range(probs.numel())}

    def predict(self, state: FileState):
        probs = self.predict_probs(state)
        label = max(probs, key=probs.get)
        return label, float(probs[label])

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """L2-normalized DistilBERT CLS vectors — used for open-vocab folder routing."""
        self.model.eval()
        with torch.no_grad():
            enc = self.tokenizer(
                texts, padding=True, truncation=True, max_length=64, return_tensors="pt"
            )
            ids = enc["input_ids"].to(self.device)
            mask = enc["attention_mask"].to(self.device)
            vec = self.model._text(ids, mask)
            return F.normalize(vec, dim=-1).cpu()

    def encode_file(self, state: FileState) -> torch.Tensor:
        """L2-normalized fused file vector (text CLS + tabular) for custom-folder similarity."""
        self.model.eval()
        with torch.no_grad():
            batch = self.preprocess([state])
            fused = self.model.fused(
                batch["input_ids"], batch["attention_mask"],
                batch["ext_indices"], batch["continuous_feats"],
            )
            text = fused[:, : self.model.transformer.config.hidden_size]
            return F.normalize(text, dim=-1)[0].cpu()

    def save(self, path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), p / "student_model.pt")
        self.tokenizer.save_pretrained(str(p))
        self.model.transformer.config.save_pretrained(str(p))

    def load(self, path: str):
        weight = Path(path) / "student_model.pt"
        if not weight.exists():
            raise FileNotFoundError(
                f"missing weights: {weight}\n"
                "student_model.pt is ~250MB and gitignored — copy it into artifacts/student_model_ckpt/"
            )
        try:
            state = torch.load(weight, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(weight, map_location=self.device)
        self.model.load_state_dict(state)
