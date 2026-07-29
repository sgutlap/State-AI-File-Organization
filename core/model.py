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


def _hf(loader, name, **kw):
    try:
        return loader(name, local_files_only=True, **kw)
    except Exception:
        return loader(name, local_files_only=False, **kw)


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
    def __init__(self, base_model_name, num_classes, tabular_embed_dim=64, dropout=0.1):
        super().__init__()
        cfg = _hf(AutoConfig.from_pretrained, base_model_name)
        self.transformer = _hf(AutoModel.from_pretrained, base_model_name, config=cfg)
        self.tabular_encoder = TabularEncoder(embed_dim=tabular_embed_dim)
        hidden = cfg.hidden_size + tabular_embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask, ext_indices, continuous_feats):
        out = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        text = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0, :]
        tab = self.tabular_encoder(ext_indices, continuous_feats)
        return self.classifier(torch.cat([text, tab], dim=-1))


# order locked to the trained checkpoint — don't reshuffle
COMMON_EXTENSIONS = [
    "<PAD>", ".pdf", ".tex", ".bib", ".txt", ".md", ".csv", ".xlsx", ".json", ".jsonl",
    ".parquet", ".py", ".ipynb", ".sh", ".js", ".ts", ".html", ".css", ".cpp", ".c", ".h",
    ".java", ".rs", ".go", ".png", ".jpg", ".jpeg", ".svg", ".gif", ".mp4", ".mov", ".mp3",
    ".wav", ".zip", ".tar.gz", ".tgz", ".rar", ".7z", ".tmp", ".log", ".bak",
]


class Student:
    def __init__(self, taxonomy=None, base_model="distilbert-base-uncased"):
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
        self.tokenizer = _hf(AutoTokenizer.from_pretrained, base_model)
        self.model = StudentNet(base_model, self.taxonomy.num_classes)
        self.model.to(self.device)

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

    def predict(self, state: FileState):
        self.model.eval()
        with torch.no_grad():
            batch = self.preprocess([state])
            logits = self.model(
                batch["input_ids"], batch["attention_mask"],
                batch["ext_indices"], batch["continuous_feats"],
            )
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        i = int(probs.argmax())
        return self.idx2label[i], float(probs[i])

    def save(self, path: str):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), p / "student_model.pt")
        self.tokenizer.save_pretrained(str(p))

    def load(self, path: str):
        weight = Path(path) / "student_model.pt"
        if weight.exists():
            try:
                state = torch.load(weight, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(weight, map_location=self.device)
            self.model.load_state_dict(state)
