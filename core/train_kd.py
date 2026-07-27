from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from core.model import Student
from core.scan import FileMeta, FileState


def load_soft_labels(path: str) -> list[FileState]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    states = []
    for item in raw:
        md = item.get("metadata", {})
        meta = FileMeta(
            filename=md.get("filename", "unknown"),
            extension=md.get("extension", ""),
            size_bytes=md.get("size_bytes", 0),
            age_days=md.get("age_days", 0),
            depth=md.get("depth", 1),
            mime_type=md.get("mime_type", "text/plain"),
            is_binary=md.get("is_binary", False),
        )
        name = meta.filename
        states.append(
            FileState(
                absolute_path=item.get("absolute_path", f"/kd/{name}"),
                relative_path=item.get("relative_path", name),
                metadata=meta,
                content_sample=item.get("content_sample", ""),
                target_class=item.get("teacher_target_class") or item.get("target_class"),
                teacher_probs=item.get("teacher_probs"),
            )
        )
    return states


def kd_loss(logits, targets, teacher_probs=None, alpha=0.5, tau=2.0):
    valid = targets >= 0
    if valid.sum() == 0:
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits[valid], targets[valid])
    if teacher_probs is None or alpha <= 0:
        return ce
    tp = teacher_probs[valid].clamp(min=1e-8)
    tp = tp / tp.sum(dim=-1, keepdim=True)
    student_log = F.log_softmax(logits[valid] / tau, dim=-1)
    soft = tp.pow(1.0 / tau)
    soft = soft / soft.sum(dim=-1, keepdim=True)
    kl = F.kl_div(student_log, soft, reduction="batchmean") * (tau ** 2)
    return (1.0 - alpha) * ce + alpha * kl


def _acc(student: Student, states: list[FileState]) -> float:
    student.model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(states), 16):
            batch_states = states[i : i + 16]
            batch = student.preprocess(batch_states)
            logits = student.model(
                batch["input_ids"], batch["attention_mask"],
                batch["ext_indices"], batch["continuous_feats"],
            )
            preds = logits.argmax(dim=-1)
            targets = batch["targets"]
            mask = targets >= 0
            correct += int((preds[mask] == targets[mask]).sum().item())
            total += int(mask.sum().item())
    return correct / max(1, total)


def train(student: Student, train_states: list[FileState], epochs=3, batch_size=16, lr=3e-4):
    data = list(train_states)
    random.shuffle(data)
    n_val = max(1, int(len(data) * 0.15))
    val, train_data = data[:n_val], data[n_val:]
    opt = torch.optim.AdamW(student.model.parameters(), lr=lr, weight_decay=0.01)

    for ep in range(1, epochs + 1):
        student.model.train()
        random.shuffle(train_data)
        total, n_batches = 0.0, 0
        for i in range(0, len(train_data), batch_size):
            batch_states = train_data[i : i + batch_size]
            batch = student.preprocess(batch_states)
            logits = student.model(
                batch["input_ids"], batch["attention_mask"],
                batch["ext_indices"], batch["continuous_feats"],
            )
            teacher = None
            if any(s.teacher_probs is not None for s in batch_states):
                k = logits.size(-1)
                rows = [s.teacher_probs if s.teacher_probs is not None else [1.0 / k] * k for s in batch_states]
                teacher = torch.tensor(rows, dtype=torch.float32, device=student.device)
            loss = kd_loss(logits, batch["targets"], teacher)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            n_batches += 1
        print(f"epoch {ep}/{epochs}  loss={total/max(1,n_batches):.4f}  val_acc={_acc(student, val)*100:.1f}%")
