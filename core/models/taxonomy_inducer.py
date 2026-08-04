"""Set-conditioned neural taxonomy induction.

The model consumes a collection of file embeddings and emits a variable number
of latent folder slots plus file-to-slot assignments.  Slot identities are
permutation-invariant, so it can learn arbitrary user taxonomies rather than a
fixed class vocabulary.
"""

from __future__ import annotations

import math
from typing import Mapping
import torch
from torch import nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class SlotTaxonomyInducer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 384, max_slots: int = 8, heads: int = 4):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or max_slots < 2 or hidden_dim % heads:
            raise ValueError("invalid inducer dimensions")
        self.max_slots = max_slots
        self.file_projection = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        layer = nn.TransformerEncoderLayer(hidden_dim, heads, hidden_dim * 2, batch_first=True, norm_first=True)
        self.collection_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.slot_queries = nn.Parameter(torch.randn(max_slots, hidden_dim) * 0.02)
        self.slot_attention = nn.MultiheadAttention(hidden_dim, heads, batch_first=True)
        self.slot_norm = nn.LayerNorm(hidden_dim)
        self.existence = nn.Linear(hidden_dim, 1)
        # Raw cosine logits live in [-1, 1] and favour a few broad slots even
        # when more slots exist. A bounded learned temperature lets semantically
        # related folders compete sharply without fixing a taxonomy vocabulary.
        self.assignment_logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, file_embeddings: torch.Tensor, padding_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Inputs [batch, files, dim]; mask is True where padded."""
        if file_embeddings.ndim != 3:
            raise ValueError("file_embeddings must have shape [batch, files, dim]")
        files = self.file_projection(file_embeddings)
        encoded = self.collection_encoder(files, src_key_padding_mask=padding_mask)
        queries = self.slot_queries.unsqueeze(0).expand(encoded.shape[0], -1, -1)
        attended, _ = self.slot_attention(queries, encoded, encoded, key_padding_mask=padding_mask, need_weights=False)
        # Retaining each learned query is essential: replacing it with only the
        # shared attended summary makes the latent slots collapse together.
        slots = self.slot_norm(queries + attended)
        assignments = self.assignment_logit_scale.exp().clamp(max=100.0) * torch.einsum(
            "bnh,bsh->bns", F.normalize(encoded, dim=-1), F.normalize(slots, dim=-1)
        )
        return {"assignment_logits": assignments, "existence_logits": self.existence(slots).squeeze(-1), "slot_embeddings": slots}


SEMANTIC_SLOT_INPUT_FORMATS = frozenset({
    "discovery_semantic_only_v2",
    "discovery_semantic_only_v3_scaled",
})


def load_slot_inducer_state(model: SlotTaxonomyInducer, checkpoint: Mapping) -> None:
    """Load a semantic slot checkpoint without changing its assignment scale.

    v2 used unscaled cosine assignments.  v3 records the learned temperature.
    Accepting v2 is intentional: the newer scale ablation did not supersede a
    stronger prior development checkpoint, so loading must reproduce rather
    than silently alter its behaviour.
    """
    input_format = checkpoint.get("input_format")
    if input_format not in SEMANTIC_SLOT_INPUT_FORMATS:
        raise ValueError("slot checkpoint is not trained on semantic-only discovery inputs")
    if input_format == "discovery_semantic_only_v2":
        model.assignment_logit_scale.data.zero_()
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
        if set(missing) != {"assignment_logit_scale"} or unexpected:
            raise ValueError("invalid legacy semantic slot checkpoint")
        return
    model.load_state_dict(checkpoint["state_dict"])


def permutation_invariant_induction_loss(
    assignment_logits: torch.Tensor, existence_logits: torch.Tensor, target_labels: torch.Tensor,
) -> torch.Tensor:
    """Match oracle folders to latent slots with a Hungarian assignment per episode."""
    if assignment_logits.ndim != 2 or target_labels.ndim != 1:
        raise ValueError("expected one episode: logits [files, slots], labels [files]")
    if assignment_logits.shape[0] != target_labels.shape[0]:
        raise ValueError("file count mismatch")
    slots = assignment_logits.shape[1]
    labels = torch.unique(target_labels)
    if len(labels) > slots or labels.numel() == 0:
        raise ValueError("target taxonomy has invalid folder count")
    log_probs = F.log_softmax(assignment_logits, dim=-1)
    costs = torch.stack([-log_probs[target_labels == label].mean(dim=0) for label in labels])
    rows, cols = linear_sum_assignment(costs.detach().cpu().numpy())
    assignment_loss = costs[torch.as_tensor(rows, device=costs.device), torch.as_tensor(cols, device=costs.device)].mean()
    exists = torch.zeros(slots, device=existence_logits.device)
    exists[torch.as_tensor(cols, device=exists.device)] = 1.0
    return assignment_loss + 0.2 * F.binary_cross_entropy_with_logits(existence_logits, exists)


def permutation_invariant_assignment_accuracy(assignment_logits: torch.Tensor, target_labels: torch.Tensor) -> float:
    """Cluster assignment accuracy after optimal slot-to-folder matching."""
    labels = torch.unique(target_labels)
    log_probs = F.log_softmax(assignment_logits, dim=-1)
    costs = torch.stack([-log_probs[target_labels == label].mean(dim=0) for label in labels])
    rows, cols = linear_sum_assignment(costs.detach().cpu().numpy())
    slot_to_label = {int(slot): labels[row] for row, slot in zip(rows, cols)}
    predictions = assignment_logits.argmax(dim=-1)
    correct = sum(int(slot_to_label.get(int(slot), torch.tensor(-1, device=target_labels.device)) == label) for slot, label in zip(predictions, target_labels))
    return correct / max(1, len(target_labels))


def calibrate_existence_threshold(validation_examples: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    """Choose a slot-existence cutoff on validation episodes only.

    Folder count is part of discovery quality, so the cutoff is selected to
    minimize mean absolute count error, not hard-coded in the UI bridge.
    Ties prefer 0.5, making the selection deterministic and conservative.
    """
    if not validation_examples:
        raise ValueError("at least one validation episode is required")
    candidates = [index / 20 for index in range(1, 20)]
    def error(threshold: float) -> float:
        return sum(
            abs(int((torch.sigmoid(existence_logits) >= threshold).sum()) - len(torch.unique(labels)))
            for existence_logits, labels in validation_examples
        ) / len(validation_examples)
    return min(candidates, key=lambda threshold: (error(threshold), abs(threshold - 0.5)))
