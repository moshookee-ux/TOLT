"""Reusable, torch-only building blocks for TriOmics models.

Tensor conventions
------------------
``B`` is batch size, ``L`` is a modality-specific token length, ``D`` is a
hidden dimension, and ``Q`` is a modality-specific latent count.  Boolean
token masks use ``True`` for a real/valid token and ``False`` for padding.
This is intentionally the opposite of PyTorch's ``key_padding_mask``; the
conversion is kept inside this module so callers have one consistent API.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _require_shape(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def safe_token_mask(token_mask: Tensor) -> Tensor:
    """Return a mask suitable for attention even when a row has no tokens.

    PyTorch attention produces undefined values when every key in a row is
    masked.  An absent modality may legitimately have such a row, so a dummy
    first token is temporarily exposed.  Callers still use the original mask
    to zero outputs or replace the row with learned missing-modality latents.
    """

    _require_shape(token_mask.ndim == 2, "token_mask must have shape [B, L]")
    _require_shape(token_mask.shape[1] > 0, "token sequences must have L > 0")
    mask = token_mask.to(dtype=torch.bool)
    empty_rows = ~mask.any(dim=1)
    if empty_rows.any():
        mask = mask.clone()
        mask[empty_rows, 0] = True
    return mask


def masked_mean(tokens: Tensor, token_mask: Tensor) -> Tensor:
    """Mean-pool ``[B, L, D]`` tokens over valid positions to ``[B, D]``."""

    _require_shape(tokens.ndim == 3, "tokens must have shape [B, L, D]")
    _require_shape(
        token_mask.shape == tokens.shape[:2],
        "token_mask must match the first two token dimensions",
    )
    mask = token_mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return (tokens * mask).sum(dim=1) / denominator


class FeatureTokenEmbedding(nn.Module):
    """Embed biological feature identity, value, state, and optional metadata.

    Parameters
    ----------
    vocab_size:
        Number of feature IDs for the modality (genes, peaks, or proteins).
    d_model:
        Output token dimension.
    num_states:
        Number of observed/masked/padding-style states understood by the data
        pipeline.  State IDs are supplied explicitly; state 0 is used when no
        state tensor is provided.
    categorical_vocab_sizes:
        Optional categorical metadata vocabularies.  ATAC can, for example,
        use ``(n_chromosomes, n_position_bins)``.  The corresponding input has
        shape ``[B, L, C]`` where ``C`` is the length of this sequence.

    Forward inputs
    --------------
    feature_ids: ``LongTensor[B, L]``
    values: ``FloatTensor[B, L]`` or ``FloatTensor[B, L, 1]``
    token_mask: ``BoolTensor[B, L]``; ``True`` means valid
    state_ids: optional ``LongTensor[B, L]``
    categorical_ids: optional ``LongTensor[B, L, C]``
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        *,
        num_states: int = 4,
        categorical_vocab_sizes: Sequence[int] = (),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _require_shape(vocab_size > 0, "vocab_size must be positive")
        _require_shape(d_model > 0, "d_model must be positive")
        _require_shape(num_states > 0, "num_states must be positive")
        _require_shape(
            all(size > 0 for size in categorical_vocab_sizes),
            "categorical vocabulary sizes must be positive",
        )

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_categorical_features = len(categorical_vocab_sizes)
        # ID 0 is reserved for padding throughout the cache and dataloader.
        # Biological feature IDs therefore occupy the inclusive range 1..V.
        self.feature_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        self.value_projection = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.state_embedding = nn.Embedding(num_states, d_model)
        self.categorical_embeddings = nn.ModuleList(
            nn.Embedding(size, d_model) for size in categorical_vocab_sizes
        )
        self.modality_embedding = nn.Parameter(torch.empty(d_model))
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.feature_embedding.weight, std=0.02)
        with torch.no_grad():
            self.feature_embedding.weight[0].zero_()
        nn.init.normal_(self.state_embedding.weight, std=0.02)
        for embedding in self.categorical_embeddings:
            nn.init.normal_(embedding.weight, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)

    def forward(
        self,
        feature_ids: Tensor,
        values: Tensor,
        token_mask: Tensor,
        *,
        state_ids: Tensor | None = None,
        categorical_ids: Tensor | None = None,
    ) -> Tensor:
        _require_shape(feature_ids.ndim == 2, "feature_ids must have shape [B, L]")
        batch_tokens = feature_ids.shape
        _require_shape(
            values.shape == batch_tokens or values.shape == (*batch_tokens, 1),
            "values must have shape [B, L] or [B, L, 1]",
        )
        _require_shape(
            token_mask.shape == batch_tokens,
            "token_mask must have the same [B, L] shape as feature_ids",
        )
        if state_ids is not None:
            _require_shape(
                state_ids.shape == batch_tokens,
                "state_ids must have the same [B, L] shape as feature_ids",
            )

        if self.num_categorical_features:
            _require_shape(
                categorical_ids is not None,
                "categorical_ids are required for this token embedding",
            )
            assert categorical_ids is not None
            _require_shape(
                categorical_ids.shape == (*batch_tokens, self.num_categorical_features),
                "categorical_ids must have shape [B, L, C] matching the configured vocabularies",
            )
        elif categorical_ids is not None:
            _require_shape(
                categorical_ids.shape[-1] == 0,
                "categorical_ids were supplied but no categorical vocabularies are configured",
            )

        feature_ids = feature_ids.to(dtype=torch.long)
        if values.ndim == 2:
            values = values.unsqueeze(-1)
        values = values.to(dtype=self.feature_embedding.weight.dtype)
        if state_ids is None:
            state_ids = torch.zeros_like(feature_ids)
        else:
            state_ids = state_ids.to(dtype=torch.long)

        tokens = self.feature_embedding(feature_ids)
        tokens = tokens + self.value_projection(values)
        tokens = tokens + self.state_embedding(state_ids)
        tokens = tokens + self.modality_embedding.view(1, 1, -1)

        if self.num_categorical_features:
            assert categorical_ids is not None
            categorical_ids = categorical_ids.to(dtype=torch.long)
            for index, embedding in enumerate(self.categorical_embeddings):
                tokens = tokens + embedding(categorical_ids[..., index])

        tokens = self.dropout(self.norm(tokens))
        mask = token_mask.to(device=tokens.device, dtype=torch.bool).unsqueeze(-1)
        return tokens.masked_fill(~mask, 0.0)


class LocalTransformerEncoder(nn.Module):
    """Pre-norm Transformer encoder for one modality.

    Input and output are both ``[B, L, D]``.  Padded output positions are
    explicitly zeroed so downstream pooling cannot accidentally consume them.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _require_shape(num_layers > 0, "num_layers must be positive")
        _require_shape(d_model % num_heads == 0, "d_model must be divisible by num_heads")
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        _require_shape(tokens.ndim == 3, "tokens must have shape [B, L, D]")
        _require_shape(
            token_mask.shape == tokens.shape[:2],
            "token_mask must match tokens [B, L]",
        )
        valid_mask = token_mask.to(device=tokens.device, dtype=torch.bool)
        attention_mask = safe_token_mask(valid_mask)
        encoded = self.encoder(tokens, src_key_padding_mask=~attention_mask)
        encoded = self.norm(encoded)
        return encoded.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


class LatentCrossAttention(nn.Module):
    """Compress a variable-length modality into learned latent queries.

    Context has shape ``[B, L, D]`` and output has shape ``[B, Q, D]``.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_latents: int,
        ffn_dim: int,
        *,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _require_shape(num_latents > 0, "num_latents must be positive")
        _require_shape(d_model % num_heads == 0, "d_model must be divisible by num_heads")
        self.num_latents = num_latents
        self.d_model = d_model
        self.latent_queries = nn.Parameter(torch.empty(num_latents, d_model))
        self.query_norm = nn.LayerNorm(d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.latent_queries, std=0.02)

    def forward(self, context: Tensor, context_mask: Tensor) -> Tensor:
        _require_shape(context.ndim == 3, "context must have shape [B, L, D]")
        _require_shape(context.shape[-1] == self.d_model, "unexpected context dimension")
        _require_shape(
            context_mask.shape == context.shape[:2],
            "context_mask must match context [B, L]",
        )
        batch_size = context.shape[0]
        valid_mask = context_mask.to(device=context.device, dtype=torch.bool)
        attention_mask = safe_token_mask(valid_mask)
        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            self.context_norm(context),
            self.context_norm(context),
            key_padding_mask=~attention_mask,
            need_weights=False,
        )
        latents = queries + attended
        latents = latents + self.ffn(self.ffn_norm(latents))
        return self.output_norm(latents)


class AttentiveProjectionPool(nn.Module):
    """Pool modality latents into a role-specific, normalized representation.

    Unlike mean pooling, the learned scalar attention can retain a small
    identity-bearing residual that is present in only a subset of latent
    queries.  ``present`` is applied *after* all affine operations so an
    unavailable modality is represented by an exact zero and cannot leak a
    learned missing-modality token into private or exact-identity objectives.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        _require_shape(input_dim > 0 and output_dim > 0, "pool dimensions must be positive")
        hidden = int(hidden_dim or min(input_dim, max(64, output_dim)))
        _require_shape(hidden > 0, "pool hidden dimension must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.normalize = bool(normalize)
        self.input_norm = nn.LayerNorm(input_dim)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False),
        )
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, latents: Tensor, present: Tensor) -> Tensor:
        _require_shape(latents.ndim == 3, "latents must have shape [B, Q, D]")
        _require_shape(latents.shape[-1] == self.input_dim, "unexpected latent dimension")
        _require_shape(present.shape == (latents.shape[0],), "present must have shape [B]")
        normalized = self.input_norm(latents)
        weights = torch.softmax(self.attention(normalized).squeeze(-1).float(), dim=1)
        pooled = torch.einsum("bq,bqd->bd", weights.to(latents.dtype), normalized)
        projected = self.output_norm(self.projection(pooled))
        if self.normalize:
            projected = torch.nn.functional.normalize(projected.float(), dim=-1).to(latents.dtype)
        return projected * present.to(
            device=projected.device,
            dtype=projected.dtype,
        ).unsqueeze(-1)


class FusionTransformer(nn.Module):
    """Fuse fixed-length modality latents with learned global query tokens.

    The input is a sequence of modality tensors ``[B, Q_m, D]``.  The returned
    tuple is ``(cell_embedding, fusion_tokens)`` with shapes ``[B, D]`` and
    ``[B, 1 + num_task_tokens + sum(Q_m), D]`` respectively.  Position zero is
    always the shared ``[CELL]`` token.  When ``num_task_tokens`` is positive,
    the following positions are distinct learned task queries.  The default is
    zero so existing checkpoints and state dictionaries remain byte-for-byte
    architecture compatible.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        ffn_dim: int,
        *,
        num_modalities: int = 3,
        num_task_tokens: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _require_shape(num_modalities > 0, "num_modalities must be positive")
        _require_shape(num_task_tokens >= 0, "num_task_tokens must be non-negative")
        _require_shape(num_layers > 0, "num_layers must be positive")
        _require_shape(d_model % num_heads == 0, "d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.num_task_tokens = num_task_tokens
        self.cell_token = nn.Parameter(torch.empty(1, 1, d_model))
        self.task_tokens = (
            nn.Parameter(torch.empty(1, num_task_tokens, d_model))
            if num_task_tokens
            else None
        )
        self.segment_embeddings = nn.Parameter(torch.empty(num_modalities + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.cell_token, std=0.02)
        if self.task_tokens is not None:
            nn.init.normal_(self.task_tokens, std=0.02)
        nn.init.normal_(self.segment_embeddings, std=0.02)

    def forward(self, modality_latents: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        _require_shape(
            len(modality_latents) == self.num_modalities,
            f"expected {self.num_modalities} modality latent tensors",
        )
        _require_shape(bool(modality_latents), "at least one modality is required")
        batch_size = modality_latents[0].shape[0]
        pieces = [
            self.cell_token.expand(batch_size, -1, -1)
            + self.segment_embeddings[0].view(1, 1, -1)
        ]
        if self.task_tokens is not None:
            pieces.append(self.task_tokens.expand(batch_size, -1, -1))
        for index, latents in enumerate(modality_latents):
            _require_shape(latents.ndim == 3, "each latent tensor must have shape [B, Q, D]")
            _require_shape(latents.shape[0] == batch_size, "all modalities must share batch size")
            _require_shape(latents.shape[-1] == self.d_model, "unexpected fusion latent dimension")
            pieces.append(
                latents + self.segment_embeddings[index + 1].view(1, 1, -1)
            )
        tokens = torch.cat(pieces, dim=1)
        tokens = self.norm(self.encoder(tokens))
        return tokens[:, 0], tokens


class MLPHead(nn.Module):
    """Layer-normalized MLP used by reconstruction and classification heads."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _require_shape(input_dim > 0 and output_dim > 0, "MLP dimensions must be positive")
        _require_shape(all(dim > 0 for dim in hidden_dims), "hidden dimensions must be positive")
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = hidden
        layers.append(nn.Linear(previous, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)
