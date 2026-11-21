"""Dense-input TriOmics Latent Transformer.

This model is the ``Dense x TOLT`` quadrant of the completion-v3 core 2x2.
Each observed modality enters as its complete, frozen-coordinate numeric
vector.  A low-rank learned expansion converts that vector into a fixed token
sequence, after which the modality-local Transformer, latent bottleneck,
fusion Transformer, and output heads are structurally identical to the pure
token TOLT control.

Dense coordinates are ordered features, not a set.  Their order is fixed by
the preprocessing artifacts and cache identity.  No feature IDs, token
states, categorical metadata, labels, conditions, or batch fields are
accepted by this model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .components import (
    FusionTransformer,
    LatentCrossAttention,
    LocalTransformerEncoder,
    MLPHead,
)
from .tolt import MODALITY_ORDER, ModalityInput


@dataclass(frozen=True)
class DenseTOLTConfig:
    """Configuration for :class:`DenseTOLTTransformer`."""

    rna_input_dim: int
    atac_lsi_input_dim: int
    adt_input_dim: int
    rna_output_dim: int | None = None
    atac_lsi_output_dim: int | None = None
    adt_output_dim: int | None = None
    num_conditions: int = 2

    rna_sequence_length: int = 512
    atac_sequence_length: int = 1024
    adt_sequence_length: int = 210
    dense_hidden_dim: int = 488
    dense_rank: int = 8

    local_d_model: int = 256
    local_heads: int = 8
    local_ffn_dim: int = 1024
    rna_local_layers: int = 2
    atac_local_layers: int = 2
    adt_local_layers: int = 1
    rna_latents: int = 48
    atac_latents: int = 64
    adt_latents: int = 24

    fusion_d_model: int = 384
    fusion_heads: int = 8
    fusion_layers: int = 6
    fusion_ffn_dim: int = 1536
    decoder_hidden_dim: int = 384
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.rna_output_dim is None:
            object.__setattr__(self, "rna_output_dim", self.rna_input_dim)
        if self.atac_lsi_output_dim is None:
            object.__setattr__(self, "atac_lsi_output_dim", self.atac_lsi_input_dim)
        if self.adt_output_dim is None:
            object.__setattr__(self, "adt_output_dim", self.adt_input_dim)
        positive = {
            "rna_input_dim": self.rna_input_dim,
            "atac_lsi_input_dim": self.atac_lsi_input_dim,
            "adt_input_dim": self.adt_input_dim,
            "rna_output_dim": self.rna_output_dim,
            "atac_lsi_output_dim": self.atac_lsi_output_dim,
            "adt_output_dim": self.adt_output_dim,
            "num_conditions": self.num_conditions,
            "rna_sequence_length": self.rna_sequence_length,
            "atac_sequence_length": self.atac_sequence_length,
            "adt_sequence_length": self.adt_sequence_length,
            "dense_hidden_dim": self.dense_hidden_dim,
            "dense_rank": self.dense_rank,
            "local_d_model": self.local_d_model,
            "local_heads": self.local_heads,
            "local_ffn_dim": self.local_ffn_dim,
            "rna_local_layers": self.rna_local_layers,
            "atac_local_layers": self.atac_local_layers,
            "adt_local_layers": self.adt_local_layers,
            "rna_latents": self.rna_latents,
            "atac_latents": self.atac_latents,
            "adt_latents": self.adt_latents,
            "fusion_d_model": self.fusion_d_model,
            "fusion_heads": self.fusion_heads,
            "fusion_layers": self.fusion_layers,
            "fusion_ffn_dim": self.fusion_ffn_dim,
            "decoder_hidden_dim": self.decoder_hidden_dim,
        }
        for name, value in positive.items():
            if value is None or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.local_d_model % self.local_heads:
            raise ValueError("local_d_model must be divisible by local_heads")
        if self.fusion_d_model % self.fusion_heads:
            raise ValueError("fusion_d_model must be divisible by fusion_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class _DenseVectorToSequence(nn.Module):
    """Expand one ordered dense vector into a fixed local-token sequence."""

    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        dense_hidden_dim: int,
        dense_rank: int,
        local_d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.dense_rank = dense_rank
        self.input_norm = nn.LayerNorm(input_dim)
        self.hidden_projection = nn.Linear(input_dim, dense_hidden_dim)
        self.coefficient_projection = nn.Linear(
            dense_hidden_dim, sequence_length * dense_rank
        )
        self.rank_projection = nn.Linear(dense_rank, local_d_model, bias=False)
        self.slot_positions = nn.Parameter(
            torch.empty(sequence_length, local_d_model)
        )
        self.modality_embedding = nn.Parameter(torch.empty(local_d_model))
        self.output_norm = nn.LayerNorm(local_d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.slot_positions, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(
                f"dense_values must have shape [B, {self.input_dim}], got "
                f"{tuple(values.shape)}"
            )
        hidden = torch.nn.functional.gelu(
            self.hidden_projection(self.input_norm(values))
        )
        hidden = self.dropout(hidden)
        coefficients = self.coefficient_projection(hidden).reshape(
            values.shape[0], self.sequence_length, self.dense_rank
        )
        tokens = self.rank_projection(coefficients)
        tokens = tokens + self.slot_positions.unsqueeze(0)
        tokens = tokens + self.modality_embedding.view(1, 1, -1)
        return self.dropout(self.output_norm(tokens))


class DenseTOLTTransformer(nn.Module):
    """Dense-vector front end followed by the pure TOLT architecture."""

    def __init__(self, config: DenseTOLTConfig) -> None:
        super().__init__()
        self.config = config
        input_dims = {
            "rna": config.rna_input_dim,
            "atac": config.atac_lsi_input_dim,
            "adt": config.adt_input_dim,
        }
        sequence_lengths = {
            "rna": config.rna_sequence_length,
            "atac": config.atac_sequence_length,
            "adt": config.adt_sequence_length,
        }
        local_layers = {
            "rna": config.rna_local_layers,
            "atac": config.atac_local_layers,
            "adt": config.adt_local_layers,
        }
        latent_counts = {
            "rna": config.rna_latents,
            "atac": config.atac_latents,
            "adt": config.adt_latents,
        }

        self.dense_frontends = nn.ModuleDict(
            {
                modality: _DenseVectorToSequence(
                    input_dims[modality],
                    sequence_lengths[modality],
                    config.dense_hidden_dim,
                    config.dense_rank,
                    config.local_d_model,
                    config.dropout,
                )
                for modality in MODALITY_ORDER
            }
        )
        self.local_encoders = nn.ModuleDict(
            {
                modality: LocalTransformerEncoder(
                    config.local_d_model,
                    config.local_heads,
                    local_layers[modality],
                    config.local_ffn_dim,
                    dropout=config.dropout,
                )
                for modality in MODALITY_ORDER
            }
        )
        self.latent_poolers = nn.ModuleDict(
            {
                modality: LatentCrossAttention(
                    config.local_d_model,
                    config.local_heads,
                    latent_counts[modality],
                    config.local_ffn_dim,
                    dropout=config.dropout,
                )
                for modality in MODALITY_ORDER
            }
        )
        self.missing_latents = nn.ParameterDict(
            {
                modality: nn.Parameter(
                    torch.empty(latent_counts[modality], config.local_d_model)
                )
                for modality in MODALITY_ORDER
            }
        )
        self.latent_projections = nn.ModuleDict(
            {
                modality: nn.Linear(config.local_d_model, config.fusion_d_model)
                for modality in MODALITY_ORDER
            }
        )
        for parameter in self.missing_latents.values():
            nn.init.normal_(parameter, std=0.02)

        self.fusion = FusionTransformer(
            config.fusion_d_model,
            config.fusion_heads,
            config.fusion_layers,
            config.fusion_ffn_dim,
            num_modalities=len(MODALITY_ORDER),
            dropout=config.dropout,
        )
        assert config.rna_output_dim is not None
        assert config.atac_lsi_output_dim is not None
        assert config.adt_output_dim is not None
        decoder_hidden = (config.decoder_hidden_dim,)
        self.rna_decoder = MLPHead(
            config.fusion_d_model,
            config.rna_output_dim,
            hidden_dims=decoder_hidden,
            dropout=config.dropout,
        )
        self.atac_lsi_decoder = MLPHead(
            config.fusion_d_model,
            config.atac_lsi_output_dim,
            hidden_dims=decoder_hidden,
            dropout=config.dropout,
        )
        self.adt_decoder = MLPHead(
            config.fusion_d_model,
            config.adt_output_dim,
            hidden_dims=decoder_hidden,
            dropout=config.dropout,
        )
        condition_hidden = max(32, min(128, config.decoder_hidden_dim))
        self.condition_head = MLPHead(
            config.fusion_d_model,
            config.num_conditions,
            hidden_dims=(condition_hidden,),
            dropout=config.dropout,
        )
        self.input_dimensions = input_dims
        self.sequence_lengths = sequence_lengths
        self.latent_counts = latent_counts

    @property
    def output_dimensions(self) -> dict[str, int]:
        assert self.config.rna_output_dim is not None
        assert self.config.atac_lsi_output_dim is not None
        assert self.config.adt_output_dim is not None
        return {
            "rna": self.config.rna_output_dim,
            "atac_lsi": self.config.atac_lsi_output_dim,
            "adt": self.config.adt_output_dim,
            "condition": self.config.num_conditions,
        }

    def _presence(self, modality_present: Tensor) -> Tensor:
        if modality_present.ndim != 2 or modality_present.shape[1] != 3:
            raise ValueError("modality_present must have shape [B, 3]")
        presence = modality_present.to(
            device=self.missing_latents["rna"].device,
            dtype=torch.bool,
        )
        if (~presence.any(dim=1)).any():
            raise ValueError("each cell must have at least one observed modality")
        return presence

    def _encode_modality(
        self,
        modality: str,
        inputs: ModalityInput | None,
        present: Tensor,
        batch_size: int,
    ) -> Tensor:
        missing = self.missing_latents[modality].unsqueeze(0).expand(
            batch_size, -1, -1
        )
        if inputs is None:
            if present.any():
                raise ValueError(
                    f"{modality} inputs are None but modality_present marks observed cells"
                )
            return self.latent_projections[modality](missing)
        if not isinstance(inputs, Mapping) or set(inputs) != {"dense_values"}:
            raise ValueError(
                f"{modality} inputs must contain only the ordered 'dense_values' vector"
            )
        values = inputs["dense_values"]
        if not isinstance(values, Tensor):
            raise TypeError(f"{modality}.dense_values must be a torch.Tensor")
        expected_width = self.input_dimensions[modality]
        if values.ndim != 2 or values.shape != (batch_size, expected_width):
            raise ValueError(
                f"{modality}.dense_values must have shape [B, {expected_width}]"
            )
        row_present = present.to(device=values.device, dtype=torch.bool)
        safe_values = torch.where(
            row_present.unsqueeze(-1), values, torch.zeros_like(values)
        )
        if not torch.isfinite(safe_values).all():
            raise ValueError(f"observed {modality}.dense_values must be finite")
        tokens = self.dense_frontends[modality](safe_values)
        token_mask = row_present.unsqueeze(-1).expand(
            batch_size, self.sequence_lengths[modality]
        )
        encoded = self.local_encoders[modality](tokens, token_mask)
        observed = self.latent_poolers[modality](encoded, token_mask)
        selected = torch.where(
            row_present.view(batch_size, 1, 1),
            observed,
            missing.to(device=observed.device, dtype=observed.dtype),
        )
        return self.latent_projections[modality](selected)

    def forward(
        self,
        rna: ModalityInput | None,
        atac: ModalityInput | None,
        adt: ModalityInput | None,
        modality_present: Tensor,
    ) -> dict[str, Tensor]:
        presence = self._presence(modality_present)
        batch_size = presence.shape[0]
        supplied = {"rna": rna, "atac": atac, "adt": adt}
        projected_latents = {
            modality: self._encode_modality(
                modality,
                supplied[modality],
                presence[:, index],
                batch_size,
            )
            for index, modality in enumerate(MODALITY_ORDER)
        }
        cell_embedding, fusion_tokens = self.fusion(
            [projected_latents[modality] for modality in MODALITY_ORDER]
        )
        return {
            "cell_embedding": cell_embedding,
            "rna_prediction": self.rna_decoder(cell_embedding),
            "atac_lsi_prediction": self.atac_lsi_decoder(cell_embedding),
            "adt_prediction": self.adt_decoder(cell_embedding),
            "condition_logits": self.condition_head(cell_embedding),
            "rna_latents": projected_latents["rna"],
            "atac_latents": projected_latents["atac"],
            "adt_latents": projected_latents["adt"],
            "fusion_tokens": fusion_tokens,
            "modality_present": presence,
        }


DenseTriOmicsLatentTransformer = DenseTOLTTransformer
