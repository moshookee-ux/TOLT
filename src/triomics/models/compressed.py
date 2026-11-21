"""Anonymous compressed-token Transformer baseline.

This baseline deliberately removes feature identity.  Each observed modality's
complete numeric vector is projected by an MLP into a small, fixed number of
anonymous tokens (RNA/ATAC-LSI/ADT by default: 16/16/8), which are fused by one
shared Transformer.  It therefore isolates the contribution of feature-aware
tokenisation and the modality-local latent bottleneck used by TOLT.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .components import FusionTransformer, MLPHead
from .tolt import MODALITY_ORDER, ModalityInput


@dataclass(frozen=True)
class CompressedTokenConfig:
    """Configuration for :class:`CompressedTokenTransformer`.

    ``*_input_dim`` refers to the complete numeric vector used as input: the
    frozen RNA target space, frozen ATAC-LSI coordinates, and frozen ADT panel.
    No feature IDs are accepted or embedded by this model.
    """

    rna_input_dim: int
    atac_lsi_input_dim: int
    adt_input_dim: int
    rna_output_dim: int | None = None
    atac_lsi_output_dim: int | None = None
    adt_output_dim: int | None = None
    num_conditions: int = 2
    rna_tokens: int = 16
    atac_tokens: int = 16
    adt_tokens: int = 8
    d_model: int = 384
    num_heads: int = 8
    num_layers: int = 4
    ffn_dim: int = 1536
    # At proposal-scale dimensions, 1024 yields ~27.2M parameters versus
    # ~27.7M for the registered TOLT-small configuration (about 1.7% lower).
    projection_hidden_dim: int = 1024
    decoder_hidden_dim: int = 512
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
            "rna_tokens": self.rna_tokens,
            "atac_tokens": self.atac_tokens,
            "adt_tokens": self.adt_tokens,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ffn_dim": self.ffn_dim,
            "projection_hidden_dim": self.projection_hidden_dim,
            "decoder_hidden_dim": self.decoder_hidden_dim,
        }
        for name, value in positive.items():
            if value is None or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class _VectorToTokens(nn.Module):
    """Project one complete vector into anonymous tokens."""

    def __init__(
        self,
        input_dim: int,
        num_tokens: int,
        d_model: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tokens * d_model),
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.token_positions = nn.Parameter(torch.empty(num_tokens, d_model))
        nn.init.normal_(self.token_positions, std=0.02)

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(
                f"dense_values must have shape [B, {self.input_dim}], got "
                f"{tuple(values.shape)}"
            )
        tokens = self.network(values).reshape(
            values.shape[0], self.num_tokens, self.d_model
        )
        return self.output_norm(tokens + self.token_positions.unsqueeze(0))


class CompressedTokenTransformer(nn.Module):
    """Feature-identity-free Transformer baseline with TOLT-compatible outputs.

    Each non-``None`` modality mapping must contain ``dense_values``.  Missing
    modalities must either be passed as ``None`` for a homogeneous batch or be
    marked false in ``modality_present[B, 3]``.  Values of cells marked missing
    are zeroed before their projector and replaced by learned missing tokens, so
    unavailable targets cannot leak into the fusion representation.
    """

    def __init__(self, config: CompressedTokenConfig) -> None:
        super().__init__()
        self.config = config
        input_dims = {
            "rna": config.rna_input_dim,
            "atac": config.atac_lsi_input_dim,
            "adt": config.adt_input_dim,
        }
        token_counts = {
            "rna": config.rna_tokens,
            "atac": config.atac_tokens,
            "adt": config.adt_tokens,
        }
        self.projectors = nn.ModuleDict(
            {
                modality: _VectorToTokens(
                    input_dims[modality],
                    token_counts[modality],
                    config.d_model,
                    config.projection_hidden_dim,
                    config.dropout,
                )
                for modality in MODALITY_ORDER
            }
        )
        self.missing_tokens = nn.ParameterDict(
            {
                modality: nn.Parameter(
                    torch.empty(token_counts[modality], config.d_model)
                )
                for modality in MODALITY_ORDER
            }
        )
        for tokens in self.missing_tokens.values():
            nn.init.normal_(tokens, std=0.02)
        self.fusion = FusionTransformer(
            config.d_model,
            config.num_heads,
            config.num_layers,
            config.ffn_dim,
            num_modalities=len(MODALITY_ORDER),
            dropout=config.dropout,
        )
        assert config.rna_output_dim is not None
        assert config.atac_lsi_output_dim is not None
        assert config.adt_output_dim is not None
        decoder_hidden = (config.decoder_hidden_dim,)
        self.rna_decoder = MLPHead(
            config.d_model,
            config.rna_output_dim,
            hidden_dims=decoder_hidden,
            dropout=config.dropout,
        )
        self.atac_lsi_decoder = MLPHead(
            config.d_model,
            config.atac_lsi_output_dim,
            hidden_dims=decoder_hidden,
            dropout=config.dropout,
        )
        self.adt_decoder = MLPHead(
            config.d_model,
            config.adt_output_dim,
            hidden_dims=decoder_hidden,
            dropout=config.dropout,
        )
        condition_hidden = max(32, min(128, config.decoder_hidden_dim))
        self.condition_head = MLPHead(
            config.d_model,
            config.num_conditions,
            hidden_dims=(condition_hidden,),
            dropout=config.dropout,
        )

    def _presence(self, modality_present: Tensor) -> Tensor:
        if modality_present.ndim != 2 or modality_present.shape[1] != 3:
            raise ValueError("modality_present must have shape [B, 3]")
        presence = modality_present.to(
            device=self.missing_tokens["rna"].device,
            dtype=torch.bool,
        )
        if (~presence.any(dim=1)).any():
            raise ValueError("each cell must have at least one observed modality")
        return presence

    def _tokens(
        self,
        modality: str,
        inputs: ModalityInput | None,
        present: Tensor,
        batch_size: int,
    ) -> Tensor:
        missing = self.missing_tokens[modality].unsqueeze(0).expand(batch_size, -1, -1)
        if inputs is None:
            if present.any():
                raise ValueError(
                    f"{modality} inputs are None but modality_present marks observed cells"
                )
            return missing
        if not isinstance(inputs, Mapping) or "dense_values" not in inputs:
            raise ValueError(f"{modality} inputs require 'dense_values'")
        values = inputs["dense_values"]
        if not isinstance(values, Tensor):
            raise TypeError(f"{modality}.dense_values must be a torch.Tensor")
        if values.ndim != 2 or values.shape[0] != batch_size:
            raise ValueError(f"{modality}.dense_values must have shape [B, D]")
        observed = present.to(device=values.device).unsqueeze(-1)
        safe_values = torch.where(observed, values, torch.zeros_like(values))
        projected = self.projectors[modality](safe_values)
        return torch.where(
            observed.unsqueeze(-1),
            projected,
            missing.to(device=projected.device, dtype=projected.dtype),
        )

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
        modality_tokens = {
            modality: self._tokens(
                modality,
                supplied[modality],
                presence[:, index],
                batch_size,
            )
            for index, modality in enumerate(MODALITY_ORDER)
        }
        cell_embedding, fusion_tokens = self.fusion(
            [modality_tokens[modality] for modality in MODALITY_ORDER]
        )
        return {
            "cell_embedding": cell_embedding,
            "rna_prediction": self.rna_decoder(cell_embedding),
            "atac_lsi_prediction": self.atac_lsi_decoder(cell_embedding),
            "adt_prediction": self.adt_decoder(cell_embedding),
            "condition_logits": self.condition_head(cell_embedding),
            "rna_embedding": modality_tokens["rna"].mean(dim=1),
            "atac_embedding": modality_tokens["atac"].mean(dim=1),
            "adt_embedding": modality_tokens["adt"].mean(dim=1),
            "fusion_tokens": fusion_tokens,
            "modality_present": presence,
        }


CompressedTransformer = CompressedTokenTransformer
