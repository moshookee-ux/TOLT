"""Torch-only baselines with the same modality-mask semantics as TOLT."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .components import FeatureTokenEmbedding, MLPHead, masked_mean
from .tolt import MODALITY_ORDER, ModalityInput


@dataclass(frozen=True)
class MLPBaselineConfig:
    """Configuration for the attention-free multimodal MLP baseline."""

    rna_vocab_size: int
    atac_vocab_size: int
    adt_vocab_size: int
    rna_output_dim: int | None = None
    atac_lsi_output_dim: int = 128
    adt_output_dim: int | None = None
    num_conditions: int = 2
    token_d_model: int = 128
    modality_hidden_dim: int = 256
    fusion_hidden_dim: int = 512
    cell_embedding_dim: int = 512
    decoder_hidden_dim: int = 512
    dropout: float = 0.1
    num_token_states: int = 4
    atac_categorical_vocab_sizes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.rna_output_dim is None:
            object.__setattr__(self, "rna_output_dim", self.rna_vocab_size)
        if self.adt_output_dim is None:
            object.__setattr__(self, "adt_output_dim", self.adt_vocab_size)
        values = {
            "rna_vocab_size": self.rna_vocab_size,
            "atac_vocab_size": self.atac_vocab_size,
            "adt_vocab_size": self.adt_vocab_size,
            "rna_output_dim": self.rna_output_dim,
            "atac_lsi_output_dim": self.atac_lsi_output_dim,
            "adt_output_dim": self.adt_output_dim,
            "num_conditions": self.num_conditions,
            "token_d_model": self.token_d_model,
            "modality_hidden_dim": self.modality_hidden_dim,
            "fusion_hidden_dim": self.fusion_hidden_dim,
            "cell_embedding_dim": self.cell_embedding_dim,
            "decoder_hidden_dim": self.decoder_hidden_dim,
            "num_token_states": self.num_token_states,
        }
        for name, value in values.items():
            if value is None or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if any(size <= 0 for size in self.atac_categorical_vocab_sizes):
            raise ValueError("ATAC categorical vocabulary sizes must be positive")


class TriOmicsMLPBaseline(nn.Module):
    """Attention-free baseline operating on the same feature-aware tokens.

    Tokens are embedded, mask-mean pooled within each modality, transformed by
    a modality MLP, concatenated, and fused by a second MLP.  Learned missing
    vectors and the per-cell ``modality_present[B, 3]`` mask match TOLT's input
    semantics, making missing-modality comparisons direct.
    """

    def __init__(self, config: MLPBaselineConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.ModuleDict(
            {
                "rna": FeatureTokenEmbedding(
                    config.rna_vocab_size,
                    config.token_d_model,
                    num_states=config.num_token_states,
                    dropout=config.dropout,
                ),
                "atac": FeatureTokenEmbedding(
                    config.atac_vocab_size,
                    config.token_d_model,
                    num_states=config.num_token_states,
                    categorical_vocab_sizes=config.atac_categorical_vocab_sizes,
                    dropout=config.dropout,
                ),
                "adt": FeatureTokenEmbedding(
                    config.adt_vocab_size,
                    config.token_d_model,
                    num_states=config.num_token_states,
                    dropout=config.dropout,
                ),
            }
        )
        self.modality_encoders = nn.ModuleDict(
            {
                modality: MLPHead(
                    config.token_d_model,
                    config.modality_hidden_dim,
                    hidden_dims=(config.modality_hidden_dim,),
                    dropout=config.dropout,
                )
                for modality in MODALITY_ORDER
            }
        )
        self.missing_vectors = nn.ParameterDict(
            {
                modality: nn.Parameter(torch.empty(config.modality_hidden_dim))
                for modality in MODALITY_ORDER
            }
        )
        for parameter in self.missing_vectors.values():
            nn.init.normal_(parameter, std=0.02)

        self.fusion_mlp = MLPHead(
            len(MODALITY_ORDER) * config.modality_hidden_dim,
            config.cell_embedding_dim,
            hidden_dims=(config.fusion_hidden_dim, config.fusion_hidden_dim),
            dropout=config.dropout,
        )
        assert config.rna_output_dim is not None
        assert config.adt_output_dim is not None
        decoder_hidden = (config.decoder_hidden_dim,)
        self.rna_decoder = MLPHead(
            config.cell_embedding_dim,
            config.rna_output_dim,
            hidden_dims=decoder_hidden,
            dropout=config.dropout,
        )
        self.atac_lsi_decoder = MLPHead(
            config.cell_embedding_dim,
            config.atac_lsi_output_dim,
            hidden_dims=decoder_hidden,
            dropout=config.dropout,
        )
        self.adt_decoder = MLPHead(
            config.cell_embedding_dim,
            config.adt_output_dim,
            hidden_dims=decoder_hidden,
            dropout=config.dropout,
        )
        condition_hidden = max(32, min(128, config.decoder_hidden_dim))
        self.condition_head = MLPHead(
            config.cell_embedding_dim,
            config.num_conditions,
            hidden_dims=(condition_hidden,),
            dropout=config.dropout,
        )

    @staticmethod
    def _required_tensor(modality: str, inputs: Mapping[str, Tensor], name: str) -> Tensor:
        try:
            value = inputs[name]
        except KeyError as error:
            raise ValueError(f"{modality} inputs require '{name}'") from error
        if not isinstance(value, Tensor):
            raise TypeError(f"{modality}.{name} must be a torch.Tensor")
        return value

    def _normalize_modality_present(self, modality_present: Tensor) -> Tensor:
        if modality_present.ndim != 2 or modality_present.shape[1] != len(MODALITY_ORDER):
            raise ValueError("modality_present must have shape [B, 3] in RNA/ATAC/ADT order")
        presence = modality_present.to(
            device=self.missing_vectors["rna"].device,
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
        missing = self.missing_vectors[modality].unsqueeze(0).expand(batch_size, -1)
        if inputs is None:
            if present.any():
                raise ValueError(
                    f"{modality} inputs are None but modality_present marks observed cells"
                )
            return missing

        feature_ids = self._required_tensor(modality, inputs, "feature_ids")
        values = self._required_tensor(modality, inputs, "values")
        token_mask = self._required_tensor(modality, inputs, "token_mask")
        if feature_ids.ndim != 2 or feature_ids.shape[0] != batch_size:
            raise ValueError(f"{modality}.feature_ids must have shape [B, L_{modality}]")
        if token_mask.shape != feature_ids.shape:
            raise ValueError(f"{modality}.token_mask must match feature_ids [B, L]")
        token_mask = token_mask.to(device=feature_ids.device, dtype=torch.bool)
        if (present.to(device=token_mask.device) & ~token_mask.any(dim=1)).any():
            raise ValueError(f"observed {modality} cells must contain at least one valid token")

        embedded = self.token_embeddings[modality](
            feature_ids,
            values,
            token_mask,
            state_ids=inputs.get("state_ids"),
            categorical_ids=inputs.get("categorical_ids"),
        )
        pooled = masked_mean(embedded, token_mask)
        observed = self.modality_encoders[modality](pooled)
        return torch.where(
            present.to(device=observed.device).unsqueeze(-1),
            observed,
            missing.to(device=observed.device, dtype=observed.dtype),
        )

    def forward(
        self,
        rna: ModalityInput | None,
        atac: ModalityInput | None,
        adt: ModalityInput | None,
        modality_present: Tensor,
    ) -> dict[str, Tensor]:
        presence = self._normalize_modality_present(modality_present)
        batch_size = presence.shape[0]
        inputs_by_modality = {"rna": rna, "atac": atac, "adt": adt}
        modality_embeddings = {
            modality: self._encode_modality(
                modality,
                inputs_by_modality[modality],
                presence[:, index],
                batch_size,
            )
            for index, modality in enumerate(MODALITY_ORDER)
        }
        fusion_input = torch.cat(
            [modality_embeddings[modality] for modality in MODALITY_ORDER],
            dim=-1,
        )
        cell_embedding = self.fusion_mlp(fusion_input)
        return {
            "cell_embedding": cell_embedding,
            "rna_prediction": self.rna_decoder(cell_embedding),
            "atac_lsi_prediction": self.atac_lsi_decoder(cell_embedding),
            "adt_prediction": self.adt_decoder(cell_embedding),
            "condition_logits": self.condition_head(cell_embedding),
            "rna_embedding": modality_embeddings["rna"],
            "atac_embedding": modality_embeddings["atac"],
            "adt_embedding": modality_embeddings["adt"],
            "modality_present": presence,
        }


MLPBaseline = TriOmicsMLPBaseline
