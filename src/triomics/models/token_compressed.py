"""Feature-aware, attention-free token-to-compressed Transformer.

This model is the ``Token x Compressed`` quadrant of the completion-v3 core
comparison.  Biological feature tokens are embedded exactly as token models,
but modality-local self-attention and latent cross-attention are deliberately
absent.  A shared element-wise ``phi`` followed by masked mean/max/log-count
aggregation is a DeepSets encoder.  A modality-specific ``rho`` maps that
invariant summary to the same fixed anonymous-token interface used by the
dense compressed baseline.

Set contract
------------
Each element is ``(feature_id, post-mask value, token state, optional
categorical fields)``.  ``token_mask=False`` is padding and its contents are
zeroed before any embedding lookup.  An unavailable modality is selected only
by ``modality_present`` and is replaced by learned missing tokens.  Valid
feature IDs must be unique within a modality/cell; duplicates are rejected.
Reordering valid elements or adding padding does not change the output.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .components import FeatureTokenEmbedding, FusionTransformer, MLPHead
from .tolt import MODALITY_ORDER, ModalityInput


@dataclass(frozen=True)
class TokenCompressedConfig:
    """Configuration for :class:`TokenCompressedTransformer`."""

    rna_vocab_size: int
    atac_vocab_size: int
    adt_vocab_size: int
    rna_output_dim: int | None = None
    atac_lsi_output_dim: int = 128
    adt_output_dim: int | None = None
    num_conditions: int = 2
    rna_tokens: int = 16
    atac_tokens: int = 16
    adt_tokens: int = 8
    element_dim: int = 256
    phi_hidden_dim: int = 1024
    rho_hidden_dim: int = 512
    fusion_d_model: int = 384
    fusion_heads: int = 8
    fusion_layers: int = 4
    fusion_ffn_dim: int = 1536
    decoder_hidden_dim: int = 512
    dropout: float = 0.1
    num_token_states: int = 4
    atac_categorical_vocab_sizes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.rna_output_dim is None:
            object.__setattr__(self, "rna_output_dim", self.rna_vocab_size)
        if self.adt_output_dim is None:
            object.__setattr__(self, "adt_output_dim", self.adt_vocab_size)
        positive = {
            "rna_vocab_size": self.rna_vocab_size,
            "atac_vocab_size": self.atac_vocab_size,
            "adt_vocab_size": self.adt_vocab_size,
            "rna_output_dim": self.rna_output_dim,
            "atac_lsi_output_dim": self.atac_lsi_output_dim,
            "adt_output_dim": self.adt_output_dim,
            "num_conditions": self.num_conditions,
            "rna_tokens": self.rna_tokens,
            "atac_tokens": self.atac_tokens,
            "adt_tokens": self.adt_tokens,
            "element_dim": self.element_dim,
            "phi_hidden_dim": self.phi_hidden_dim,
            "rho_hidden_dim": self.rho_hidden_dim,
            "fusion_d_model": self.fusion_d_model,
            "fusion_heads": self.fusion_heads,
            "fusion_layers": self.fusion_layers,
            "fusion_ffn_dim": self.fusion_ffn_dim,
            "decoder_hidden_dim": self.decoder_hidden_dim,
            "num_token_states": self.num_token_states,
        }
        for name, value in positive.items():
            if value is None or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.fusion_d_model % self.fusion_heads:
            raise ValueError("fusion_d_model must be divisible by fusion_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if any(size <= 0 for size in self.atac_categorical_vocab_sizes):
            raise ValueError("ATAC categorical vocabulary sizes must be positive")


class _InvariantSetCompressor(nn.Module):
    """Shared-phi DeepSets aggregation with a modality-specific rho."""

    def __init__(
        self,
        element_dim: int,
        num_tokens: int,
        fusion_dim: int,
        rho_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.element_dim = element_dim
        self.num_tokens = num_tokens
        self.fusion_dim = fusion_dim
        # mean[D] + max[D] + log(1 + valid cardinality)[1]
        pooled_dim = 2 * element_dim + 1
        self.rho = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, rho_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rho_hidden_dim, num_tokens * fusion_dim),
        )
        self.token_positions = nn.Parameter(torch.empty(num_tokens, fusion_dim))
        self.output_norm = nn.LayerNorm(fusion_dim)
        nn.init.normal_(self.token_positions, std=0.02)

    def forward(self, embedded: Tensor, token_mask: Tensor) -> Tensor:
        if embedded.ndim != 3 or embedded.shape[-1] != self.element_dim:
            raise ValueError("embedded tokens must have shape [B, L, element_dim]")
        if token_mask.shape != embedded.shape[:2]:
            raise ValueError("token_mask must match embedded tokens [B, L]")
        valid = token_mask.to(device=embedded.device, dtype=torch.bool)
        encoded = embedded.masked_fill(~valid.unsqueeze(-1), 0.0)
        count = valid.sum(dim=1)
        safe_count = count.clamp_min(1).to(dtype=encoded.dtype).unsqueeze(-1)
        mean = encoded.sum(dim=1) / safe_count
        maximum = encoded.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(dim=1)
        maximum = torch.where(count.unsqueeze(-1) > 0, maximum, torch.zeros_like(maximum))
        log_count = torch.log1p(count.to(dtype=encoded.dtype)).unsqueeze(-1)
        pooled = torch.cat((mean, maximum, log_count), dim=-1)
        tokens = self.rho(pooled).reshape(
            embedded.shape[0], self.num_tokens, self.fusion_dim
        )
        return self.output_norm(tokens + self.token_positions.unsqueeze(0))


class TokenCompressedTransformer(nn.Module):
    """Feature-aware DeepSets compressor with S5-compatible fusion/heads."""

    def __init__(self, config: TokenCompressedConfig) -> None:
        super().__init__()
        self.config = config
        vocabulary_sizes = {
            "rna": config.rna_vocab_size,
            "atac": config.atac_vocab_size,
            "adt": config.adt_vocab_size,
        }
        token_counts = {
            "rna": config.rna_tokens,
            "atac": config.atac_tokens,
            "adt": config.adt_tokens,
        }
        self.token_embeddings = nn.ModuleDict(
            {
                "rna": FeatureTokenEmbedding(
                    config.rna_vocab_size,
                    config.element_dim,
                    num_states=config.num_token_states,
                    dropout=config.dropout,
                ),
                "atac": FeatureTokenEmbedding(
                    config.atac_vocab_size,
                    config.element_dim,
                    num_states=config.num_token_states,
                    categorical_vocab_sizes=config.atac_categorical_vocab_sizes,
                    dropout=config.dropout,
                ),
                "adt": FeatureTokenEmbedding(
                    config.adt_vocab_size,
                    config.element_dim,
                    num_states=config.num_token_states,
                    dropout=config.dropout,
                ),
            }
        )
        # One element-wise phi is shared across every set element and modality.
        # Modality and feature identity already enter through the embeddings.
        self.shared_phi = nn.Sequential(
            nn.LayerNorm(config.element_dim),
            nn.Linear(config.element_dim, config.phi_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.phi_hidden_dim, config.element_dim),
            nn.Dropout(config.dropout),
        )
        self.compressors = nn.ModuleDict(
            {
                modality: _InvariantSetCompressor(
                    config.element_dim,
                    token_counts[modality],
                    config.fusion_d_model,
                    config.rho_hidden_dim,
                    config.dropout,
                )
                for modality in MODALITY_ORDER
            }
        )
        self.missing_tokens = nn.ParameterDict(
            {
                modality: nn.Parameter(
                    torch.empty(token_counts[modality], config.fusion_d_model)
                )
                for modality in MODALITY_ORDER
            }
        )
        for tokens in self.missing_tokens.values():
            nn.init.normal_(tokens, std=0.02)

        self.fusion = FusionTransformer(
            config.fusion_d_model,
            config.fusion_heads,
            config.fusion_layers,
            config.fusion_ffn_dim,
            num_modalities=len(MODALITY_ORDER),
            dropout=config.dropout,
        )
        assert config.rna_output_dim is not None
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
        self.vocabulary_sizes = vocabulary_sizes
        self.token_counts = token_counts

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

    @staticmethod
    def _required_tensor(modality: str, inputs: ModalityInput, name: str) -> Tensor:
        try:
            value = inputs[name]
        except KeyError as error:
            raise ValueError(f"{modality} inputs require '{name}'") from error
        if not isinstance(value, Tensor):
            raise TypeError(f"{modality}.{name} must be a torch.Tensor")
        return value

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
        if not isinstance(inputs, Mapping):
            raise TypeError(f"{modality} inputs must be a mapping")
        feature_ids = self._required_tensor(modality, inputs, "feature_ids")
        values = self._required_tensor(modality, inputs, "values")
        token_mask = self._required_tensor(modality, inputs, "token_mask")
        if feature_ids.ndim != 2 or feature_ids.shape[0] != batch_size:
            raise ValueError(f"{modality}.feature_ids must have shape [B, L]")
        if token_mask.shape != feature_ids.shape:
            raise ValueError(f"{modality}.token_mask must match feature_ids [B, L]")
        if values.shape not in (feature_ids.shape, (*feature_ids.shape, 1)):
            raise ValueError(f"{modality}.values must have shape [B, L] or [B, L, 1]")

        row_present = present.to(device=feature_ids.device, dtype=torch.bool)
        valid = token_mask.to(device=feature_ids.device, dtype=torch.bool)
        valid = valid & row_present.unsqueeze(-1)
        if (row_present & ~valid.any(dim=1)).any():
            raise ValueError(f"observed {modality} cells must contain a valid token")

        # Sanitize every ignored element before embedding lookup.  Padding and
        # missing-row contents therefore cannot influence output or even raise
        # an out-of-vocabulary error.
        safe_feature_ids = torch.where(valid, feature_ids, torch.zeros_like(feature_ids))
        value_mask = valid if values.ndim == 2 else valid.unsqueeze(-1)
        safe_values = torch.where(value_mask, values, torch.zeros_like(values))
        state_ids = inputs.get("state_ids")
        safe_state_ids = None
        if state_ids is not None:
            if state_ids.shape != feature_ids.shape:
                raise ValueError(f"{modality}.state_ids must match feature_ids [B, L]")
            safe_state_ids = torch.where(valid, state_ids, torch.zeros_like(state_ids))
        categorical_ids = inputs.get("categorical_ids")
        safe_categorical_ids = None
        if categorical_ids is not None:
            if categorical_ids.ndim != 3 or categorical_ids.shape[:2] != feature_ids.shape:
                raise ValueError(
                    f"{modality}.categorical_ids must have shape [B, L, C]"
                )
            safe_categorical_ids = torch.where(
                valid.unsqueeze(-1),
                categorical_ids,
                torch.zeros_like(categorical_ids),
            )

        # Canonicalize the unique valid feature set by biological ID.  The
        # train-defined caches forbid duplicate IDs within a cell; enforcing
        # that boundary here makes common tuple permutations produce the same
        # reduction order (and hence bitwise-stable eval-mode outputs).
        sentinel = torch.iinfo(safe_feature_ids.dtype).max
        order = torch.argsort(
            torch.where(valid, safe_feature_ids, sentinel),
            dim=1,
            stable=True,
        )
        safe_feature_ids = safe_feature_ids.gather(1, order)
        valid = valid.gather(1, order)
        duplicate = (
            valid[:, 1:]
            & valid[:, :-1]
            & safe_feature_ids[:, 1:].eq(safe_feature_ids[:, :-1])
        )
        if duplicate.any():
            raise ValueError(f"duplicate valid {modality} feature ID within a cell")
        if safe_values.ndim == 2:
            safe_values = safe_values.gather(1, order)
        else:
            safe_values = safe_values.gather(
                1, order.unsqueeze(-1).expand_as(safe_values)
            )
        if safe_state_ids is not None:
            safe_state_ids = safe_state_ids.gather(1, order)
        if safe_categorical_ids is not None:
            safe_categorical_ids = safe_categorical_ids.gather(
                1,
                order.unsqueeze(-1).expand_as(safe_categorical_ids),
            )
        embedded = self.token_embeddings[modality](
            safe_feature_ids,
            safe_values,
            valid,
            state_ids=safe_state_ids,
            categorical_ids=safe_categorical_ids,
        )
        encoded = embedded + self.shared_phi(embedded)
        observed = self.compressors[modality](encoded, valid)
        return torch.where(
            row_present.view(batch_size, 1, 1),
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
            "rna_tokens": modality_tokens["rna"],
            "atac_tokens": modality_tokens["atac"],
            "adt_tokens": modality_tokens["adt"],
            "fusion_tokens": fusion_tokens,
            "modality_present": presence,
        }


TokenToCompressedTransformer = TokenCompressedTransformer
