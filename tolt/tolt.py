"""Feature-aware TriOmics Latent Transformer (TOLT).

The public forward interface deliberately mirrors the three biological
modalities.  Each modality input is either ``None`` (the whole batch is
missing that modality) or a mapping containing:

``feature_ids``
    ``LongTensor[B, L_m]``
``values``
    ``FloatTensor[B, L_m]``
``token_mask``
    ``BoolTensor[B, L_m]`` with ``True`` for valid tokens
``state_ids``
    optional ``LongTensor[B, L_m]``
``categorical_ids``
    optional ``LongTensor[B, L_m, C_m]``; currently useful for ATAC

``modality_present`` is a ``BoolTensor[B, 3]`` in the fixed order
``(RNA, ATAC, ADT)``.  This per-cell mask is independent of token padding and
allows arbitrary observed-modality combinations inside one batch.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .components import (
    AttentiveProjectionPool,
    FeatureTokenEmbedding,
    FusionTransformer,
    LatentCrossAttention,
    LocalTransformerEncoder,
    MLPHead,
)


MODALITY_ORDER = ("rna", "atac", "adt")
ModalityInput = Mapping[str, Tensor]


@dataclass(frozen=True)
class TOLTConfig:
    """Configuration for :class:`TriOmicsLatentTransformer`.

    Output dimensions default to the corresponding feature vocabulary for RNA
    and ADT.  ``atac_lsi_output_dim`` is intentionally separate from the ATAC
    peak vocabulary because the stable primary ATAC target is low-dimensional
    LSI rather than a dense peak vector.
    """

    rna_vocab_size: int
    atac_vocab_size: int
    adt_vocab_size: int
    rna_output_dim: int | None = None
    atac_lsi_output_dim: int = 128
    adt_output_dim: int | None = None
    num_conditions: int = 2

    local_d_model: int = 256
    local_heads: int = 8
    local_ffn_dim: int = 1024
    rna_local_layers: int = 2
    atac_local_layers: int = 2
    adt_local_layers: int = 1

    rna_latents: int = 64
    atac_latents: int = 96
    adt_latents: int = 32

    fusion_d_model: int = 512
    fusion_heads: int = 8
    fusion_layers: int = 8
    fusion_ffn_dim: int = 2048
    decoder_hidden_dim: int = 512
    dropout: float = 0.1

    num_token_states: int = 4
    atac_categorical_vocab_sizes: tuple[int, ...] = ()
    enable_peak_auxiliary: bool = False
    task_conditioned_decoders: bool = False
    observed_summary_side_channel: bool = False
    observed_summary_hidden_dim: int = 64
    factorized_representation: bool = False
    private_dim: int = 128
    identity_dim: int = 128
    modality_shared_dim: int = 128
    factorized_pool_hidden_dim: int = 192
    heteroscedastic_decoders: bool = False
    cross_modal_decoder_adapters: bool = False
    conditional_private_decoders: bool = False
    conditional_private_attention_heads: int = 4
    conditional_private_ffn_dim: int = 256
    conditional_private_residual_gating: bool = False
    conditional_private_gate_init_logit: float = -4.0
    technical_nuisance_representation: bool = False
    technical_dim: int = 64
    num_technical_domains: int = 7

    def __post_init__(self) -> None:
        if self.rna_output_dim is None:
            object.__setattr__(self, "rna_output_dim", self.rna_vocab_size)
        if self.adt_output_dim is None:
            object.__setattr__(self, "adt_output_dim", self.adt_vocab_size)

        positive_values = {
            "rna_vocab_size": self.rna_vocab_size,
            "atac_vocab_size": self.atac_vocab_size,
            "adt_vocab_size": self.adt_vocab_size,
            "rna_output_dim": self.rna_output_dim,
            "atac_lsi_output_dim": self.atac_lsi_output_dim,
            "adt_output_dim": self.adt_output_dim,
            "num_conditions": self.num_conditions,
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
            "num_token_states": self.num_token_states,
            "observed_summary_hidden_dim": self.observed_summary_hidden_dim,
        }
        if self.factorized_representation:
            positive_values.update(
                {
                    "private_dim": self.private_dim,
                    "identity_dim": self.identity_dim,
                    "modality_shared_dim": self.modality_shared_dim,
                    "factorized_pool_hidden_dim": self.factorized_pool_hidden_dim,
                }
            )
        if self.conditional_private_decoders:
            positive_values.update(
                {
                    "conditional_private_attention_heads": (
                        self.conditional_private_attention_heads
                    ),
                    "conditional_private_ffn_dim": self.conditional_private_ffn_dim,
                }
            )
        if self.technical_nuisance_representation:
            positive_values.update(
                {
                    "technical_dim": self.technical_dim,
                    "num_technical_domains": self.num_technical_domains,
                }
            )
        for name, value in positive_values.items():
            if value is None or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.local_d_model % self.local_heads:
            raise ValueError("local_d_model must be divisible by local_heads")
        if self.fusion_d_model % self.fusion_heads:
            raise ValueError("fusion_d_model must be divisible by fusion_heads")
        if self.conditional_private_decoders and not self.factorized_representation:
            raise ValueError(
                "conditional_private_decoders requires factorized_representation"
            )
        if (
            self.conditional_private_residual_gating
            and not self.conditional_private_decoders
        ):
            raise ValueError(
                "conditional_private_residual_gating requires "
                "conditional_private_decoders"
            )
        if not math.isfinite(self.conditional_private_gate_init_logit):
            raise ValueError("conditional_private_gate_init_logit must be finite")
        if (
            self.conditional_private_decoders
            and self.private_dim % self.conditional_private_attention_heads
        ):
            raise ValueError(
                "private_dim must be divisible by conditional_private_attention_heads"
            )
        if self.technical_nuisance_representation and not self.factorized_representation:
            raise ValueError(
                "technical_nuisance_representation requires factorized_representation"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if any(size <= 0 for size in self.atac_categorical_vocab_sizes):
            raise ValueError("ATAC categorical vocabulary sizes must be positive")


class TriOmicsLatentTransformer(nn.Module):
    """Feature-aware latent Transformer for paired RNA, ATAC, and ADT.

    Output mapping
    --------------
    ``cell_embedding``: ``[B, fusion_d_model]``
    ``rna_prediction``: ``[B, rna_output_dim]``
    ``atac_lsi_prediction``: ``[B, atac_lsi_output_dim]``
    ``adt_prediction``: ``[B, adt_output_dim]``
    ``condition_logits``: ``[B, num_conditions]``
    ``{modality}_latents``: ``[B, Q_m, fusion_d_model]``
    ``fusion_tokens``: ``[B, 1 + Q_task + Q_rna + Q_atac + Q_adt, fusion_d_model]``
    ``task_embeddings``: optional ``[B, 3, fusion_d_model]`` in the fixed
    reconstruction order RNA, ATAC-LSI, ADT
    ``modality_present``: normalized boolean ``[B, 3]`` mask
    ``peak_logits``: optional ``[B, S]`` scores for exactly the supplied
    ``peak_ids``; a full cell-by-peak-vocabulary tensor is never constructed
    """

    def __init__(self, config: TOLTConfig) -> None:
        super().__init__()
        self.config = config
        local_layers = {
            "rna": config.rna_local_layers,
            "atac": config.atac_local_layers,
            "adt": config.adt_local_layers,
        }
        vocabulary_sizes = {
            "rna": config.rna_vocab_size,
            "atac": config.atac_vocab_size,
            "adt": config.adt_vocab_size,
        }
        latent_counts = {
            "rna": config.rna_latents,
            "atac": config.atac_latents,
            "adt": config.adt_latents,
        }

        self.token_embeddings = nn.ModuleDict(
            {
                "rna": FeatureTokenEmbedding(
                    config.rna_vocab_size,
                    config.local_d_model,
                    num_states=config.num_token_states,
                    dropout=config.dropout,
                ),
                "atac": FeatureTokenEmbedding(
                    config.atac_vocab_size,
                    config.local_d_model,
                    num_states=config.num_token_states,
                    categorical_vocab_sizes=config.atac_categorical_vocab_sizes,
                    dropout=config.dropout,
                ),
                "adt": FeatureTokenEmbedding(
                    config.adt_vocab_size,
                    config.local_d_model,
                    num_states=config.num_token_states,
                    dropout=config.dropout,
                ),
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
            num_task_tokens=len(MODALITY_ORDER) if config.task_conditioned_decoders else 0,
            dropout=config.dropout,
        )
        self.observed_summary_projection = (
            nn.Sequential(
                nn.LayerNorm(18),
                nn.Linear(18, config.observed_summary_hidden_dim),
                nn.GELU(),
                nn.Linear(config.observed_summary_hidden_dim, config.fusion_d_model),
            )
            if config.observed_summary_side_channel
            else None
        )
        assert config.rna_output_dim is not None
        assert config.adt_output_dim is not None
        decoder_hidden = (config.decoder_hidden_dim,)
        if config.factorized_representation:
            self.private_poolers: nn.ModuleDict | None = nn.ModuleDict(
                {
                    modality: AttentiveProjectionPool(
                        config.fusion_d_model,
                        config.private_dim,
                        hidden_dim=config.factorized_pool_hidden_dim,
                        dropout=config.dropout,
                        normalize=False,
                    )
                    for modality in MODALITY_ORDER
                }
            )
            self.identity_poolers: nn.ModuleDict | None = nn.ModuleDict(
                {
                    modality: AttentiveProjectionPool(
                        config.fusion_d_model,
                        config.identity_dim,
                        hidden_dim=config.factorized_pool_hidden_dim,
                        dropout=config.dropout,
                    )
                    for modality in MODALITY_ORDER
                }
            )
            self.modality_shared_poolers: nn.ModuleDict | None = nn.ModuleDict(
                {
                    modality: AttentiveProjectionPool(
                        config.fusion_d_model,
                        config.modality_shared_dim,
                        hidden_dim=config.factorized_pool_hidden_dim,
                        dropout=config.dropout,
                    )
                    for modality in MODALITY_ORDER
                }
            )
            self.private_decoder_adapters: nn.ModuleDict | None = nn.ModuleDict(
                {
                    "rna": MLPHead(
                        config.private_dim,
                        config.rna_output_dim,
                        hidden_dims=(config.factorized_pool_hidden_dim,),
                        dropout=config.dropout,
                    ),
                    "atac": MLPHead(
                        config.private_dim,
                        config.atac_lsi_output_dim,
                        hidden_dims=(config.factorized_pool_hidden_dim,),
                        dropout=config.dropout,
                    ),
                    "adt": MLPHead(
                        config.private_dim,
                        config.adt_output_dim,
                        hidden_dims=(config.factorized_pool_hidden_dim,),
                        dropout=config.dropout,
                    ),
                }
            )
            for adapter in self.private_decoder_adapters.values():
                final = adapter.network[-1]
                assert isinstance(final, nn.Linear)
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
            factorized_decoder_input_dim = config.fusion_d_model + config.private_dim
            self.cross_modal_decoder_adapters: nn.ModuleDict | None = (
                nn.ModuleDict(
                    {
                        "rna": MLPHead(
                            factorized_decoder_input_dim,
                            config.rna_output_dim,
                            hidden_dims=(config.factorized_pool_hidden_dim,),
                            dropout=config.dropout,
                        ),
                        "atac": MLPHead(
                            factorized_decoder_input_dim,
                            config.atac_lsi_output_dim,
                            hidden_dims=(config.factorized_pool_hidden_dim,),
                            dropout=config.dropout,
                        ),
                        "adt": MLPHead(
                            factorized_decoder_input_dim,
                            config.adt_output_dim,
                            hidden_dims=(config.factorized_pool_hidden_dim,),
                            dropout=config.dropout,
                        ),
                    }
                )
                if config.cross_modal_decoder_adapters
                else None
            )
            if self.cross_modal_decoder_adapters is not None:
                for adapter in self.cross_modal_decoder_adapters.values():
                    final = adapter.network[-1]
                    assert isinstance(final, nn.Linear)
                    nn.init.zeros_(final.weight)
                    nn.init.zeros_(final.bias)
            if config.conditional_private_decoders:
                self.conditional_private_target_queries: nn.ParameterDict | None = (
                    nn.ParameterDict(
                        {
                            modality: nn.Parameter(torch.empty(1, config.private_dim))
                            for modality in MODALITY_ORDER
                        }
                    )
                )
                self.conditional_private_null_tokens: nn.ParameterDict | None = (
                    nn.ParameterDict(
                        {
                            modality: nn.Parameter(torch.empty(1, config.private_dim))
                            for modality in MODALITY_ORDER
                        }
                    )
                )
                self.conditional_private_source_modalities: nn.Parameter | None = (
                    nn.Parameter(torch.empty(len(MODALITY_ORDER), config.private_dim))
                )
                self.conditional_private_context_projections: nn.ModuleDict | None = (
                    nn.ModuleDict(
                        {
                            modality: nn.Linear(
                                config.modality_shared_dim,
                                config.private_dim,
                            )
                            for modality in MODALITY_ORDER
                        }
                    )
                )
                self.conditional_private_attentions: nn.ModuleDict | None = nn.ModuleDict(
                    {
                        modality: nn.MultiheadAttention(
                            config.private_dim,
                            config.conditional_private_attention_heads,
                            dropout=config.dropout,
                            batch_first=True,
                        )
                        for modality in MODALITY_ORDER
                    }
                )
                self.conditional_private_norms: nn.ModuleDict | None = nn.ModuleDict(
                    {
                        modality: nn.LayerNorm(config.private_dim)
                        for modality in MODALITY_ORDER
                    }
                )
                self.conditional_private_refiners: nn.ModuleDict | None = nn.ModuleDict(
                    {
                        modality: nn.Sequential(
                            nn.LayerNorm(config.private_dim),
                            nn.Linear(
                                config.private_dim,
                                config.conditional_private_ffn_dim,
                            ),
                            nn.GELU(),
                            nn.Dropout(config.dropout),
                            nn.Linear(
                                config.conditional_private_ffn_dim,
                                config.private_dim,
                            ),
                        )
                        for modality in MODALITY_ORDER
                    }
                )
                self.conditional_private_residual_gate_logits: nn.Parameter | None = (
                    nn.Parameter(
                        torch.full(
                            (len(MODALITY_ORDER), 2 ** len(MODALITY_ORDER)),
                            float(config.conditional_private_gate_init_logit),
                        )
                    )
                    if config.conditional_private_residual_gating
                    else None
                )
                for parameter in self.conditional_private_target_queries.values():
                    nn.init.normal_(parameter, std=0.02)
                for parameter in self.conditional_private_null_tokens.values():
                    nn.init.normal_(parameter, std=0.02)
                nn.init.normal_(self.conditional_private_source_modalities, std=0.02)
            else:
                self.conditional_private_target_queries = None
                self.conditional_private_null_tokens = None
                self.conditional_private_source_modalities = None
                self.conditional_private_context_projections = None
                self.conditional_private_attentions = None
                self.conditional_private_norms = None
                self.conditional_private_refiners = None
                self.conditional_private_residual_gate_logits = None
            if config.technical_nuisance_representation:
                self.technical_encoder: nn.Module | None = MLPHead(
                    config.fusion_d_model,
                    config.technical_dim,
                    hidden_dims=(config.factorized_pool_hidden_dim,),
                    dropout=config.dropout,
                )
                self.technical_to_shared: nn.Linear | None = nn.Linear(
                    config.technical_dim,
                    config.fusion_d_model,
                    bias=False,
                )
                self.technical_domain_head: nn.Module | None = MLPHead(
                    config.technical_dim,
                    config.num_technical_domains,
                    hidden_dims=(config.factorized_pool_hidden_dim,),
                    dropout=config.dropout,
                )
                nn.init.zeros_(self.technical_to_shared.weight)
            else:
                self.technical_encoder = None
                self.technical_to_shared = None
                self.technical_domain_head = None
        else:
            self.private_poolers = None
            self.identity_poolers = None
            self.modality_shared_poolers = None
            self.private_decoder_adapters = None
            self.cross_modal_decoder_adapters = None
            self.conditional_private_target_queries = None
            self.conditional_private_null_tokens = None
            self.conditional_private_source_modalities = None
            self.conditional_private_context_projections = None
            self.conditional_private_attentions = None
            self.conditional_private_norms = None
            self.conditional_private_refiners = None
            self.conditional_private_residual_gate_logits = None
            self.technical_encoder = None
            self.technical_to_shared = None
            self.technical_domain_head = None
            factorized_decoder_input_dim = config.fusion_d_model
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
        if config.heteroscedastic_decoders:
            self.rna_log_variance_decoder: MLPHead | None = MLPHead(
                factorized_decoder_input_dim,
                config.rna_output_dim,
                hidden_dims=decoder_hidden,
                dropout=config.dropout,
            )
            self.atac_lsi_log_variance_decoder: MLPHead | None = MLPHead(
                factorized_decoder_input_dim,
                config.atac_lsi_output_dim,
                hidden_dims=decoder_hidden,
                dropout=config.dropout,
            )
            self.adt_log_variance_decoder: MLPHead | None = MLPHead(
                factorized_decoder_input_dim,
                config.adt_output_dim,
                hidden_dims=decoder_hidden,
                dropout=config.dropout,
            )
            for variance_decoder in (
                self.rna_log_variance_decoder,
                self.atac_lsi_log_variance_decoder,
                self.adt_log_variance_decoder,
            ):
                final = variance_decoder.network[-1]
                assert isinstance(final, nn.Linear)
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)
        else:
            self.rna_log_variance_decoder = None
            self.atac_lsi_log_variance_decoder = None
            self.adt_log_variance_decoder = None
        condition_hidden = max(32, min(128, config.decoder_hidden_dim))
        self.condition_head = MLPHead(
            config.fusion_d_model,
            config.num_conditions,
            hidden_dims=(condition_hidden,),
            dropout=config.dropout,
        )
        if config.enable_peak_auxiliary:
            # Reuse the ATAC feature table: sampled target IDs and input-token
            # IDs therefore live in exactly the same train-defined vocabulary.
            self.peak_cell_projection: nn.Linear | None = nn.Linear(
                factorized_decoder_input_dim,
                config.local_d_model,
            )
            self.peak_bias: nn.Embedding | None = nn.Embedding(
                config.atac_vocab_size + 1,
                1,
                padding_idx=0,
            )
            nn.init.zeros_(self.peak_bias.weight)
        else:
            self.peak_cell_projection = None
            self.peak_bias = None

        # Retain these names in the module state for easier run inspection.
        self.vocabulary_sizes = vocabulary_sizes
        self.latent_counts = latent_counts

    @property
    def output_dimensions(self) -> dict[str, int]:
        """Return decoder dimensions without requiring a dummy forward pass."""

        assert self.config.rna_output_dim is not None
        assert self.config.adt_output_dim is not None
        return {
            "rna": self.config.rna_output_dim,
            "atac_lsi": self.config.atac_lsi_output_dim,
            "adt": self.config.adt_output_dim,
            "condition": self.config.num_conditions,
        }

    def _normalize_modality_present(self, modality_present: Tensor) -> Tensor:
        if modality_present.ndim != 2 or modality_present.shape[1] != len(MODALITY_ORDER):
            raise ValueError("modality_present must have shape [B, 3] in RNA/ATAC/ADT order")
        presence = modality_present.to(
            device=self.missing_latents["rna"].device,
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

    def _encode_modality(
        self,
        modality: str,
        inputs: ModalityInput | None,
        present: Tensor,
        batch_size: int,
    ) -> Tensor:
        missing = self.missing_latents[modality].unsqueeze(0).expand(batch_size, -1, -1)
        if inputs is None:
            if present.any():
                raise ValueError(
                    f"{modality} inputs are None but modality_present marks observed cells"
                )
            return self.latent_projections[modality](missing)

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

        state_ids = inputs.get("state_ids")
        categorical_ids = inputs.get("categorical_ids")
        embedded = self.token_embeddings[modality](
            feature_ids,
            values,
            token_mask,
            state_ids=state_ids,
            categorical_ids=categorical_ids,
        )
        encoded = self.local_encoders[modality](embedded, token_mask)
        observed = self.latent_poolers[modality](encoded, token_mask)
        selected = torch.where(
            present.to(device=observed.device).view(batch_size, 1, 1),
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
        peak_ids: Tensor | None = None,
        observed_summary: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Encode modalities and optionally score only caller-supplied peaks."""
        presence = self._normalize_modality_present(modality_present)
        batch_size = presence.shape[0]
        inputs_by_modality = {"rna": rna, "atac": atac, "adt": adt}
        projected_latents = {
            modality: self._encode_modality(
                modality,
                inputs_by_modality[modality],
                presence[:, index],
                batch_size,
            )
            for index, modality in enumerate(MODALITY_ORDER)
        }
        cell_embedding, fusion_tokens = self.fusion(
            [projected_latents[modality] for modality in MODALITY_ORDER]
        )
        if self.observed_summary_projection is not None:
            if observed_summary is None:
                raise ValueError(
                    "observed-summary side channel is enabled but no summary was supplied"
                )
            if observed_summary.shape != (batch_size, 18):
                raise ValueError("observed_summary must have shape [B, 18]")
            if not torch.isfinite(observed_summary).all():
                raise ValueError("observed_summary contains non-finite values")
            summary_residual = self.observed_summary_projection(
                observed_summary.to(
                    device=cell_embedding.device,
                    dtype=cell_embedding.dtype,
                )
            )
            cell_embedding = cell_embedding + summary_residual
            fusion_tokens = fusion_tokens.clone()
            fusion_tokens[:, 0] = cell_embedding
        elif observed_summary is not None:
            raise ValueError(
                "observed_summary was supplied while the side channel is disabled"
            )
        raw_fused_embedding = cell_embedding
        technical_embedding: Tensor | None = None
        technical_residual: Tensor | None = None
        if self.technical_encoder is not None:
            assert self.technical_to_shared is not None
            technical_embedding = self.technical_encoder(raw_fused_embedding)
            technical_residual = self.technical_to_shared(technical_embedding)
            cell_embedding = raw_fused_embedding - technical_residual
            fusion_tokens = fusion_tokens.clone()
            fusion_tokens[:, 0] = cell_embedding
        if self.config.task_conditioned_decoders:
            task_embeddings = fusion_tokens[:, 1 : 1 + len(MODALITY_ORDER)]
            if task_embeddings.shape[1] != len(MODALITY_ORDER):
                raise RuntimeError("task-conditioned fusion omitted a reconstruction query")
            decoder_embeddings = {
                modality: task_embeddings[:, index]
                for index, modality in enumerate(MODALITY_ORDER)
            }
        else:
            task_embeddings = None
            decoder_embeddings = {modality: cell_embedding for modality in MODALITY_ORDER}
        private_embeddings: Tensor | None = None
        identity_embeddings: Tensor | None = None
        modality_shared_embeddings: Tensor | None = None
        cross_source_private: dict[str, Tensor] = {}
        cross_source_available: dict[str, Tensor] = {}
        conditional_private: dict[str, Tensor] = {}
        conditional_private_attention: dict[str, Tensor] = {}
        conditional_private_residual_gates: dict[str, Tensor] = {}
        if self.private_poolers is not None:
            assert self.identity_poolers is not None
            assert self.modality_shared_poolers is not None
            private_embeddings = torch.stack(
                [
                    self.private_poolers[modality](
                        projected_latents[modality], presence[:, index]
                    )
                    for index, modality in enumerate(MODALITY_ORDER)
                ],
                dim=1,
            )
            identity_embeddings = torch.stack(
                [
                    self.identity_poolers[modality](
                        projected_latents[modality], presence[:, index]
                    )
                    for index, modality in enumerate(MODALITY_ORDER)
                ],
                dim=1,
            )
            modality_shared_embeddings = torch.stack(
                [
                    self.modality_shared_poolers[modality](
                        projected_latents[modality], presence[:, index]
                    )
                    for index, modality in enumerate(MODALITY_ORDER)
                ],
                dim=1,
            )
            for target_index, modality in enumerate(MODALITY_ORDER):
                source_mask = presence.clone()
                source_mask[:, target_index] = False
                source_weights = source_mask.to(private_embeddings.dtype).unsqueeze(-1)
                source_count = source_weights.sum(dim=1).clamp_min(1.0)
                cross_source_private[modality] = (
                    private_embeddings * source_weights
                ).sum(dim=1) / source_count
                cross_source_available[modality] = source_mask.any(dim=1)
            if self.conditional_private_attentions is not None:
                assert self.conditional_private_target_queries is not None
                assert self.conditional_private_null_tokens is not None
                assert self.conditional_private_source_modalities is not None
                assert self.conditional_private_context_projections is not None
                assert self.conditional_private_norms is not None
                assert self.conditional_private_refiners is not None
                source_tokens = private_embeddings + self.conditional_private_source_modalities[
                    None, :, :
                ]
                presence_code = (
                    presence.to(dtype=torch.long)
                    * torch.tensor(
                        (1, 2, 4),
                        device=presence.device,
                        dtype=torch.long,
                    ).unsqueeze(0)
                ).sum(dim=1)
                for target_index, modality in enumerate(MODALITY_ORDER):
                    source_mask = presence.clone()
                    source_mask[:, target_index] = False
                    source_weights = source_mask.to(
                        modality_shared_embeddings.dtype
                    ).unsqueeze(-1)
                    source_count = source_weights.sum(dim=1).clamp_min(1.0)
                    shared_context = (
                        modality_shared_embeddings * source_weights
                    ).sum(dim=1) / source_count
                    query = self.conditional_private_target_queries[modality].expand(
                        batch_size, -1
                    ) + self.conditional_private_context_projections[modality](
                        shared_context
                    )
                    null_token = self.conditional_private_null_tokens[modality].expand(
                        batch_size, -1
                    )
                    key_value = torch.cat(
                        [source_tokens, null_token.unsqueeze(1)],
                        dim=1,
                    )
                    key_padding_mask = torch.cat(
                        [
                            ~source_mask,
                            torch.zeros(
                                batch_size,
                                1,
                                device=presence.device,
                                dtype=torch.bool,
                            ),
                        ],
                        dim=1,
                    )
                    attended, attention = self.conditional_private_attentions[modality](
                        query.unsqueeze(1),
                        key_value,
                        key_value,
                        key_padding_mask=key_padding_mask,
                        need_weights=True,
                    )
                    inferred = self.conditional_private_norms[modality](
                        query + attended.squeeze(1)
                    )
                    inferred = inferred + self.conditional_private_refiners[modality](
                        inferred
                    )
                    conditional_private[modality] = inferred
                    conditional_private_attention[modality] = attention.squeeze(1)
                    if self.conditional_private_residual_gate_logits is None:
                        gate = torch.ones(
                            batch_size,
                            device=presence.device,
                            dtype=inferred.dtype,
                        )
                    else:
                        gate = torch.sigmoid(
                            self.conditional_private_residual_gate_logits[
                                target_index
                            ].gather(0, presence_code)
                        ).to(dtype=inferred.dtype)
                    conditional_private_residual_gates[modality] = gate
                decoder_private_embeddings = torch.stack(
                    [
                        torch.where(
                            presence[:, index].unsqueeze(-1),
                            private_embeddings[:, index],
                            conditional_private[modality],
                        )
                        for index, modality in enumerate(MODALITY_ORDER)
                    ],
                    dim=1,
                )
            else:
                decoder_private_embeddings = torch.stack(
                    [
                        torch.where(
                            presence[:, index].unsqueeze(-1),
                            private_embeddings[:, index],
                            cross_source_private[modality],
                        )
                        if self.config.cross_modal_decoder_adapters
                        else private_embeddings[:, index]
                        for index, modality in enumerate(MODALITY_ORDER)
                    ],
                    dim=1,
                )
            factorized_decoder_embeddings = {
                modality: torch.cat(
                    [decoder_embeddings[modality], decoder_private_embeddings[:, index]],
                    dim=-1,
                )
                for index, modality in enumerate(MODALITY_ORDER)
            }
        else:
            factorized_decoder_embeddings = decoder_embeddings
            decoder_private_embeddings = None
        predictions = {
            "rna": self.rna_decoder(decoder_embeddings["rna"]),
            "atac": self.atac_lsi_decoder(decoder_embeddings["atac"]),
            "adt": self.adt_decoder(decoder_embeddings["adt"]),
        }
        if self.private_decoder_adapters is not None:
            assert private_embeddings is not None
            assert decoder_private_embeddings is not None
            for index, modality in enumerate(MODALITY_ORDER):
                residual = self.private_decoder_adapters[modality](
                    decoder_private_embeddings[:, index]
                )
                if self.conditional_private_attentions is not None:
                    residual_available = presence[:, index] | cross_source_available[modality]
                    residual_gate = torch.where(
                        presence[:, index],
                        torch.ones_like(conditional_private_residual_gates[modality]),
                        conditional_private_residual_gates[modality],
                    )
                else:
                    residual_available = presence[:, index]
                    residual_gate = torch.ones(
                        batch_size,
                        device=residual.device,
                        dtype=residual.dtype,
                    )
                predictions[modality] = predictions[modality] + residual * (
                    residual_available.to(
                        device=residual.device,
                        dtype=residual.dtype,
                    )
                    * residual_gate.to(device=residual.device, dtype=residual.dtype)
                ).unsqueeze(-1)
        if self.cross_modal_decoder_adapters is not None:
            for modality in MODALITY_ORDER:
                cross_input = torch.cat(
                    [decoder_embeddings[modality], cross_source_private[modality]], dim=-1
                )
                residual = self.cross_modal_decoder_adapters[modality](cross_input)
                predictions[modality] = predictions[modality] + residual * cross_source_available[
                    modality
                ].to(device=residual.device, dtype=residual.dtype).unsqueeze(-1)
        outputs = {
            "cell_embedding": cell_embedding,
            "rna_prediction": predictions["rna"],
            "atac_lsi_prediction": predictions["atac"],
            "adt_prediction": predictions["adt"],
            "condition_logits": self.condition_head(cell_embedding),
            "rna_latents": projected_latents["rna"],
            "atac_latents": projected_latents["atac"],
            "adt_latents": projected_latents["adt"],
            "fusion_tokens": fusion_tokens,
            "modality_present": presence,
        }
        if technical_embedding is not None:
            assert technical_residual is not None
            assert self.technical_domain_head is not None
            outputs["raw_fused_embedding"] = raw_fused_embedding
            outputs["technical_embedding"] = technical_embedding
            outputs["technical_residual"] = technical_residual
            outputs["technical_domain_logits"] = self.technical_domain_head(
                technical_embedding
            )
        if private_embeddings is not None:
            assert identity_embeddings is not None
            assert modality_shared_embeddings is not None
            outputs["shared_embedding"] = cell_embedding
            outputs["private_embeddings"] = private_embeddings
            outputs["identity_embeddings"] = identity_embeddings
            outputs["modality_shared_embeddings"] = modality_shared_embeddings
            assert decoder_private_embeddings is not None
            outputs["decoder_private_embeddings"] = decoder_private_embeddings
            if conditional_private:
                outputs["conditional_private_embeddings"] = torch.stack(
                    [conditional_private[modality] for modality in MODALITY_ORDER],
                    dim=1,
                )
                outputs["conditional_private_attention"] = torch.stack(
                    [
                        conditional_private_attention[modality]
                        for modality in MODALITY_ORDER
                    ],
                    dim=1,
                )
                outputs["conditional_private_available"] = torch.stack(
                    [cross_source_available[modality] for modality in MODALITY_ORDER],
                    dim=1,
                )
                outputs["conditional_private_residual_gates"] = torch.stack(
                    [
                        conditional_private_residual_gates[modality]
                        for modality in MODALITY_ORDER
                    ],
                    dim=1,
                )
        if self.rna_log_variance_decoder is not None:
            assert self.atac_lsi_log_variance_decoder is not None
            assert self.adt_log_variance_decoder is not None
            outputs["rna_log_variance"] = self.rna_log_variance_decoder(
                factorized_decoder_embeddings["rna"]
            ).clamp(-8.0, 6.0)
            outputs["atac_lsi_log_variance"] = self.atac_lsi_log_variance_decoder(
                factorized_decoder_embeddings["atac"]
            ).clamp(-8.0, 6.0)
            outputs["adt_log_variance"] = self.adt_log_variance_decoder(
                factorized_decoder_embeddings["adt"]
            ).clamp(-8.0, 6.0)
        if task_embeddings is not None:
            outputs["task_embeddings"] = task_embeddings
        if observed_summary is not None:
            outputs["observed_summary"] = observed_summary
        if self.peak_cell_projection is not None:
            outputs["atac_peak_query"] = self.peak_cell_projection(
                factorized_decoder_embeddings["atac"]
            )
        if peak_ids is not None:
            if self.peak_cell_projection is None or self.peak_bias is None:
                raise ValueError(
                    "peak_ids were supplied but TOLT peak auxiliary decoding is disabled"
                )
            normalized_ids, logits = self.score_peak_ids(
                outputs["atac_peak_query"], peak_ids
            )
            outputs["peak_ids"] = normalized_ids
            outputs["peak_logits"] = logits
        return outputs

    def score_peak_ids(
        self,
        peak_query: Tensor,
        peak_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Score arbitrary peak-ID chunks from one already encoded cell query."""

        if self.peak_cell_projection is None or self.peak_bias is None:
            raise ValueError("peak scoring is disabled for this TOLT configuration")
        if peak_query.ndim != 2 or peak_query.shape[1] != self.config.local_d_model:
            raise ValueError("peak_query must have shape [B, local_d_model]")
        if peak_ids.ndim != 2 or peak_ids.shape[0] != peak_query.shape[0]:
            raise ValueError("peak_ids must have shape [B, sampled_peaks]")
        peak_ids = peak_ids.to(device=peak_query.device, dtype=torch.long)
        if peak_ids.numel() and (
            int(peak_ids.min()) < 0
            or int(peak_ids.max()) > self.config.atac_vocab_size
        ):
            raise ValueError(f"peak_ids must lie in [0, {self.config.atac_vocab_size}]")
        peak_embeddings = self.token_embeddings["atac"].feature_embedding(peak_ids)
        peak_logits = torch.einsum("bd,bsd->bs", peak_query, peak_embeddings)
        peak_logits = peak_logits / math.sqrt(self.config.local_d_model)
        peak_logits = peak_logits + self.peak_bias(peak_ids).squeeze(-1)
        return peak_ids, peak_logits


# Short alias for config-driven construction and experiment labels.
TOLT = TriOmicsLatentTransformer


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable scalar parameters in ``model``."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
