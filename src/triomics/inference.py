"""Public frozen-backbone inference and cross-fitted Δ²-simplex calibration API for TOLT v4."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from triomics.models import TOLTConfig, TriOmicsLatentTransformer
from triomics.v4.cohort_loaders import seurat_clr
from triomics.v4.core_methods import fit_predict_tolt_adapter_suite
from triomics.v4.immunophenotyping import fit_bimodal_1d_gate


def default_tolt_base110_config() -> TOLTConfig:
    """Return the exact 30,989,097-parameter TOLT-Base-110 frozen encoder configuration."""
    return TOLTConfig(
        rna_vocab_size=2048,
        atac_vocab_size=30000,
        adt_vocab_size=210,
        rna_output_dim=2048,
        atac_lsi_output_dim=128,
        adt_output_dim=210,
        local_d_model=256,
        local_heads=8,
        local_ffn_dim=1024,
        rna_local_layers=2,
        atac_local_layers=2,
        adt_local_layers=1,
        rna_latents=48,
        atac_latents=64,
        adt_latents=24,
        fusion_d_model=384,
        fusion_heads=8,
        fusion_layers=6,
        fusion_ffn_dim=1536,
        decoder_hidden_dim=384,
        enable_peak_auxiliary=True,
        factorized_representation=True,
        private_dim=128,
        identity_dim=128,
        modality_shared_dim=128,
        factorized_pool_hidden_dim=192,
        heteroscedastic_decoders=True,
    )


def build_tolt_base110() -> TriOmicsLatentTransformer:
    """Instantiate the 31.0M-parameter Perceiver-style TOLT-Base-110 encoder."""
    return TriOmicsLatentTransformer(default_tolt_base110_config())


@dataclass
class TOLTPanelCalibrator:
    """Three-fold cross-fitted Cholesky ridge + residual + Δ²-simplex target-panel calibrator."""

    seed: int = 42

    def fit_predict(
        self,
        svd_ref: np.ndarray,
        svd_query: np.ndarray,
        emb_ref: np.ndarray,
        emb_query: np.ndarray,
        adt_clr_ref: np.ndarray,
    ) -> dict[str, np.ndarray]:
        suite = fit_predict_tolt_adapter_suite(
            svd_ref, svd_query, emb_ref, emb_query, adt_clr_ref, seed=self.seed
        )
        return {
            "pred": suite["TOLT"]["pred"],
            "weights": suite["TOLT"]["weights"],
            "pred_direct": suite["TOLT-Direct"]["pred"],
            "pred_residual": suite["TOLT-Residual"]["pred"],
            "pred_local_svd": suite["SVD-Ridge"]["pred"],
        }


def quickstart_cli() -> None:
    """Console entry point verifying TOLT-Base-110 instantiation and OOF Δ²-simplex calibration."""
    rng = np.random.default_rng(7)
    counts = rng.poisson(4.0, size=(120, 6)).astype(np.float32)
    clr = seurat_clr(counts)
    cal = TOLTPanelCalibrator(seed=42)
    out = cal.fit_predict(
        rng.normal(size=(80, 16)).astype(np.float32),
        rng.normal(size=(40, 16)).astype(np.float32),
        rng.normal(size=(80, 32)).astype(np.float32),
        rng.normal(size=(40, 32)).astype(np.float32),
        clr[:80],
    )
    tau, prev = fit_bimodal_1d_gate(
        np.concatenate([rng.normal(0.2, 0.08, 180), rng.normal(2.0, 0.22, 20)]),
        nonzero_only=False,
    )
    print(
        f"TOLT v4.5 ready | pred={out['pred'].shape} | simplex_sum={out['weights'].sum(axis=1).mean():.4f} | 2-GMM tau*={tau:.3f} ({prev*100:.1f}%)"
    )
