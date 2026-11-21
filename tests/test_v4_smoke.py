"""Unit smoke tests for TOLT core methods, compositional CLR, and 2-GMM gating."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from triomics import TOLTPanelCalibrator, __version__, build_tolt_base110
from triomics.inference import quickstart_cli
from triomics.models import TOLTConfig, TriOmicsLatentTransformer, count_trainable_parameters
from triomics.v4.cohort_loaders import seurat_clr
from triomics.v4.core_methods import fit_predict_tolt_adapter_suite
from triomics.v4.immunophenotyping import fit_bimodal_1d_gate


def test_tolt_base110_parameter_count() -> None:
    cfg = TOLTConfig(
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
    model = TriOmicsLatentTransformer(cfg)
    assert count_trainable_parameters(model) == 30_989_097


def test_clr_and_oof_simplex_convexity() -> None:
    rng = np.random.default_rng(7)
    counts = rng.poisson(5.0, size=(120, 8)).astype(np.float32)
    clr = seurat_clr(counts)
    assert clr.shape == (120, 8) and np.all(clr >= 0.0)

    svd_tr = rng.normal(size=(80, 16)).astype(np.float32)
    svd_te = rng.normal(size=(40, 16)).astype(np.float32)
    emb_tr = rng.normal(size=(80, 32)).astype(np.float32)
    emb_te = rng.normal(size=(40, 32)).astype(np.float32)
    y_tr = clr[:80]
    out = fit_predict_tolt_adapter_suite(svd_tr, svd_te, emb_tr, emb_te, y_tr, seed=42)
    w = out["TOLT"]["weights"]
    assert w.shape == (8, 3)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-4)
    assert np.all(w >= -1e-6)


def test_bimodal_2gmm_gate_separates_modes() -> None:
    rng = np.random.default_rng(11)
    x = np.concatenate([rng.normal(0.2, 0.08, 900), rng.normal(2.1, 0.25, 100)])
    thr, pos_frac = fit_bimodal_1d_gate(x, nonzero_only=False)
    assert 0.6 < thr < 1.6
    assert 0.07 < pos_frac < 0.14


def test_public_inference_api() -> None:
    assert __version__ == "4.5.0"
    model = build_tolt_base110()
    assert count_trainable_parameters(model) == 30_989_097
    quickstart_cli()
    cal = TOLTPanelCalibrator(seed=42)
    rng = np.random.default_rng(5)
    out = cal.fit_predict(
        rng.normal(size=(60, 12)).astype(np.float32),
        rng.normal(size=(20, 12)).astype(np.float32),
        rng.normal(size=(60, 24)).astype(np.float32),
        rng.normal(size=(20, 24)).astype(np.float32),
        seurat_clr(rng.poisson(3.0, size=(60, 4)).astype(np.float32)),
    )
    assert out["pred"].shape == (20, 4)
    assert np.allclose(out["weights"].sum(axis=1), 1.0, atol=1e-4)
