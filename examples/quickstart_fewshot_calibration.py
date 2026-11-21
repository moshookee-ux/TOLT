"""Minimal end-to-end example: 3-fold OOF Cholesky Δ²-simplex calibration & 2-GMM gating.

Demonstrates how TOLT adapts frozen 384-d tri-omic representations + reference
singular subspaces to a custom P-antibody panel in milliseconds and fits
prevalence-invariant Bayesian 2-GMM surface gates.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from triomics.v4.cohort_loaders import seurat_clr
from triomics.v4.core_methods import fit_predict_tolt_adapter_suite
from triomics.v4.immunophenotyping import fit_bimodal_1d_gate


def main() -> None:
    rng = np.random.default_rng(42)
    n_ref, n_query, n_proteins = 250, 2000, 25

    # Simulate coupled biological latent state shared across reference & unseen donor
    bio_ref = rng.normal(size=(n_ref, 12)).astype(np.float32)
    bio_qry = rng.normal(size=(n_query, 12)).astype(np.float32)

    w_svd = rng.normal(scale=0.35, size=(12, 64)).astype(np.float32)
    w_emb = rng.normal(scale=0.55, size=(12, 384)).astype(np.float32)
    w_adt = rng.uniform(0.2, 1.4, size=(12, n_proteins)).astype(np.float32)

    svd_ref = bio_ref @ w_svd + rng.normal(scale=0.5, size=(n_ref, 64)).astype(np.float32)
    svd_qry = bio_qry @ w_svd + rng.normal(scale=0.5, size=(n_query, 64)).astype(np.float32)
    emb_ref = bio_ref @ w_emb + rng.normal(scale=0.3, size=(n_ref, 384)).astype(np.float32)
    emb_qry = bio_qry @ w_emb + rng.normal(scale=0.3, size=(n_query, 384)).astype(np.float32)

    raw_counts_ref = rng.poisson(np.exp(np.clip(bio_ref @ w_adt, -1.5, 3.5))).astype(np.float32)
    raw_counts_qry = rng.poisson(np.exp(np.clip(bio_qry @ w_adt, -1.5, 3.5))).astype(np.float32)
    y_ref = seurat_clr(raw_counts_ref)
    y_qry = seurat_clr(raw_counts_qry)

    t0 = time.perf_counter()
    suite = fit_predict_tolt_adapter_suite(svd_ref, svd_qry, emb_ref, emb_qry, y_ref, seed=42)
    dt = time.perf_counter() - t0

    pred_qry = suite["TOLT"]["pred"]
    weights = suite["TOLT"]["weights"]
    pccs = [np.corrcoef(y_qry[:, p], pred_qry[:, p])[0, 1] for p in range(n_proteins)]

    # Bimodal 2-GMM equal-posterior gate on rare ~10%-positive lineage receptor
    bimodal_ref = np.concatenate([rng.normal(0.22, 0.09, 225), rng.normal(2.05, 0.24, 25)])
    tau_star, pos_frac = fit_bimodal_1d_gate(bimodal_ref, nonzero_only=False)

    print(f"[TOLT Quickstart] Calibrated {n_proteins} ADTs from n_ref={n_ref} -> {n_query} query cells in {dt:.3f} s")
    print(f"  Median query PCC : {float(np.median(pccs)):.3f}")
    print(f"  Mean Δ² weights  : w_loc={weights[:,0].mean():.3f}, w_dir={weights[:,1].mean():.3f}, w_cor={weights[:,2].mean():.3f}")
    print(f"  2-GMM Bayes gate : tau*={tau_star:.3f} CLR (estimated ref prevalence={pos_frac*100:.1f}%)")


if __name__ == "__main__":
    main()
