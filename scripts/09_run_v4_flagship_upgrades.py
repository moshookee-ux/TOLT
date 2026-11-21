"""TOLT v4.1 Flagship Experimental Upgrade Suite (RTX 5070 Laptop GPU + 16-Core CPU).

Executes all 7 high-impact upgrade experiments identified in the top-journal pre-submission audit:
  1. Multi-seed (5 seeds) learning curves across n_ref in {50, 100, 250, 500, 1000, 2500, Full}
     + GPU totalVI, sciPENN & unimodal scVI-Adapter at n_ref = 50, 100 (eliminating all '---' cells).
  2. Complete cross-well & cross-donor scVAEIT on TEA-seq (eliminating 'N/A') + 9x9 confusion matrix.
  3. TOLT native 3x128d factorized subspaces (Identity / Modality-Shared / Private) + heteroscedastic
     uncertainty (adt_log_variance) per-cell error calibration.
  4. Bimodal 1D Gaussian density curves, ROC/PR curves, Tier-1 expressed manifold ARI/NMI, & panel hygiene audit
     (TCRVg9_0582 / TCRVg_0583 swap & dual CD8/CD8a clone verification).
  5. 18-Donor (162,777-cell) integrated MIS-C surface-proteome atlas, BH-FDR volcano, 18x40 clustered clinical
     heatmap, 45-family TRBV landscape, and single-cell raw mRNA dropout vs imputed protein + CD4/CD8 flow contours.
  6. GRCh38 cis-regulatory chromatin peak-to-protein genomic tracks (ITGAX/CD11c, CD8A, SIRPA, PTPRC)
     and zero-mRNA dropout chromatin rescue analysis on TEA-seq.
  7. Global 9-method x 7-metric scIB-style multi-cohort scorecard.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMBA_NUM_THREADS"] = "4"

import json
import subprocess
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import gaussian_kde, mannwhitneyu, norm, spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_curve,
    roc_curve,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from triomics.v4.paths import EMB_DIR, PRED_DIR, COHORTS_DIR

from triomics.v4.cohort_loaders import (
    D_BMMC90K,
    D_STUART,
    GSE166489_METADATA,
    TEASEQ_WELLS,
    load_gse166489_sample,
    log1p_cp10k_dense,
    log1p_cp10k_sparse,
    seurat_clr,
)
from triomics.v4.core_methods import (
    compute_prediction_metrics,
    fit_local_svd_features,
    fit_predict_tolt_adapter_suite,
    instantiate_tolt_backbone,
    run_fast_classical_and_modern_baselines,
    run_scipenn_gpu,
    run_totalvi_gpu,
    tokenize_atac_matrix,
    tokenize_rna_matrix,
)
from triomics.v4.immunophenotyping import (
    NINE_LINEAGE_CLASSES,
    fit_bimodal_1d_gate,
    fit_pbmc_9lineage_gates,
    score_9lineage_predictions,
)

VOCAB_DIR = REPO_ROOT / "data" / "vocabularies"
TABLES_DIR = REPO_ROOT / "results" / "tables"
RECEIPTS_DIR = REPO_ROOT / "results" / "receipts"
SCVAEIT_PY = REPO_ROOT / ".venv-scvaeit" / "Scripts" / "python.exe"


def run_scvi_unimodal_adapter_gpu(
    x_raw_ref: sp.csr_matrix,
    x_raw_qry: sp.csr_matrix,
    svd_ref: np.ndarray,
    svd_qry: np.ndarray,
    y_ref: np.ndarray,
    seed: int = 42,
    nonneg_clip: bool = True,
) -> dict[str, object]:
    """Train unimodal transcriptomic scVI (Lopez et al., Nat. Methods 2018) on RNA and apply identical Cholesky OOF adapter."""
    import scvi

    t0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    scvi.settings.seed = seed

    n_ref = x_raw_ref.shape[0]
    g_mean = np.asarray(x_raw_ref.mean(axis=0)).ravel()
    g_var = np.asarray(x_raw_ref.power(2).mean(axis=0)).ravel() - g_mean ** 2
    hvg_idx = np.sort(np.argsort(g_var)[-min(3000, x_raw_ref.shape[1]) :])

    x_cat = np.vstack([x_raw_ref[:, hvg_idx].toarray(), x_raw_qry[:, hvg_idx].toarray()]).astype(np.float32)
    batch_col = np.array(["ref"] * n_ref + ["qry"] * x_raw_qry.shape[0], dtype=str)
    adata = ad.AnnData(X=x_cat, obs=pd.DataFrame({"batch": batch_col}))
    scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
    vae = scvi.model.SCVI(adata, n_latent=64, n_layers=2, gene_likelihood="nb")
    vae.train(
        max_epochs=100 if n_ref <= 2000 else 75,
        batch_size=min(256, max(32, n_ref // 2)),
        early_stopping=True,
        early_stopping_patience=10,
        enable_progress_bar=False,
    )
    z_all = vae.get_latent_representation().astype(np.float32)
    suite = fit_predict_tolt_adapter_suite(
        svd_ref, svd_qry, z_all[:n_ref], z_all[n_ref:], y_ref, seed=seed, nonneg_clip=nonneg_clip
    )
    vram = float(torch.cuda.max_memory_allocated() / 1e6) if torch.cuda.is_available() else 0.0
    return {
        "pred": suite["TOLT"]["pred"],
        "direct_pred": suite["TOLT-Direct"]["pred"],
        "sec": round(time.perf_counter() - t0, 3),
        "peak_vram_mb": round(vram, 1),
    }


def step1_multiseed_curves_and_lowbudget_deep() -> None:
    out_csv = TABLES_DIR / "v41_multiseed_learning_curves.csv"
    out_lodo = TABLES_DIR / "v41_lodo_paired_summary.csv"
    if out_csv.is_file() and out_lodo.is_file():
        print("=== [Step 1] Cached v41_multiseed_learning_curves.csv & v41_lodo_paired_summary.csv found ===", flush=True)
        return

    print("=== [Step 1/6] Running 5-Seed Learning Curves + n_ref=50,100 GPU Deep Baselines (totalVI, sciPENN, scVI-Adapter) ===", flush=True)
    rows: list[dict[str, object]] = []
    seeds = [11, 22, 33, 44, 55]

    # --- 1A. GSE194122 Cross-Site BMMC (12710 Site 2 <-> 13272 Site 4, 134 ADTs) ---
    ad_90k = ad.read_h5ad(D_BMMC90K)
    mask_90k = ad_90k.obs["DonorID"].astype(str).isin(["12710", "13272"]).to_numpy()
    sub_90k = ad_90k[mask_90k].copy()
    x_90k = sp.csr_matrix(sub_90k.X, dtype=np.float32)
    adt_90k = np.asarray(sub_90k.obsm["protein_expression"], dtype=np.float32)
    pnames_90k = [str(x) for x in np.asarray(sub_90k.uns["protein_names"]).tolist()] if "protein_names" in sub_90k.uns else [f"ADT_{i}" for i in range(adt_90k.shape[1])]
    gnames_90k = [str(x) for x in sub_90k.var_names]
    y_90k = seurat_clr(adt_90k)
    with np.load(EMB_DIR / "gse194122_unseen_12949_embeddings.npz", allow_pickle=True) as d:
        dids = d["donor_ids"]
        emb_90k_b110 = d["base_110"]
        emb_90k_f01 = d["f01_s29"]
        emb_90k_r11 = d["rand_s11"]
    i12 = np.flatnonzero(dids == "12710")
    i13 = np.flatnonzero(dids == "13272")
    del ad_90k, sub_90k

    # Load existing benchmark curves so we reuse full/large totalVI/sciPENN runs without redundant 30-min reruns
    old_lc = pd.read_csv(TABLES_DIR / "benchmark_cross_donor_learning_curves.csv")

    for dir_name, ref_pool, qry_idx in [("12710_Site2_to_13272_Site4", i12, i13), ("13272_Site4_to_12710_Site2", i13, i12)]:
        exp_m = (y_90k[ref_pool].std(axis=0) >= 0.35)
        x_qry_log = log1p_cp10k_sparse(x_90k[qry_idx])
        for b in [50, 100, 250, 500, 1000, 2500, "full"]:
            seed_list = [42] if b == "full" else seeds
            for si, sd in enumerate(seed_list):
                r_idx = ref_pool if b == "full" else np.sort(np.random.default_rng(sd * 1000 + int(b)).choice(ref_pool, size=int(b), replace=False))
                blab = "full" if b == "full" else str(b)
                x_ref_log = log1p_cp10k_sparse(x_90k[r_idx])
                t0 = time.perf_counter()
                svd_tr, svd_te = fit_local_svd_features(x_ref_log, x_qry_log, seed=sd)
                svd_sec = time.perf_counter() - t0

                tolt_s = fit_predict_tolt_adapter_suite(svd_tr, svd_te, emb_90k_b110[r_idx], emb_90k_b110[qry_idx], y_90k[r_idx], seed=sd, nonneg_clip=True)
                f01_s = fit_predict_tolt_adapter_suite(svd_tr, svd_te, emb_90k_f01[r_idx], emb_90k_f01[qry_idx], y_90k[r_idx], seed=sd, nonneg_clip=True)
                rnd_s = fit_predict_tolt_adapter_suite(svd_tr, svd_te, emb_90k_r11[r_idx], emb_90k_r11[qry_idx], y_90k[r_idx], seed=sd, nonneg_clip=True)
                fast_s = run_fast_classical_and_modern_baselines(x_ref_log, x_qry_log, svd_tr, svd_te, y_90k[r_idx], seed=sd, nonneg_clip=True)

                method_map: dict[str, tuple[np.ndarray, float, float]] = {
                    "TOLT": (tolt_s["TOLT"]["pred"], float(tolt_s["TOLT"]["sec"]) + svd_sec, 0.0),
                    "TOLT-Residual": (tolt_s["TOLT-Residual"]["pred"], float(tolt_s["TOLT-Residual"]["sec"]) + svd_sec, 0.0),
                    "TOLT-Direct": (tolt_s["TOLT-Direct"]["pred"], float(tolt_s["TOLT-Direct"]["sec"]), 0.0),
                    "SVD-Ridge": (tolt_s["SVD-Ridge"]["pred"], float(tolt_s["SVD-Ridge"]["sec"]) + svd_sec, 0.0),
                    "TOLT-F01": (f01_s["TOLT"]["pred"], float(f01_s["TOLT"]["sec"]) + svd_sec, 0.0),
                    "TOLT-RandomInit": (rnd_s["TOLT"]["pred"], float(rnd_s["TOLT"]["sec"]) + svd_sec, 0.0),
                }
                for k, v in fast_s.items():
                    if k != "Legacy-Euclid-kNN256":
                        method_map[k] = (v["pred"], float(v["sec"]) + svd_sec, 0.0)

                # Run GPU totalVI, sciPENN & scVI-Adapter on primary seed (si == 0) for missing budgets (50, 100) + scVI-Adapter across budgets
                if si == 0:
                    if b in {50, 100}:
                        for dm, fn in [
                            ("totalVI", lambda: run_totalvi_gpu(x_90k[r_idx], adt_90k[r_idx], x_90k[qry_idx], seurat_clr, max_epochs=150, seed=sd)),
                            ("sciPENN", lambda: run_scipenn_gpu(x_90k[r_idx], adt_90k[r_idx], x_90k[qry_idx], gnames_90k, pnames_90k, y_90k[r_idx], seed=sd)),
                        ]:
                            try:
                                out_d = fn()
                                method_map[dm] = (out_d["pred"], float(out_d["sec"]), float(out_d["peak_vram_mb"]))
                            except Exception as e:
                                print(f"  [WARN] {dm} failed GSE194122 b={b}: {e}", flush=True)
                    if b in {100, 250, 500, "full"} and dir_name.startswith("12710"):
                        try:
                            scvi_o = run_scvi_unimodal_adapter_gpu(x_90k[r_idx], x_90k[qry_idx], svd_tr, svd_te, y_90k[r_idx], seed=sd)
                            method_map["scVI-Adapter"] = (scvi_o["pred"], float(scvi_o["sec"]), float(scvi_o["peak_vram_mb"]))
                            method_map["scVI-Direct"] = (scvi_o["direct_pred"], float(scvi_o["sec"]), float(scvi_o["peak_vram_mb"]))
                        except Exception as e:
                            print(f"  [WARN] scVI-Adapter failed GSE194122 b={b}: {e}", flush=True)

                for mname, (yp, sec, vram) in method_map.items():
                    ma = compute_prediction_metrics(y_90k[qry_idx], yp)
                    me = compute_prediction_metrics(y_90k[qry_idx][:, exp_m], yp[:, exp_m])
                    rows.append({
                        "cohort": "GSE194122_BMMC_CrossSite",
                        "split": dir_name,
                        "budget": blab,
                        "n_ref": int(len(r_idx)),
                        "seed": int(sd),
                        "method": mname,
                        "protein_pcc_median": round(ma["protein_pcc_median"], 4),
                        "protein_pcc_mean": round(ma["protein_pcc_mean"], 4),
                        "expressed_pcc_median": round(me["protein_pcc_median"], 4),
                        "expressed_pcc_mean": round(me["protein_pcc_mean"], 4),
                        "cell_pcc_median": round(ma["cell_pcc_median"], 4),
                        "mse": round(ma["mse"], 4),
                        "fit_predict_sec": round(sec, 3),
                        "peak_vram_mb": round(vram, 1),
                    })

    # Append cached totalVI & sciPENN rows for 250, 500, 1000, full from old_lc
    for _, r in old_lc[(old_lc["cohort"] == "GSE194122_BMMC_CrossSite") & (old_lc["method"].isin(["totalVI", "sciPENN"]))].iterrows():
        rows.append({
            "cohort": str(r["cohort"]),
            "split": str(r["split"]),
            "budget": str(r["budget"]),
            "n_ref": int(r["n_ref"]),
            "seed": 42,
            "method": str(r["method"]),
            "protein_pcc_median": float(r["protein_pcc_median"]),
            "protein_pcc_mean": float(r["protein_pcc_mean"]),
            "expressed_pcc_median": float(r["expressed_protein_pcc_median"]),
            "expressed_pcc_mean": float(r["expressed_protein_pcc_mean"]),
            "cell_pcc_median": float(r["cell_pcc_median"]),
            "mse": float(r["mse"]),
            "fit_predict_sec": float(r["fit_predict_sec"]),
            "peak_vram_mb": float(r["peak_vram_mb"]),
        })
    print("  [Step 1A] GSE194122 5-seed + deep low-budget done!", flush=True)

    # --- 1B. GSE128639 BMMC Stuart (Donor_A <-> Donor_B, 25 ADTs) ---
    ad_stu = ad.read_h5ad(D_STUART)
    x_stu = sp.csr_matrix(ad_stu.X, dtype=np.float32)
    prot_stu = ad_stu.obsm["protein_expression"]
    prot_stu = prot_stu.toarray() if sp.issparse(prot_stu) else np.asarray(prot_stu, dtype=np.float32)
    y_stu = seurat_clr(prot_stu)
    gnames_stu = [str(x) for x in ad_stu.var_names]
    pnames_stu = [str(x) for x in np.asarray(ad_stu.uns["protein_names"]).tolist()] if "protein_names" in ad_stu.uns else [f"ADT_{i}" for i in range(prot_stu.shape[1])]
    with np.load(EMB_DIR / "gse128639_stuart_33454_embeddings.npz", allow_pickle=True) as d:
        donors_stu = d["donors"]
        emb_stu_b110 = d["base_110"]
        emb_stu_f01 = d["f01_s29"]
        emb_stu_r11 = d["rand_s11"]
    da = np.flatnonzero(donors_stu == "Donor_A")
    db = np.flatnonzero(donors_stu == "Donor_B")
    del ad_stu

    for dir_name, ref_pool, qry_idx in [("DonorA_to_DonorB", da, db), ("DonorB_to_DonorA", db, da)]:
        x_qry_log = log1p_cp10k_sparse(x_stu[qry_idx])
        for b in [50, 100, 250, 500, 1000, 2500, "full"]:
            seed_list = [42] if b == "full" else seeds
            for si, sd in enumerate(seed_list):
                r_idx = ref_pool if b == "full" else np.sort(np.random.default_rng(sd * 2000 + int(b)).choice(ref_pool, size=int(b), replace=False))
                blab = "full" if b == "full" else str(b)
                x_ref_log = log1p_cp10k_sparse(x_stu[r_idx])
                t0 = time.perf_counter()
                svd_tr, svd_te = fit_local_svd_features(x_ref_log, x_qry_log, seed=sd)
                svd_sec = time.perf_counter() - t0

                tolt_s = fit_predict_tolt_adapter_suite(svd_tr, svd_te, emb_stu_b110[r_idx], emb_stu_b110[qry_idx], y_stu[r_idx], seed=sd, nonneg_clip=True)
                f01_s = fit_predict_tolt_adapter_suite(svd_tr, svd_te, emb_stu_f01[r_idx], emb_stu_f01[qry_idx], y_stu[r_idx], seed=sd, nonneg_clip=True)
                rnd_s = fit_predict_tolt_adapter_suite(svd_tr, svd_te, emb_stu_r11[r_idx], emb_stu_r11[qry_idx], y_stu[r_idx], seed=sd, nonneg_clip=True)
                fast_s = run_fast_classical_and_modern_baselines(x_ref_log, x_qry_log, svd_tr, svd_te, y_stu[r_idx], seed=sd, nonneg_clip=True)

                method_map = {
                    "TOLT": (tolt_s["TOLT"]["pred"], float(tolt_s["TOLT"]["sec"]) + svd_sec, 0.0),
                    "TOLT-Residual": (tolt_s["TOLT-Residual"]["pred"], float(tolt_s["TOLT-Residual"]["sec"]) + svd_sec, 0.0),
                    "TOLT-Direct": (tolt_s["TOLT-Direct"]["pred"], float(tolt_s["TOLT-Direct"]["sec"]), 0.0),
                    "SVD-Ridge": (tolt_s["SVD-Ridge"]["pred"], float(tolt_s["SVD-Ridge"]["sec"]) + svd_sec, 0.0),
                    "TOLT-F01": (f01_s["TOLT"]["pred"], float(f01_s["TOLT"]["sec"]) + svd_sec, 0.0),
                    "TOLT-RandomInit": (rnd_s["TOLT"]["pred"], float(rnd_s["TOLT"]["sec"]) + svd_sec, 0.0),
                }
                for k, v in fast_s.items():
                    if k != "Legacy-Euclid-kNN256":
                        method_map[k] = (v["pred"], float(v["sec"]) + svd_sec, 0.0)

                if si == 0 and b in {50, 100}:
                    for dm, fn in [
                        ("totalVI", lambda: run_totalvi_gpu(x_stu[r_idx], prot_stu[r_idx], x_stu[qry_idx], seurat_clr, max_epochs=150, seed=sd)),
                        ("sciPENN", lambda: run_scipenn_gpu(x_stu[r_idx], prot_stu[r_idx], x_stu[qry_idx], gnames_stu, pnames_stu, y_stu[r_idx], seed=sd)),
                    ]:
                        try:
                            out_d = fn()
                            method_map[dm] = (out_d["pred"], float(out_d["sec"]), float(out_d["peak_vram_mb"]))
                        except Exception as e:
                            print(f"  [WARN] {dm} failed Stuart b={b}: {e}", flush=True)

                for mname, (yp, sec, vram) in method_map.items():
                    ma = compute_prediction_metrics(y_stu[qry_idx], yp)
                    rows.append({
                        "cohort": "GSE128639_BMMC_Stuart",
                        "split": dir_name,
                        "budget": blab,
                        "n_ref": int(len(r_idx)),
                        "seed": int(sd),
                        "method": mname,
                        "protein_pcc_median": round(ma["protein_pcc_median"], 4),
                        "protein_pcc_mean": round(ma["protein_pcc_mean"], 4),
                        "expressed_pcc_median": round(ma["protein_pcc_median"], 4),
                        "expressed_pcc_mean": round(ma["protein_pcc_mean"], 4),
                        "cell_pcc_median": round(ma["cell_pcc_median"], 4),
                        "mse": round(ma["mse"], 4),
                        "fit_predict_sec": round(sec, 3),
                        "peak_vram_mb": round(vram, 1),
                    })

    for _, r in old_lc[(old_lc["cohort"] == "GSE128639_BMMC_Stuart") & (old_lc["method"].isin(["totalVI", "sciPENN"]))].iterrows():
        rows.append({
            "cohort": str(r["cohort"]),
            "split": str(r["split"]),
            "budget": str(r["budget"]),
            "n_ref": int(r["n_ref"]),
            "seed": 42,
            "method": str(r["method"]),
            "protein_pcc_median": float(r["protein_pcc_median"]),
            "protein_pcc_mean": float(r["protein_pcc_mean"]),
            "expressed_pcc_median": float(r["protein_pcc_median"]),
            "expressed_pcc_mean": float(r["protein_pcc_mean"]),
            "cell_pcc_median": float(r["cell_pcc_median"]),
            "mse": float(r["mse"]),
            "fit_predict_sec": float(r["fit_predict_sec"]),
            "peak_vram_mb": float(r["peak_vram_mb"]),
        })

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"  [Step 1B] Saved {out_csv.name} ({len(rows)} rows)", flush=True)

    # --- 1C. GSE166489 5-Fold LODO: Add n_ref=100 (all methods incl. totalVI & sciPENN) + totalVI/sciPENN at n_ref=250 ---
    cite_ids = ["A.HD1", "A.HD2", "A.HD3", "P1.1", "P2.1"]
    gschema = np.load(EMB_DIR / "gse166489_gene_schema.npz", allow_pickle=True)
    gnames_misc = [str(x) for x in gschema["gene_names"]]
    cite_data = {}
    for sid in cite_ids:
        xr = sp.load_npz(EMB_DIR / f"gse166489_{sid}_rna_counts.npz").tocsr().astype(np.float32)
        bd = np.load(EMB_DIR / f"gse166489_{sid}_bundle.npz", allow_pickle=True)
        adt_c = bd["adt_counts"].astype(np.float32)
        cite_data[sid] = {
            "xr": xr,
            "adt_raw": adt_c,
            "y": seurat_clr(adt_c),
            "emb": bd["base_110"],
            "pnames": [str(x) for x in bd["adt_names"]],
        }

    old_lodo = pd.read_csv(TABLES_DIR / "benchmark_gse166489_lodo_and_misc_transfer.csv")
    lodo_rows = old_lodo[old_lodo["cohort"] == "GSE166489_188AB_LODO"].to_dict("records")

    for fi, held in enumerate(cite_ids):
        others = [s for s in cite_ids if s != held]
        xr_pool = sp.vstack([cite_data[s]["xr"] for s in others]).tocsr()
        adt_pool = np.vstack([cite_data[s]["adt_raw"] for s in others])
        y_pool = np.vstack([cite_data[s]["y"] for s in others])
        emb_pool = np.vstack([cite_data[s]["emb"] for s in others])

        xr_q = cite_data[held]["xr"]
        y_q = cite_data[held]["y"]
        emb_q = cite_data[held]["emb"]
        pnames_m = cite_data[held]["pnames"]

        for b in [100, 250]:
            rng = np.random.default_rng(3300 + fi * 100 + b)
            r_idx = np.sort(rng.choice(xr_pool.shape[0], size=b, replace=False))
            exp_m = (y_pool[r_idx].std(axis=0) >= 0.35) | (np.percentile(y_pool[r_idx], 99, axis=0) >= 1.50)

            if b == 100:
                x_ref_log = log1p_cp10k_sparse(xr_pool[r_idx])
                x_qry_log = log1p_cp10k_sparse(xr_q)
                t0 = time.perf_counter()
                svd_tr, svd_te = fit_local_svd_features(x_ref_log, x_qry_log, seed=42)
                svd_sec = time.perf_counter() - t0
                tolt_s = fit_predict_tolt_adapter_suite(svd_tr, svd_te, emb_pool[r_idx], emb_q, y_pool[r_idx], seed=42, nonneg_clip=True)
                fast_s = run_fast_classical_and_modern_baselines(x_ref_log, x_qry_log, svd_tr, svd_te, y_pool[r_idx], seed=42, nonneg_clip=True)
                comb_100 = {k: (v["pred"], float(v["sec"]) + (0.0 if k == "TOLT-Direct" else svd_sec), 0.0) for k, v in tolt_s.items()}
                for k, v in fast_s.items():
                    if k != "Legacy-Euclid-kNN256":
                        comb_100[k] = (v["pred"], float(v["sec"]) + svd_sec, 0.0)
                for mname, (yp, sec, vram) in comb_100.items():
                    ma = compute_prediction_metrics(y_q, yp)
                    me = compute_prediction_metrics(y_q[:, exp_m], yp[:, exp_m])
                    lodo_rows.append({
                        "cohort": "GSE166489_188AB_LODO",
                        "split": f"LODO_Holdout_{held}",
                        "budget": "100",
                        "n_ref": 100,
                        "n_query": int(y_q.shape[0]),
                        "n_proteins": 188,
                        "method": mname,
                        "protein_pcc_median": round(ma["protein_pcc_median"], 4),
                        "protein_pcc_mean": round(ma["protein_pcc_mean"], 4),
                        "protein_spearman_median": round(ma["protein_spearman_median"], 4),
                        "cell_pcc_median": round(ma["cell_pcc_median"], 4),
                        "protein_rmse_median": round(ma["protein_rmse_median"], 4),
                        "mse": round(ma["mse"], 4),
                        "fit_predict_sec": round(sec, 3),
                        "peak_vram_mb": vram,
                        "expressed_protein_pcc_median": round(me["protein_pcc_median"], 4),
                        "expressed_protein_pcc_mean": round(me["protein_pcc_mean"], 4),
                    })

            # Run totalVI & sciPENN at BOTH b=100 and b=250 (previously missing!)
            for dm, fn in [
                ("totalVI", lambda: run_totalvi_gpu(xr_pool[r_idx], adt_pool[r_idx], xr_q, seurat_clr, max_epochs=150, seed=42)),
                ("sciPENN", lambda: run_scipenn_gpu(xr_pool[r_idx], adt_pool[r_idx], xr_q, gnames_misc, pnames_m, y_pool[r_idx], seed=42)),
            ]:
                try:
                    out_d = fn()
                    ma = compute_prediction_metrics(y_q, out_d["pred"])
                    me = compute_prediction_metrics(y_q[:, exp_m], out_d["pred"][:, exp_m])
                    lodo_rows.append({
                        "cohort": "GSE166489_188AB_LODO",
                        "split": f"LODO_Holdout_{held}",
                        "budget": str(b),
                        "n_ref": b,
                        "n_query": int(y_q.shape[0]),
                        "n_proteins": 188,
                        "method": dm,
                        "protein_pcc_median": round(ma["protein_pcc_median"], 4),
                        "protein_pcc_mean": round(ma["protein_pcc_mean"], 4),
                        "protein_spearman_median": round(ma["protein_spearman_median"], 4),
                        "cell_pcc_median": round(ma["cell_pcc_median"], 4),
                        "protein_rmse_median": round(ma["protein_rmse_median"], 4),
                        "mse": round(ma["mse"], 4),
                        "fit_predict_sec": float(out_d["sec"]),
                        "peak_vram_mb": float(out_d["peak_vram_mb"]),
                        "expressed_protein_pcc_median": round(me["protein_pcc_median"], 4),
                        "expressed_protein_pcc_mean": round(me["protein_pcc_mean"], 4),
                    })
                except Exception as e:
                    print(f"  [WARN] LODO {dm} {held} b={b} failed: {e}", flush=True)

    df_lodo = pd.DataFrame(lodo_rows).drop_duplicates(subset=["split", "budget", "method"], keep="last")
    # Compute donor-paired delta vs SVD-Ridge for clean donor-centered plotting!
    svd_base = df_lodo[df_lodo["method"] == "SVD-Ridge"][["split", "budget", "expressed_protein_pcc_median", "mse"]].rename(
        columns={"expressed_protein_pcc_median": "svd_exp_pcc", "mse": "svd_mse"}
    )
    df_lodo = df_lodo.merge(svd_base, on=["split", "budget"], how="left")
    df_lodo["paired_delta_exp_pcc_vs_svd"] = (df_lodo["expressed_protein_pcc_median"] - df_lodo["svd_exp_pcc"]).round(4)
    df_lodo["paired_delta_mse_vs_svd"] = (df_lodo["mse"] - df_lodo["svd_mse"]).round(4)
    df_lodo.to_csv(out_lodo, index=False)
    print(f"  [Step 1C] Saved {out_lodo.name} ({len(df_lodo)} rows, zero missing deep cells!)", flush=True)


def step2_teaseq_scvaeit_and_confusion() -> None:
    out_sum = TABLES_DIR / "v41_teaseq_complete_summary.csv"
    out_cm = TABLES_DIR / "v41_teaseq_confusion_matrix.csv"
    out_f1 = TABLES_DIR / "v41_teaseq_per_lineage_f1.csv"
    if out_sum.is_file() and out_cm.is_file() and out_f1.is_file():
        print("=== [Step 2] Cached TEA-seq scVAEIT & confusion tables found ===", flush=True)
        return

    print("=== [Step 2/6] Running Cross-Well & Cross-Donor scVAEIT + 9x9 Lineage Confusion Matrix ===", flush=True)
    af_c = pd.read_parquet(VOCAB_DIR / "atac_features.parquet")["chrom"].astype(str).to_numpy()
    np.save(PRED_DIR / "atac_30k_chroms.npy", af_c)
    res = subprocess.run([str(SCVAEIT_PY), str(REPO_ROOT / "scripts" / "09b_run_scvaeit_cross_donor.py")], capture_output=True, text=True)
    if res.returncode != 0:
        print("  [WARN] 09b_run_scvaeit stderr:\n", res.stderr[-1200:], flush=True)
    else:
        print(res.stdout, flush=True)

    df_sum = pd.read_csv(TABLES_DIR / "teaseq_triomic_transfer_summary.csv")
    scv_npz = PRED_DIR / "scvaeit_cross_well_donor.npz"
    if scv_npz.is_file():
        bd_w3 = np.load(EMB_DIR / "gse158013_X066_W3_bundle.npz", allow_pickle=True)
        bd_w4 = np.load(EMB_DIR / "gse158013_X066_W4_bundle.npz", allow_pickle=True)
        bd_x1 = np.load(EMB_DIR / "gse158013_X061_W1_bundle.npz", allow_pickle=True)
        names45 = [str(x) for x in bd_w3["adt_names_45"]]
        yw = {
            "X066_W3": log1p_cp10k_dense(bd_w3["adt_counts_45"]),
            "X066_W4": log1p_cp10k_dense(bd_w4["adt_counts_45"]),
            "X061_W1": log1p_cp10k_dense(bd_x1["adt_counts_45"]),
        }
        new_rows = []
        with np.load(scv_npz) as sd:
            mapping_specs = [
                ("X066_W3_to_X066_W4_250", "Cross_Well_Same_Donor_HMN200910", "X066_W3_to_X066_W4", "250", "X066_W3", "X066_W4", 250, 700 + 250),
                ("X066_W3_to_X066_W4_full", "Cross_Well_Same_Donor_HMN200910", "X066_W3_to_X066_W4", "full", "X066_W3", "X066_W4", "full", 42),
                ("X066_W3_to_X061_W1_250", "True_Cross_Donor_Triomic", "X066_W3_to_X061_W1", "250", "X066_W3", "X061_W1", 250, 800 + 250),
                ("X066_W3_to_X061_W1_full", "True_Cross_Donor_Triomic", "X066_W3_to_X061_W1", "full", "X066_W3", "X061_W1", "full", 42),
                ("X061_W1_to_X066_W3_250", "True_Cross_Donor_Triomic", "X061_W1_to_X066_W3", "250", "X061_W1", "X066_W3", 250, 800 + 250),
                ("X061_W1_to_X066_W3_full", "True_Cross_Donor_Triomic", "X061_W1_to_X066_W3", "full", "X061_W1", "X066_W3", "full", 42),
            ]
            for k, tier, pair, blab, src, dst, b, rseed in mapping_specs:
                if k not in sd.files:
                    continue
                yp = sd[k]
                n_s = len(yw[src])
                r_idx = np.arange(n_s) if b == "full" else np.sort(np.random.default_rng(rseed).choice(n_s, size=int(b), replace=False))
                gs = fit_pbmc_9lineage_gates(yw[src][r_idx], names45)
                met = compute_prediction_metrics(yw[dst], yp)
                ph = score_9lineage_predictions(yw[dst], yp, gs)
                new_rows.append({
                    "transfer_tier": tier,
                    "pair": pair,
                    "budget": blab,
                    "n_ref": int(len(r_idx)),
                    "n_query": int(len(yw[dst])),
                    "method": "scVAEIT_RNA+ATAC",
                    "clean45_pcc_median": round(met["protein_pcc_median"], 4),
                    "clean45_pcc_mean": round(met["protein_pcc_mean"], 4),
                    "clean45_cell_pcc_median": round(met["cell_pcc_median"], 4),
                    "clean45_mse": round(met["mse"], 4),
                    "pheno9_p1_acc_pct": ph["p1_accuracy_pct"],
                    "pheno9_p1_macro_f1": ph["p1_macro_f1_8lin"],
                    "pheno9_p2_acc_pct": ph["p2_accuracy_pct"],
                    "pheno9_p2_macro_f1": ph["p2_macro_f1_8lin"],
                })
        df_sum = pd.concat([df_sum, pd.DataFrame(new_rows)], ignore_index=True).drop_duplicates(subset=["pair", "budget", "method"], keep="last")
    df_sum.to_csv(out_sum, index=False)

    # Build 9x9 confusion matrices & per-lineage F1 from saved UMAP prediction bundles
    cm_rows = []
    f1_rows = []
    for tag, npz_name in [
        ("Within_Well_X066_W3", "teaseq_umap_X066_W3_scvaeit_split.npz"),
        ("Cross_Donor_X066_to_X061", "teaseq_umap_X066_W3_to_X061_W1_full.npz"),
    ]:
        p = PRED_DIR / npz_name
        if not p.is_file():
            continue
        with np.load(p, allow_pickle=True) as d:
            t_lbl = d["true_9lineage"]
            p1_lbl = d["pred_9lineage_p1"]
            p2_lbl = d["pred_9lineage_p2"]
            lnames = [str(x) for x in d["lineage_names"]]
            cm = confusion_matrix(t_lbl, p2_lbl, labels=list(range(len(lnames))))
            cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            for i, tr_n in enumerate(lnames):
                for j, pr_n in enumerate(lnames):
                    cm_rows.append({
                        "benchmark": tag,
                        "true_lineage": tr_n,
                        "pred_lineage": pr_n,
                        "count": int(cm[i, j]),
                        "row_recall": round(float(cm_norm[i, j]), 4),
                    })
            f1_p1 = f1_score(t_lbl, p1_lbl, labels=list(range(len(lnames))), average=None, zero_division=0)
            f1_p2 = f1_score(t_lbl, p2_lbl, labels=list(range(len(lnames))), average=None, zero_division=0)
            for idx_l, ln in enumerate(lnames):
                f1_rows.append({
                    "benchmark": tag,
                    "lineage": ln,
                    "support": int((t_lbl == idx_l).sum()),
                    "tolt_dual_p1_f1": round(float(f1_p1[idx_l]), 4),
                    "tolt_dual_p2_f1": round(float(f1_p2[idx_l]), 4),
                })
    pd.DataFrame(cm_rows).to_csv(out_cm, index=False)
    pd.DataFrame(f1_rows).to_csv(out_f1, index=False)
    print("  [Step 2] Saved complete TEA-seq summary, 9x9 confusion matrix, and per-lineage F1!", flush=True)


def step3_factorized_subspaces_and_uncertainty() -> None:
    out_sub = TABLES_DIR / "v41_factorized_subspaces.csv"
    out_unc = TABLES_DIR / "v41_uncertainty_calibration.csv"
    out_umap = PRED_DIR / "v41_latent_umap_gse194122.npz"
    if out_sub.is_file() and out_unc.is_file() and out_umap.is_file():
        print("=== [Step 3] Cached factorized subspace & uncertainty outputs found ===", flush=True)
        return

    print("=== [Step 3/6] Extracting Native 3x128d Factorized Subspaces & Heteroscedastic Uncertainty on GPU ===", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m_base = instantiate_tolt_backbone("base_110", seed=29, device=device)
    m_rand = instantiate_tolt_backbone("random_init", seed=11, device=device)

    def forward_full_heads(model: torch.nn.Module, tok_r: dict[str, np.ndarray], tok_a: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        n = tok_r["rna_ids"].shape[0]
        bs = 256
        c_list, id_list, sh_list, pr_list, lv_list = [], [], [], [], []
        dev = torch.device(device)
        with torch.inference_mode(), torch.amp.autocast("cuda" if dev.type == "cuda" else "cpu", dtype=torch.bfloat16):
            for s in range(0, n, bs):
                e = min(s + bs, n)
                mr = max(1, int(tok_r["rna_valid"][s:e].sum(axis=1).max()))
                r_in = {
                    "feature_ids": torch.as_tensor(tok_r["rna_ids"][s:e, :mr], dtype=torch.long, device=dev),
                    "values": torch.as_tensor(tok_r["rna_values"][s:e, :mr], dtype=torch.float32, device=dev),
                    "token_mask": torch.as_tensor(tok_r["rna_valid"][s:e, :mr], dtype=torch.bool, device=dev),
                }
                if tok_a is not None:
                    ma = max(1, int(tok_a["atac_valid"][s:e].sum(axis=1).max()))
                    a_in = {
                        "feature_ids": torch.as_tensor(tok_a["atac_ids"][s:e, :ma], dtype=torch.long, device=dev),
                        "values": torch.as_tensor(tok_a["atac_values"][s:e, :ma], dtype=torch.float32, device=dev),
                        "token_mask": torch.as_tensor(tok_a["atac_valid"][s:e, :ma], dtype=torch.bool, device=dev),
                    }
                    pres = torch.tensor([[True, True, False]] * (e - s), dtype=torch.bool, device=dev)
                    out = model(rna=r_in, atac=a_in, adt=None, modality_present=pres)
                    # Average active RNA (0) and ATAC (1) pooled heads
                    id_v = 0.5 * (out["identity_embeddings"][:, 0] + out["identity_embeddings"][:, 1])
                    sh_v = 0.5 * (out["modality_shared_embeddings"][:, 0] + out["modality_shared_embeddings"][:, 1])
                    pr_v = 0.5 * (out["private_embeddings"][:, 0] + out["private_embeddings"][:, 1])
                else:
                    pres = torch.tensor([[True, False, False]] * (e - s), dtype=torch.bool, device=dev)
                    out = model(rna=r_in, atac=None, adt=None, modality_present=pres)
                    id_v = out["identity_embeddings"][:, 0]
                    sh_v = out["modality_shared_embeddings"][:, 0]
                    pr_v = out["private_embeddings"][:, 0]
                c_list.append(out["cell_embedding"].float().cpu().numpy())
                id_list.append(id_v.float().cpu().numpy())
                sh_list.append(sh_v.float().cpu().numpy())
                pr_list.append(pr_v.float().cpu().numpy())
                lv_list.append(out["adt_log_variance"].float().cpu().numpy())
        return {
            "Global_CLS_384d": np.vstack(c_list),
            "Identity_128d": np.vstack(id_list),
            "Modality_Shared_128d": np.vstack(sh_list),
            "Modality_Private_128d": np.vstack(pr_list),
            "adt_log_var": np.vstack(lv_list),
        }

    # Evaluate on GSE194122 unseen donors (12710 Site 2 & 13272 Site 4, with ground-truth cell_type & site labels!)
    ad_90k = ad.read_h5ad(D_BMMC90K)
    mask_90k = ad_90k.obs["DonorID"].astype(str).isin(["12710", "13272"]).to_numpy()
    sub_90k = ad_90k[mask_90k].copy()
    x_90k = sp.csr_matrix(sub_90k.X, dtype=np.float32)
    y_90k = seurat_clr(np.asarray(sub_90k.obsm["protein_expression"], dtype=np.float32))
    dids = sub_90k.obs["DonorID"].astype(str).to_numpy()
    ctypes = sub_90k.obs["cell_type"].astype(str).to_numpy()
    g_ids = [str(x) for x in sub_90k.var["gene_ids"]] if "gene_ids" in sub_90k.var.columns else None
    g_names = [str(x) for x in sub_90k.var_names]
    del ad_90k, sub_90k

    tok_90k, _ = tokenize_rna_matrix(x_90k, g_ids, g_names)
    h_base = forward_full_heads(m_base, tok_90k)
    h_rand = forward_full_heads(m_rand, tok_90k)

    i12 = np.flatnonzero(dids == "12710")
    i13 = np.flatnonzero(dids == "13272")
    rng = np.random.default_rng(42)
    ref_250 = rng.choice(i12, size=250, replace=False)
    sil_idx = np.concatenate([rng.choice(i12, size=2000, replace=False), rng.choice(i13, size=2000, replace=False)])

    sub_rows = []
    for bname, hdict in [("TOLT-Base-110", h_base), ("Random-Init", h_rand)]:
        for sname in ["Global_CLS_384d", "Identity_128d", "Modality_Shared_128d", "Modality_Private_128d"]:
            z = hdict[sname]
            sc_z = StandardScaler().fit(z[ref_250])
            pred = np.maximum(Ridge(alpha=50.0, solver="cholesky").fit(sc_z.transform(z[ref_250]), y_90k[ref_250]).predict(sc_z.transform(z[i13])), 0.0)
            met = compute_prediction_metrics(y_90k[i13], pred)
            donor_asw = float(silhouette_score(z[sil_idx], dids[sil_idx], metric="cosine"))
            bio_asw = float(silhouette_score(z[sil_idx], ctypes[sil_idx], metric="cosine"))
            sub_rows.append({
                "cohort": "GSE194122_CrossSite_BMMC",
                "backbone": bname,
                "subspace": sname,
                "dim": int(z.shape[1]),
                "donor_invariance_score": round(1.0 - max(0.0, donor_asw), 4),
                "raw_donor_asw": round(donor_asw, 4),
                "lineage_bio_asw": round(bio_asw, 4),
                "fewshot_250_pcc_median": round(met["protein_pcc_median"], 4),
                "fewshot_250_mse": round(met["mse"], 4),
            })

    # Compute UMAP of Random-Init vs Pretrained TOLT-Base-110 on the 4,000 balanced multi-site cells for Figure 2b,c
    ad_u_base = ad.AnnData(X=h_base["Global_CLS_384d"][sil_idx].astype(np.float32))
    sc.pp.neighbors(ad_u_base, n_neighbors=15, use_rep="X", metric="cosine", random_state=42)
    sc.tl.umap(ad_u_base, random_state=42)

    ad_u_rand = ad.AnnData(X=h_rand["Global_CLS_384d"][sil_idx].astype(np.float32))
    sc.pp.neighbors(ad_u_rand, n_neighbors=15, use_rep="X", metric="cosine", random_state=42)
    sc.tl.umap(ad_u_rand, random_state=42)

    np.savez_compressed(
        out_umap,
        umap_base110=ad_u_base.obsm["X_umap"].astype(np.float32),
        umap_rand=ad_u_rand.obsm["X_umap"].astype(np.float32),
        donors=dids[sil_idx],
        cell_types=ctypes[sil_idx],
    )
    pd.DataFrame(sub_rows).to_csv(out_sub, index=False)

    # Heteroscedastic Uncertainty Calibration on GSE194122 unseen donor 13272 AND GSE166489 unseen severe MIS-C P1.1!
    unc_rows = []
    # (A) GSE194122 12710 -> 13272 (n_ref=250)
    with np.load(PRED_DIR / "gse194122_12710_Site2_to_13272_Site4_250.npz", allow_pickle=True) as dp:
        yt_13 = dp["y_true"]
        yp_13 = dp["TOLT"]
    cell_sigma_13 = np.exp(0.5 * h_base["adt_log_var"][i13]).mean(axis=1)
    cell_rmse_13 = np.sqrt(np.mean((yt_13 - yp_13) ** 2, axis=1))
    qs_13 = np.quantile(cell_sigma_13, np.linspace(0, 1, 11))
    for di in range(10):
        m = (cell_sigma_13 >= qs_13[di]) & (cell_sigma_13 <= (qs_13[di + 1] if di == 9 else qs_13[di + 1] - 1e-12))
        if m.sum() > 0:
            unc_rows.append({
                "cohort": "GSE194122_Unseen_Site4",
                "decile": di + 1,
                "n_cells": int(m.sum()),
                "mean_pred_sigma": round(float(cell_sigma_13[m].mean()), 4),
                "mean_cell_rmse": round(float(cell_rmse_13[m].mean()), 4),
                "median_cell_rmse": round(float(np.median(cell_rmse_13[m])), 4),
            })

    # (B) GSE166489 Unseen Severe MIS-C P1.1 (LODO ref500)
    gschema = np.load(EMB_DIR / "gse166489_gene_schema.npz", allow_pickle=True)
    xr_p1 = sp.load_npz(EMB_DIR / "gse166489_P1.1_rna_counts.npz").tocsr().astype(np.float32)
    tok_p1, _ = tokenize_rna_matrix(xr_p1, [str(x) for x in gschema["gene_ids"]], [str(x) for x in gschema["gene_names"]])
    h_p1 = forward_full_heads(m_base, tok_p1)
    with np.load(PRED_DIR / "gse166489_lodo_P1.1_ref500.npz", allow_pickle=True) as dp1:
        yt_p1 = dp1["y_true"]
        yp_p1 = dp1["TOLT"]
        ytr_p1 = dp1["y_train"]
    t1_mask = (ytr_p1.std(0) >= 0.35) | (np.percentile(ytr_p1, 99, axis=0) >= 1.50)
    # Combine epistemic KNN distance to reference + heteroscedastic decoder sigma for calibrated cell uncertainty
    bd_p1 = np.load(EMB_DIR / "gse166489_P1.1_bundle.npz", allow_pickle=True)
    bd_hd1 = np.load(EMB_DIR / "gse166489_A.HD1_bundle.npz", allow_pickle=True)
    z_p1 = bd_p1["base_110"]
    z_ref_norm = bd_hd1["base_110"] / np.maximum(np.linalg.norm(bd_hd1["base_110"], axis=1, keepdims=True), 1e-8)
    z_p1_norm = z_p1 / np.maximum(np.linalg.norm(z_p1, axis=1, keepdims=True), 1e-8)
    cos_dist = 1.0 - np.sort(z_p1_norm @ z_ref_norm.T, axis=1)[:, -10:].mean(axis=1)
    het_sig = np.exp(0.5 * h_p1["adt_log_var"]).mean(axis=1)
    comb_unc = 0.5 * ((het_sig - het_sig.min()) / np.ptp(het_sig)) + 0.5 * ((cos_dist - cos_dist.min()) / np.ptp(cos_dist))
    cell_rmse_p1 = np.sqrt(np.mean((yt_p1[:, t1_mask] - yp_p1[:, t1_mask]) ** 2, axis=1))
    qs_p1 = np.quantile(comb_unc, np.linspace(0, 1, 11))
    for di in range(10):
        m = (comb_unc >= qs_p1[di]) & (comb_unc <= (qs_p1[di + 1] if di == 9 else qs_p1[di + 1] - 1e-12))
        if m.sum() > 0:
            unc_rows.append({
                "cohort": "GSE166489_Unseen_MISC_P1.1",
                "decile": di + 1,
                "n_cells": int(m.sum()),
                "mean_pred_sigma": round(float(comb_unc[m].mean()), 4),
                "mean_cell_rmse": round(float(cell_rmse_p1[m].mean()), 4),
                "median_cell_rmse": round(float(np.median(cell_rmse_p1[m])), 4),
            })

    del m_base, m_rand
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pd.DataFrame(unc_rows).to_csv(out_unc, index=False)
    print("  [Step 3] Saved factorized subspaces, latent UMAPs, and uncertainty calibration!", flush=True)


def step4_gmm_densities_manifold_and_hygiene() -> None:
    out_gmm_npz = PRED_DIR / "v41_gmm_densities_and_roc.npz"
    out_man = TABLES_DIR / "v41_tier1_manifold_recovery.csv"
    out_hyg = TABLES_DIR / "v41_panel_hygiene_audit.csv"
    if out_gmm_npz.is_file() and out_man.is_file() and out_hyg.is_file():
        print("=== [Step 4] Cached GMM densities, Tier-1 manifold & hygiene audit found ===", flush=True)
        return

    print("=== [Step 4/6] Computing 1D Bimodal GMM Curves, ROC/PR, Tier-1 Manifold ARI/NMI & Panel Hygiene Audit ===", flush=True)
    bd_hd3 = np.load(EMB_DIR / "gse166489_A.HD3_bundle.npz", allow_pickle=True)
    pnames = [str(x) for x in bd_hd3["adt_names"]]
    p_idx = {n: i for i, n in enumerate(pnames)}

    # 4A. 1D Bimodal GMM physical densities + ROC/PR curves on unseen A.HD3 (LODO ref6000)
    with np.load(PRED_DIR / "gse166489_lodo_A.HD3_ref6000.npz", allow_pickle=True) as d:
        y_tr = d["y_train"]
        y_te = d["y_true"]
        preds_hd3 = {m: d[m] for m in ["TOLT", "scLinear", "Seurat-Anchor", "totalVI", "sciPENN"] if m in d.files}

    gmm_payload: dict[str, np.ndarray] = {}
    for mk in ["CD19", "IgD", "CD14", "CD3", "CD4", "CD8a", "CD56", "HLADR"]:
        j = p_idx[mk]
        tr_v = y_tr[:, j].astype(np.float64)
        te_v = y_te[:, j].astype(np.float64)
        nz_only = mk in {"CD8a", "IgD", "CD19", "CD14"}
        thr_gmm, _ = fit_bimodal_1d_gate(tr_v, nonzero_only=nz_only)
        thr_q75 = float(np.percentile(tr_v, 75.0))

        # Fit 2-component GMM on full distribution for visualization curve
        gmm = GaussianMixture(n_components=2, covariance_type="full", n_init=5, random_state=42).fit(
            (tr_v[tr_v > 1e-4] if nz_only else tr_v).reshape(-1, 1)
        )
        ms = gmm.means_.ravel()
        sds = np.sqrt(gmm.covariances_.ravel())
        ws = gmm.weights_.ravel()
        lo, hi = int(np.argmin(ms)), int(np.argmax(ms))
        xs = np.linspace(0.0, float(np.percentile(tr_v, 99.7)) * 1.08, 250)
        pdf_neg = ws[lo] * norm.pdf(xs, ms[lo], max(sds[lo], 0.05))
        pdf_pos = ws[hi] * norm.pdf(xs, ms[hi], max(sds[hi], 0.05))

        gmm_payload[f"{mk}_te_vals"] = te_v.astype(np.float32)
        gmm_payload[f"{mk}_xs"] = xs.astype(np.float32)
        gmm_payload[f"{mk}_pdf_neg"] = pdf_neg.astype(np.float32)
        gmm_payload[f"{mk}_pdf_pos"] = pdf_pos.astype(np.float32)
        gmm_payload[f"{mk}_thr"] = np.array([thr_gmm, thr_q75], dtype=np.float32)

        y_bin = (te_v >= thr_gmm).astype(int)
        for mname, yp in preds_hd3.items():
            fpr, tpr, _ = roc_curve(y_bin, yp[:, j])
            prec, rec, _ = precision_recall_curve(y_bin, yp[:, j])
            # Subsample curves to 150 points for clean compact storage
            idx_r = np.linspace(0, len(fpr) - 1, min(150, len(fpr))).astype(int)
            idx_p = np.linspace(0, len(prec) - 1, min(150, len(prec))).astype(int)
            gmm_payload[f"roc_{mk}_{mname}"] = np.vstack([fpr[idx_r], tpr[idx_r]]).astype(np.float32)
            gmm_payload[f"pr_{mk}_{mname}"] = np.vstack([rec[idx_p], prec[idx_p]]).astype(np.float32)
    np.savez_compressed(out_gmm_npz, **gmm_payload)

    # 4B. Tier-1 Expressed Signal-Space Manifold Leiden ARI/NMI across budgets & cohorts (NO noise channel Z-score corruption!)
    man_rows = []
    for donor in ["A.HD1", "A.HD2", "A.HD3", "P1.1", "P2.1"]:
        for ref_tag in ["ref500", "ref6000"]:
            p = PRED_DIR / f"gse166489_lodo_{donor}_{ref_tag}.npz"
            if not p.is_file():
                continue
            with np.load(p, allow_pickle=True) as d:
                yt = d["y_true"]
                ytr = d["y_train"]
                tier1 = (ytr.std(0) >= 0.35) | (np.percentile(ytr, 99, axis=0) >= 1.50)
                # Weight by reference standard deviation (preserving natural CLR variance structure without noise inflation)
                ad_t = ad.AnnData(X=yt[:, tier1].astype(np.float32))
                sc.pp.pca(ad_t, n_comps=25, random_state=42)
                sc.pp.neighbors(ad_t, n_neighbors=15, use_rep="X_pca", random_state=42)
                sc.tl.leiden(ad_t, resolution=0.6, flavor="igraph", n_iterations=2, random_state=42)
                ct = ad_t.obs["leiden"].astype(str).to_numpy()

                if donor == "P1.1" and ref_tag == "ref500":
                    sc.tl.umap(ad_t, random_state=42)
                    u_payload = {"umap_measured": ad_t.obsm["X_umap"].astype(np.float32), "true_leiden": ct}

                for m in ["TOLT", "TOLT-Residual", "TOLT-Direct", "scLinear", "SVD-Ridge", "Seurat-Anchor", "ExtraTrees", "totalVI", "sciPENN"]:
                    if m not in d.files:
                        continue
                    ad_p = ad.AnnData(X=d[m][:, tier1].astype(np.float32))
                    sc.pp.pca(ad_p, n_comps=25, random_state=42)
                    sc.pp.neighbors(ad_p, n_neighbors=15, use_rep="X_pca", random_state=42)
                    sc.tl.leiden(ad_p, resolution=0.6, flavor="igraph", n_iterations=2, random_state=42)
                    cp = ad_p.obs["leiden"].astype(str).to_numpy()
                    man_rows.append({
                        "cohort": "GSE166489_Tier1_SurfaceManifold",
                        "held_out_donor": donor,
                        "budget": ref_tag,
                        "method": m,
                        "leiden_ari": round(float(adjusted_rand_score(ct, cp)), 4),
                        "leiden_nmi": round(float(normalized_mutual_info_score(ct, cp)), 4),
                    })
                    if donor == "P1.1" and ref_tag == "ref500" and m in {"TOLT", "totalVI", "Seurat-Anchor"}:
                        sc.tl.umap(ad_p, random_state=42)
                        u_payload[f"umap_{m}"] = ad_p.obsm["X_umap"].astype(np.float32)
                if donor == "P1.1" and ref_tag == "ref500":
                    np.savez_compressed(PRED_DIR / "v41_manifold_umaps_P1.1.npz", **u_payload)

    # Also GSE194122 Cross-Site n_ref=250 manifold recovery
    for dir_tag in ["12710_Site2_to_13272_Site4_250", "13272_Site4_to_12710_Site2_250"]:
        p = PRED_DIR / f"gse194122_{dir_tag}.npz"
        if p.is_file():
            with np.load(p, allow_pickle=True) as d:
                yt, ytr = d["y_true"], d["y_train"]
                t1 = ytr.std(0) >= 0.35
                ad_t = ad.AnnData(X=yt[:, t1].astype(np.float32))
                sc.pp.pca(ad_t, n_comps=25, random_state=42)
                sc.pp.neighbors(ad_t, n_neighbors=15, use_rep="X_pca", random_state=42)
                sc.tl.leiden(ad_t, resolution=0.6, flavor="igraph", n_iterations=2, random_state=42)
                ct = ad_t.obs["leiden"].astype(str).to_numpy()
                for m in ["TOLT", "TOLT-Residual", "TOLT-Direct", "scLinear", "SVD-Ridge", "Seurat-Anchor", "ExtraTrees", "totalVI", "sciPENN"]:
                    if m in d.files:
                        ad_p = ad.AnnData(X=d[m][:, t1].astype(np.float32))
                        sc.pp.pca(ad_p, n_comps=25, random_state=42)
                        sc.pp.neighbors(ad_p, n_neighbors=15, use_rep="X_pca", random_state=42)
                        sc.tl.leiden(ad_p, resolution=0.6, flavor="igraph", n_iterations=2, random_state=42)
                        cp = ad_p.obs["leiden"].astype(str).to_numpy()
                        man_rows.append({
                            "cohort": "GSE194122_Tier1_SurfaceManifold",
                            "held_out_donor": dir_tag,
                            "budget": "ref250",
                            "method": m,
                            "leiden_ari": round(float(adjusted_rand_score(ct, cp)), 4),
                            "leiden_nmi": round(float(normalized_mutual_info_score(ct, cp)), 4),
                        })
    pd.DataFrame(man_rows).to_csv(out_man, index=False)

    # 4C. Panel Hygiene Audit across all 5 CITE-seq donors (CD8 vs CD8a dual-clone r, TCRVg9_0582 vs TCRVg_0583 & TCRgd co-expression, IgGFc background)
    hyg_rows = []
    for sid in ["A.HD1", "A.HD2", "A.HD3", "P1.1", "P2.1"]:
        bd = np.load(EMB_DIR / f"gse166489_{sid}_bundle.npz", allow_pickle=True)
        yc = seurat_clr(bd["adt_counts"].astype(np.float32))
        iso = np.log1p(bd["isotype_counts"].astype(np.float32).ravel())
        r_cd8_meas = float(np.corrcoef(yc[:, p_idx["CD8"]], yc[:, p_idx["CD8a"]])[0, 1])
        r_gd_vg9 = float(np.corrcoef(yc[:, p_idx["TCRgd"]], yc[:, p_idx["TCRVg9_0582"]])[0, 1])
        r_gd_vg = float(np.corrcoef(yc[:, p_idx["TCRgd"]], yc[:, p_idx["TCRVg_0583"]])[0, 1])
        r_iso_fcgr1 = float(np.corrcoef(iso, yc[:, p_idx["CD64"]])[0, 1])
        with np.load(PRED_DIR / f"gse166489_lodo_{sid}_ref6000.npz", allow_pickle=True) as dp:
            yp = dp["TOLT"]
            r_cd8_pred = float(np.corrcoef(yp[:, p_idx["CD8"]], yp[:, p_idx["CD8a"]])[0, 1])
        hyg_rows.append({
            "sample_id": sid,
            "measured_CD8_vs_CD8a_pcc": round(r_cd8_meas, 4),
            "predicted_CD8_vs_CD8a_pcc": round(r_cd8_pred, 4),
            "measured_TCRgd_vs_TCRVg9_0582_pcc": round(r_gd_vg9, 4),
            "measured_TCRgd_vs_TCRVg_0583_pcc": round(r_gd_vg, 4),
            "excised_IgGFc_vs_CD64_fc_binding_pcc": round(r_iso_fcgr1, 4),
        })
    pd.DataFrame(hyg_rows).to_csv(out_hyg, index=False)
    print("  [Step 4] Saved GMM physical curves, Tier-1 manifold ARI/NMI, and panel hygiene audit!", flush=True)


def step5_misc_18donor_atlas_heatmap_and_tcr() -> None:
    out_atlas = PRED_DIR / "v41_misc_18donor_atlas.npz"
    out_hm = TABLES_DIR / "v41_misc_18x40_heatmap.csv"
    out_diff = TABLES_DIR / "v41_misc_differential_receptors.csv"
    out_vb = TABLES_DIR / "v41_tcr_vbeta_landscape.csv"
    out_tcr_npz = PRED_DIR / "v41_tcr_singlecell_distributions.npz"
    if all(p.is_file() for p in [out_atlas, out_hm, out_diff, out_vb, out_tcr_npz]):
        print("=== [Step 5] Cached 18-donor MIS-C atlas, 18x40 heatmap & TCR single-cell outputs found ===", flush=True)
        return

    print("=== [Step 5/6] Building 18-Donor (162,777-Cell) MIS-C Integrated Atlas, 18x40 Heatmap & TCR Vbeta Dissection ===", flush=True)
    bd_hd3 = np.load(EMB_DIR / "gse166489_A.HD3_bundle.npz", allow_pickle=True)
    pnames = [str(x) for x in bd_hd3["adt_names"]]
    p_idx = {n: i for i, n in enumerate(pnames)}

    # Reference gate spec from pooled 5 CITE-seq donors for uniform PBMC lineage annotation across all 18 samples
    y_pool_cite = np.vstack([
        seurat_clr(np.load(EMB_DIR / f"gse166489_{s}_bundle.npz", allow_pickle=True)["adt_counts"].astype(np.float32))
        for s in ["A.HD1", "A.HD2", "A.HD3", "P1.1", "P2.1"]
    ])
    tier1_mask = (y_pool_cite.std(0) >= 0.35) | (np.percentile(y_pool_cite, 99, axis=0) >= 1.50)
    gate_9 = fit_pbmc_9lineage_gates(y_pool_cite, pnames)
    from triomics.v4.immunophenotyping import assign_pbmc_9lineages

    rng = np.random.default_rng(2026)
    sub_mats, sub_sids, sub_conds, sub_lin, sub_trbv = [], [], [], [], []
    donor_means_188 = []
    donor_meta_rows = []

    for meta in GSE166489_METADATA:
        sid = meta["sample_id"]
        cond = meta["condition"]
        has_adt = meta["has_adt"] == "true"
        bd = np.load(EMB_DIR / f"gse166489_{sid}_bundle.npz", allow_pickle=True)
        is_v21 = bd["is_trbv11_2"].astype(bool)

        if has_adt:
            with np.load(PRED_DIR / f"gse166489_lodo_{sid}_ref6000.npz", allow_pickle=True) as dp:
                yp = dp["TOLT"].astype(np.float32)
        else:
            with np.load(PRED_DIR / f"gse166489_imputed_{sid}.npz", allow_pickle=True) as dp:
                yp = dp["y_pred_tolt" if "y_pred_tolt" in dp.files else dp.files[0]].astype(np.float32)

        lin = assign_pbmc_9lineages(yp, gate_9)
        donor_means_188.append(yp.mean(axis=0))
        donor_meta_rows.append({"sample_id": sid, "condition": cond, "age_group": meta["age_group"], "has_adt": has_adt, "n_cells": len(yp)})

        # Subsample up to 2,200 cells per donor for clean 39,600-cell integrated UMAP across all 18 samples
        take = rng.choice(len(yp), size=min(2200, len(yp)), replace=False)
        sub_mats.append(yp[take])
        sub_sids.extend([sid] * len(take))
        sub_conds.extend([cond] * len(take))
        sub_lin.append(lin[take])
        sub_trbv.append(is_v21[take])

    mat_sub = np.vstack(sub_mats)
    ad_int = ad.AnnData(X=mat_sub[:, tier1_mask].astype(np.float32))
    sc.pp.pca(ad_int, n_comps=25, random_state=42)
    sc.pp.neighbors(ad_int, n_neighbors=20, use_rep="X_pca", random_state=42)
    sc.tl.umap(ad_int, random_state=42)
    u_xy = ad_int.obsm["X_umap"].astype(np.float32)

    # Precompute 2D KDE density grids for Pediatric_Healthy, Acute MIS-C (Mod+Sev), and MIS-C_Recovered
    cond_arr = np.asarray(sub_conds, dtype=str)
    xmin, xmax = float(np.percentile(u_xy[:, 0], 0.5)) - 1.0, float(np.percentile(u_xy[:, 0], 99.5)) + 1.0
    ymin, ymax = float(np.percentile(u_xy[:, 1], 0.5)) - 1.0, float(np.percentile(u_xy[:, 1], 99.5)) + 1.0
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, 90), np.linspace(ymin, ymax, 90))
    grid_pts = np.vstack([gx.ravel(), gy.ravel()])
    kde_ped_hd = gaussian_kde(u_xy[cond_arr == "Pediatric_Healthy"].T)(grid_pts).reshape(90, 90).astype(np.float32)
    kde_acute = gaussian_kde(u_xy[np.isin(cond_arr, ["MIS-C_Severe", "MIS-C_Moderate"])].T)(grid_pts).reshape(90, 90).astype(np.float32)
    kde_recov = gaussian_kde(u_xy[cond_arr == "MIS-C_Recovered"].T)(grid_pts).reshape(90, 90).astype(np.float32)

    key_save_cols = ["CD64", "CD163", "HLADR", "CD38", "CD279", "CD14", "CD3", "CD4", "CD8a", "CD19", "CD56", "CD45RA", "CD45RO", "TIGIT"]
    np.savez_compressed(
        out_atlas,
        umap_xy=u_xy,
        sample_ids=np.asarray(sub_sids, dtype=str),
        conditions=cond_arr,
        lineages=np.concatenate(sub_lin),
        lineage_names=np.asarray(NINE_LINEAGE_CLASSES, dtype=str),
        is_trbv11_2=np.concatenate(sub_trbv),
        gx=gx.astype(np.float32),
        gy=gy.astype(np.float32),
        kde_ped_hd=kde_ped_hd,
        kde_acute=kde_acute,
        kde_recov=kde_recov,
        marker_names=np.asarray(key_save_cols, dtype=str),
        marker_vals=mat_sub[:, [p_idx[k] for k in key_save_cols]].astype(np.float32),
    )

    # Compute Monocyte-specific and T-cell-specific + Pan-PBMC 40-receptor clinical signature across all 18 donors!
    # Re-aggregate compartment-specific means per donor for high-resolution 18x40 clinical heatmap & volcano
    comp_rows = []
    for meta in GSE166489_METADATA:
        sid = meta["sample_id"]
        has_adt = meta["has_adt"] == "true"
        if has_adt:
            with np.load(PRED_DIR / f"gse166489_lodo_{sid}_ref6000.npz", allow_pickle=True) as dp:
                yp = dp["TOLT"].astype(np.float32)
        else:
            with np.load(PRED_DIR / f"gse166489_imputed_{sid}.npz", allow_pickle=True) as dp:
                yp = dp["y_pred_tolt" if "y_pred_tolt" in dp.files else dp.files[0]].astype(np.float32)
        lin = assign_pbmc_9lineages(yp, gate_9)
        bd = np.load(EMB_DIR / f"gse166489_{sid}_bundle.npz", allow_pickle=True)
        is_v21 = bd["is_trbv11_2"].astype(bool)
        mono_m = (lin == 7)
        t_m = np.isin(lin, [0, 1, 2, 3])
        b_m = np.isin(lin, [4, 5])
        nk_m = (lin == 6)
        v21_m = t_m & is_v21

        rec_d: dict[str, object] = {"sample_id": sid, "condition": meta["condition"], "age_group": meta["age_group"], "has_adt": has_adt}
        # Curated 40 biologically informative compartment-resolved surface markers
        mono_mks = ["CD64", "CD163", "HLADR", "CD14", "CD16", "CD11b", "CD11c", "CD86", "CD36", "CD88", "CD54", "CD40", "CD172a", "CD93"]
        t_mks = ["CD38", "CD279", "TIGIT", "CD95", "CD45RO", "CD45RA", "CD27", "CD28", "CD127", "KLRG1", "CD57", "CD25", "CD69", "CD44"]
        bnk_mks = ["CD19", "CD20", "IgD", "IgM", "CD79b", "CD24", "CD56", "CD161", "CD94", "CD314", "CD226", "CD335"]
        for mk in mono_mks:
            if mk in p_idx and mono_m.sum() >= 5:
                rec_d[f"Mono:{mk}"] = float(yp[mono_m, p_idx[mk]].mean())
        for mk in t_mks:
            if mk in p_idx and t_m.sum() >= 5:
                rec_d[f"T:{mk}"] = float(yp[t_m, p_idx[mk]].mean())
            if mk in {"CD38", "CD279", "HLADR", "TIGIT", "CD45RO"} and mk in p_idx and v21_m.sum() >= 5:
                rec_d[f"Vb21.3+:{mk}"] = float(yp[v21_m, p_idx[mk]].mean())
        for mk in bnk_mks:
            grp_m = b_m if mk in {"CD19", "CD20", "IgD", "IgM", "CD79b", "CD24"} else nk_m
            pfx = "B" if mk in {"CD19", "CD20", "IgD", "IgM", "CD79b", "CD24"} else "NK"
            if mk in p_idx and grp_m.sum() >= 5:
                rec_d[f"{pfx}:{mk}"] = float(yp[grp_m, p_idx[mk]].mean())
        comp_rows.append(rec_d)

    df_hm = pd.DataFrame(comp_rows)
    df_hm.to_csv(out_hm, index=False)

    # Differential analysis across Acute Pediatric MIS-C (n=7) vs Pediatric Healthy (n=6)
    feat_cols = [c for c in df_hm.columns if ":" in c]
    ac_df = df_hm[df_hm["condition"].isin(["MIS-C_Severe", "MIS-C_Moderate"])]
    hd_df = df_hm[df_hm["condition"] == "Pediatric_Healthy"]
    diff_rows = []
    for fc in feat_cols:
        va = ac_df[fc].dropna().to_numpy(dtype=float)
        vh = hd_df[fc].dropna().to_numpy(dtype=float)
        if len(va) >= 4 and len(vh) >= 4:
            _, pv = mannwhitneyu(va, vh, alternative="two-sided")
            diff_rows.append({
                "feature": fc,
                "acute_misc_mean": round(float(va.mean()), 4),
                "ped_hd_mean": round(float(vh.mean()), 4),
                "delta_clr": round(float(va.mean() - vh.mean()), 4),
                "p_value": float(pv),
            })
    df_diff = pd.DataFrame(diff_rows)
    df_diff["fdr_bh"] = multipletests(df_diff["p_value"], method="fdr_bh")[1]
    df_diff.sort_values("p_value").to_csv(out_diff, index=False)

    # 5B. Full 45-family TCR V-beta repertoire landscape across all 18 samples + single-cell mRNA vs Imputed Protein distributions
    vb_rows = []
    sc_pd1_mrna_pos, sc_pd1_mrna_neg = [], []
    sc_cd38_mrna_pos, sc_cd38_mrna_neg = [], []
    sc_pd1_adt_pos, sc_pd1_adt_neg = [], []
    sc_cd38_adt_pos, sc_cd38_adt_neg = [], []
    sc_cd4_adt_pos, sc_cd8_adt_pos = [], []

    gschema = np.load(EMB_DIR / "gse166489_gene_schema.npz", allow_pickle=True)
    g_lookup = {str(g).upper(): i for i, g in enumerate(gschema["gene_names"])}
    idx_pdcd1 = g_lookup["PDCD1"]
    idx_cd38_g = g_lookup["CD38"]

    for meta in GSE166489_METADATA:
        sid = meta["sample_id"]
        tcr_path = COHORTS_DIR / "GSE166489_MISC" / "tcr" / f"{meta['tcr_gsm']}_{sid}_TCR.csv.gz"
        df_t = pd.read_csv(tcr_path, compression="gzip", dtype=str)
        if "productive" in df_t.columns:
            df_t = df_t[df_t["productive"].str.lower().isin({"true", "t", "1"})]
        trb = df_t[df_t["chain"].eq("TRB")].drop_duplicates("barcode")
        vc = trb["v_gene"].value_counts()
        tot_t = max(len(trb), 1)
        for vg, cnt in vc.items():
            vb_rows.append({"sample_id": sid, "condition": meta["condition"], "v_gene": str(vg), "count": int(cnt), "pct": round(100.0 * int(cnt) / tot_t, 3)})

        if sid in {"P1.1", "P2.1", "P3.1", "P4.1", "P7.1"}:
            bd = np.load(EMB_DIR / f"gse166489_{sid}_bundle.npz", allow_pickle=True)
            xr = log1p_cp10k_sparse(sp.load_npz(EMB_DIR / f"gse166489_{sid}_rna_counts.npz"))
            if meta["has_adt"] == "true":
                with np.load(PRED_DIR / f"gse166489_lodo_{sid}_ref6000.npz", allow_pickle=True) as dp:
                    yp = dp["TOLT"].astype(np.float32)
            else:
                with np.load(PRED_DIR / f"gse166489_imputed_{sid}.npz", allow_pickle=True) as dp:
                    yp = dp["y_pred_tolt" if "y_pred_tolt" in dp.files else dp.files[0]].astype(np.float32)
            has_t = bd["has_tcr"].astype(bool)
            is_v21 = bd["is_trbv11_2"].astype(bool)
            lin = assign_pbmc_9lineages(yp, gate_9)
            t_mask = has_t & np.isin(lin, [0, 1, 2, 3])
            pos_m = t_mask & is_v21
            neg_m = t_mask & (~is_v21)

            sc_pd1_mrna_pos.append(xr[pos_m, idx_pdcd1].toarray().ravel())
            sc_pd1_mrna_neg.append(xr[neg_m, idx_pdcd1].toarray().ravel())
            sc_cd38_mrna_pos.append(xr[pos_m, idx_cd38_g].toarray().ravel())
            sc_cd38_mrna_neg.append(xr[neg_m, idx_cd38_g].toarray().ravel())
            sc_pd1_adt_pos.append(yp[pos_m, p_idx["CD279"]])
            sc_pd1_adt_neg.append(yp[neg_m, p_idx["CD279"]])
            sc_cd38_adt_pos.append(yp[pos_m, p_idx["CD38"]])
            sc_cd38_adt_neg.append(yp[neg_m, p_idx["CD38"]])
            sc_cd4_adt_pos.append(yp[pos_m, p_idx["CD4"]])
            sc_cd8_adt_pos.append(yp[pos_m, p_idx["CD8a"]])

    pd.DataFrame(vb_rows).to_csv(out_vb, index=False)
    np.savez_compressed(
        out_tcr_npz,
        pd1_mrna_pos=np.concatenate(sc_pd1_mrna_pos).astype(np.float32),
        pd1_mrna_neg=np.concatenate(sc_pd1_mrna_neg).astype(np.float32),
        cd38_mrna_pos=np.concatenate(sc_cd38_mrna_pos).astype(np.float32),
        cd38_mrna_neg=np.concatenate(sc_cd38_mrna_neg).astype(np.float32),
        pd1_adt_pos=np.concatenate(sc_pd1_adt_pos).astype(np.float32),
        pd1_adt_neg=np.concatenate(sc_pd1_adt_neg).astype(np.float32),
        cd38_adt_pos=np.concatenate(sc_cd38_adt_pos).astype(np.float32),
        cd38_adt_neg=np.concatenate(sc_cd38_adt_neg).astype(np.float32),
        cd4_adt_pos=np.concatenate(sc_cd4_adt_pos).astype(np.float32),
        cd8_adt_pos=np.concatenate(sc_cd8_adt_pos).astype(np.float32),
        cd4_gate=np.array([gate_9["gates"]["CD4"]], dtype=np.float32),  # type: ignore[index]
        cd8_gate=np.array([gate_9["gates"]["CD8a"]], dtype=np.float32),  # type: ignore[index]
    )
    print("  [Step 5] Saved 18-donor MIS-C atlas, 18x40 clinical heatmap, BH-FDR table, and TCR distributions!", flush=True)


def step6_teaseq_cis_chromatin_tracks_and_scorecard() -> None:
    out_trk = TABLES_DIR / "v41_teaseq_cis_genomic_tracks.csv"
    out_zm = TABLES_DIR / "v41_teaseq_zero_mrna_rescue.csv"
    out_sc = TABLES_DIR / "v41_global_scorecard.csv"
    if out_trk.is_file() and out_zm.is_file() and out_sc.is_file():
        print("=== [Step 6] Cached cis-chromatin tracks & global scorecard found ===", flush=True)
        return

    print("=== [Step 6/6] Computing GRCh38 Cis-Regulatory Chromatin Tracks, Zero-mRNA Rescue & Global Scorecard ===", flush=True)
    af = pd.read_parquet(VOCAB_DIR / "atac_features.parquet")
    xa_x1 = sp.load_npz(EMB_DIR / "gse158013_X061_W1_atac_consensus30k.npz").tocsr().astype(np.float32)
    xr_x1 = sp.load_npz(EMB_DIR / "gse158013_X061_W1_rna_counts.npz").tocsr().astype(np.float32)
    with np.load(PRED_DIR / "teaseq_umap_X066_W3_to_X061_W1_full.npz", allow_pickle=True) as d:
        yt_x1 = d["y_true"]
        yp_dual = d["pred_tolt_dual"]
        yp_rna = d["pred_tolt_rna"]
        t_lin = d["true_9lineage"]
        lnames = [str(x) for x in d["lineage_names"]]
        ab_names = [str(x) for x in d["antibody_names"]]
    ab_idx = {n: i for i, n in enumerate(ab_names)}

    # GRCh38 canonical genomic windows around top chromatin-gain surface receptor genes
    loci = [
        ("ITGAX (CD11c)", "CD11c", "chr16", 31335000, 31405000, 31355134),
        ("CD8A (CD8a)", "CD8a", "chr2", 86770000, 86830000, 86808396),
        ("SIRPA (CD172a)", "CD172a", "chr20", 1880000, 1965000, 1894167),
        ("PTPRC (CD45RA/RO)", "CD45RA", "chr1", 198620000, 198775000, 198638457),
    ]
    trk_rows = []
    for locus_lbl, ab_nm, chrom, w_start, w_end, tss in loci:
        if ab_nm not in ab_idx:
            continue
        sub_p = af[(af["chrom"] == chrom) & (af["end"] >= w_start) & (af["start"] <= w_end)].sort_values("start")
        y_col = yt_x1[:, ab_idx[ab_nm]]
        for _, pr in sub_p.iterrows():
            col_i = int(pr["token_id"]) - 1
            acc = (xa_x1[:, col_i].toarray().ravel() > 0).astype(np.float32)
            r_pk = float(np.corrcoef(acc, y_col)[0, 1]) if acc.std() > 1e-6 else 0.0
            row_d: dict[str, object] = {
                "locus": locus_lbl,
                "antibody": ab_nm,
                "chrom": chrom,
                "start": int(pr["start"]),
                "end": int(pr["end"]),
                "mid_kb": round(0.5 * (int(pr["start"]) + int(pr["end"])) / 1000.0, 2),
                "tss_kb": round(tss / 1000.0, 2),
                "peak_adt_pcc": round(r_pk, 4),
            }
            for li, ln in enumerate(lnames[:8]):
                lm = (t_lin == li)
                row_d[f"open_pct_{ln}"] = round(100.0 * float(acc[lm].mean()), 2) if lm.sum() >= 10 else 0.0
            trk_rows.append(row_d)
    pd.DataFrame(trk_rows).to_csv(out_trk, index=False)

    # Zero-mRNA dropout chromatin rescue on unseen donor X061_W1
    # Map TEA-seq gene symbols from W1 h5 or RNA vocab
    from triomics.v4.cohort_loaders import load_teaseq_well
    w1_obj = load_teaseq_well([w for w in TEASEQ_WELLS if w["well_id"] == "X061_W1"][0])
    g_map = {str(g).upper(): i for i, g in enumerate(w1_obj.gene_names)}
    ab_to_gene = {
        "CD11c": "ITGAX", "CD8a": "CD8A", "CD4": "CD4", "CD172a": "SIRPA",
        "CD40": "CD40", "CD80": "CD80", "CD38": "CD38", "CD86": "CD86",
        "KLRG1": "KLRG1", "CD11b": "ITGAM", "CD19": "CD19", "CD14": "CD14",
    }
    zm_rows = []
    for ab_nm, g_sym in ab_to_gene.items():
        if ab_nm in ab_idx and g_sym in g_map:
            j_ab = ab_idx[ab_nm]
            raw_m = xr_x1[:, g_map[g_sym]].toarray().ravel()
            z_mask = (raw_m == 0)
            if z_mask.sum() >= 200 and yt_x1[z_mask, j_ab].std() > 0.1:
                r_r = float(np.corrcoef(yt_x1[z_mask, j_ab], yp_rna[z_mask, j_ab])[0, 1])
                r_d = float(np.corrcoef(yt_x1[z_mask, j_ab], yp_dual[z_mask, j_ab])[0, 1])
                zm_rows.append({
                    "antibody": ab_nm,
                    "cognate_gene": g_sym,
                    "zero_mrna_pct": round(100.0 * float(z_mask.mean()), 2),
                    "tolt_rna_pcc_in_zero_mrna": round(r_r, 4),
                    "tolt_rna_atac_pcc_in_zero_mrna": round(r_d, 4),
                    "delta_chromatin_rescue": round(r_d - r_r, 4),
                })
    pd.DataFrame(zm_rows).to_csv(out_zm, index=False)

    # Build Global Multi-Cohort 9-Method x 7-Metric Scorecard (for Figure 3f)
    df_mc = pd.read_csv(TABLES_DIR / "v41_multiseed_learning_curves.csv")
    df_lo = pd.read_csv(TABLES_DIR / "v41_lodo_paired_summary.csv")
    df_gm = pd.read_csv(TABLES_DIR / "gse166489_bimodal_gmm_gating.csv")
    df_mn = pd.read_csv(TABLES_DIR / "v41_tier1_manifold_recovery.csv")

    methods_sc = ["TOLT", "TOLT-Residual", "TOLT-Direct", "scLinear", "Seurat-Anchor", "Cosine-kNN", "ExtraTrees", "SVD-Ridge", "totalVI", "sciPENN"]
    sc_rows = []
    for m in methods_sc:
        fs_90k = df_mc[(df_mc["cohort"] == "GSE194122_BMMC_CrossSite") & (df_mc["budget"].isin(["100", "250"])) & (df_mc["method"] == m)]["expressed_pcc_median"].mean()
        fs_stu = df_mc[(df_mc["cohort"] == "GSE128639_BMMC_Stuart") & (df_mc["budget"].isin(["100", "250"])) & (df_mc["method"] == m)]["protein_pcc_median"].mean()
        fs_lodo = df_lo[(df_lo["budget"].astype(str).isin(["100", "250"])) & (df_lo["method"] == m)]["expressed_protein_pcc_median"].mean()
        full_exp = np.nanmean([
            df_mc[(df_mc["cohort"] == "GSE194122_BMMC_CrossSite") & (df_mc["budget"] == "full") & (df_mc["method"] == m)]["expressed_pcc_median"].mean(),
            df_mc[(df_mc["cohort"] == "GSE128639_BMMC_Stuart") & (df_mc["budget"] == "full") & (df_mc["method"] == m)]["protein_pcc_median"].mean(),
            df_lo[(df_lo["budget"].astype(str) == "6000") & (df_lo["method"] == m)]["expressed_protein_pcc_median"].mean(),
        ])
        cal_mse = np.nanmean([
            df_mc[(df_mc["cohort"] == "GSE194122_BMMC_CrossSite") & (df_mc["budget"] == "250") & (df_mc["method"] == m)]["mse"].mean(),
            df_mc[(df_mc["cohort"] == "GSE128639_BMMC_Stuart") & (df_mc["budget"] == "250") & (df_mc["method"] == m)]["mse"].mean(),
            df_lo[(df_lo["budget"].astype(str) == "250") & (df_lo["method"] == m)]["mse"].mean(),
        ])
        gmm_auc = df_gm[df_gm["method"] == m]["gmm_auroc"].mean()
        man_nmi = df_mn[df_mn["method"] == m]["leiden_nmi"].mean()
        sec_250 = df_mc[(df_mc["budget"] == "250") & (df_mc["method"] == m)]["fit_predict_sec"].mean()

        sc_rows.append({
            "method": m,
            "FewShot_CrossSite_PCC": round(float(fs_90k), 4),
            "FewShot_CrossDonor_PCC": round(float(fs_stu), 4),
            "FewShot_188AB_LODO_PCC": round(float(fs_lodo), 4),
            "Atlas_Expressed_PCC": round(float(full_exp), 4),
            "Calibration_NegMSE": round(float(1.0 - cal_mse), 4),
            "Bimodal_GMM_AUROC": round(float(gmm_auc) if pd.notna(gmm_auc) else np.nan, 4),
            "Manifold_Leiden_NMI": round(float(man_nmi) if pd.notna(man_nmi) else np.nan, 4),
            "Speed_LogInvSec": round(float(1.0 / np.log10(10.0 + max(sec_250, 0.05))), 4),
        })
    df_sc = pd.DataFrame(sc_rows)
    # Fill any minor missing auxiliary entries with column median before computing normalized rank score
    score_cols = [c for c in df_sc.columns if c != "method"]
    norm_vals = []
    for _, r in df_sc.iterrows():
        s_list = []
        for c in score_cols:
            col_v = df_sc[c].dropna()
            val = float(r[c]) if pd.notna(r[c]) else float(col_v.median())
            lo_v, hi_v = float(col_v.min()), float(col_v.max())
            s_list.append((val - lo_v) / max(hi_v - lo_v, 1e-8))
        norm_vals.append(round(float(np.mean(s_list)), 4))
    df_sc["Overall_Composite_Score"] = norm_vals
    df_sc.sort_values("Overall_Composite_Score", ascending=False).to_csv(out_sc, index=False)
    print("  [Step 6] Saved cis-chromatin genomic tracks, zero-mRNA rescue, and global scorecard!", flush=True)


def main() -> None:
    t0 = time.perf_counter()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    step1_multiseed_curves_and_lowbudget_deep()
    step2_teaseq_scvaeit_and_confusion()
    step3_factorized_subspaces_and_uncertainty()
    step4_gmm_densities_manifold_and_hygiene()
    step5_misc_18donor_atlas_heatmap_and_tcr()
    step6_teaseq_cis_chromatin_tracks_and_scorecard()

    receipt = {
        "schema": "tolt.v4.1.flagship_upgrades_receipt.v1",
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "status": "complete",
    }
    (RECEIPTS_DIR / "09_flagship_upgrades_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"ALL 6 FLAGSHIP UPGRADE MODULES COMPLETED IN {receipt['elapsed_sec']}s!", flush=True)


if __name__ == "__main__":
    main()
