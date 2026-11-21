"""GSE166489 18-Donor Clinical Immunology & Multi-Omic Discovery Pipeline.

Performs:
  1. 188-Antibody Expression Stratification + Isotype Specificity + Cognate mRNA-Protein Coupling Analysis
  2. Cross-Donor Zero-Leakage Bimodal GMM Gating (AUROC/AUPRC/F1 across canonical & activation markers)
  3. Surface-Proteome Manifold Reconstruction (Scanpy PCA + UMAP + Leiden ARI/NMI vs ground-truth ADT)
  4. 18-Sample Cohort Extension (5 CITE-seq + 13 Pediatric GEX-only):
     - Age-matched Pediatric Healthy (C.HD1-6) vs Acute Moderate/Severe MIS-C (P1.1-P7.1)
     - Within-child Paired Longitudinal Acute -> Recovered Resolution (P3.1->P3.2 & P4.1->P4.2)
     - Single-cell TCR TRBV11-2 (Vb21.3) Superantigen Clone Surface Activation vs mRNA Dropout Analysis
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from triomics.v4.paths import EMB_DIR, PRED_DIR

from triomics.v4.cohort_loaders import GSE166489_METADATA, log1p_cp10k_sparse, seurat_clr
from triomics.v4.core_methods import (
    compute_prediction_metrics,
    fit_local_svd_features,
    fit_predict_tolt_adapter_suite,
)
from triomics.v4.immunophenotyping import evaluate_gmm_marker_gates, fit_bimodal_1d_gate

TABLES_DIR = REPO_ROOT / "results" / "tables"
RECEIPTS_DIR = REPO_ROOT / "results" / "receipts"

COGNATE_GENE_MAP: dict[str, str] = {
    "CD3": "CD3E", "CD4": "CD4", "CD8": "CD8A", "CD8a": "CD8A", "CD14": "CD14",
    "CD16": "FCGR3A", "CD19": "CD19", "CD20": "MS4A1", "CD56": "NCAM1",
    "HLADR": "HLA-DRA", "CD38": "CD38", "CD64": "FCGR1A", "CD169": "SIGLEC1",
    "CD163": "CD163", "CD86": "CD86", "CD279": "PDCD1", "TIGIT": "TIGIT",
    "CD95": "FAS", "KLRG1": "KLRG1", "CD27": "CD27", "CD28": "CD28",
    "CD45RA": "PTPRC", "CD45RO": "PTPRC", "CD57": "B3GAT1", "CD127": "IL7R",
    "CD25": "IL2RA", "CD11c": "ITGAX", "CD11b": "ITGAM", "CD123": "IL3RA",
    "IgD": "IGHD", "CD269": "TNFRSF17", "CD274": "CD274", "CD366": "HAVCR2",
}


def compute_adt_leiden_agreement(y_true: np.ndarray, pred_map: dict[str, np.ndarray], resolution: float = 0.6) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    """Compute Leiden cluster ARI/NMI on predicted 188-ADT space vs measured ground-truth 188-ADT space."""
    ad_true = ad.AnnData(X=y_true.astype(np.float32))
    sc.pp.pca(ad_true, n_comps=30, random_state=42)
    sc.pp.neighbors(ad_true, n_neighbors=15, use_rep="X_pca", random_state=42)
    sc.tl.umap(ad_true, random_state=42)
    sc.tl.leiden(ad_true, resolution=resolution, flavor="igraph", n_iterations=2, random_state=42)
    true_clusters = ad_true.obs["leiden"].astype(str).to_numpy()
    umap_true = ad_true.obsm["X_umap"].astype(np.float32)

    rows = []
    umap_tolt = umap_true
    for mname, yp in pred_map.items():
        ad_p = ad.AnnData(X=yp.astype(np.float32))
        sc.pp.pca(ad_p, n_comps=30, random_state=42)
        sc.pp.neighbors(ad_p, n_neighbors=15, use_rep="X_pca", random_state=42)
        sc.tl.leiden(ad_p, resolution=resolution, flavor="igraph", n_iterations=2, random_state=42)
        p_clusters = ad_p.obs["leiden"].astype(str).to_numpy()
        if mname == "TOLT":
            sc.tl.umap(ad_p, random_state=42)
            umap_tolt = ad_p.obsm["X_umap"].astype(np.float32)
        rows.append({
            "method": mname,
            "n_true_clusters": int(len(np.unique(true_clusters))),
            "n_pred_clusters": int(len(np.unique(p_clusters))),
            "leiden_ari": round(float(adjusted_rand_score(true_clusters, p_clusters)), 4),
            "leiden_nmi": round(float(normalized_mutual_info_score(true_clusters, p_clusters)), 4),
        })
    return rows, umap_true, umap_tolt


def main() -> None:
    t0 = time.perf_counter()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    cite_ids = ["A.HD1", "A.HD2", "A.HD3", "P1.1", "P2.1"]
    gschema = np.load(EMB_DIR / "gse166489_gene_schema.npz", allow_pickle=True)
    gene_names = [str(x) for x in gschema["gene_names"]]
    gene_lookup = {g.upper(): i for i, g in enumerate(gene_names)}

    b_hd3 = np.load(EMB_DIR / "gse166489_A.HD3_bundle.npz", allow_pickle=True)
    protein_names = [str(x) for x in b_hd3["adt_names"]]
    prot_lookup = {p: i for i, p in enumerate(protein_names)}

    # =========================================================================
    # 1. Analyze Saved 5-Donor LODO Predictions: Stratification, GMM Gates & Manifold ARI
    # =========================================================================
    run_step1 = not (TABLES_DIR / "gse166489_per_antibody_coupling.csv").is_file() or not (PRED_DIR / "umap_coords_P1.1.npz").is_file()
    if run_step1:
        print("=== [1/3] 188-Antibody Stratification, GMM Gating & Manifold Clustering ===", flush=True)
    strat_rows: list[dict[str, object]] = []
    gmm_rows: list[dict[str, object]] = []
    manifold_rows: list[dict[str, object]] = []
    per_ab_rows: list[dict[str, object]] = []

    key_markers = ["CD3", "CD4", "CD8", "CD8a", "CD14", "CD16", "CD19", "CD56", "IgD", "HLADR", "CD38", "CD64", "CD169", "CD279"]

    for held_out in (cite_ids if run_step1 else []):
        for ref_tag in ["ref500", "ref6000"]:
            npz_path = PRED_DIR / f"gse166489_lodo_{held_out}_{ref_tag}.npz"
            if not npz_path.is_file():
                continue
            with np.load(npz_path, allow_pickle=True) as d:
                y_true = d["y_true"]
                y_train = d["y_train"]
                weights = d["tolt_simplex_weights"] if "tolt_simplex_weights" in d else None
                methods = [k for k in d.files if k not in {"y_true", "y_train", "tolt_simplex_weights", "Legacy-Euclid-kNN256"}]
                preds = {m: d[m] for m in methods}

            # Objective training-only abundance tiers (zero test leakage)
            tr_std = y_train.std(axis=0)
            tr_p99 = np.percentile(y_train, 99, axis=0)
            tier1 = (tr_std >= 0.35) | (tr_p99 >= 1.50)
            tier2 = (~tier1) & ((tr_std >= 0.22) | (tr_p99 >= 1.00))
            tier3 = (~tier1) & (~tier2)

            for mname, yp in preds.items():
                for tname, tmask in [("All_188_Human", np.ones(len(protein_names), dtype=bool)),
                                     ("Tier1_Expressed_Receptors", tier1),
                                     ("Tier2_Rare_Inducible", tier2),
                                     ("Tier3_Background", tier3)]:
                    if not tmask.any():
                        continue
                    met = compute_prediction_metrics(y_true[:, tmask], yp[:, tmask])
                    strat_rows.append({
                        "held_out_donor": held_out,
                        "reference_budget": ref_tag,
                        "tier": tname,
                        "n_antibodies": int(tmask.sum()),
                        "method": mname,
                        "protein_pcc_median": round(met["protein_pcc_median"], 4),
                        "protein_pcc_mean": round(met["protein_pcc_mean"], 4),
                        "mse": round(met["mse"], 4),
                    })

            if ref_tag == "ref6000":
                g_rows = evaluate_gmm_marker_gates(y_train, y_true, preds, protein_names, key_markers)
                for gr in g_rows:
                    gr["held_out_donor"] = held_out
                gmm_rows.extend(g_rows)

                # Manifold Leiden clustering agreement on A.HD3 and P1.1
                if held_out in {"A.HD3", "P1.1"}:
                    m_sub = {k: preds[k] for k in ["TOLT", "totalVI", "sciPENN", "scLinear", "Seurat-Anchor", "SVD-Ridge"] if k in preds}
                    l_rows, u_true, u_tolt = compute_adt_leiden_agreement(y_true, m_sub)
                    for lr in l_rows:
                        lr["held_out_donor"] = held_out
                    manifold_rows.extend(l_rows)
                    np.savez_compressed(
                        PRED_DIR / f"umap_coords_{held_out}.npz",
                        umap_measured=u_true,
                        umap_tolt=u_tolt,
                        y_true=y_true,
                        y_tolt=preds["TOLT"],
                        protein_names=np.asarray(protein_names, dtype=str),
                    )

            # Per-antibody mRNA-protein coupling & weights on ref500
            if ref_tag == "ref500" and weights is not None:
                xr_qry = log1p_cp10k_sparse(sp.load_npz(EMB_DIR / f"gse166489_{held_out}_rna_counts.npz"))
                pcc_tolt = compute_prediction_metrics(y_true, preds["TOLT"])["per_protein_pcc"]
                pcc_svd = compute_prediction_metrics(y_true, preds["SVD-Ridge"])["per_protein_pcc"]
                for j, pname in enumerate(protein_names):
                    cog_g = COGNATE_GENE_MAP.get(pname, pname).upper()
                    mrna_r = np.nan
                    if cog_g in gene_lookup:
                        gv = xr_qry[:, gene_lookup[cog_g]].toarray().ravel()
                        if gv.std() > 1e-6 and y_true[:, j].std() > 1e-6:
                            mrna_r = float(np.corrcoef(gv, y_true[:, j])[0, 1])
                    per_ab_rows.append({
                        "held_out_donor": held_out,
                        "protein": pname,
                        "tier": "Tier1" if tier1[j] else ("Tier2" if tier2[j] else "Tier3"),
                        "tolt_pcc": round(float(pcc_tolt[j]), 4),
                        "svd_pcc": round(float(pcc_svd[j]), 4),
                        "delta_pcc_tolt_minus_svd": round(float(pcc_tolt[j] - pcc_svd[j]), 4),
                        "w_local_svd": round(float(weights[j, 0]), 4),
                        "w_tolt_direct": round(float(weights[j, 1]), 4),
                        "w_tolt_residual": round(float(weights[j, 2]), 4),
                        "w_tolt_total": round(float(weights[j, 1] + weights[j, 2]), 4),
                        "cognate_mrna_protein_pcc": round(mrna_r, 4) if np.isfinite(mrna_r) else np.nan,
                    })

    if run_step1:
        pd.DataFrame(strat_rows).to_csv(TABLES_DIR / "gse166489_antibody_stratification.csv", index=False)
        pd.DataFrame(gmm_rows).to_csv(TABLES_DIR / "gse166489_bimodal_gmm_gating.csv", index=False)
        pd.DataFrame(manifold_rows).to_csv(TABLES_DIR / "gse166489_manifold_clustering.csv", index=False)
        pd.DataFrame(per_ab_rows).to_csv(TABLES_DIR / "gse166489_per_antibody_coupling.csv", index=False)

    # =========================================================================
    # 2. Full 18-Donor Cohort Surface-Proteome Inference & Age-Matched / Longitudinal MIS-C Discovery
    # =========================================================================
    print("=== [2/3] Imputing 188 Surface Proteins Across All 18 GSE166489 Samples ===", flush=True)
    # Build balanced 5-donor CITE-seq reference (1,500 cells per donor = 7,500 cells) for the 13 completely unseen RNA-only pediatric samples;
    # for the 5 CITE-seq donors themselves, strictly use their LODO out-of-donor predictions (zero self-donor leakage!).
    x_cite_list, y_cite_list, e_cite_list = [], [], []
    rng_ref = np.random.default_rng(20261120)
    for sid in cite_ids:
        xr = log1p_cp10k_sparse(sp.load_npz(EMB_DIR / f"gse166489_{sid}_rna_counts.npz"))
        bd = np.load(EMB_DIR / f"gse166489_{sid}_bundle.npz", allow_pickle=True)
        yc = seurat_clr(bd["adt_counts"].astype(np.float32))
        eb = bd["base_110"]
        sel = np.sort(rng_ref.choice(len(yc), size=min(1500, len(yc)), replace=False))
        x_cite_list.append(xr[sel])
        y_cite_list.append(yc[sel])
        e_cite_list.append(eb[sel])

    x_ref_7500 = sp.vstack(x_cite_list, format="csr")
    y_ref_7500 = np.vstack(y_cite_list)
    e_ref_7500 = np.vstack(e_cite_list)

    # Fit monocyte & T-cell lineage gates strictly on reference ground-truth ADT
    cd14_thr, _ = fit_bimodal_1d_gate(y_ref_7500[:, prot_lookup["CD14"]], nonzero_only=True)
    cd3_thr, _ = fit_bimodal_1d_gate(y_ref_7500[:, prot_lookup["CD3"]], nonzero_only=False)
    cd19_thr, _ = fit_bimodal_1d_gate(y_ref_7500[:, prot_lookup["CD19"]], nonzero_only=True)

    donor_pathology_rows: list[dict[str, object]] = []
    tcr_clone_rows: list[dict[str, object]] = []

    idx_cd169 = prot_lookup["CD169"]
    idx_cd64 = prot_lookup["CD64"]
    idx_hladr = prot_lookup["HLADR"]
    idx_cd163 = prot_lookup["CD163"]
    idx_cd38 = prot_lookup["CD38"]
    idx_pd1 = prot_lookup["CD279"]
    idx_tigit = prot_lookup["TIGIT"]
    idx_cd45ro = prot_lookup["CD45RO"]

    g_cd38 = gene_lookup.get("CD38")
    g_pd1 = gene_lookup.get("PDCD1")
    g_siglec1 = gene_lookup.get("SIGLEC1")

    for meta in GSE166489_METADATA:
        sid = meta["sample_id"]
        xr_raw = sp.load_npz(EMB_DIR / f"gse166489_{sid}_rna_counts.npz")
        xr_log = log1p_cp10k_sparse(xr_raw)
        bd = np.load(EMB_DIR / f"gse166489_{sid}_bundle.npz", allow_pickle=True)
        emb_s = bd["base_110"]

        if meta["has_adt"] == "true":
            with np.load(PRED_DIR / f"gse166489_lodo_{sid}_ref6000.npz", allow_pickle=True) as ld:
                pred_adt = ld["TOLT"]
                meas_adt = ld["y_true"]
        else:
            svd_r, svd_q = fit_local_svd_features(x_ref_7500, xr_log, seed=42)
            suite_q = fit_predict_tolt_adapter_suite(svd_r, svd_q, e_ref_7500, emb_s, y_ref_7500, seed=42, nonneg_clip=True)
            pred_adt = suite_q["TOLT"]["pred"]
            meas_adt = None
            np.savez_compressed(PRED_DIR / f"gse166489_imputed_{sid}.npz", pred_tolt=pred_adt)

        # Identify predicted CD14+ Monocytes and CD3+ T cells
        mono_mask = (pred_adt[:, prot_lookup["CD14"]] >= cd14_thr * 0.90) & (pred_adt[:, prot_lookup["CD3"]] < cd3_thr)
        t_mask = (pred_adt[:, prot_lookup["CD3"]] >= cd3_thr * 0.90) & (pred_adt[:, prot_lookup["CD19"]] < cd19_thr)

        rec: dict[str, object] = {
            "sample_id": sid,
            "patient": meta["patient"],
            "condition": meta["condition"],
            "age_group": meta["age_group"],
            "timepoint": meta["timepoint"],
            "has_measured_adt": meta["has_adt"] == "true",
            "n_cells": int(len(pred_adt)),
            "n_monocytes": int(mono_mask.sum()),
            "n_t_cells": int(t_mask.sum()),
            "pred_mono_CD169_mean": round(float(pred_adt[mono_mask, idx_cd169].mean()), 4) if mono_mask.sum() >= 10 else np.nan,
            "pred_mono_CD64_mean": round(float(pred_adt[mono_mask, idx_cd64].mean()), 4) if mono_mask.sum() >= 10 else np.nan,
            "pred_mono_HLADR_mean": round(float(pred_adt[mono_mask, idx_hladr].mean()), 4) if mono_mask.sum() >= 10 else np.nan,
            "pred_mono_CD163_mean": round(float(pred_adt[mono_mask, idx_cd163].mean()), 4) if mono_mask.sum() >= 10 else np.nan,
            "pred_tcell_CD38_mean": round(float(pred_adt[t_mask, idx_cd38].mean()), 4) if t_mask.sum() >= 10 else np.nan,
            "pred_tcell_PD1_mean": round(float(pred_adt[t_mask, idx_pd1].mean()), 4) if t_mask.sum() >= 10 else np.nan,
            "pred_tcell_HLADR_mean": round(float(pred_adt[t_mask, idx_hladr].mean()), 4) if t_mask.sum() >= 10 else np.nan,
        }
        if meas_adt is not None:
            m_mono = (meas_adt[:, prot_lookup["CD14"]] >= cd14_thr) & (meas_adt[:, prot_lookup["CD3"]] < cd3_thr)
            m_t = (meas_adt[:, prot_lookup["CD3"]] >= cd3_thr) & (meas_adt[:, prot_lookup["CD19"]] < cd19_thr)
            rec["meas_mono_CD169_mean"] = round(float(meas_adt[m_mono, idx_cd169].mean()), 4)
            rec["meas_mono_CD64_mean"] = round(float(meas_adt[m_mono, idx_cd64].mean()), 4)
            rec["meas_mono_HLADR_mean"] = round(float(meas_adt[m_mono, idx_hladr].mean()), 4)
            rec["meas_tcell_CD38_mean"] = round(float(meas_adt[m_t, idx_cd38].mean()), 4)
            rec["meas_tcell_PD1_mean"] = round(float(meas_adt[m_t, idx_pd1].mean()), 4)
        donor_pathology_rows.append(rec)

        # Single-cell TCR TRBV11-2 superantigen clone surface phenotyping & mRNA dropout analysis
        if "has_tcr" in bd and "is_trbv11_2" in bd:
            has_tcr = bd["has_tcr"].astype(bool) & t_mask
            is_v11 = bd["is_trbv11_2"].astype(bool)
            for clone_grp, cmask in [("TRBV11-2_pos", has_tcr & is_v11), ("TRBV11-2_neg", has_tcr & (~is_v11))]:
                if cmask.sum() < 5:
                    continue
                cd38_rna_zero_pct = 100.0 * float((xr_raw[cmask, g_cd38].toarray().ravel() == 0).mean()) if g_cd38 is not None else np.nan
                pd1_rna_zero_pct = 100.0 * float((xr_raw[cmask, g_pd1].toarray().ravel() == 0).mean()) if g_pd1 is not None else np.nan
                tcr_clone_rows.append({
                    "sample_id": sid,
                    "condition": meta["condition"],
                    "age_group": meta["age_group"],
                    "clone_group": clone_grp,
                    "n_t_cells": int(cmask.sum()),
                    "pred_CD38_mean": round(float(pred_adt[cmask, idx_cd38].mean()), 4),
                    "pred_HLADR_mean": round(float(pred_adt[cmask, idx_hladr].mean()), 4),
                    "pred_PD1_mean": round(float(pred_adt[cmask, idx_pd1].mean()), 4),
                    "pred_TIGIT_mean": round(float(pred_adt[cmask, idx_tigit].mean()), 4),
                    "pred_CD45RO_mean": round(float(pred_adt[cmask, idx_cd45ro].mean()), 4),
                    "raw_CD38_mRNA_dropout_pct": round(cd38_rna_zero_pct, 2),
                    "raw_PDCD1_mRNA_dropout_pct": round(pd1_rna_zero_pct, 2),
                })

    df_path = pd.DataFrame(donor_pathology_rows)
    df_tcr = pd.DataFrame(tcr_clone_rows)
    df_path.to_csv(TABLES_DIR / "gse166489_18donor_misc_pathology.csv", index=False)
    df_tcr.to_csv(TABLES_DIR / "gse166489_tcr_trbv11_2_phenotyping.csv", index=False)

    # Compute donor-level statistical comparisons (Age-matched Pediatric Acute MIS-C n=7 vs Pediatric Healthy C.HD1-6 n=6)
    ped_acute = df_path[df_path["condition"].isin(["MIS-C_Severe", "MIS-C_Moderate"])]
    ped_hd = df_path[df_path["condition"] == "Pediatric_Healthy"]
    stats_summary = {}
    for col in ["pred_mono_CD169_mean", "pred_mono_CD64_mean", "pred_mono_HLADR_mean", "pred_tcell_CD38_mean", "pred_tcell_PD1_mean"]:
        a_vals = ped_acute[col].dropna().to_numpy()
        h_vals = ped_hd[col].dropna().to_numpy()
        if len(a_vals) >= 3 and len(h_vals) >= 3:
            u_stat, p_val = mannwhitneyu(a_vals, h_vals, alternative="two-sided")
            stats_summary[col] = {
                "pediatric_acute_misc_mean": round(float(a_vals.mean()), 4),
                "pediatric_healthy_mean": round(float(h_vals.mean()), 4),
                "mannwhitney_p": float(p_val),
            }

    receipt = {
        "schema": "tolt.v4.gse166489_clinical_receipt.v1",
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "age_matched_pediatric_stats": stats_summary,
        "n_donors_analyzed": len(df_path),
        "n_tcr_strata_rows": len(df_tcr),
    }
    (RECEIPTS_DIR / "05_gse166489_clinical_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"DONE GSE166489 18-Donor Clinical Discovery in {receipt['elapsed_sec']}s!", flush=True)


if __name__ == "__main__":
    main()
