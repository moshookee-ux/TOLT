"""Run systematic pretrained vs random-initialized vs ancestor (F01) backbone ablations across all 4 cohorts.

Produces:
  - results/tables/ablation_backbone_representations.csv
  - results/tables/ablation_simplex_weights.csv
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
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from triomics.v4.paths import EMB_DIR

from triomics.v4.cohort_loaders import D_BMMC90K, D_STUART, log1p_cp10k_dense, log1p_cp10k_sparse, seurat_clr
from triomics.v4.core_methods import (
    compute_prediction_metrics,
    fit_local_svd_features,
    fit_predict_tolt_adapter_suite,
)

TABLES_DIR = REPO_ROOT / "results" / "tables"
RECEIPTS_DIR = REPO_ROOT / "results" / "receipts"


def evaluate_task(
    task_name: str,
    modality: str,
    x_tr_log: sp.csr_matrix,
    x_te_log: sp.csr_matrix,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    emb_tr_map: dict[str, np.ndarray],
    emb_te_map: dict[str, np.ndarray],
    budgets: list[int | str],
    x_atac_tr: sp.csr_matrix | None = None,
    x_atac_te: sp.csr_matrix | None = None,
    protein_subset_mask: np.ndarray | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    perf_rows: list[dict[str, object]] = []
    weight_rows: list[dict[str, object]] = []

    rng = np.random.default_rng(20261120)
    n_full = x_tr_log.shape[0]

    for b in budgets:
        if b == "full" or (isinstance(b, int) and b >= n_full):
            sub_idx = np.arange(n_full)
            b_label = f"full_{n_full}"
            n_ref = n_full
        else:
            n_ref = int(b)
            sub_idx = np.sort(rng.choice(n_full, size=n_ref, replace=False))
            b_label = str(n_ref)

        svd_tr, svd_te = fit_local_svd_features(
            x_tr_log[sub_idx],
            x_te_log,
            seed=42,
            x_atac_train=x_atac_tr[sub_idx] if x_atac_tr is not None else None,
            x_atac_test=x_atac_te if x_atac_te is not None else None,
        )

        # Evaluate Base-110, F01, and 3 Random-Init seeds
        for bbone in ["base_110", "f01_s29", "rand_s11", "rand_s29", "rand_s47"]:
            suite = fit_predict_tolt_adapter_suite(
                svd_tr,
                svd_te,
                emb_tr_map[bbone][sub_idx],
                emb_te_map[bbone],
                y_tr[sub_idx],
                seed=42,
                nonneg_clip=True,
            )
            w = suite["TOLT"]["weights"]  # (n_prot, 3): [w_svd, w_direct, w_residual]
            weight_rows.append({
                "task": task_name,
                "modality": modality,
                "budget": b_label,
                "n_ref": n_ref,
                "backbone": bbone,
                "mean_w_local_svd": round(float(w[:, 0].mean()), 4),
                "mean_w_tolt_direct": round(float(w[:, 1].mean()), 4),
                "mean_w_tolt_residual": round(float(w[:, 2].mean()), 4),
            })

            for head_name in (["SVD-Ridge", "TOLT-Direct", "TOLT-Residual", "TOLT"] if bbone == "base_110" else ["TOLT-Direct", "TOLT-Residual", "TOLT"]):
                if bbone != "base_110" and head_name == "SVD-Ridge":
                    continue
                pred = suite[head_name]["pred"]
                yt_eval = y_te[:, protein_subset_mask] if protein_subset_mask is not None else y_te
                yp_eval = pred[:, protein_subset_mask] if protein_subset_mask is not None else pred
                m = compute_prediction_metrics(yt_eval, yp_eval)
                perf_rows.append({
                    "task": task_name,
                    "modality": modality,
                    "budget": b_label,
                    "n_ref": n_ref,
                    "n_query": int(y_te.shape[0]),
                    "n_proteins_scored": int(yt_eval.shape[1]),
                    "backbone": "none" if head_name == "SVD-Ridge" else bbone,
                    "readout": head_name,
                    "protein_pcc_median": round(m["protein_pcc_median"], 4),
                    "protein_pcc_mean": round(m["protein_pcc_mean"], 4),
                    "protein_spearman_median": round(m["protein_spearman_median"], 4),
                    "cell_pcc_median": round(m["cell_pcc_median"], 4),
                    "mse": round(m["mse"], 4),
                })

    return perf_rows, weight_rows


def main() -> None:
    t0 = time.perf_counter()
    all_perf: list[dict[str, object]] = []
    all_weights: list[dict[str, object]] = []

    # 1. GSE128639 BMMC Stuart Cross-Donor (Donor A -> Donor B)
    print("=== Ablation 1: BMMC Stuart Cross-Donor (Donor A -> Donor B) ===", flush=True)
    ad_stu = ad.read_h5ad(D_STUART)
    x_stu_log = log1p_cp10k_sparse(sp.csr_matrix(ad_stu.X, dtype=np.float32))
    prot_stu = ad_stu.obsm["protein_expression"]
    prot_stu = prot_stu.toarray() if sp.issparse(prot_stu) else np.asarray(prot_stu, dtype=np.float32)
    y_stu_clr = seurat_clr(prot_stu)
    with np.load(EMB_DIR / "gse128639_stuart_33454_embeddings.npz") as d:
        donors = d["donors"]
        da = np.flatnonzero(donors == "Donor_A")
        db = np.flatnonzero(donors == "Donor_B")
        e_tr = {k: d[k][da] for k in ["base_110", "f01_s29", "rand_s11", "rand_s29", "rand_s47"]}
        e_te = {k: d[k][db] for k in ["base_110", "f01_s29", "rand_s11", "rand_s29", "rand_s47"]}
    p, w = evaluate_task("BMMC_Stuart_A_to_B", "RNA", x_stu_log[da], x_stu_log[db], y_stu_clr[da], y_stu_clr[db], e_tr, e_te, [100, 250, 500, 1500, "full"])
    all_perf.extend(p); all_weights.extend(w)
    del ad_stu

    # 2. GSE194122 Unseen Cross-Donor + Cross-Site (12710 Site 2 -> 13272 Site 4)
    print("=== Ablation 2: GSE194122 Unseen Cross-Donor + Cross-Site (12710 -> 13272) ===", flush=True)
    ad_90k = ad.read_h5ad(D_BMMC90K)
    unseen_mask = ad_90k.obs["DonorID"].astype(str).isin(["12710", "13272"]).to_numpy()
    sub_90k = ad_90k[unseen_mask].copy()
    x_90k_log = log1p_cp10k_sparse(sp.csr_matrix(sub_90k.X, dtype=np.float32))
    prot_90k = np.asarray(sub_90k.obsm["protein_expression"], dtype=np.float32)
    y_90k_clr = seurat_clr(prot_90k)
    with np.load(EMB_DIR / "gse194122_unseen_12949_embeddings.npz") as d:
        dids = d["donor_ids"]
        i12 = np.flatnonzero(dids == "12710")
        i13 = np.flatnonzero(dids == "13272")
        e_tr = {k: d[k][i12] for k in ["base_110", "f01_s29", "rand_s11", "rand_s29", "rand_s47"]}
        e_te = {k: d[k][i13] for k in ["base_110", "f01_s29", "rand_s11", "rand_s29", "rand_s47"]}
    p, w = evaluate_task("GSE194122_12710_to_13272", "RNA", x_90k_log[i12], x_90k_log[i13], y_90k_clr[i12], y_90k_clr[i13], e_tr, e_te, [100, 250, 500, 1500, "full"])
    all_perf.extend(p); all_weights.extend(w)
    del ad_90k, sub_90k

    # 3. GSE166489 Cross-Condition Transfer (Healthy A.HD1+A.HD2 -> Severe MIS-C P1.1, Expressed Tier-1 subset & All 188)
    print("=== Ablation 3: GSE166489 Healthy (A.HD1+2) -> Severe MIS-C (P1.1) ===", flush=True)
    x_hd1 = log1p_cp10k_sparse(sp.load_npz(EMB_DIR / "gse166489_A.HD1_rna_counts.npz"))
    x_hd2 = log1p_cp10k_sparse(sp.load_npz(EMB_DIR / "gse166489_A.HD2_rna_counts.npz"))
    x_p1 = log1p_cp10k_sparse(sp.load_npz(EMB_DIR / "gse166489_P1.1_rna_counts.npz"))
    b_hd1 = np.load(EMB_DIR / "gse166489_A.HD1_bundle.npz", allow_pickle=True)
    b_hd2 = np.load(EMB_DIR / "gse166489_A.HD2_bundle.npz", allow_pickle=True)
    b_p1 = np.load(EMB_DIR / "gse166489_P1.1_bundle.npz", allow_pickle=True)

    x_tr_m = sp.vstack([x_hd1, x_hd2], format="csr")
    y_tr_m = seurat_clr(np.vstack([b_hd1["adt_counts"], b_hd2["adt_counts"]]))
    y_te_m = seurat_clr(b_p1["adt_counts"])
    expressed_mask = (y_tr_m.std(axis=0) >= 0.35) | (np.percentile(y_tr_m, 99, axis=0) >= 1.50)

    e_tr_m = {k: np.vstack([b_hd1[k], b_hd2[k]]) for k in ["base_110", "f01_s29", "rand_s11", "rand_s29", "rand_s47"]}
    e_te_m = {k: b_p1[k] for k in ["base_110", "f01_s29", "rand_s11", "rand_s29", "rand_s47"]}
    p, w = evaluate_task("GSE166489_HD12_to_MISC_P1_Expressed", "RNA", x_tr_m, x_p1, y_tr_m, y_te_m, e_tr_m, e_te_m, [100, 250, 500, 1500, 4000], protein_subset_mask=expressed_mask)
    all_perf.extend(p); all_weights.extend(w)

    # 4. GSE158013 True Cross-Donor TEA-seq (X066_W3 -> X061_W1, 45 clean ADTs, RNA & RNA+ATAC)
    print("=== Ablation 4: GSE158013 Cross-Donor TEA-seq (X066_W3 -> X061_W1, RNA & RNA+ATAC) ===", flush=True)
    xr_w3 = log1p_cp10k_sparse(sp.load_npz(EMB_DIR / "gse158013_X066_W3_rna_counts.npz"))
    xa_w3 = sp.load_npz(EMB_DIR / "gse158013_X066_W3_atac_consensus30k.npz")
    xr_x1 = log1p_cp10k_sparse(sp.load_npz(EMB_DIR / "gse158013_X061_W1_rna_counts.npz"))
    xa_x1 = sp.load_npz(EMB_DIR / "gse158013_X061_W1_atac_consensus30k.npz")
    bw3 = np.load(EMB_DIR / "gse158013_X066_W3_bundle.npz", allow_pickle=True)
    bx1 = np.load(EMB_DIR / "gse158013_X061_W1_bundle.npz", allow_pickle=True)
    yw3 = log1p_cp10k_dense(bw3["adt_counts_45"])
    yx1 = log1p_cp10k_dense(bx1["adt_counts_45"])

    for mod_tag, use_a in [("RNA", False), ("RNA+ATAC", True)]:
        suf = "_rna_atac" if use_a else "_rna"
        e_tr_t = {k: bw3[f"{k}{suf}"] for k in ["base_110", "f01_s29", "rand_s11", "rand_s29", "rand_s47"]}
        e_te_t = {k: bx1[f"{k}{suf}"] for k in ["base_110", "f01_s29", "rand_s11", "rand_s29", "rand_s47"]}
        p, w = evaluate_task(
            "TEAseq_CrossDonor_X066W3_to_X061W1",
            mod_tag,
            xr_w3,
            xr_x1,
            yw3,
            yx1,
            e_tr_t,
            e_te_t,
            [100, 250, 500, 1500, "full"],
            x_atac_tr=xa_w3 if use_a else None,
            x_atac_te=xa_x1 if use_a else None,
        )
        all_perf.extend(p); all_weights.extend(w)

    df_perf = pd.DataFrame(all_perf)
    df_weights = pd.DataFrame(all_weights)
    df_perf.to_csv(TABLES_DIR / "ablation_backbone_representations.csv", index=False)
    df_weights.to_csv(TABLES_DIR / "ablation_simplex_weights.csv", index=False)

    receipt = {
        "schema": "tolt.v4.backbone_ablation_receipt.v1",
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "n_perf_rows": len(df_perf),
        "n_weight_rows": len(df_weights),
    }
    (RECEIPTS_DIR / "03_backbone_ablation_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"DONE Backbone Ablations ({len(df_perf)} rows) in {receipt['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
