"""GSE158013 Multi-Well & Cross-Donor Triomic (RNA+ATAC -> 45 Clean ADTs) Benchmark & 9-Lineage PBMC Immunophenotyping.

Strictly separates:
  1. Cross-Well Technical Batch Transfer (Donor HMN200910: X066_W3 -> X066_W4, X066_W5, X066_W6)
  2. True Cross-Donor Triomic Transfer (Donor HMN200910 X066_W3 <-> Independent Donor X061_W1, plus Pooled X066 -> X061_W1)
  3. Within-Well Reference Comparison on X066_W3 (including official scVAEIT v1.1.1 baseline)

Outputs:
  - results/tables/teaseq_triomic_transfer_summary.csv
  - results/tables/teaseq_per_antibody_chromatin_gain.csv
  - results/tables/teaseq_9lineage_immunophenotyping.csv
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
from scipy.stats import wilcoxon
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from triomics.v4.paths import EMB_DIR, PRED_DIR, TEASEQ_REFERENCE_DIR

from triomics.v4.cohort_loaders import TEASEQ_WELLS, log1p_cp10k_dense, log1p_cp10k_sparse
from triomics.v4.core_methods import (
    compute_prediction_metrics,
    fit_local_svd_features,
    fit_predict_tolt_adapter_suite,
    run_fast_classical_and_modern_baselines,
)
from triomics.v4.immunophenotyping import (
    NINE_LINEAGE_CLASSES,
    fit_pbmc_9lineage_gates,
    score_9lineage_predictions,
)

TABLES_DIR = REPO_ROOT / "results" / "tables"
RECEIPTS_DIR = REPO_ROOT / "results" / "receipts"


def load_well_bundle(wid: str) -> dict[str, object]:
    xr = log1p_cp10k_sparse(sp.load_npz(EMB_DIR / f"gse158013_{wid}_rna_counts.npz"))
    xa = sp.load_npz(EMB_DIR / f"gse158013_{wid}_atac_consensus30k.npz")
    bd = np.load(EMB_DIR / f"gse158013_{wid}_bundle.npz", allow_pickle=True)
    raw45 = bd["adt_counts_45"].astype(np.float32)
    y45 = log1p_cp10k_dense(raw45)
    raw_iso = bd["isotype_counts"].astype(np.float32)
    # Isotype in log1p(count / total_46_sum * 1e4)
    tot46 = np.maximum(bd["adt_counts_all46"].astype(np.float32).sum(axis=1, keepdims=True), 1.0)
    y_iso = np.log1p(raw_iso / tot46 * 10000.0).astype(np.float32)
    return {
        "xr": xr,
        "xa": xa,
        "y45": y45,
        "y_iso": y_iso,
        "names45": [str(x) for x in bd["adt_names_45"]],
        "emb_rna": bd["base_110_rna"],
        "emb_dual": bd["base_110_rna_atac"],
        "f01_rna": bd["f01_s29_rna"],
        "f01_dual": bd["f01_s29_rna_atac"],
    }


def evaluate_triomic_pair(
    transfer_tier: str,
    pair_name: str,
    budget_label: str,
    ref: dict[str, object],
    qry: dict[str, object],
    ref_idx: np.ndarray,
    qry_idx: np.ndarray,
    seed: int = 42,
    save_umap_key: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    summary_rows: list[dict[str, object]] = []
    gain_rows: list[dict[str, object]] = []
    pheno_rows: list[dict[str, object]] = []

    xr_tr = ref["xr"][ref_idx]      # type: ignore[index]
    xa_tr = ref["xa"][ref_idx]      # type: ignore[index]
    y_tr = ref["y45"][ref_idx]      # type: ignore[index]
    xr_te = qry["xr"][qry_idx]      # type: ignore[index]
    xa_te = qry["xa"][qry_idx]      # type: ignore[index]
    y_te = qry["y45"][qry_idx]      # type: ignore[index]
    names45: list[str] = ref["names45"]  # type: ignore[assignment]

    # 1. RNA-only local & TOLT
    svd_r_tr, svd_r_te = fit_local_svd_features(xr_tr, xr_te, seed=seed)
    suite_rna = fit_predict_tolt_adapter_suite(
        svd_r_tr, svd_r_te, ref["emb_rna"][ref_idx], qry["emb_rna"][qry_idx], y_tr, seed=seed  # type: ignore[index]
    )
    fast_rna = run_fast_classical_and_modern_baselines(xr_tr, xr_te, svd_r_tr, svd_r_te, y_tr, seed=seed)

    # 2. RNA+ATAC dual local & TOLT
    svd_d_tr, svd_d_te = fit_local_svd_features(xr_tr, xr_te, seed=seed, x_atac_train=xa_tr, x_atac_test=xa_te)
    suite_dual = fit_predict_tolt_adapter_suite(
        svd_d_tr, svd_d_te, ref["emb_dual"][ref_idx], qry["emb_dual"][qry_idx], y_tr, seed=seed  # type: ignore[index]
    )
    fast_dual = run_fast_classical_and_modern_baselines(xr_tr, xr_te, svd_d_tr, svd_d_te, y_tr, seed=seed)

    method_preds: dict[str, tuple[np.ndarray, np.ndarray | None]] = {
        "Classical_RNA": (suite_rna["SVD-Ridge"]["pred"], suite_rna["SVD-Ridge"]["oof"]),
        "Classical_RNA+ATAC": (suite_dual["SVD-Ridge"]["pred"], suite_dual["SVD-Ridge"]["oof"]),
        "scLinear_RNA": (fast_rna["scLinear"]["pred"], None),
        "Cosine-kNN_RNA+ATAC": (fast_dual["Cosine-kNN"]["pred"], None),
        "ExtraTrees_RNA+ATAC": (fast_dual["ExtraTrees"]["pred"], None),
        "TOLT_RNA": (suite_rna["TOLT"]["pred"], suite_rna["TOLT"]["oof"]),
        "TOLT_RNA+ATAC": (suite_dual["TOLT"]["pred"], suite_dual["TOLT"]["oof"]),
    }

    gate_spec = fit_pbmc_9lineage_gates(y_tr, names45)

    per_ab_pccs: dict[str, np.ndarray] = {}
    for mname, (yp, yoof) in method_preds.items():
        met = compute_prediction_metrics(y_te, yp)
        per_ab_pccs[mname] = met["per_protein_pcc"]
        ph = score_9lineage_predictions(y_te, yp, gate_spec, pred_oof=yoof)
        summary_rows.append({
            "transfer_tier": transfer_tier,
            "pair": pair_name,
            "budget": budget_label,
            "n_ref": int(len(ref_idx)),
            "n_query": int(len(qry_idx)),
            "method": mname,
            "clean45_pcc_median": round(met["protein_pcc_median"], 4),
            "clean45_pcc_mean": round(met["protein_pcc_mean"], 4),
            "clean45_cell_pcc_median": round(met["cell_pcc_median"], 4),
            "clean45_mse": round(met["mse"], 4),
            "pheno9_p1_acc_pct": ph["p1_accuracy_pct"],
            "pheno9_p1_macro_f1": ph["p1_macro_f1_8lin"],
            "pheno9_p2_acc_pct": ph["p2_accuracy_pct"],
            "pheno9_p2_macro_f1": ph["p2_macro_f1_8lin"],
        })
        pheno_rows.append({
            "transfer_tier": transfer_tier,
            "pair": pair_name,
            "budget": budget_label,
            "method": mname,
            **{k: v for k, v in ph.items() if not k.endswith("_labels")},
        })

    # Per-antibody chromatin gain (TOLT_RNA+ATAC vs TOLT_RNA)
    r_rna = per_ab_pccs["TOLT_RNA"]
    r_dual = per_ab_pccs["TOLT_RNA+ATAC"]
    r_cl_dual = per_ab_pccs["Classical_RNA+ATAC"]
    for j, pname in enumerate(names45):
        gain_rows.append({
            "transfer_tier": transfer_tier,
            "pair": pair_name,
            "budget": budget_label,
            "antibody": pname,
            "tolt_rna_pcc": round(float(r_rna[j]), 4),
            "tolt_rna_atac_pcc": round(float(r_dual[j]), 4),
            "classical_rna_atac_pcc": round(float(r_cl_dual[j]), 4),
            "delta_atac_tolt": round(float(r_dual[j] - r_rna[j]), 4),
            "delta_tolt_vs_classical_dual": round(float(r_dual[j] - r_cl_dual[j]), 4),
        })

    if save_umap_key is not None:
        ad_t = ad.AnnData(X=y_te.astype(np.float32))
        sc.pp.pca(ad_t, n_comps=25, random_state=42)
        sc.pp.neighbors(ad_t, n_neighbors=15, use_rep="X_pca", random_state=42)
        sc.tl.umap(ad_t, random_state=42)
        true_9 = score_9lineage_predictions(y_te, method_preds["TOLT_RNA+ATAC"][0], gate_spec, method_preds["TOLT_RNA+ATAC"][1])
        np.savez_compressed(
            PRED_DIR / f"{save_umap_key}.npz",
            umap_measured=ad_t.obsm["X_umap"].astype(np.float32),
            y_true=y_te,
            pred_tolt_dual=method_preds["TOLT_RNA+ATAC"][0],
            pred_tolt_rna=method_preds["TOLT_RNA"][0],
            pred_classical_dual=method_preds["Classical_RNA+ATAC"][0],
            true_9lineage=true_9["true_labels"],
            pred_9lineage_p1=true_9["p1_labels"],
            pred_9lineage_p2=true_9["p2_labels"],
            lineage_names=np.asarray(NINE_LINEAGE_CLASSES, dtype=str),
            antibody_names=np.asarray(names45, dtype=str),
        )

    return summary_rows, gain_rows, pheno_rows


def main() -> None:
    t0 = time.perf_counter()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    wells = {w["well_id"]: load_well_bundle(w["well_id"]) for w in TEASEQ_WELLS}

    all_sum: list[dict[str, object]] = []
    all_gain: list[dict[str, object]] = []
    all_pheno: list[dict[str, object]] = []

    # 1. Cross-Well Technical Batch Transfer (Same donor HMN200910: X066_W3 -> W4, W5, W6)
    print("=== [1/3] Cross-Well Technical Transfer (X066_W3 -> W4, W5, W6) ===", flush=True)
    for target_w in ["X066_W4", "X066_W5", "X066_W6"]:
        n_r = len(wells["X066_W3"]["y45"])  # type: ignore[arg-type]
        n_q = len(wells[target_w]["y45"])   # type: ignore[arg-type]
        for b in [250, 500, "full"]:
            r_idx = np.arange(n_r) if b == "full" else np.sort(np.random.default_rng(700 + int(b)).choice(n_r, size=int(b), replace=False))
            s, g, p = evaluate_triomic_pair(
                "Cross_Well_Same_Donor_HMN200910",
                f"X066_W3_to_{target_w}",
                str(b),
                wells["X066_W3"],
                wells[target_w],
                r_idx,
                np.arange(n_q),
            )
            all_sum.extend(s); all_gain.extend(g); all_pheno.extend(p)

    # 2. True Cross-Donor Triomic Transfer (HMN200910 X066 <-> Independent Donor X061_W1)
    print("=== [2/3] True Cross-Donor Triomic Transfer (X066 <-> X061_W1) ===", flush=True)
    for pair_lbl, src_w, dst_w in [("X066_W3_to_X061_W1", "X066_W3", "X061_W1"), ("X061_W1_to_X066_W3", "X061_W1", "X066_W3")]:
        n_r = len(wells[src_w]["y45"])  # type: ignore[arg-type]
        n_q = len(wells[dst_w]["y45"])  # type: ignore[arg-type]
        for b in [100, 250, 500, 1500, "full"]:
            r_idx = np.arange(n_r) if b == "full" else np.sort(np.random.default_rng(800 + int(b)).choice(n_r, size=int(b), replace=False))
            save_u = f"teaseq_umap_{pair_lbl}_{b}" if (pair_lbl == "X066_W3_to_X061_W1" and b == "full") else None
            s, g, p = evaluate_triomic_pair(
                "True_Cross_Donor_Triomic",
                pair_lbl,
                str(b),
                wells[src_w],
                wells[dst_w],
                r_idx,
                np.arange(n_q),
                save_umap_key=save_u,
            )
            all_sum.extend(s); all_gain.extend(g); all_pheno.extend(p)

    # 3. Benchmark on X066_W3 exact split (including archived official scVAEIT v1.1.1 predictions for head-to-head comparison)
    print("=== [3/3] X066_W3 Benchmark Split + Official scVAEIT v1.1.1 Comparison ===", flush=True)
    with np.load(TEASEQ_REFERENCE_DIR / "target_and_split.npz") as c2d:
        c2_tr, c2_te = c2d["train"], c2d["test"]
    s, g, p = evaluate_triomic_pair(
        "Within_Well_scVAEIT_Reference",
        "X066_W3_50_50",
        "full_3534",
        wells["X066_W3"],
        wells["X066_W3"],
        c2_tr,
        c2_te,
        save_umap_key="teaseq_umap_X066_W3_scvaeit_split",
    )
    # Also score archived official scVAEIT v1.1.1 predictions on the identical 45 non-isotype antibodies!
    scv_pred_files = list((TEASEQ_REFERENCE_DIR / "scvaeit").rglob("*.npz")) + list(TEASEQ_REFERENCE_DIR.glob("*scvaeit*.npz"))
    for spf in scv_pred_files:
        try:
            with np.load(spf) as sd:
                for key in sd.files:
                    arr = sd[key]
                    if arr.ndim == 2 and arr.shape == (len(c2_te), 46):
                        with np.load(TEASEQ_REFERENCE_DIR / "target_and_split.npz") as c2d2:
                            p46 = [str(x) for x in c2d2["protein_names"]]
                        keep45 = np.array(["Isotype" not in x for x in p46], dtype=bool)
                        yp_scv = arr[:, keep45].astype(np.float32)
                        yt_w3 = wells["X066_W3"]["y45"][c2_te]  # type: ignore[index]
                        met_scv = compute_prediction_metrics(yt_w3, yp_scv)
                        gs_w3 = fit_pbmc_9lineage_gates(wells["X066_W3"]["y45"][c2_tr], wells["X066_W3"]["names45"])  # type: ignore[index,arg-type]
                        ph_scv = score_9lineage_predictions(yt_w3, yp_scv, gs_w3)
                        s.append({
                            "transfer_tier": "Within_Well_scVAEIT_Reference",
                            "pair": "X066_W3_50_50",
                            "budget": "full_3534",
                            "n_ref": int(len(c2_tr)),
                            "n_query": int(len(c2_te)),
                            "method": "scVAEIT_RNA+ATAC",
                            "clean45_pcc_median": round(met_scv["protein_pcc_median"], 4),
                            "clean45_pcc_mean": round(met_scv["protein_pcc_mean"], 4),
                            "clean45_cell_pcc_median": round(met_scv["cell_pcc_median"], 4),
                            "clean45_mse": round(met_scv["mse"], 4),
                            "pheno9_p1_acc_pct": ph_scv["p1_accuracy_pct"],
                            "pheno9_p1_macro_f1": ph_scv["p1_macro_f1_8lin"],
                            "pheno9_p2_acc_pct": ph_scv["p2_accuracy_pct"],
                            "pheno9_p2_macro_f1": ph_scv["p2_macro_f1_8lin"],
                        })
                        pheno_rows_scv = {
                            "transfer_tier": "Within_Well_scVAEIT_Reference",
                            "pair": "X066_W3_50_50",
                            "budget": "full_3534",
                            "method": "scVAEIT_RNA+ATAC",
                            **{k: v for k, v in ph_scv.items() if not k.endswith("_labels")},
                        }
                        p.append(pheno_rows_scv)
                        break
        except Exception:
            continue

    all_sum.extend(s); all_gain.extend(g); all_pheno.extend(p)

    df_sum = pd.DataFrame(all_sum)
    df_gain = pd.DataFrame(all_gain)
    df_pheno = pd.DataFrame(all_pheno)
    df_sum.to_csv(TABLES_DIR / "teaseq_triomic_transfer_summary.csv", index=False)
    df_gain.to_csv(TABLES_DIR / "teaseq_per_antibody_chromatin_gain.csv", index=False)
    df_pheno.to_csv(TABLES_DIR / "teaseq_9lineage_immunophenotyping.csv", index=False)

    # Wilcoxon stats on True Cross-Donor full transfer
    cd_sub = df_gain[(df_gain["pair"] == "X066_W3_to_X061_W1") & (df_gain["budget"] == "full")]
    n_improved = int((cd_sub["delta_atac_tolt"] > 0).sum())
    _, wilcox_p = wilcoxon(cd_sub["tolt_rna_atac_pcc"], cd_sub["tolt_rna_pcc"], alternative="greater")

    receipt = {
        "schema": "tolt.v4.teaseq_triomic_receipt.v1",
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "summary_rows": len(df_sum),
        "cross_donor_X066W3_to_X061W1_atac_improved_channels": f"{n_improved}/45",
        "cross_donor_atac_wilcoxon_p": float(wilcox_p),
    }
    (RECEIPTS_DIR / "06_teaseq_triomic_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"DONE Multi-Well & Cross-Donor TEA-seq in {receipt['elapsed_sec']}s! ATAC improved {n_improved}/45 (p={wilcox_p:.2e})", flush=True)


if __name__ == "__main__":
    main()
