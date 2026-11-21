"""Run comprehensive Cross-Donor & Cross-Site Benchmarks + Modern Deep Baselines (totalVI, sciPENN, scLinear) + Compute Pareto Frontier.

Covers:
  1. GSE194122 Multi-Site Bone Marrow (12710 Site 2 <-> 13272 Site 4, 134 ADTs; budgets 100, 250, 500, 1000, 2500, Full)
  2. GSE128639 BMMC Stuart (Donor A <-> Donor B, 25 ADTs; budgets 100, 250, 500, 1000, 2500, Full)
  3. GSE166489 5-Donor 188-Antibody CITE-seq Leave-One-Donor-Out (A.HD1, A.HD2, A.HD3, P1.1, P2.1)
  4. GSE166489 Healthy (A.HD1-3) + Few-Shot MIS-C Reference (0, 100, 250, 500 cells) -> Unseen Severe MIS-C Patient (P1.1 & P2.1)

Outputs:
  - results/tables/benchmark_cross_donor_learning_curves.csv
  - results/tables/benchmark_gse166489_lodo_and_misc_transfer.csv
  - results/tables/compute_pareto_frontier.csv
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
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from triomics.v4.paths import EMB_DIR, PRED_DIR

from triomics.v4.cohort_loaders import D_BMMC90K, D_STUART, log1p_cp10k_sparse, seurat_clr
from triomics.v4.core_methods import (
    compute_prediction_metrics,
    fit_local_svd_features,
    fit_predict_tolt_adapter_suite,
    run_fast_classical_and_modern_baselines,
    run_scipenn_gpu,
    run_totalvi_gpu,
)

TABLES_DIR = REPO_ROOT / "results" / "tables"
RECEIPTS_DIR = REPO_ROOT / "results" / "receipts"


def evaluate_all_methods_on_split(
    cohort_name: str,
    split_name: str,
    budget_label: str,
    x_raw_ref: sp.csr_matrix,
    adt_raw_ref: np.ndarray,
    x_raw_qry: sp.csr_matrix,
    y_ref: np.ndarray,
    y_qry: np.ndarray,
    emb_ref: np.ndarray,
    emb_qry: np.ndarray,
    gene_names: list[str],
    protein_names: list[str],
    run_deep_gpu: bool = True,
    expressed_mask: np.ndarray | None = None,
    seed: int = 42,
    save_predictions_key: str | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    pareto_rows: list[dict[str, object]] = []

    x_ref_log = log1p_cp10k_sparse(x_raw_ref)
    x_qry_log = log1p_cp10k_sparse(x_raw_qry)

    t_svd_prep0 = time.perf_counter()
    svd_ref, svd_qry = fit_local_svd_features(x_ref_log, x_qry_log, seed=seed)
    svd_prep_sec = time.perf_counter() - t_svd_prep0

    tolt_res = fit_predict_tolt_adapter_suite(
        svd_ref, svd_qry, emb_ref, emb_qry, y_ref, seed=seed, nonneg_clip=True
    )
    fast_res = run_fast_classical_and_modern_baselines(
        x_ref_log, x_qry_log, svd_ref, svd_qry, y_ref, seed=seed, nonneg_clip=True
    )

    combined: dict[str, dict[str, object]] = {}
    for k, v in tolt_res.items():
        add_prep = svd_prep_sec if k != "TOLT-Direct" else 0.0
        combined[k] = {"pred": v["pred"], "oof": v.get("oof"), "sec": round(float(v["sec"]) + add_prep, 3), "peak_vram_mb": 0.0}
    for k, v in fast_res.items():
        combined[k] = {"pred": v["pred"], "sec": round(float(v["sec"]) + svd_prep_sec, 3), "peak_vram_mb": 0.0}

    if run_deep_gpu:
        try:
            tvi = run_totalvi_gpu(
                x_raw_ref,
                adt_raw_ref,
                x_raw_qry,
                target_transform_fn=seurat_clr,
                max_epochs=180 if x_raw_ref.shape[0] <= 8000 else 120,
                seed=seed,
            )
            combined["totalVI"] = tvi
        except Exception as exc:
            print(f"  [WARN] totalVI failed on {split_name} ({budget_label}): {exc}", flush=True)

        try:
            spn = run_scipenn_gpu(
                x_raw_ref,
                adt_raw_ref,
                x_raw_qry,
                gene_names=gene_names,
                protein_names=protein_names,
                y_train_target=y_ref,
                seed=seed,
            )
            combined["sciPENN"] = spn
        except Exception as exc:
            print(f"  [WARN] sciPENN failed on {split_name} ({budget_label}): {exc}", flush=True)

    if save_predictions_key is not None:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        save_p = {m: d["pred"] for m, d in combined.items() if "pred" in d}
        save_p["y_true"] = y_qry
        save_p["y_train"] = y_ref
        if "TOLT" in tolt_res and "weights" in tolt_res["TOLT"]:
            save_p["tolt_simplex_weights"] = tolt_res["TOLT"]["weights"]
        np.savez_compressed(PRED_DIR / f"{save_predictions_key}.npz", **save_p)

    for method_name, payload in combined.items():
        pred = payload["pred"]
        m_all = compute_prediction_metrics(y_qry, pred)
        rec: dict[str, object] = {
            "cohort": cohort_name,
            "split": split_name,
            "budget": budget_label,
            "n_ref": int(x_raw_ref.shape[0]),
            "n_query": int(x_raw_qry.shape[0]),
            "n_proteins": int(y_qry.shape[1]),
            "method": method_name,
            "protein_pcc_median": round(m_all["protein_pcc_median"], 4),
            "protein_pcc_mean": round(m_all["protein_pcc_mean"], 4),
            "protein_spearman_median": round(m_all["protein_spearman_median"], 4),
            "cell_pcc_median": round(m_all["cell_pcc_median"], 4),
            "protein_rmse_median": round(m_all["protein_rmse_median"], 4),
            "mse": round(m_all["mse"], 4),
            "fit_predict_sec": payload["sec"],
            "peak_vram_mb": payload["peak_vram_mb"],
        }
        if expressed_mask is not None and expressed_mask.any():
            m_exp = compute_prediction_metrics(y_qry[:, expressed_mask], pred[:, expressed_mask])
            rec["expressed_protein_pcc_median"] = round(m_exp["protein_pcc_median"], 4)
            rec["expressed_protein_pcc_mean"] = round(m_exp["protein_pcc_mean"], 4)
        rows.append(rec)
        pareto_rows.append(rec)

    return rows, pareto_rows


def main(*, fast: bool = False) -> None:
    t_start = time.perf_counter()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    curve_rows: list[dict[str, object]] = []
    misc_rows: list[dict[str, object]] = []
    pareto_rows: list[dict[str, object]] = []

    # =========================================================================
    # 1. GSE194122 Multi-Site Bone Marrow (12710 Site 2 <-> 13272 Site 4, 134 ADTs)
    # =========================================================================
    print("=== [1/3] GSE194122 Unseen Cross-Donor + Cross-Site Learning Curves ===", flush=True)
    ad_90k = ad.read_h5ad(D_BMMC90K)
    unseen_mask = ad_90k.obs["DonorID"].astype(str).isin(["12710", "13272"]).to_numpy()
    sub_90k = ad_90k[unseen_mask].copy()
    x_90k = sp.csr_matrix(sub_90k.X, dtype=np.float32)
    adt_90k = np.asarray(sub_90k.obsm["protein_expression"], dtype=np.float32)
    # Remove the 6 mouse/rat isotype controls in GSE194122 if named isotype, or keep clean human markers
    pnames_90k = [f"ADT_{i}" for i in range(adt_90k.shape[1])]
    if "protein_names" in sub_90k.uns:
        pnames_90k = [str(x) for x in np.asarray(sub_90k.uns["protein_names"]).tolist()]
    y_90k = seurat_clr(adt_90k)
    gnames_90k = [str(x) for x in sub_90k.var_names]
    with np.load(EMB_DIR / "gse194122_unseen_12949_embeddings.npz", allow_pickle=True) as d:
        dids = d["donor_ids"]
        emb_90k = d["base_110"]
    i12 = np.flatnonzero(dids == "12710")
    i13 = np.flatnonzero(dids == "13272")
    del ad_90k, sub_90k

    for dir_name, ref_pool, qry_idx in [
        ("12710_Site2_to_13272_Site4", i12, i13),
        ("13272_Site4_to_12710_Site2", i13, i12),
    ]:
        exp_m = (y_90k[ref_pool].std(axis=0) >= 0.35)
        for b in [100, 250, 500, 1000, 2500, "full"]:
            rng = np.random.default_rng(1001 + (0 if b == "full" else int(b)))
            if b == "full" or int(b) >= len(ref_pool):
                r_idx = ref_pool
                blab = "full"
            else:
                r_idx = np.sort(rng.choice(ref_pool, size=int(b), replace=False))
                blab = str(b)
            # Run GPU totalVI & sciPENN at representative key budgets (250, 500, 1000, full) on primary direction & (250, full) on reverse
            do_deep = (b in {250, 500, 1000, "full"}) if dir_name.startswith("12710") else (b in {250, 500, "full"})
            save_k = f"gse194122_{dir_name}_{blab}" if blab in {"250", "500", "full"} else None
            r, p = evaluate_all_methods_on_split(
                "GSE194122_BMMC_CrossSite",
                dir_name,
                blab,
                x_90k[r_idx],
                adt_90k[r_idx],
                x_90k[qry_idx],
                y_90k[r_idx],
                y_90k[qry_idx],
                emb_90k[r_idx],
                emb_90k[qry_idx],
                gnames_90k,
                pnames_90k,
                run_deep_gpu=do_deep and not fast,
                expressed_mask=exp_m,
                save_predictions_key=save_k,
            )
            curve_rows.extend(r)
            if do_deep:
                pareto_rows.extend(p)
            top_m = max(r, key=lambda d: float(d["protein_pcc_median"]))
            print(f"  [{dir_name} | budget={blab:4s}] winner={top_m['method']} ({top_m['protein_pcc_median']:.4f})", flush=True)

    # =========================================================================
    # 2. GSE128639 BMMC Stuart (Donor A <-> Donor B, 25 ADTs)
    # =========================================================================
    print("=== [2/3] GSE128639 BMMC Stuart Cross-Donor Learning Curves ===", flush=True)
    ad_stu = ad.read_h5ad(D_STUART)
    x_stu = sp.csr_matrix(ad_stu.X, dtype=np.float32)
    prot_stu = ad_stu.obsm["protein_expression"]
    prot_stu = prot_stu.toarray() if sp.issparse(prot_stu) else np.asarray(prot_stu, dtype=np.float32)
    y_stu = seurat_clr(prot_stu)
    gnames_stu = [str(x) for x in ad_stu.var_names]
    pnames_stu = [str(x) for x in np.asarray(ad_stu.uns["protein_names"]).tolist()] if "protein_names" in ad_stu.uns else [f"ADT_{i}" for i in range(prot_stu.shape[1])]
    with np.load(EMB_DIR / "gse128639_stuart_33454_embeddings.npz", allow_pickle=True) as d:
        donors_stu = d["donors"]
        # Note: F01 and Base-110 both evaluated; use Base-110 for uniform protocol + log F01 in ablation
        emb_stu = d["base_110"]
    da = np.flatnonzero(donors_stu == "Donor_A")
    db = np.flatnonzero(donors_stu == "Donor_B")
    del ad_stu

    for dir_name, ref_pool, qry_idx in [("DonorA_to_DonorB", da, db), ("DonorB_to_DonorA", db, da)]:
        for b in [100, 250, 500, 1000, 2500, "full"]:
            rng = np.random.default_rng(2002 + (0 if b == "full" else int(b)))
            if b == "full" or int(b) >= len(ref_pool):
                r_idx = ref_pool
                blab = "full"
            else:
                r_idx = np.sort(rng.choice(ref_pool, size=int(b), replace=False))
                blab = str(b)
            do_deep = (b in {250, 500, 1000, "full"}) if dir_name == "DonorA_to_DonorB" else (b in {250, "full"})
            r, p = evaluate_all_methods_on_split(
                "GSE128639_BMMC_Stuart",
                dir_name,
                blab,
                x_stu[r_idx],
                prot_stu[r_idx],
                x_stu[qry_idx],
                y_stu[r_idx],
                y_stu[qry_idx],
                emb_stu[r_idx],
                emb_stu[qry_idx],
                gnames_stu,
                pnames_stu,
                run_deep_gpu=do_deep and not fast,
            )
            curve_rows.extend(r)
            if do_deep:
                pareto_rows.extend(p)
            top_m = max(r, key=lambda d: float(d["protein_pcc_median"]))
            print(f"  [{dir_name} | budget={blab:4s}] winner={top_m['method']} ({top_m['protein_pcc_median']:.4f})", flush=True)

    # =========================================================================
    # 3. GSE166489 5-Donor 188-Antibody CITE-seq LODO & Healthy->MIS-C Transfer
    # =========================================================================
    print("=== [3/3] GSE166489 5-Donor 188-AB LODO & Healthy->MIS-C Few-Shot ===", flush=True)
    cite_ids = ["A.HD1", "A.HD2", "A.HD3", "P1.1", "P2.1"]
    gschema = np.load(EMB_DIR / "gse166489_gene_schema.npz", allow_pickle=True)
    gnames_misc = [str(x) for x in gschema["gene_names"]]

    cite_data: dict[str, dict[str, object]] = {}
    for sid in cite_ids:
        xr = sp.load_npz(EMB_DIR / f"gse166489_{sid}_rna_counts.npz")
        bd = np.load(EMB_DIR / f"gse166489_{sid}_bundle.npz", allow_pickle=True)
        adt_c = bd["adt_counts"].astype(np.float32)
        cite_data[sid] = {
            "x_raw": xr,
            "adt_raw": adt_c,
            "y_clr": seurat_clr(adt_c),
            "emb": bd["base_110"],
            "adt_names": [str(x) for x in bd["adt_names"]],
        }
    pnames_188 = cite_data["A.HD3"]["adt_names"]  # type: ignore[assignment]

    # 3A. 5-Donor Leave-One-Donor-Out (LODO) at n_ref=500 (few-shot) and n_ref=6000 (balanced multi-donor reference)
    for held_out in cite_ids:
        train_donors = [d for d in cite_ids if d != held_out]
        x_tr_all = sp.vstack([cite_data[d]["x_raw"] for d in train_donors], format="csr")
        adt_tr_all = np.vstack([cite_data[d]["adt_raw"] for d in train_donors])
        y_tr_all = np.vstack([cite_data[d]["y_clr"] for d in train_donors])
        emb_tr_all = np.vstack([cite_data[d]["emb"] for d in train_donors])

        exp_mask = (y_tr_all.std(axis=0) >= 0.35) | (np.percentile(y_tr_all, 99, axis=0) >= 1.50)

        for b_lodo in [250, 500, 6000]:
            rng = np.random.default_rng(3003 + b_lodo)
            sub_i = np.sort(rng.choice(len(y_tr_all), size=min(b_lodo, len(y_tr_all)), replace=False))
            save_k = f"gse166489_lodo_{held_out}_ref{b_lodo}" if b_lodo in {500, 6000} else None
            r, p = evaluate_all_methods_on_split(
                "GSE166489_188AB_LODO",
                f"LODO_Holdout_{held_out}",
                str(b_lodo),
                x_tr_all[sub_i],
                adt_tr_all[sub_i],
                cite_data[held_out]["x_raw"],  # type: ignore[arg-type]
                y_tr_all[sub_i],
                cite_data[held_out]["y_clr"],  # type: ignore[arg-type]
                emb_tr_all[sub_i],
                cite_data[held_out]["emb"],    # type: ignore[arg-type]
                gnames_misc,
                pnames_188,
                run_deep_gpu=(not fast and b_lodo in {500, 6000}),
                expressed_mask=exp_mask,
                save_predictions_key=save_k,
            )
            misc_rows.extend(r)
            if b_lodo in {500, 6000}:
                pareto_rows.extend(p)
            top_m = max(r, key=lambda d: float(d.get("expressed_protein_pcc_median", d["protein_pcc_median"])))
            print(
                f"  [LODO -> {held_out:5s} | ref={b_lodo:4d}] top_expressed={top_m['method']} "
                f"({top_m.get('expressed_protein_pcc_median', 0.0):.4f})",
                flush=True,
            )

    # 3B. Healthy (A.HD1-3) + k MIS-C calibration cells from OTHER MIS-C donor -> Unseen Severe MIS-C Donor
    hd_ids = ["A.HD1", "A.HD2", "A.HD3"]
    x_hd_pool = sp.vstack([cite_data[d]["x_raw"] for d in hd_ids], format="csr")
    adt_hd_pool = np.vstack([cite_data[d]["adt_raw"] for d in hd_ids])
    y_hd_pool = np.vstack([cite_data[d]["y_clr"] for d in hd_ids])
    emb_hd_pool = np.vstack([cite_data[d]["emb"] for d in hd_ids])
    rng_hd = np.random.default_rng(4040)
    hd_3k = np.sort(rng_hd.choice(len(y_hd_pool), size=min(4500, len(y_hd_pool)), replace=False))

    for target_misc, helper_misc in [("P1.1", "P2.1"), ("P2.1", "P1.1")]:
        for k_misc in [0, 100, 250, 500]:
            if k_misc == 0:
                x_r = x_hd_pool[hd_3k]
                adt_r = adt_hd_pool[hd_3k]
                y_r = y_hd_pool[hd_3k]
                e_r = emb_hd_pool[hd_3k]
            else:
                h_n = len(cite_data[helper_misc]["y_clr"])  # type: ignore[arg-type]
                h_i = np.sort(np.random.default_rng(5000 + k_misc).choice(h_n, size=min(k_misc, h_n), replace=False))
                x_r = sp.vstack([x_hd_pool[hd_3k], cite_data[helper_misc]["x_raw"][h_i]], format="csr")  # type: ignore[index]
                adt_r = np.vstack([adt_hd_pool[hd_3k], cite_data[helper_misc]["adt_raw"][h_i]])          # type: ignore[index]
                y_r = np.vstack([y_hd_pool[hd_3k], cite_data[helper_misc]["y_clr"][h_i]])                # type: ignore[index]
                e_r = np.vstack([emb_hd_pool[hd_3k], cite_data[helper_misc]["emb"][h_i]])                # type: ignore[index]

            exp_mask = (y_r.std(axis=0) >= 0.35) | (np.percentile(y_r, 99, axis=0) >= 1.50)
            save_k = f"gse166489_hd_plus_{k_misc}misc_to_{target_misc}" if k_misc in {0, 250} else None
            r, _ = evaluate_all_methods_on_split(
                "GSE166489_Healthy_to_MISC",
                f"HD_plus_{helper_misc}_to_{target_misc}",
                f"misc_SpikeIn_{k_misc}",
                x_r,
                adt_r,
                cite_data[target_misc]["x_raw"],  # type: ignore[arg-type]
                y_r,
                cite_data[target_misc]["y_clr"],  # type: ignore[arg-type]
                e_r,
                cite_data[target_misc]["emb"],    # type: ignore[arg-type]
                gnames_misc,
                pnames_188,
                run_deep_gpu=(not fast and k_misc in {0, 250}),
                expressed_mask=exp_mask,
                save_predictions_key=save_k,
            )
            misc_rows.extend(r)

    pd.DataFrame(curve_rows).to_csv(TABLES_DIR / "benchmark_cross_donor_learning_curves.csv", index=False)
    pd.DataFrame(misc_rows).to_csv(TABLES_DIR / "benchmark_gse166489_lodo_and_misc_transfer.csv", index=False)
    pd.DataFrame(pareto_rows).to_csv(TABLES_DIR / "compute_pareto_frontier.csv", index=False)

    receipt = {
        "schema": "tolt.v4.cross_donor_benchmarks_receipt.v1",
        "elapsed_sec": round(time.perf_counter() - t_start, 2),
        "deep_gpu_baselines": not fast,
        "learning_curve_rows": len(curve_rows),
        "gse166489_rows": len(misc_rows),
        "pareto_rows": len(pareto_rows),
    }
    (RECEIPTS_DIR / "04_cross_donor_benchmarks_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"DONE Cross-Donor Benchmarks in {receipt['elapsed_sec']}s!", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="Run the classical and TOLT models without training deep GPU baselines.")
    args = parser.parse_args()
    if args.fast:
        output = REPO_ROOT / "results" / "offline_fast"
        TABLES_DIR = output / "tables"
        RECEIPTS_DIR = output / "receipts"
        PRED_DIR = output / "predictions"
    main(fast=args.fast)
