"""Extract and cache GPU BF16 embeddings across all cohorts and backbone variants on RTX 5070 Laptop GPU.

Covers:
1. GSE194122 Multi-Site Bone Marrow unseen donors (12710 Site 2 & 13272 Site 4; 12,949 cells, 134 ADTs)
2. GSE128639 BMMC Stuart two-donor cohort (Donor A 15,437 & Donor B 18,017; 33,454 cells, 25 ADTs)
3. GSE166489 MIS-C Clinical Cohort (all 18 donors/timepoints: 5 paired CITE-seq + 13 pediatric RNA-only)
4. GSE158013 Multi-Well & Cross-Donor TEA-seq (all 5 wells: X066 W3-W6 + independent donor X061 W1; RNA & RNA+ATAC)
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

from triomics.v4.paths import EMB_DIR

from triomics.v4.cohort_loaders import (
    D_BMMC90K,
    D_STUART,
    GSE166489_METADATA,
    TEASEQ_WELLS,
    load_gse166489_sample,
    load_teaseq_well,
)
from triomics.v4.core_methods import (
    forward_tolt_embeddings,
    instantiate_tolt_backbone,
    tokenize_atac_matrix,
    tokenize_rna_matrix,
)

TABLES_DIR = REPO_ROOT / "results" / "tables"
RECEIPTS_DIR = REPO_ROOT / "results" / "receipts"


def main() -> None:
    t_global = time.perf_counter()
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"Running TOLT v4 embedding extraction on device={device} ({gpu_name})", flush=True)

    # Instantiate all backbone models once
    models = {
        "base_110": instantiate_tolt_backbone("base_110", seed=29, device=device),
        "f01_s29": instantiate_tolt_backbone("f01", seed=29, device=device),
        "rand_s11": instantiate_tolt_backbone("random_init", seed=11, device=device),
        "rand_s29": instantiate_tolt_backbone("random_init", seed=29, device=device),
        "rand_s47": instantiate_tolt_backbone("random_init", seed=47, device=device),
    }

    throughput_rows: list[dict[str, object]] = []

    # =========================================================================
    # 1. GSE194122 Unseen Cross-Donor + Cross-Site Bone Marrow (12710 & 13272)
    # =========================================================================
    print("=== [1/4] Processing GSE194122 unseen donors (12710 & 13272) ===", flush=True)
    ad_90k = ad.read_h5ad(D_BMMC90K)
    unseen_mask = ad_90k.obs["DonorID"].astype(str).isin(["12710", "13272"]).to_numpy()
    sub_90k = ad_90k[unseen_mask].copy()
    x_90k = sp.csr_matrix(sub_90k.X, dtype=np.float32)
    g_names_90k = [str(x) for x in sub_90k.var_names]
    g_ids_90k = [str(x) for x in sub_90k.var["gene_ids"]] if "gene_ids" in sub_90k.var.columns else None

    t0 = time.perf_counter()
    tok_90k, aud_90k = tokenize_rna_matrix(x_90k, g_ids_90k, g_names_90k)
    tok_sec_90k = round(time.perf_counter() - t0, 4)

    emb_90k_payload: dict[str, np.ndarray] = {
        "cell_ids": np.asarray([str(x) for x in sub_90k.obs_names], dtype=str),
        "donor_ids": np.asarray([str(x) for x in sub_90k.obs["DonorID"]], dtype=str),
        "sites": np.asarray([str(x) for x in sub_90k.obs["Site"]], dtype=str),
        "cell_types": np.asarray([str(x) for x in sub_90k.obs["cell_type"]], dtype=str),
    }
    for mname, mod in models.items():
        emb, st = forward_tolt_embeddings(mod, tok_90k, use_atac=False, batch_size=512, device=device)
        emb_90k_payload[mname] = emb
        if mname == "base_110":
            throughput_rows.append({
                "cohort": "GSE194122_Unseen_BMMC",
                "modality": "RNA",
                "n_cells": int(x_90k.shape[0]),
                "mapped_features": aud_90k["mapped_genes"],
                "tokenize_sec": tok_sec_90k,
                "gpu_inference_sec": st["inference_sec"],
                "peak_vram_mb": st["peak_vram_mb"],
                "cells_per_sec": round(x_90k.shape[0] / max(st["inference_sec"], 1e-4), 1),
            })
    np.savez_compressed(EMB_DIR / "gse194122_unseen_12949_embeddings.npz", **emb_90k_payload)
    del ad_90k, sub_90k

    # =========================================================================
    # 2. GSE128639 BMMC Stuart (33,454 cells, Donor A & Donor B)
    # =========================================================================
    print("=== [2/4] Processing GSE128639 BMMC Stuart (33,454 cells) ===", flush=True)
    ad_stu = ad.read_h5ad(D_STUART)
    x_stu = sp.csr_matrix(ad_stu.X, dtype=np.float32)
    stu_cells = [str(x) for x in ad_stu.obs_names]
    stu_donors = np.array(["Donor_A" if c.startswith("a_") else "Donor_B" for c in stu_cells], dtype=str)
    g_names_stu = [str(x) for x in ad_stu.var_names]

    t0 = time.perf_counter()
    tok_stu, aud_stu = tokenize_rna_matrix(x_stu, None, g_names_stu)
    tok_sec_stu = round(time.perf_counter() - t0, 4)

    emb_stu_payload: dict[str, np.ndarray] = {
        "cell_ids": np.asarray(stu_cells, dtype=str),
        "donors": stu_donors,
    }
    for mname, mod in models.items():
        emb, st = forward_tolt_embeddings(mod, tok_stu, use_atac=False, batch_size=512, device=device)
        emb_stu_payload[mname] = emb
        if mname == "base_110":
            throughput_rows.append({
                "cohort": "GSE128639_BMMC_Stuart",
                "modality": "RNA",
                "n_cells": int(x_stu.shape[0]),
                "mapped_features": aud_stu["mapped_genes"],
                "tokenize_sec": tok_sec_stu,
                "gpu_inference_sec": st["inference_sec"],
                "peak_vram_mb": st["peak_vram_mb"],
                "cells_per_sec": round(x_stu.shape[0] / max(st["inference_sec"], 1e-4), 1),
            })
    np.savez_compressed(EMB_DIR / "gse128639_stuart_33454_embeddings.npz", **emb_stu_payload)
    del ad_stu

    # =========================================================================
    # 3. GSE166489 MIS-C Clinical Cohort (All 18 Samples: 5 CITE-seq + 13 Pediatric GEX)
    # =========================================================================
    print("=== [3/4] Processing GSE166489 MIS-C Cohort (18 samples) ===", flush=True)
    misc_summary_rows: list[dict[str, object]] = []
    total_misc_cells = 0
    total_misc_inf_sec = 0.0
    max_misc_vram = 0.0

    for meta in GSE166489_METADATA:
        samp = load_gse166489_sample(meta)
        t0 = time.perf_counter()
        tok_s, aud_s = tokenize_rna_matrix(samp.rna_counts, samp.gene_ids, samp.gene_names)
        tok_s_sec = round(time.perf_counter() - t0, 4)

        save_dict: dict[str, object] = {
            "barcodes": samp.barcodes,
            "condition": np.array(samp.condition),
            "age_group": np.array(samp.age_group),
        }
        variants_to_run = list(models.keys()) if meta["has_adt"] == "true" else ["base_110", "f01_s29"]
        inf_b110 = 0.0
        for mname in variants_to_run:
            emb, st = forward_tolt_embeddings(models[mname], tok_s, use_atac=False, batch_size=512, device=device)
            save_dict[mname] = emb
            if mname == "base_110":
                inf_b110 = st["inference_sec"]
                total_misc_cells += samp.rna_counts.shape[0]
                total_misc_inf_sec += inf_b110
                max_misc_vram = max(max_misc_vram, st["peak_vram_mb"])

        if samp.adt_counts is not None and samp.adt_names is not None:
            save_dict["adt_counts"] = samp.adt_counts
            save_dict["adt_names"] = np.asarray(samp.adt_names, dtype=str)
            save_dict["isotype_counts"] = samp.isotype_counts
            save_dict["isotype_names"] = np.asarray(samp.isotype_names, dtype=str)

        n_tcr = 0
        n_trbv11_2 = 0
        if samp.tcr_df is not None:
            bc_to_trbv = dict(zip(samp.tcr_df["barcode"], samp.tcr_df["is_trbv11_2"]))
            has_tcr = np.array([b in bc_to_trbv for b in samp.barcodes], dtype=bool)
            is_v11 = np.array([bool(bc_to_trbv.get(b, False)) for b in samp.barcodes], dtype=bool)
            save_dict["has_tcr"] = has_tcr
            save_dict["is_trbv11_2"] = is_v11
            n_tcr = int(has_tcr.sum())
            n_trbv11_2 = int(is_v11.sum())

        sp.save_npz(EMB_DIR / f"gse166489_{samp.sample_id}_rna_counts.npz", samp.rna_counts)
        if meta["sample_id"] == "A.HD3":
            np.savez_compressed(
                EMB_DIR / "gse166489_gene_schema.npz",
                gene_ids=samp.gene_ids,
                gene_names=samp.gene_names,
            )
        np.savez_compressed(EMB_DIR / f"gse166489_{samp.sample_id}_bundle.npz", **save_dict)

        misc_summary_rows.append({
            "sample_id": samp.sample_id,
            "gsm": meta["gsm"],
            "condition": samp.condition,
            "age_group": samp.age_group,
            "patient": meta["patient"],
            "timepoint": meta["timepoint"],
            "has_paired_adt": meta["has_adt"] == "true",
            "qc_cells": int(samp.rna_counts.shape[0]),
            "mapped_tolt_genes": int(aud_s["mapped_genes"]),
            "tcr_matched_cells": n_tcr,
            "trbv11_2_cells": n_trbv11_2,
            "trbv11_2_pct_of_tcr": round(100.0 * n_trbv11_2 / max(n_tcr, 1), 2),
            "tokenize_sec": tok_s_sec,
            "gpu_base110_sec": inf_b110,
        })
        print(
            f"  [GSE166489 {samp.sample_id:5s}] {samp.condition:17s} | cells={samp.rna_counts.shape[0]:5d} "
            f"| TCR={n_tcr:4d} (TRBV11-2={n_trbv11_2:3d}) | GPU={inf_b110:.3f}s",
            flush=True,
        )

    pd.DataFrame(misc_summary_rows).to_csv(TABLES_DIR / "cohort_gse166489_sample_summary.csv", index=False)
    throughput_rows.append({
        "cohort": "GSE166489_All18_Samples",
        "modality": "RNA",
        "n_cells": total_misc_cells,
        "mapped_features": int(misc_summary_rows[0]["mapped_tolt_genes"]),
        "tokenize_sec": round(sum(float(r["tokenize_sec"]) for r in misc_summary_rows), 3),
        "gpu_inference_sec": round(total_misc_inf_sec, 3),
        "peak_vram_mb": round(max_misc_vram, 2),
        "cells_per_sec": round(total_misc_cells / max(total_misc_inf_sec, 1e-4), 1),
    })

    # =========================================================================
    # 4. GSE158013 Multi-Well & Cross-Donor TEA-seq (All 5 Wells: RNA & RNA+ATAC)
    # =========================================================================
    print("=== [4/4] Processing GSE158013 TEA-seq (5 wells: RNA & RNA+ATAC) ===", flush=True)
    tea_summary_rows: list[dict[str, object]] = []
    total_tea_cells = 0
    total_tea_rna_sec = 0.0
    total_tea_dual_sec = 0.0
    max_tea_vram = 0.0

    for wmeta in TEASEQ_WELLS:
        well = load_teaseq_well(wmeta)
        t0 = time.perf_counter()
        tok_r, aud_r = tokenize_rna_matrix(well.rna_counts, well.gene_ids, well.gene_names)
        tok_a, aud_a = tokenize_atac_matrix(well.atac_counts, well.peak_names)
        tok_both = {**tok_r, **tok_a}
        tok_w_sec = round(time.perf_counter() - t0, 3)

        w_save: dict[str, object] = {
            "barcodes": well.barcodes,
            "donor_id": np.array(well.donor_id),
            "adt_counts_45": well.adt_counts_45,
            "adt_names_45": np.asarray(well.adt_names_45, dtype=str),
            "isotype_counts": well.isotype_counts,
            "adt_counts_all46": well.adt_counts_all46,
            "adt_names_all46": np.asarray(well.adt_names_all46, dtype=str),
        }

        variants_w = list(models.keys()) if well.well_id in {"X066_W3", "X061_W1"} else ["base_110", "f01_s29"]
        sec_r_b110, sec_d_b110 = 0.0, 0.0
        for mname in variants_w:
            emb_r, st_r = forward_tolt_embeddings(models[mname], tok_both, use_atac=False, batch_size=512, device=device)
            emb_d, st_d = forward_tolt_embeddings(models[mname], tok_both, use_atac=True, batch_size=192, device=device)
            w_save[f"{mname}_rna"] = emb_r
            w_save[f"{mname}_rna_atac"] = emb_d
            if mname == "base_110":
                sec_r_b110 = st_r["inference_sec"]
                sec_d_b110 = st_d["inference_sec"]
                total_tea_cells += well.rna_counts.shape[0]
                total_tea_rna_sec += sec_r_b110
                total_tea_dual_sec += sec_d_b110
                max_tea_vram = max(max_tea_vram, st_d["peak_vram_mb"])

        sp.save_npz(EMB_DIR / f"gse158013_{well.well_id}_rna_counts.npz", well.rna_counts)
        sp.save_npz(EMB_DIR / f"gse158013_{well.well_id}_atac_counts.npz", well.atac_counts)
        np.savez_compressed(EMB_DIR / f"gse158013_{well.well_id}_bundle.npz", **w_save)

        tea_summary_rows.append({
            "well_id": well.well_id,
            "gsm": wmeta["gsm"],
            "donor_id": well.donor_id,
            "batch_group": wmeta["batch_group"],
            "retained_cells": int(well.rna_counts.shape[0]),
            "rna_features": int(well.rna_counts.shape[1]),
            "source_atac_peaks": int(well.atac_counts.shape[1]),
            "mapped_tolt_genes": int(aud_r["mapped_genes"]),
            "covered_tolt_peaks": int(aud_a["frozen_features_covered"]),
            "clean_human_adts": len(well.adt_names_45),
            "tokenize_sec": tok_w_sec,
            "gpu_rna_sec": sec_r_b110,
            "gpu_rna_atac_sec": sec_d_b110,
        })
        print(
            f"  [TEA-seq {well.well_id}] donor={well.donor_id:10s} | cells={well.rna_counts.shape[0]:4d} "
            f"| peaks={well.atac_counts.shape[1]:6d} -> {aud_a['frozen_features_covered']}/30000 "
            f"| GPU RNA={sec_r_b110:.3f}s, RNA+ATAC={sec_d_b110:.3f}s",
            flush=True,
        )

    pd.DataFrame(tea_summary_rows).to_csv(TABLES_DIR / "cohort_gse158013_well_summary.csv", index=False)
    throughput_rows.append({
        "cohort": "GSE158013_5Wells_TEAseq",
        "modality": "RNA+ATAC",
        "n_cells": total_tea_cells,
        "mapped_features": int(tea_summary_rows[0]["covered_tolt_peaks"]),
        "tokenize_sec": round(sum(float(r["tokenize_sec"]) for r in tea_summary_rows), 3),
        "gpu_inference_sec": round(total_tea_dual_sec, 3),
        "peak_vram_mb": round(max_tea_vram, 2),
        "cells_per_sec": round(total_tea_cells / max(total_tea_dual_sec, 1e-4), 1),
    })

    pd.DataFrame(throughput_rows).to_csv(TABLES_DIR / "gpu_inference_throughput.csv", index=False)
    receipt = {
        "schema": "tolt.v4.embedding_extraction_receipt.v1",
        "device": device,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "total_elapsed_sec": round(time.perf_counter() - t_global, 2),
        "gse166489_total_cells": total_misc_cells,
        "gse158013_total_cells": total_tea_cells,
        "throughput_summary": throughput_rows,
    }
    (RECEIPTS_DIR / "02_embedding_extraction_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"COMPLETED ALL EMBEDDING EXTRACTIONS in {receipt['total_elapsed_sec']}s!", flush=True)


if __name__ == "__main__":
    main()
