"""Download and verify public cohorts (GSE166489 MIS-C + GSE158013 Multi-Well TEA-seq) to the configured TOLT asset directory.

All filenames have been verified directly against NCBI GEO HTTPS directories.
Downloads run in parallel via curl.exe with automatic resume/retry and archive header verification.
"""
from __future__ import annotations

import concurrent.futures
import gzip
import hashlib
import json
import subprocess
import tarfile
import time
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from triomics.v4.paths import COHORTS_DIR
CITE_DIR = COHORTS_DIR / "GSE166489_MISC" / "cite_seq"
PED_GEX_DIR = COHORTS_DIR / "GSE166489_MISC" / "pediatric_gex"
TCR_DIR = COHORTS_DIR / "GSE166489_MISC" / "tcr"
TEA_DIR = COHORTS_DIR / "GSE158013_TEAseq"
RECEIPT_PATH = REPO_ROOT / "results" / "receipts" / "01_download_receipt.json"

DOWNLOAD_MANIFEST: list[dict[str, str]] = [
    # --- GSE166489 Paired CITE-seq (GEX + 189 ADT): 5 donors total (A.HD3 already copied locally) ---
    {
        "cohort": "GSE166489_CITE",
        "sample_id": "P1.1",
        "condition": "MIS-C_Severe",
        "age_group": "Pediatric",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073055/suppl/GSM5073055_P1.1_filtered_feature_bc_matrix.tar.gz",
        "dest": str(CITE_DIR / "GSM5073055_P1.1_filtered_feature_bc_matrix.tar.gz"),
    },
    {
        "cohort": "GSE166489_CITE",
        "sample_id": "P2.1",
        "condition": "MIS-C_Severe",
        "age_group": "Pediatric",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073056/suppl/GSM5073056_P2.1_filtered_feature_bc_matrix.tar.gz",
        "dest": str(CITE_DIR / "GSM5073056_P2.1_filtered_feature_bc_matrix.tar.gz"),
    },
    {
        "cohort": "GSE166489_CITE",
        "sample_id": "A.HD1",
        "condition": "Healthy",
        "age_group": "Adult",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073070/suppl/GSM5073070_A.HD1_filtered_feature_bc_matrix.tar.gz",
        "dest": str(CITE_DIR / "GSM5073070_A.HD1_filtered_feature_bc_matrix.tar.gz"),
    },
    {
        "cohort": "GSE166489_CITE",
        "sample_id": "A.HD2",
        "condition": "Healthy",
        "age_group": "Adult",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073071/suppl/GSM5073071_A.HD2_filtered_feature_bc_matrix.tar.gz",
        "dest": str(CITE_DIR / "GSM5073071_A.HD2_filtered_feature_bc_matrix.tar.gz"),
    },
    {
        "cohort": "GSE166489_CITE",
        "sample_id": "A.HD3",
        "condition": "Healthy",
        "age_group": "Adult",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073072/suppl/GSM5073072_A.HD3_filtered_feature_bc_matrix.tar.gz",
        "dest": str(CITE_DIR / "GSM5073072_A.HD3_filtered_feature_bc_matrix.tar.gz"),
    },
    # --- GSE166489 13 Pediatric GEX-only Samples (Acute Severe/Moderate MIS-C, Longitudinal Recovered, Pediatric HD) ---
    {
        "cohort": "GSE166489_GEX",
        "sample_id": "P3.1",
        "condition": "MIS-C_Severe",
        "age_group": "Pediatric",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073057/suppl/GSM5073057_P3.1_filtered_feature_bc_matrix.tar.gz",
        "dest": str(PED_GEX_DIR / "GSM5073057_P3.1_filtered_feature_bc_matrix.tar.gz"),
    },
    {
        "cohort": "GSE166489_GEX",
        "sample_id": "P4.1",
        "condition": "MIS-C_Moderate",
        "age_group": "Pediatric",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073058/suppl/GSM5073058_P4.1_filtered_feature_bc_matrix.tar.gz",
        "dest": str(PED_GEX_DIR / "GSM5073058_P4.1_filtered_feature_bc_matrix.tar.gz"),
    },
    {
        "cohort": "GSE166489_GEX",
        "sample_id": "P5.1",
        "condition": "MIS-C_Moderate",
        "age_group": "Pediatric",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073059/suppl/GSM5073059_P5.1_filtered_feature_bc_matrix.tar.gz",
        "dest": str(PED_GEX_DIR / "GSM5073059_P5.1_filtered_feature_bc_matrix.tar.gz"),
    },
    {
        "cohort": "GSE166489_GEX",
        "sample_id": "P6.1",
        "condition": "MIS-C_Severe",
        "age_group": "Pediatric",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073060/suppl/GSM5073060_P6.1_filtered_feature_bc_matrix.tar.gz",
        "dest": str(PED_GEX_DIR / "GSM5073060_P6.1_filtered_feature_bc_matrix.tar.gz"),
    },
    {
        "cohort": "GSE166489_GEX",
        "sample_id": "P7.1",
        "condition": "MIS-C_Severe",
        "age_group": "Pediatric",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073061/suppl/GSM5073061_P7.1_filtered_feature_bc_matrix.tar.gz",
        "dest": str(PED_GEX_DIR / "GSM5073061_P7.1_filtered_feature_bc_matrix.tar.gz"),
    },
    {
        "cohort": "GSE166489_GEX",
        "sample_id": "P3.2",
        "condition": "MIS-C_Recovered",
        "age_group": "Pediatric",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073062/suppl/GSM5073062_P3.2_filtered_feature_bc_matrix.tar.gz",
        "dest": str(PED_GEX_DIR / "GSM5073062_P3.2_filtered_feature_bc_matrix.tar.gz"),
    },
    {
        "cohort": "GSE166489_GEX",
        "sample_id": "P4.2",
        "condition": "MIS-C_Recovered",
        "age_group": "Pediatric",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073063/suppl/GSM5073063_P4.2_filtered_feature_bc_matrix.tar.gz",
        "dest": str(PED_GEX_DIR / "GSM5073063_P4.2_filtered_feature_bc_matrix.tar.gz"),
    },
    {
        "cohort": "GSE166489_GEX",
        "sample_id": "C.HD1",
        "condition": "Pediatric_Healthy",
        "age_group": "Pediatric",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073064/suppl/GSM5073064_C.HD1_filtered_feature_bc_matrix.tar.gz",
        "dest": str(PED_GEX_DIR / "GSM5073064_C.HD1_filtered_feature_bc_matrix.tar.gz"),
    },
    {
        "cohort": "GSE166489_GEX",
        "sample_id": "C.HD2",
        "condition": "Pediatric_Healthy",
        "age_group": "Pediatric",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073065/suppl/GSM5073065_C.HD2_filtered_feature_bc_matrix.tar.gz",
        "dest": str(PED_GEX_DIR / "GSM5073065_C.HD2_filtered_feature_bc_matrix.tar.gz"),
    },
    {
        "cohort": "GSE166489_GEX",
        "sample_id": "C.HD3",
        "condition": "Pediatric_Healthy",
        "age_group": "Pediatric",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073066/suppl/GSM5073066_C.HD3_filtered_feature_bc_matrix.tar.gz",
        "dest": str(PED_GEX_DIR / "GSM5073066_C.HD3_filtered_feature_bc_matrix.tar.gz"),
    },
    {
        "cohort": "GSE166489_GEX",
        "sample_id": "C.HD4",
        "condition": "Pediatric_Healthy",
        "age_group": "Pediatric",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073067/suppl/GSM5073067_C.HD4_filtered_feature_bc_matrix.tar.gz",
        "dest": str(PED_GEX_DIR / "GSM5073067_C.HD4_filtered_feature_bc_matrix.tar.gz"),
    },
    {
        "cohort": "GSE166489_GEX",
        "sample_id": "C.HD5",
        "condition": "Pediatric_Healthy",
        "age_group": "Pediatric",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073068/suppl/GSM5073068_C.HD5_filtered_feature_bc_matrix.tar.gz",
        "dest": str(PED_GEX_DIR / "GSM5073068_C.HD5_filtered_feature_bc_matrix.tar.gz"),
    },
    {
        "cohort": "GSE166489_GEX",
        "sample_id": "C.HD6",
        "condition": "Pediatric_Healthy",
        "age_group": "Pediatric",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073069/suppl/GSM5073069_C.HD6_filtered_feature_bc_matrix.tar.gz",
        "dest": str(PED_GEX_DIR / "GSM5073069_C.HD6_filtered_feature_bc_matrix.tar.gz"),
    },
    # --- GSE166489 18 Matched Single-Cell TCR Contig Annotations ---
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "P1.1",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073073/suppl/GSM5073073_PBMC_P1.1_MIS-C_Severe_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073073_P1.1_TCR.csv.gz"),
    },
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "P2.1",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073074/suppl/GSM5073074_PBMC_P2.1_MIS-C_Severe_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073074_P2.1_TCR.csv.gz"),
    },
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "P3.1",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073075/suppl/GSM5073075_PBMC_P3.1_MIS-C_Severe_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073075_P3.1_TCR.csv.gz"),
    },
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "P4.1",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073076/suppl/GSM5073076_PBMC_P4.1_MIS-C_Moderate_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073076_P4.1_TCR.csv.gz"),
    },
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "P5.1",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073077/suppl/GSM5073077_PBMC_P5.1_MIS-C_Moderate_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073077_P5.1_TCR.csv.gz"),
    },
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "P6.1",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073078/suppl/GSM5073078_PBMC_P6.1_MIS-C_Severe_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073078_P6.1_TCR.csv.gz"),
    },
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "P7.1",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073079/suppl/GSM5073079_PBMC_P7.1_MIS-C_Severe_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073079_P7.1_TCR.csv.gz"),
    },
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "P3.2",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073080/suppl/GSM5073080_PBMC_P3.2_MIS-C_Severe_Recovered_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073080_P3.2_TCR.csv.gz"),
    },
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "P4.2",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073081/suppl/GSM5073081_PBMC_P4.2_MIS-C_Moderate_Recovered_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073081_P4.2_TCR.csv.gz"),
    },
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "C.HD1",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073082/suppl/GSM5073082_PBMC_C.HD1_Pediatric_healthy_donor_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073082_C.HD1_TCR.csv.gz"),
    },
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "C.HD2",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073083/suppl/GSM5073083_PBMC_C.HD2_Pediatric_healthy_donor_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073083_C.HD2_TCR.csv.gz"),
    },
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "C.HD3",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073084/suppl/GSM5073084_PBMC_C.HD3_Pediatric_healthy_donor_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073084_C.HD3_TCR.csv.gz"),
    },
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "C.HD4",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073085/suppl/GSM5073085_PBMC_C.HD4_Pediatric_healthy_donor_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073085_C.HD4_TCR.csv.gz"),
    },
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "C.HD5",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073086/suppl/GSM5073086_PBMC_C.HD5_Pediatric_healthy_donor_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073086_C.HD5_TCR.csv.gz"),
    },
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "C.HD6",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073087/suppl/GSM5073087_PBMC_C.HD6_Pediatric_healthy_donor_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073087_C.HD6_TCR.csv.gz"),
    },
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "A.HD1",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073088/suppl/GSM5073088_PBMC_A.HD1_Adult_healthy_donor_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073088_A.HD1_TCR.csv.gz"),
    },
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "A.HD2",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073089/suppl/GSM5073089_PBMC_A.HD2_Adult_healthy_donor_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073089_A.HD2_TCR.csv.gz"),
    },
    {
        "cohort": "GSE166489_TCR",
        "sample_id": "A.HD3",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5073nnn/GSM5073090/suppl/GSM5073090_PBMC_A.HD3_Adult_healthy_donor_TCR_filtered_contig_annotations.csv.gz",
        "dest": str(TCR_DIR / "GSM5073090_A.HD3_TCR.csv.gz"),
    },
    # --- GSE158013 Multi-Well & Cross-Donor TEA-seq (W4, W5, W6 = Donor HMN200910; X061-W1 = Independent Donor) ---
    {
        "cohort": "GSE158013_TEA",
        "sample_id": "X066_W4",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5123nnn/GSM5123952/suppl/GSM5123952_X066-MP0C1W4_leukopak_perm-cells_tea_200M_cellranger-arc_filtered_feature_bc_matrix.h5",
        "dest": str(TEA_DIR / "GSM5123952_X066-MP0C1W4_leukopak_perm-cells_tea_200M_cellranger-arc_filtered_feature_bc_matrix.h5"),
    },
    {
        "cohort": "GSE158013_TEA",
        "sample_id": "X066_W4",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5123nnn/GSM5123952/suppl/GSM5123952_X066-MP0C1W4_leukopak_perm-cells_tea_48M_adt_counts.csv.gz",
        "dest": str(TEA_DIR / "GSM5123952_X066-MP0C1W4_leukopak_perm-cells_tea_48M_adt_counts.csv.gz"),
    },
    {
        "cohort": "GSE158013_TEA",
        "sample_id": "X066_W4",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5123nnn/GSM5123952/suppl/GSM5123952_X066-MP0C1W4_leukopak_perm-cells_tea_200M_atac_filtered_metadata.csv.gz",
        "dest": str(TEA_DIR / "GSM5123952_X066-MP0C1W4_leukopak_perm-cells_tea_200M_atac_filtered_metadata.csv.gz"),
    },
    {
        "cohort": "GSE158013_TEA",
        "sample_id": "X066_W5",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5123nnn/GSM5123953/suppl/GSM5123953_X066-MP0C1W5_leukopak_perm-cells_tea_200M_cellranger-arc_filtered_feature_bc_matrix.h5",
        "dest": str(TEA_DIR / "GSM5123953_X066-MP0C1W5_leukopak_perm-cells_tea_200M_cellranger-arc_filtered_feature_bc_matrix.h5"),
    },
    {
        "cohort": "GSE158013_TEA",
        "sample_id": "X066_W5",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5123nnn/GSM5123953/suppl/GSM5123953_X066-MP0C1W5_leukopak_perm-cells_tea_48M_adt_counts.csv.gz",
        "dest": str(TEA_DIR / "GSM5123953_X066-MP0C1W5_leukopak_perm-cells_tea_48M_adt_counts.csv.gz"),
    },
    {
        "cohort": "GSE158013_TEA",
        "sample_id": "X066_W5",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5123nnn/GSM5123953/suppl/GSM5123953_X066-MP0C1W5_leukopak_perm-cells_tea_200M_atac_filtered_metadata.csv.gz",
        "dest": str(TEA_DIR / "GSM5123953_X066-MP0C1W5_leukopak_perm-cells_tea_200M_atac_filtered_metadata.csv.gz"),
    },
    {
        "cohort": "GSE158013_TEA",
        "sample_id": "X066_W6",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5123nnn/GSM5123954/suppl/GSM5123954_X066-MP0C1W6_leukopak_perm-cells_tea_200M_cellranger-arc_filtered_feature_bc_matrix.h5",
        "dest": str(TEA_DIR / "GSM5123954_X066-MP0C1W6_leukopak_perm-cells_tea_200M_cellranger-arc_filtered_feature_bc_matrix.h5"),
    },
    {
        "cohort": "GSE158013_TEA",
        "sample_id": "X066_W6",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5123nnn/GSM5123954/suppl/GSM5123954_X066-MP0C1W6_leukopak_perm-cells_tea_48M_adt_counts.csv.gz",
        "dest": str(TEA_DIR / "GSM5123954_X066-MP0C1W6_leukopak_perm-cells_tea_48M_adt_counts.csv.gz"),
    },
    {
        "cohort": "GSE158013_TEA",
        "sample_id": "X066_W6",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5123nnn/GSM5123954/suppl/GSM5123954_X066-MP0C1W6_leukopak_perm-cells_tea_200M_atac_filtered_metadata.csv.gz",
        "dest": str(TEA_DIR / "GSM5123954_X066-MP0C1W6_leukopak_perm-cells_tea_200M_atac_filtered_metadata.csv.gz"),
    },
    {
        "cohort": "GSE158013_TEA",
        "sample_id": "X061_W1",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4949nnn/GSM4949911/suppl/GSM4949911_X061-AP0C1W1_leukopak_perm-cells_tea_fulldepth_cellranger-arc_filtered_feature_bc_matrix.h5",
        "dest": str(TEA_DIR / "GSM4949911_X061-AP0C1W1_leukopak_perm-cells_tea_fulldepth_cellranger-arc_filtered_feature_bc_matrix.h5"),
    },
    {
        "cohort": "GSE158013_TEA",
        "sample_id": "X061_W1",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4949nnn/GSM4949911/suppl/GSM4949911_X061-AP0C1W1_leukopak_perm-cells_tea_fulldepth_adt_counts.csv.gz",
        "dest": str(TEA_DIR / "GSM4949911_X061-AP0C1W1_leukopak_perm-cells_tea_fulldepth_adt_counts.csv.gz"),
    },
    {
        "cohort": "GSE158013_TEA",
        "sample_id": "X061_W1",
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4949nnn/GSM4949911/suppl/GSM4949911_X061-AP0C1W1_leukopak_perm-cells_tea_fulldepth_atac_filtered_metadata.csv.gz",
        "dest": str(TEA_DIR / "GSM4949911_X061-AP0C1W1_leukopak_perm-cells_tea_fulldepth_atac_filtered_metadata.csv.gz"),
    },
]


def verify_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    name = path.name.lower()
    try:
        if name.endswith(".tar.gz"):
            with tarfile.open(path, "r:gz") as t:
                members = t.getnames()
                return len(members) >= 3
        elif name.endswith(".gz"):
            with gzip.open(path, "rb") as f:
                f.read(256)
            return True
        elif name.endswith(".h5"):
            with path.open("rb") as f:
                sig = f.read(8)
            return sig == b"\x89HDF\r\n\x1a\n"
        return True
    except Exception:
        return False


def download_one(entry: dict[str, str]) -> dict[str, object]:
    dest = Path(entry["dest"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    if verify_file(dest):
        status = "cached"
    else:
        part = dest.with_suffix(dest.suffix + ".part")
        cmd = [
            "curl.exe",
            "-sS",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "3",
            "-C",
            "-",
            "-o",
            str(part),
            entry["url"],
        ]
        for attempt in range(3):
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0 and part.is_file() and part.stat().st_size > 1024:
                part.replace(dest)
                if verify_file(dest):
                    status = "downloaded"
                    break
            part.unlink(missing_ok=True)
            time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"Failed downloading {entry['url']}: {res.stderr}")

    h = hashlib.sha256()
    with dest.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    mb = round(dest.stat().st_size / (1024 * 1024), 3)
    print(f"[{status.upper()}] {dest.name} ({mb} MB)", flush=True)
    return {
        **entry,
        "status": status,
        "size_mb": mb,
        "sha256": h.hexdigest(),
    }


def main() -> None:
    t0 = time.perf_counter()
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futures = [ex.submit(download_one, e) for e in DOWNLOAD_MANIFEST]
        for fut in concurrent.futures.as_completed(futures):
            records.append(fut.result())

    records.sort(key=lambda r: (str(r["cohort"]), str(r["sample_id"]), str(r["dest"])))
    total_mb = round(sum(float(r["size_mb"]) for r in records), 2)
    receipt = {
        "schema": "tolt.v4.cohort_download_receipt.v1",
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "file_count": len(records),
        "total_size_mb": total_mb,
        "files": records,
    }
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ALL {len(records)} FILES VERIFIED ({total_mb} MB) in {receipt['elapsed_sec']}s -> {RECEIPT_PATH}", flush=True)


if __name__ == "__main__":
    main()
