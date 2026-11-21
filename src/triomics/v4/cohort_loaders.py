"""Clean, zero-leakage cohort loaders for GSE194122, GSE128639, GSE166489 (18 samples + TCR), and GSE158013 (5 TEA-seq wells)."""
from __future__ import annotations

import gzip
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .paths import COHORTS_DIR, D_BMMC90K, D_STUART
from scipy.io import mmread


@dataclass(frozen=True)
class TenXMatrix:
    """Standard 10x feature-by-cell H5 materialized as cell-by-feature CSR."""
    values: sp.csr_matrix
    barcodes: tuple[str, ...]
    feature_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_types: tuple[str, ...]


def _decode_strings(values: Any, label: str) -> tuple[str, ...]:
    arr = np.asarray(values)
    return tuple(raw.decode("utf-8") if isinstance(raw, bytes) else str(raw) for raw in arr.tolist())


def read_10x_feature_h5(path: str | Path) -> TenXMatrix:
    """Read standard Cell Ranger ARC CSC H5 payload into cell-by-feature CSR."""
    with h5py.File(Path(path), "r") as handle:
        shape = tuple(int(v) for v in np.asarray(handle["matrix/shape"]))
        csc = sp.csc_matrix(
            (
                np.asarray(handle["matrix/data"]),
                np.asarray(handle["matrix/indices"], dtype=np.int64),
                np.asarray(handle["matrix/indptr"], dtype=np.int64),
            ),
            shape=shape,
        )
        barcodes = _decode_strings(handle["matrix/barcodes"], "barcodes")
        f_ids = _decode_strings(handle["matrix/features/id"], "feature_ids")
        f_names = _decode_strings(handle["matrix/features/name"], "feature_names")
        f_types = _decode_strings(handle["matrix/features/feature_type"], "feature_types")
    vals = csc.T.tocsr()
    vals.sum_duplicates()
    vals.sort_indices()
    return TenXMatrix(vals, barcodes, f_ids, f_names, f_types)


def modality_view(payload: TenXMatrix, feature_type: str) -> TenXMatrix:
    sel = np.flatnonzero(np.asarray(payload.feature_types, dtype=object) == str(feature_type))
    return TenXMatrix(
        payload.values[:, sel].tocsr(),
        payload.barcodes,
        tuple(payload.feature_ids[i] for i in sel),
        tuple(payload.feature_names[i] for i in sel),
        tuple(payload.feature_types[i] for i in sel),
    )



def seurat_clr(counts: np.ndarray) -> np.ndarray:
    """Standard Seurat non-negative centered log-ratio transform per cell across panel."""
    c = np.maximum(np.asarray(counts, dtype=np.float32), 0.0)
    return np.log1p(c / np.exp(np.log1p(c).mean(axis=1, keepdims=True))).astype(np.float32)


def log1p_cp10k_dense(counts: np.ndarray) -> np.ndarray:
    """Per-cell CP10k normalization followed by log1p on dense ADT count matrix."""
    c = np.maximum(np.asarray(counts, dtype=np.float32), 0.0)
    rs = np.maximum(c.sum(axis=1, keepdims=True), 1.0)
    return np.log1p(c / rs * 10000.0).astype(np.float32)


def log1p_cp10k_sparse(x: sp.csr_matrix) -> sp.csr_matrix:
    """Per-cell CP10k normalization followed by log1p on sparse RNA count matrix."""
    mat = x.tocsr().astype(np.float32, copy=True)
    rs = np.asarray(mat.sum(axis=1)).ravel()
    scale = np.where(rs > 0, 10000.0 / np.maximum(rs, 1.0), 0.0).astype(np.float32)
    mat = sp.diags(scale) @ mat
    mat.data = np.log1p(mat.data)
    return mat.tocsr()


@dataclass
class CITESample:
    sample_id: str
    condition: str
    age_group: str
    barcodes: np.ndarray
    gene_ids: np.ndarray
    gene_names: np.ndarray
    rna_counts: sp.csr_matrix
    adt_counts: np.ndarray | None
    adt_names: list[str] | None
    isotype_counts: np.ndarray | None
    isotype_names: list[str] | None
    tcr_df: pd.DataFrame | None


def read_10x_mtx_tar_gz(tar_path: Path) -> tuple[np.ndarray, pd.DataFrame, sp.csr_matrix]:
    """Read 10x CellRanger filtered_feature_bc_matrix.tar.gz into (barcodes, features_df, cells_x_features_csr)."""
    with tarfile.open(tar_path, "r:gz") as t:
        names = {Path(m.name).name: m for m in t.getmembers() if m.isfile()}
        with gzip.GzipFile(fileobj=t.extractfile(names["features.tsv.gz"])) as f:
            feat = pd.read_csv(f, sep="\t", header=None, dtype=str)
        with gzip.GzipFile(fileobj=t.extractfile(names["barcodes.tsv.gz"])) as f:
            barcodes = pd.read_csv(f, sep="\t", header=None, dtype=str)[0].astype(str).to_numpy()
        with gzip.GzipFile(fileobj=t.extractfile(names["matrix.mtx.gz"])) as f:
            mat = sp.csr_matrix(mmread(f).T, dtype=np.float32)
    return barcodes, feat, mat


def clean_gse166489_adt_panel(feat_adt: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray, list[str]]:
    """Apply the three verified panel hygiene rules to GSE166489's 189 Antibody Capture features:
    1. Separate non-specific `IgGFc` (`0375_anti_human_IgGFc`) as negative control, leaving 188 human targets.
    2. Correct upstream GEO feature_id/clone swap on TCRVd2 (`0582_anti_human_TCRVg9`) and TCRVg9 (`0583_anti_human_TCR_Vg`).
    3. Disambiguate duplicate CD8 clones (`CD8`=`0046` vs `CD8a`=`0080`).
    """
    raw_ids = feat_adt[0].astype(str).tolist()
    raw_clones = feat_adt[1].astype(str).tolist()
    clean_indices: list[int] = []
    clean_names: list[str] = []
    iso_indices: list[int] = []
    iso_names: list[str] = []

    for idx, (fid, clone) in enumerate(zip(raw_ids, raw_clones)):
        if fid == "IgGFc" or "IgGFc" in clone or "Isotype" in clone:
            iso_indices.append(idx)
            iso_names.append(fid)
            continue
        # Fix upstream GEO label swap (verified in antibody_panel.tsv lines 150-151)
        if fid == "TCRVd2" and "0582_anti_human_TCRVg9" in clone:
            corrected = "TCRVg9_0582"
        elif fid == "TCRVg9" and "0583_anti_human_TCR_Vg" in clone:
            corrected = "TCRVg_0583"
        else:
            corrected = fid
        clean_indices.append(idx)
        clean_names.append(corrected)

    return (
        np.asarray(clean_indices, dtype=np.int64),
        clean_names,
        np.asarray(iso_indices, dtype=np.int64),
        iso_names,
    )


def load_tcr_table(tcr_csv_gz: Path) -> pd.DataFrame:
    """Load 10x V(D)J filtered contig annotations and summarize per-barcode TCR beta V-gene (incl. TRBV11-2)."""
    df = pd.read_csv(tcr_csv_gz, compression="gzip", dtype=str)
    if "productive" in df.columns:
        df = df[df["productive"].str.lower().isin({"true", "t", "1"})].copy()
    trb = df[df["chain"].eq("TRB")].copy() if "chain" in df.columns else df.copy()
    trb["is_trbv11_2"] = trb["v_gene"].astype(str).str.contains("TRBV11-2", regex=False)
    # Aggregate per cell barcode
    grouped = (
        trb.groupby("barcode", sort=False)
        .agg(
            trb_v_genes=("v_gene", lambda s: ";".join(sorted(set(str(x) for x in s.dropna())))),
            is_trbv11_2=("is_trbv11_2", "any"),
            clonotype_id=("raw_clonotype_id", "first"),
        )
        .reset_index()
    )
    return grouped


GSE166489_METADATA: list[dict[str, str]] = [
    {"sample_id": "A.HD1", "gsm": "GSM5073070", "tcr_gsm": "GSM5073088", "condition": "Adult_Healthy", "age_group": "Adult", "has_adt": "true", "patient": "A.HD1", "timepoint": "Baseline"},
    {"sample_id": "A.HD2", "gsm": "GSM5073071", "tcr_gsm": "GSM5073089", "condition": "Adult_Healthy", "age_group": "Adult", "has_adt": "true", "patient": "A.HD2", "timepoint": "Baseline"},
    {"sample_id": "A.HD3", "gsm": "GSM5073072", "tcr_gsm": "GSM5073090", "condition": "Adult_Healthy", "age_group": "Adult", "has_adt": "true", "patient": "A.HD3", "timepoint": "Baseline"},
    {"sample_id": "P1.1", "gsm": "GSM5073055", "tcr_gsm": "GSM5073073", "condition": "MIS-C_Severe", "age_group": "Pediatric", "has_adt": "true", "patient": "P1", "timepoint": "Acute"},
    {"sample_id": "P2.1", "gsm": "GSM5073056", "tcr_gsm": "GSM5073074", "condition": "MIS-C_Severe", "age_group": "Pediatric", "has_adt": "true", "patient": "P2", "timepoint": "Acute"},
    {"sample_id": "P3.1", "gsm": "GSM5073057", "tcr_gsm": "GSM5073075", "condition": "MIS-C_Severe", "age_group": "Pediatric", "has_adt": "false", "patient": "P3", "timepoint": "Acute"},
    {"sample_id": "P4.1", "gsm": "GSM5073058", "tcr_gsm": "GSM5073076", "condition": "MIS-C_Moderate", "age_group": "Pediatric", "has_adt": "false", "patient": "P4", "timepoint": "Acute"},
    {"sample_id": "P5.1", "gsm": "GSM5073059", "tcr_gsm": "GSM5073077", "condition": "MIS-C_Moderate", "age_group": "Pediatric", "has_adt": "false", "patient": "P5", "timepoint": "Acute"},
    {"sample_id": "P6.1", "gsm": "GSM5073060", "tcr_gsm": "GSM5073078", "condition": "MIS-C_Severe", "age_group": "Pediatric", "has_adt": "false", "patient": "P6", "timepoint": "Acute"},
    {"sample_id": "P7.1", "gsm": "GSM5073061", "tcr_gsm": "GSM5073079", "condition": "MIS-C_Severe", "age_group": "Pediatric", "has_adt": "false", "patient": "P7", "timepoint": "Acute"},
    {"sample_id": "P3.2", "gsm": "GSM5073062", "tcr_gsm": "GSM5073080", "condition": "MIS-C_Recovered", "age_group": "Pediatric", "has_adt": "false", "patient": "P3", "timepoint": "Recovered"},
    {"sample_id": "P4.2", "gsm": "GSM5073063", "tcr_gsm": "GSM5073081", "condition": "MIS-C_Recovered", "age_group": "Pediatric", "has_adt": "false", "patient": "P4", "timepoint": "Recovered"},
    {"sample_id": "C.HD1", "gsm": "GSM5073064", "tcr_gsm": "GSM5073082", "condition": "Pediatric_Healthy", "age_group": "Pediatric", "has_adt": "false", "patient": "C.HD1", "timepoint": "Baseline"},
    {"sample_id": "C.HD2", "gsm": "GSM5073065", "tcr_gsm": "GSM5073083", "condition": "Pediatric_Healthy", "age_group": "Pediatric", "has_adt": "false", "patient": "C.HD2", "timepoint": "Baseline"},
    {"sample_id": "C.HD3", "gsm": "GSM5073066", "tcr_gsm": "GSM5073084", "condition": "Pediatric_Healthy", "age_group": "Pediatric", "has_adt": "false", "patient": "C.HD3", "timepoint": "Baseline"},
    {"sample_id": "C.HD4", "gsm": "GSM5073067", "tcr_gsm": "GSM5073085", "condition": "Pediatric_Healthy", "age_group": "Pediatric", "has_adt": "false", "patient": "C.HD4", "timepoint": "Baseline"},
    {"sample_id": "C.HD5", "gsm": "GSM5073068", "tcr_gsm": "GSM5073086", "condition": "Pediatric_Healthy", "age_group": "Pediatric", "has_adt": "false", "patient": "C.HD5", "timepoint": "Baseline"},
    {"sample_id": "C.HD6", "gsm": "GSM5073069", "tcr_gsm": "GSM5073087", "condition": "Pediatric_Healthy", "age_group": "Pediatric", "has_adt": "false", "patient": "C.HD6", "timepoint": "Baseline"},
]


def load_gse166489_sample(meta: dict[str, str], min_genes: int = 200, max_mt_pct: float = 20.0) -> CITESample:
    """Load one GSE166489 sample (CITE-seq or Pediatric GEX-only) with standard PBMC QC and matched TCR."""
    sid = meta["sample_id"]
    gsm = meta["gsm"]
    has_adt = meta["has_adt"] == "true"
    sub_dir = "cite_seq" if has_adt else "pediatric_gex"
    tar_path = COHORTS_DIR / "GSE166489_MISC" / sub_dir / f"{gsm}_{sid}_filtered_feature_bc_matrix.tar.gz"
    barcodes, feat, mat = read_10x_mtx_tar_gz(tar_path)

    rna_mask = feat[2].eq("Gene Expression").to_numpy()
    rna_feat = feat.loc[rna_mask].reset_index(drop=True)
    x_rna = mat[:, rna_mask].tocsr()

    # Standard single-cell PBMC QC filter
    n_genes = np.diff(x_rna.indptr)
    rna_sum = np.asarray(x_rna.sum(axis=1)).ravel()
    mt_cols = rna_feat[1].astype(str).str.upper().str.startswith("MT-").to_numpy()
    mt_sum = np.asarray(x_rna[:, mt_cols].sum(axis=1)).ravel() if mt_cols.any() else np.zeros_like(rna_sum)
    mt_pct = np.where(rna_sum > 0, 100.0 * mt_sum / np.maximum(rna_sum, 1.0), 100.0)
    keep = (rna_sum > 0) & (n_genes >= min_genes) & (mt_pct <= max_mt_pct)

    adt_counts = None
    adt_names = None
    iso_counts = None
    iso_names = None
    if has_adt:
        adt_mask = feat[2].eq("Antibody Capture").to_numpy()
        adt_feat = feat.loc[adt_mask].reset_index(drop=True)
        raw_adt = mat[:, adt_mask].toarray().astype(np.float32)
        keep = keep & (raw_adt.sum(axis=1) > 0) & (raw_adt.sum(axis=1) < 1e5)
        c_idx, adt_names, i_idx, iso_names = clean_gse166489_adt_panel(adt_feat)
        adt_counts = raw_adt[keep][:, c_idx]
        iso_counts = raw_adt[keep][:, i_idx]

    x_rna = x_rna[keep].tocsr()
    kept_barcodes = barcodes[keep]

    tcr_path = COHORTS_DIR / "GSE166489_MISC" / "tcr" / f"{meta['tcr_gsm']}_{sid}_TCR.csv.gz"
    tcr_df = load_tcr_table(tcr_path) if tcr_path.is_file() else None

    return CITESample(
        sample_id=sid,
        condition=meta["condition"],
        age_group=meta["age_group"],
        barcodes=kept_barcodes,
        gene_ids=rna_feat[0].astype(str).to_numpy(),
        gene_names=rna_feat[1].astype(str).to_numpy(),
        rna_counts=x_rna,
        adt_counts=adt_counts,
        adt_names=adt_names,
        isotype_counts=iso_counts,
        isotype_names=iso_names,
        tcr_df=tcr_df,
    )


TEASEQ_WELLS: list[dict[str, str]] = [
    {"well_id": "X066_W3", "gsm": "GSM5123951", "donor_id": "HMN200910", "batch_group": "X066", "prefix": "GSM5123951_X066-MP0C1W3_leukopak_perm-cells_tea_"},
    {"well_id": "X066_W4", "gsm": "GSM5123952", "donor_id": "HMN200910", "batch_group": "X066", "prefix": "GSM5123952_X066-MP0C1W4_leukopak_perm-cells_tea_"},
    {"well_id": "X066_W5", "gsm": "GSM5123953", "donor_id": "HMN200910", "batch_group": "X066", "prefix": "GSM5123953_X066-MP0C1W5_leukopak_perm-cells_tea_"},
    {"well_id": "X066_W6", "gsm": "GSM5123954", "donor_id": "HMN200910", "batch_group": "X066", "prefix": "GSM5123954_X066-MP0C1W6_leukopak_perm-cells_tea_"},
    {"well_id": "X061_W1", "gsm": "GSM4949911", "donor_id": "X061_Donor", "batch_group": "X061", "prefix": "GSM4949911_X061-AP0C1W1_leukopak_perm-cells_tea_"},
]


@dataclass
class TEASeqWell:
    well_id: str
    donor_id: str
    barcodes: np.ndarray
    gene_ids: np.ndarray
    gene_names: np.ndarray
    peak_names: np.ndarray
    rna_counts: sp.csr_matrix
    atac_counts: sp.csr_matrix
    adt_counts_45: np.ndarray
    adt_names_45: list[str]
    isotype_counts: np.ndarray
    adt_counts_all46: np.ndarray
    adt_names_all46: list[str]


def load_teaseq_well(meta: dict[str, str]) -> TEASeqWell:
    """Load one GSE158013 TEA-seq well aligning ARC .h5, ATAC QC metadata, and streaming 48M ADT CSV."""
    tea_dir = COHORTS_DIR / "GSE158013_TEAseq"
    prefix = meta["prefix"]
    h5_path = next(tea_dir.glob(f"{prefix}*filtered_feature_bc_matrix.h5"))
    adt_path = next(tea_dir.glob(f"{prefix}*adt_counts.csv.gz"))
    meta_path = next(tea_dir.glob(f"{prefix}*atac_filtered_metadata.csv.gz"))

    payload = read_10x_feature_h5(h5_path)
    rna = modality_view(payload, "Gene Expression")
    atac = modality_view(payload, "Peaks")

    # Match ATAC QC metadata barcodes
    meta_header = pd.read_csv(meta_path, compression="gzip", nrows=0).columns.tolist()
    bc_col = "original_barcodes" if "original_barcodes" in meta_header else "barcodes"
    meta_df = pd.read_csv(meta_path, compression="gzip", usecols=[bc_col], dtype=str)
    qc_bcs = set(meta_df[bc_col].dropna().astype(str))
    qc_cores = {b[:16] for b in qc_bcs}

    wanted_by_core: dict[str, str] = {}
    for raw_bc in rna.barcodes:
        s = str(raw_bc)
        core = s[:16]
        if s in qc_bcs or core in qc_cores:
            wanted_by_core[core] = s

    # Stream ADT counts CSV in 100k chunks
    adt_header = pd.read_csv(adt_path, compression="gzip", nrows=0).columns.tolist()
    all_46_names = [c for c in adt_header if c not in ("cell_barcode", "total")]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        adt_path,
        compression="gzip",
        usecols=["cell_barcode", *all_46_names],
        chunksize=100_000,
        dtype={"cell_barcode": str},
    ):
        core_series = chunk["cell_barcode"].astype(str).str.slice(stop=16)
        sub = chunk[core_series.isin(wanted_by_core)].copy()
        if len(sub):
            sub["h5_barcode"] = core_series.loc[sub.index].map(wanted_by_core)
            pieces.append(sub)
    adt_df = pd.concat(pieces, axis=0, ignore_index=True).drop_duplicates("h5_barcode").set_index("h5_barcode")

    kept_set = set(adt_df.index)
    ordered_ids = np.array([str(c) for c in rna.barcodes if str(c) in kept_set], dtype=str)
    h5_lookup = {str(c): i for i, c in enumerate(rna.barcodes)}
    idx = np.array([h5_lookup[c] for c in ordered_ids], dtype=np.int64)

    x_rna = rna.values[idx].tocsr().astype(np.float32)
    x_atac = atac.values[idx].tocsr().astype(np.float32)
    raw_adt46 = adt_df.loc[ordered_ids, all_46_names].to_numpy(dtype=np.float32)

    pos = (np.asarray(x_rna.sum(axis=1)).ravel() > 0) & (np.asarray(x_atac.sum(axis=1)).ravel() > 0) & (raw_adt46.sum(axis=1) > 0)
    x_rna, x_atac, raw_adt46, ordered_ids = x_rna[pos], x_atac[pos], raw_adt46[pos], ordered_ids[pos]

    non_iso_mask = np.array(["Isotype" not in n for n in all_46_names], dtype=bool)
    iso_mask = ~non_iso_mask
    names_45 = [n for n, m in zip(all_46_names, non_iso_mask) if m]

    return TEASeqWell(
        well_id=meta["well_id"],
        donor_id=meta["donor_id"],
        barcodes=ordered_ids,
        gene_ids=np.asarray(rna.feature_ids, dtype=str),
        gene_names=np.asarray(rna.feature_names, dtype=str),
        peak_names=np.asarray(atac.feature_names, dtype=str),
        rna_counts=x_rna,
        atac_counts=x_atac,
        adt_counts_45=raw_adt46[:, non_iso_mask],
        adt_names_45=names_45,
        isotype_counts=raw_adt46[:, iso_mask],
        adt_counts_all46=raw_adt46,
        adt_names_all46=all_46_names,
    )
