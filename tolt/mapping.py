from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import pandas as pd
from scipy import sparse

_PEAK_RE = re.compile(r"^(chr[^:]+)[:_](\d+)[-_](\d+)$", re.IGNORECASE)

def parse_peak(value: str) -> tuple[str, int, int] | None:
    match = _PEAK_RE.match(str(value).strip())
    if not match:
        return None
    chrom, start, end = match.group(1), int(match.group(2)), int(match.group(3))
    if end <= start:
        return None
    return chrom, start, end

_GRCH38_INTERVAL_RE = re.compile(
    r"^([^:]+)[:_-](\d+)[-_](\d+)$", re.IGNORECASE
)

def normalize_grch38_contig(value: str) -> str | None:
    """Normalize an unambiguous GRCh38 primary contig; reject all other aliases."""

    contig = str(value).strip()
    if contig.lower().startswith("chr"):
        core = contig[3:]
    else:
        core = contig
    if core.isdigit() and 1 <= int(core) <= 22:
        return f"chr{int(core)}"
    if core.upper() in {"X", "Y"}:
        return f"chr{core.upper()}"
    return None

def normalize_grch38_interval(
    value: str | tuple[str, int, int],
) -> tuple[str, int, int] | None:
    parsed = parse_peak(value) if isinstance(value, str) else value
    if parsed is None and isinstance(value, str):
        # ``parse_peak`` intentionally follows the historical chr-prefixed
        # DOGMA schema. Phase-1 additionally accepts unprefixed primary contigs
        # and the standard 10x ``chr1-100-200`` spelling so that the explicit
        # GRCh38 normalizer owns this decision.
        match = _GRCH38_INTERVAL_RE.fullmatch(value.strip())
        if match is not None:
            parsed = (match.group(1), int(match.group(2)), int(match.group(3)))
    if parsed is None:
        return None
    chrom, start, end = parsed
    normalized = normalize_grch38_contig(chrom)
    if normalized is None or start < 0 or end <= start:
        return None
    return normalized, int(start), int(end)

def _normalize_rna(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    result = matrix.astype(np.float32, copy=True)
    totals = np.asarray(result.sum(axis=1)).ravel()
    result = sparse.diags(1.0e4 / np.maximum(totals, 1.0)) @ result
    result.data = np.log1p(result.data)
    return result.tocsr()

def _rna_tokens(
    normalized: sparse.csr_matrix,
    standardized_target: np.ndarray,
    token_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = normalized.shape[0]
    ids = np.zeros((n, token_count), dtype=np.int32)
    values = np.zeros((n, token_count), dtype=np.float32)
    valid = np.zeros((n, token_count), dtype=bool)
    for row in range(n):
        start, stop = normalized.indptr[row], normalized.indptr[row + 1]
        columns = normalized.indices[start:stop]
        data = normalized.data[start:stop]
        if len(columns) > token_count:
            keep = np.argpartition(data, -token_count)[-token_count:]
            keep = keep[np.argsort(data[keep])[::-1]]
            columns = columns[keep]
        length = len(columns)
        ids[row, :length] = columns + 1
        values[row, :length] = standardized_target[row, columns]
        valid[row, :length] = True
    return ids, values, valid

def _interval_lookup(consensus: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    lookup: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for chrom, group in consensus.groupby("chrom", sort=False):
        group = group.sort_values(["start", "end"], kind="stable")
        lookup[str(chrom)] = (
            group["start"].to_numpy(dtype=np.int64),
            group["end"].to_numpy(dtype=np.int64),
            group.index.to_numpy(dtype=np.int64),
        )
    return lookup

def _peak_mapping(
    features: pd.DataFrame,
    consensus: pd.DataFrame,
) -> tuple[sparse.csr_matrix, float]:
    """Return a sparse source-peak to consensus-peak overlap matrix.

    One source peak may overlap multiple consensus intervals (especially when a
    holdout peak bridges two training-derived regions).  Every positive
    half-open interval overlap is retained; boundary touching alone is not.
    """

    lookup = _interval_lookup(consensus)
    rows: list[int] = []
    columns: list[int] = []
    parseable = 0
    mapped_sources = 0
    for row, value in enumerate(features["feature_name"].astype(str)):
        parsed = normalize_grch38_interval(value)
        if parsed is None:
            continue
        parseable += 1
        chrom, start, end = parsed
        if chrom not in lookup:
            continue
        starts, ends, consensus_indices = lookup[chrom]
        left = int(np.searchsorted(ends, start, side="right"))
        right = int(np.searchsorted(starts, end, side="left"))
        source_was_mapped = False
        for candidate in range(left, right):
            if starts[candidate] >= end or ends[candidate] <= start:
                continue
            rows.append(row)
            columns.append(int(consensus_indices[candidate]))
            source_was_mapped = True
        mapped_sources += int(source_was_mapped)
    mapping = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(features), len(consensus)),
    )
    mapping.sum_duplicates()
    mapping.data[:] = 1.0
    coverage = mapped_sources / max(parseable, 1)
    return mapping, coverage

def _tfidf(matrix: sparse.csr_matrix, idf: np.ndarray) -> sparse.csr_matrix:
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    tf = sparse.diags(1.0 / np.maximum(totals, 1.0)) @ matrix
    return tf.multiply(idf).tocsr().astype(np.float32)

def _atac_tokens(
    tfidf: sparse.csr_matrix,
    token_count: int,
    clip: float,
    mean: float,
    std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = tfidf.shape[0]
    ids = np.zeros((n, token_count), dtype=np.int32)
    values = np.zeros((n, token_count), dtype=np.float32)
    valid = np.zeros((n, token_count), dtype=bool)
    for row in range(n):
        start, stop = tfidf.indptr[row], tfidf.indptr[row + 1]
        columns = tfidf.indices[start:stop]
        data = tfidf.data[start:stop]
        if len(columns) > token_count:
            keep = np.argpartition(data, -token_count)[-token_count:]
            keep = keep[np.argsort(data[keep])[::-1]]
            columns, data = columns[keep], data[keep]
        length = len(columns)
        transformed = np.log1p(1.0e4 * data)
        transformed = (np.minimum(transformed, clip) - mean) / max(std, 1.0e-6)
        ids[row, :length] = columns + 1
        values[row, :length] = transformed
        valid[row, :length] = True
    return ids, values, valid

def exact_rna_indices(
    source_ids: Sequence[str],
    source_names: Sequence[str],
    frozen_ids: Sequence[str],
    frozen_names: Sequence[str],
) -> np.ndarray:
    """Map frozen RNA features by their exact Ensembl-ID/name identity pair.

    Gene symbols are not identifiers: one processed TEA panel legitimately
    contains the same symbol on rows with different Ensembl IDs.  Those rows
    remain distinct.  Duplicate Ensembl IDs are rejected because summing them
    without transcript/version provenance would change the count definition.
    """

    if len(source_ids) != len(source_names) or len(frozen_ids) != len(frozen_names):
        raise ValueError("RNA ID/name columns are misaligned")
    source_pairs = [
        (str(feature_id), str(name))
        for feature_id, name in zip(source_ids, source_names, strict=True)
    ]
    if len({feature_id for feature_id, _ in source_pairs}) != len(source_pairs):
        raise ValueError(
            "source RNA Ensembl IDs must be unique; duplicate-ID rows cannot be safely aggregated"
        )
    if len(set(frozen_ids)) != len(frozen_ids) or len(set(frozen_names)) != len(frozen_names):
        raise ValueError("frozen RNA IDs and names must each be unique")
    by_pair = {pair: index for index, pair in enumerate(source_pairs)}
    selected: list[int] = []
    for feature_id, feature_name in zip(frozen_ids, frozen_names, strict=True):
        source_index = by_pair.get((str(feature_id), str(feature_name)))
        if source_index is None:
            raise ValueError(
                "every frozen RNA feature must match one source column by both exact ID and name"
            )
        selected.append(source_index)
    return np.asarray(selected, dtype=np.int64)

def peak_overlap_projection(
    source_names: Sequence[str], frozen_features: pd.DataFrame
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Map source peaks to every positive half-open frozen-peak overlap."""

    source = pd.DataFrame({"feature_name": [str(value) for value in source_names]})
    mapping, source_coverage = _peak_mapping(source, frozen_features)
    source_mapped = int(np.count_nonzero(np.diff(mapping.indptr)))
    frozen_hit = np.asarray(mapping.sum(axis=0)).ravel() > 0
    return mapping, {
        "source_features": int(len(source)),
        "source_features_mapped": source_mapped,
        "source_mapping_fraction": float(source_coverage),
        "frozen_features": int(len(frozen_features)),
        "frozen_features_covered": int(frozen_hit.sum()),
        "frozen_coverage_fraction": float(frozen_hit.mean()),
        "coordinate_system": "GRCh38_zero_based_half_open",
        "overlap_rule": "positive_overlap_only_boundary_touch_excluded_all_overlaps_retained",
    }

def _transform_rna(
    counts: sparse.csr_matrix,
    source_ids: Sequence[str],
    source_names: Sequence[str],
    frozen_features: pd.DataFrame,
    scaler_path: Path,
    token_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected = exact_rna_indices(
        source_ids,
        source_names,
        frozen_features["feature_id"].astype(str).tolist(),
        frozen_features["feature_name"].astype(str).tolist(),
    )
    totals = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("every retained TEA cell must have positive RNA library size")
    normalized_full = _normalize_rna(counts)
    normalized = normalized_full[:, selected].tocsr()
    with np.load(scaler_path, allow_pickle=False) as scaler:
        mean = np.asarray(scaler["mean"], dtype=np.float32)
        std = np.asarray(scaler["std"], dtype=np.float32)
    if mean.shape != (len(frozen_features),) or std.shape != mean.shape or np.any(std <= 0):
        raise ValueError("frozen RNA scaler dimensions changed")
    dense = normalized.toarray().astype(np.float32)
    dense = np.clip((dense - mean) / std, -10.0, 10.0).astype(np.float32)
    ids, values, valid = _rna_tokens(normalized, dense, token_count)
    if (~valid.any(axis=1)).any():
        raise ValueError("a retained TEA RNA row has no frozen token")
    return {
        "rna_ids": ids.astype(np.int32, copy=False),
        "rna_values": values.astype(np.float32, copy=False),
        "rna_valid": valid.astype(bool, copy=False),
        "rna_dense": dense,
    }, {
        "source_features": len(source_ids),
        "source_unique_ensembl_IDs": len(set(str(value) for value in source_ids)),
        "source_unique_gene_names": len(set(str(value) for value in source_names)),
        "source_gene_name_duplicate_excess_rows": len(source_names)
        - len(set(str(value) for value in source_names)),
        "source_identity_rule": "exact_Ensembl_ID_and_name_pair_gene_names_may_repeat",
        "frozen_features": len(frozen_features),
        "exact_ID_and_name_matches": len(selected),
        "normalization": "full_raw_RNA_library_CPTT_log1p_before_frozen_2048_selection",
        "standardization": "frozen_clean_mean_std_clip_minus10_plus10",
        "token_count": token_count,
    }

def map_inputs(rna, atac, gene_ids, gene_names, peak_names, asset_dir):
    asset_dir = Path(asset_dir)
    rt, rr = _transform_rna(rna, gene_ids, gene_names, pd.read_parquet(asset_dir / "rna_features.parquet"), asset_dir / "rna_scaler.npz", 512)
    frozen = pd.read_parquet(asset_dir / "atac_features.parquet")
    mapping, ar = peak_overlap_projection(peak_names, frozen)
    projected = (atac @ mapping).tocsr()
    projected.sum_duplicates()
    projected.data[:] = 1.0
    projected.eliminate_zeros()
    projected.sort_indices()
    if np.any(np.diff(projected.indptr) <= 0):
        raise ValueError("A cell has no mapped ATAC peaks")
    with np.load(asset_dir / "atac_tfidf_scaler.npz") as scale:
        ids, values, valid = _atac_tokens(_tfidf(projected, scale["idf"]), 1024, float(scale["token_clip"]), float(scale["token_mean"]), float(scale["token_std"]))
    return {**rt, "atac_ids": ids, "atac_values": values, "atac_valid": valid}, {"RNA": rr, "ATAC": ar}
