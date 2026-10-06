"""Reproducible CITE-BMMC-Stuart random-holdout benchmark.

This module implements the four metrics used by Li et al. (Genome Biology,
2025) and a compact set of strong, leakage-bounded RNA-to-protein baselines.
The public article specifies five 50/50 random holdouts but does not publish
the cell assignments.  We therefore freeze an explicit reconstruction using
``train_test_split(..., random_state=seed)`` for seeds 0..4 and hash every
ordered cell-ID vector.

All learned RNA preprocessing and supervised estimators are fit on the outer
training half.  OOF residuals and stack features are generated without using
outer-test protein values.  Evaluation is deliberately separate from fitting.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from .optimization import _simplex_quadratic_solution


BMMC_STUART_SHA256 = "9c668dd707efa8844aa61f0ec78441b89a46f98593796fe32cb55c82e02cef64"
BMMC_STUART_CELLS = 33_454
BMMC_STUART_GENES = 17_009
BMMC_STUART_PROTEINS = 25
BMMC_STUART_SEEDS = (0, 1, 2, 3, 4)
BMMC_STUART_SCHEMA = "triomics.v3_110.bmmc-stuart-random-holdout.v1"

METHODS = (
    "mean",
    "svd_ridge",
    "svd_knn",
    "tolt_ridge",
    "tolt_residual",
    "oof_stack",
    "per_protein_oof_simplex",
)

PER_PROTEIN_OOF_SIMPLEX_BASES = (
    "svd_ridge",
    "tolt_ridge",
    "tolt_residual",
)
PER_PROTEIN_OOF_SIMPLEX_FOLDS = 3

METRIC_DIRECTIONS = {
    "protein_pcc": "higher",
    "protein_rmse": "lower",
    "cell_pcc": "higher",
    "cell_rmse": "lower",
}


@dataclass(frozen=True)
class BMMCStuartConfig:
    """Compute-conscious defaults for one 50/50 holdout."""

    target_sum: float = 10_000.0
    n_top_genes: int = 6_000
    svd_components: int = 256
    svd_iterations: int = 7
    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)
    ridge_cv_folds: int = 3
    knn_neighbors: int = 30
    stack_alpha: float = 10.0
    clip_min: float = 0.0
    n_jobs: int = -1

    def validate(self) -> None:
        if not math.isfinite(self.target_sum) or self.target_sum <= 0:
            raise ValueError("target_sum must be positive")
        if self.n_top_genes < 2 or self.svd_components < 2:
            raise ValueError("n_top_genes and svd_components must be at least 2")
        if self.svd_iterations < 1 or self.ridge_cv_folds < 2:
            raise ValueError("svd_iterations must be positive and ridge_cv_folds at least 2")
        if not self.ridge_alphas or any(
            (not math.isfinite(value) or value <= 0) for value in self.ridge_alphas
        ):
            raise ValueError("ridge_alphas must be a non-empty positive sequence")
        if self.knn_neighbors < 1 or not math.isfinite(self.stack_alpha) or self.stack_alpha <= 0:
            raise ValueError("knn_neighbors and stack_alpha must be positive")
        if not math.isfinite(self.clip_min):
            raise ValueError("clip_min must be finite")


@dataclass(frozen=True)
class BMMCStuartData:
    cell_ids: np.ndarray
    protein_names: np.ndarray
    rna: sp.csr_matrix | None
    protein: np.ndarray
    source_path: Path
    source_sha256: str | None


@dataclass(frozen=True)
class HoldoutSplit:
    seed: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_cell_ids: np.ndarray
    test_cell_ids: np.ndarray
    train_ids_sha256: str
    test_ids_sha256: str
    assignment_sha256: str


@dataclass(frozen=True)
class MethodFit:
    prediction: np.ndarray
    recipe: Mapping[str, object]
    oof_prediction: np.ndarray | None = None


def file_sha256(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _as_string_vector(value: object, *, name: str, unique: bool = True) -> np.ndarray:
    result = np.asarray(value).astype(str)
    if result.ndim != 1 or result.size == 0 or any(item == "" for item in result.tolist()):
        raise ValueError(f"{name} must be a non-empty one-dimensional string vector")
    if unique and len(set(result.tolist())) != result.size:
        raise ValueError(f"{name} contains duplicates")
    return result


def _validate_nonnegative_finite(matrix: object, *, name: str, integer: bool = False) -> None:
    values = matrix.data if sp.issparse(matrix) else np.asarray(matrix).ravel()
    values = np.asarray(values)
    if values.size and (not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError(f"{name} contains negative or non-finite values")
    if integer and values.size and not np.allclose(values, np.rint(values), atol=1e-6, rtol=0):
        raise ValueError(f"{name} is not raw integer count data")


def read_bmmc_stuart(
    path: str | Path,
    *,
    load_rna: bool = True,
    verify_sha256: bool = True,
) -> BMMCStuartData:
    """Read and validate the frozen Zenodo random-holdout dataset.

    ``load_rna=False`` uses backed mode and is intended for metadata/mean-model
    preflight; it avoids materializing the 33,454 x 17,009 RNA matrix.
    """

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = file_sha256(source) if verify_sha256 else None
    if digest is not None and digest != BMMC_STUART_SHA256:
        raise ValueError(
            f"CITE-BMMC-Stuart SHA256 mismatch: expected {BMMC_STUART_SHA256}, got {digest}"
        )

    dataset = ad.read_h5ad(source, backed=None if load_rna else "r")
    try:
        if dataset.shape != (BMMC_STUART_CELLS, BMMC_STUART_GENES):
            raise ValueError(
                "CITE-BMMC-Stuart shape mismatch: "
                f"expected {(BMMC_STUART_CELLS, BMMC_STUART_GENES)}, got {dataset.shape}"
            )
        cell_ids = _as_string_vector(dataset.obs_names, name="cell IDs")
        if "cell_name" in dataset.obs:
            declared = _as_string_vector(dataset.obs["cell_name"], name="obs.cell_name")
            if not np.array_equal(cell_ids, declared):
                raise ValueError("obs_names and obs.cell_name differ")
        if "protein_expression" not in dataset.obsm:
            raise ValueError("dataset lacks obsm['protein_expression']")
        protein_raw = dataset.obsm["protein_expression"]
        if protein_raw.shape != (BMMC_STUART_CELLS, BMMC_STUART_PROTEINS):
            raise ValueError("protein_expression shape mismatch")
        protein = (
            protein_raw.toarray() if sp.issparse(protein_raw) else np.asarray(protein_raw)
        ).astype(np.float32, copy=False)
        _validate_nonnegative_finite(protein, name="protein_expression", integer=True)
        if "protein_name" not in dataset.uns:
            raise ValueError("dataset lacks uns['protein_name']")
        protein_names = _as_string_vector(dataset.uns["protein_name"], name="protein names")
        if protein_names.size != BMMC_STUART_PROTEINS:
            raise ValueError("protein_name length mismatch")

        rna: sp.csr_matrix | None = None
        if load_rna:
            raw = dataset.X
            if not sp.issparse(raw):
                raise ValueError("CITE-BMMC-Stuart X must be sparse raw counts")
            rna = sp.csr_matrix(raw, dtype=np.float32)
            _validate_nonnegative_finite(rna, name="RNA X", integer=True)
        return BMMCStuartData(
            cell_ids=cell_ids,
            protein_names=protein_names,
            rna=rna,
            protein=protein,
            source_path=source,
            source_sha256=digest,
        )
    finally:
        if getattr(dataset, "isbacked", False):
            dataset.file.close()


def make_holdout_split(cell_ids: Sequence[object], seed: int) -> HoldoutSplit:
    """Freeze one exact 50/50 sklearn random-holdout reconstruction."""

    ids = _as_string_vector(cell_ids, name="cell IDs")
    indices = np.arange(ids.size, dtype=np.int64)
    train_index, test_index = train_test_split(
        indices,
        test_size=0.5,
        random_state=int(seed),
        shuffle=True,
    )
    train_index = np.asarray(train_index, dtype=np.int64)
    test_index = np.asarray(test_index, dtype=np.int64)
    if train_index.size + test_index.size != ids.size:
        raise AssertionError("holdout dropped rows")
    if np.intersect1d(train_index, test_index).size:
        raise AssertionError("holdout train/test overlap")
    train_ids = ids[train_index]
    test_ids = ids[test_index]
    train_hash = sha256_lines(train_ids)
    test_hash = sha256_lines(test_ids)
    assignment = sha256_json(
        {
            "algorithm": "sklearn.model_selection.train_test_split",
            "random_state": int(seed),
            "shuffle": True,
            "test_size": 0.5,
            "train_ids_sha256": train_hash,
            "test_ids_sha256": test_hash,
        }
    )
    return HoldoutSplit(
        seed=int(seed),
        train_indices=train_index,
        test_indices=test_index,
        train_cell_ids=train_ids,
        test_cell_ids=test_ids,
        train_ids_sha256=train_hash,
        test_ids_sha256=test_hash,
        assignment_sha256=assignment,
    )


def _correlation_rows(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    pred_centered = prediction - prediction.mean(axis=1, keepdims=True)
    truth_centered = truth - truth.mean(axis=1, keepdims=True)
    numerator = np.sum(pred_centered * truth_centered, axis=1, dtype=np.float64)
    denominator = np.sqrt(
        np.sum(pred_centered * pred_centered, axis=1, dtype=np.float64)
        * np.sum(truth_centered * truth_centered, axis=1, dtype=np.float64)
    )
    result = np.full(prediction.shape[0], np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _zscore_columns(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, ddof=0, keepdims=True)
    # StandardScaler semantics for a constant feature: scale=1 and all zeros.
    safe = np.where(scale > 0, scale, 1.0)
    return centered / safe


def _l2_rows(values: np.ndarray, *, denominator_epsilon: float) -> np.ndarray:
    """Mirror the public evaluation notebook's asymmetric L2 convention."""

    norms = np.linalg.norm(values, axis=1, keepdims=True) + denominator_epsilon
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def official_metric_distributions(
    prediction: object,
    truth: object,
) -> dict[str, np.ndarray]:
    """Calculate the paper's protein/cell PCC and normalized RMSE arrays."""

    pred = np.asarray(prediction, dtype=np.float64)
    actual = np.asarray(truth, dtype=np.float64)
    if pred.shape != actual.shape or pred.ndim != 2 or min(pred.shape, default=0) <= 0:
        raise ValueError("prediction and truth must be same-shaped non-empty matrices")
    if not np.isfinite(pred).all() or not np.isfinite(actual).all():
        raise ValueError("prediction/truth contains non-finite values")

    protein_pcc = _correlation_rows(pred.T, actual.T)
    protein_rmse = np.sqrt(np.mean((_zscore_columns(pred) - _zscore_columns(actual)) ** 2, axis=0))
    cell_pcc = _correlation_rows(pred, actual)
    # All published method-evaluation notebooks add 1e-8 to only the
    # predicted row norm; the truth row norm is used without an epsilon.
    cell_rmse = np.sqrt(
        np.mean(
            (_l2_rows(pred, denominator_epsilon=1.0e-8) - _l2_rows(actual, denominator_epsilon=0.0))
            ** 2,
            axis=1,
        )
    )
    return {
        "protein_pcc": protein_pcc,
        "protein_rmse": protein_rmse,
        "cell_pcc": cell_pcc,
        "cell_rmse": cell_rmse,
    }


def metric_summary(distributions: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for name in METRIC_DIRECTIONS:
        values = np.asarray(distributions[name], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[name] = {
            "median": float(np.median(finite)) if finite.size else None,
            "mean": float(np.mean(finite)) if finite.size else None,
            "n": int(values.size),
            "n_finite": int(finite.size),
            "direction": METRIC_DIRECTIONS[name],
        }
    return result


def rank_method_summaries(
    summaries: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, dict[str, float]]:
    """Rank one experiment's method set and calculate protein/cell ARS.

    The best method receives score ``n_methods`` and the worst score 1.
    Ties receive their average rank.  Missing/non-finite values rank worst.
    ARS is meaningful only within the exact declared method set.
    """

    names = sorted(summaries)
    if not names:
        raise ValueError("cannot rank an empty method set")
    scores: dict[str, dict[str, float]] = {name: {} for name in names}
    for metric, direction in METRIC_DIRECTIONS.items():
        raw = np.asarray(
            [summaries[name][metric].get("median") for name in names],
            dtype=object,
        )
        numeric = np.asarray(
            [float(value) if value is not None else np.nan for value in raw],
            dtype=np.float64,
        )
        finite = np.isfinite(numeric)
        if not finite.any():
            ranked = np.ones(len(names), dtype=np.float64)
        else:
            if direction == "higher":
                worst = float(np.min(numeric[finite]) - max(1.0, np.ptp(numeric[finite]) + 1.0))
                ranking_values = np.where(finite, numeric, worst)
            else:
                worst = float(np.max(numeric[finite]) + max(1.0, np.ptp(numeric[finite]) + 1.0))
                ranking_values = np.where(finite, -numeric, -worst)
            ranked = rankdata(ranking_values, method="average")
        for name, score in zip(names, ranked, strict=True):
            scores[name][f"{metric}_rank_score"] = float(score)
    for name in names:
        scores[name]["protein_ars"] = 0.5 * (
            scores[name]["protein_pcc_rank_score"] + scores[name]["protein_rmse_rank_score"]
        )
        scores[name]["cell_ars"] = 0.5 * (
            scores[name]["cell_pcc_rank_score"] + scores[name]["cell_rmse_rank_score"]
        )
        scores[name]["combined_ars_descriptive"] = 0.5 * (
            scores[name]["protein_ars"] + scores[name]["cell_ars"]
        )
    return scores


def align_embeddings(
    dataset_cell_ids: Sequence[object],
    embedding_cell_ids: Sequence[object],
    embeddings: object,
) -> np.ndarray:
    dataset_ids = _as_string_vector(dataset_cell_ids, name="dataset cell IDs")
    source_ids = _as_string_vector(embedding_cell_ids, name="embedding cell IDs")
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != source_ids.size or not np.isfinite(values).all():
        raise ValueError("embedding matrix is invalid or disagrees with embedding cell IDs")
    positions = {cell_id: index for index, cell_id in enumerate(source_ids.tolist())}
    missing = [cell_id for cell_id in dataset_ids.tolist() if cell_id not in positions]
    if missing:
        raise ValueError(
            f"embedding archive lacks {len(missing)} dataset cells; first={missing[0]}"
        )
    return values[np.asarray([positions[cell_id] for cell_id in dataset_ids], dtype=np.int64)]


def load_aligned_embeddings(
    path: str | Path, dataset_cell_ids: Sequence[object]
) -> tuple[np.ndarray, dict[str, object]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as payload:
        id_key = next((key for key in ("cell_ids", "obs_names", "cells") if key in payload), None)
        value_key = next(
            (
                key
                for key in ("embeddings", "rna_embeddings", "embedding", "latent")
                if key in payload
            ),
            None,
        )
        if id_key is None or value_key is None:
            raise ValueError("embedding NPZ needs cell_ids and embeddings-compatible keys")
        aligned = align_embeddings(dataset_cell_ids, payload[id_key], payload[value_key])
    return aligned, {
        "path": str(source),
        "sha256": file_sha256(source),
        "cell_id_key": id_key,
        "embedding_key": value_key,
        "shape": list(aligned.shape),
        "aligned_cell_ids_sha256": sha256_lines(dataset_cell_ids),
    }


def _normalize_log1p(matrix: sp.csr_matrix, target_sum: float) -> sp.csr_matrix:
    result = sp.csr_matrix(matrix, dtype=np.float32, copy=True)
    totals = np.asarray(result.sum(axis=1)).ravel().astype(np.float64)
    factors = np.divide(
        target_sum,
        totals,
        out=np.zeros_like(totals),
        where=totals > 0,
    ).astype(np.float32)
    result = result.multiply(factors[:, None]).tocsr()
    np.log1p(result.data, out=result.data)
    return result


def fit_rna_representation(
    x_train: sp.csr_matrix,
    x_test: sp.csr_matrix,
    *,
    seed: int,
    config: BMMCStuartConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Fit normalize/log1p, train-only variable genes, SVD, and scaler."""

    config.validate()
    train = _normalize_log1p(x_train, config.target_sum)
    test = _normalize_log1p(x_test, config.target_sum)
    mean = np.asarray(train.mean(axis=0)).ravel().astype(np.float64)
    second = np.asarray(train.power(2).mean(axis=0)).ravel().astype(np.float64)
    variance = np.maximum(second - mean * mean, 0.0)
    n_top = min(config.n_top_genes, train.shape[1])
    hvg = np.argsort(-variance, kind="stable")[:n_top].astype(np.int64)
    components = min(config.svd_components, train.shape[0] - 1, n_top - 1)
    if components < 2:
        raise ValueError("insufficient rows/features for requested RNA SVD")
    svd = TruncatedSVD(
        n_components=components,
        n_iter=config.svd_iterations,
        random_state=int(seed),
    )
    train_latent = svd.fit_transform(train[:, hvg]).astype(np.float32, copy=False)
    test_latent = svd.transform(test[:, hvg]).astype(np.float32, copy=False)
    scaler = StandardScaler().fit(train_latent)
    train_latent = scaler.transform(train_latent).astype(np.float32, copy=False)
    test_latent = scaler.transform(test_latent).astype(np.float32, copy=False)
    return (
        train_latent,
        test_latent,
        {
            "normalization": "per-cell library-size to target_sum, then log1p",
            "target_sum": config.target_sum,
            "hvg_scope": "outer_train_only_sparse_variance",
            "hvg_count": int(hvg.size),
            "hvg_indices_sha256": sha256_lines(hvg),
            "svd_components": int(components),
            "svd_iterations": int(config.svd_iterations),
            "svd_random_state": int(seed),
            "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
            "scaler_scope": "outer_train_only",
        },
    )


def _safe_selection_key(summary: Mapping[str, Mapping[str, object]]) -> tuple[float, ...]:
    def value(name: str, *, default: float) -> float:
        raw = summary[name]["median"]
        return float(raw) if raw is not None and math.isfinite(float(raw)) else default

    protein_pcc = value("protein_pcc", default=-math.inf)
    cell_pcc = value("cell_pcc", default=-math.inf)
    protein_rmse = value("protein_rmse", default=math.inf)
    cell_rmse = value("cell_rmse", default=math.inf)
    return (
        0.5 * (protein_pcc + cell_pcc),
        protein_pcc,
        cell_pcc,
        -0.5 * (protein_rmse + cell_rmse),
        -protein_rmse,
        -cell_rmse,
    )


def _ridge_oof_and_fit(
    train_features: np.ndarray,
    test_features: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    alphas: Sequence[float],
    folds: int,
) -> tuple[np.ndarray, np.ndarray, float, list[dict[str, object]]]:
    splitter = KFold(n_splits=folds, shuffle=True, random_state=int(seed) + 20_003)
    assignments = list(splitter.split(train_features))
    best: tuple[tuple[float, ...], float, np.ndarray, dict[str, object]] | None = None
    screen: list[dict[str, object]] = []
    for alpha in alphas:
        oof = np.empty_like(y_train, dtype=np.float32)
        for fit_index, valid_index in assignments:
            model = Ridge(alpha=float(alpha), solver="lsqr")
            model.fit(train_features[fit_index], y_train[fit_index])
            oof[valid_index] = model.predict(train_features[valid_index]).astype(np.float32)
        summary = metric_summary(official_metric_distributions(oof, y_train))
        row = {"alpha": float(alpha), "metrics": summary}
        screen.append(row)
        candidate = (_safe_selection_key(summary), -float(alpha), oof, row)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    selected_alpha = float(best[3]["alpha"])
    selected_oof = best[2]
    final = Ridge(alpha=selected_alpha, solver="lsqr").fit(train_features, y_train)
    prediction = final.predict(test_features).astype(np.float32)
    return selected_oof, prediction, selected_alpha, screen


def _knn_oof_and_fit(
    train_features: np.ndarray,
    test_features: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    neighbors: int,
    folds: int,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = KFold(n_splits=folds, shuffle=True, random_state=int(seed) + 40_009)
    oof = np.empty_like(y_train, dtype=np.float32)
    for fit_index, valid_index in splitter.split(train_features):
        k = min(neighbors, fit_index.size)
        model = KNeighborsRegressor(
            n_neighbors=k,
            weights="distance",
            algorithm="brute",
            n_jobs=n_jobs,
        ).fit(train_features[fit_index], y_train[fit_index])
        oof[valid_index] = model.predict(train_features[valid_index]).astype(np.float32)
    model = KNeighborsRegressor(
        n_neighbors=min(neighbors, train_features.shape[0]),
        weights="distance",
        algorithm="brute",
        n_jobs=n_jobs,
    ).fit(train_features, y_train)
    prediction = model.predict(test_features).astype(np.float32)
    return oof, prediction


def _fit_per_protein_oof_simplex(
    *,
    oof_predictions: Sequence[np.ndarray],
    test_predictions: Sequence[np.ndarray],
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one deterministic convex OOF blend per protein.

    The function deliberately has no outer-test target argument.  Each
    protein's non-negative weights sum to one and minimize squared error on
    outer-train cross-fitted predictions only.  Because every simplex vertex
    is feasible, the fitted OOF squared error cannot exceed the best supplied
    base member for that protein (up to floating-point tolerance).
    """

    target = np.asarray(y_train, dtype=np.float64)
    if target.ndim != 2 or min(target.shape, default=0) <= 0:
        raise ValueError("y_train must be a non-empty two-dimensional matrix")
    if not np.isfinite(target).all():
        raise ValueError("y_train contains non-finite values")
    if len(oof_predictions) != len(test_predictions) or len(oof_predictions) < 2:
        raise ValueError("simplex blending requires matching OOF/test predictions")

    oof = tuple(np.asarray(value, dtype=np.float64) for value in oof_predictions)
    test = tuple(np.asarray(value, dtype=np.float64) for value in test_predictions)
    test_shape = test[0].shape
    if any(value.shape != target.shape for value in oof):
        raise ValueError("OOF prediction shapes must match y_train")
    if (
        len(test_shape) != 2
        or test_shape[1] != target.shape[1]
        or any(value.shape != test_shape for value in test)
    ):
        raise ValueError("test predictions must share rows and protein columns")
    if any(not np.isfinite(value).all() for value in (*oof, *test)):
        raise ValueError("simplex base predictions contain non-finite values")

    n_sources = len(oof)
    n_rows, n_proteins = target.shape
    weights = np.empty((n_proteins, n_sources), dtype=np.float64)
    oof_blend = np.empty_like(target)
    test_blend = np.empty(test_shape, dtype=np.float64)
    for protein in range(n_proteins):
        design = np.column_stack([value[:, protein] for value in oof])
        test_design = np.column_stack([value[:, protein] for value in test])
        gram = design.T @ design / float(n_rows)
        linear = design.T @ target[:, protein] / float(n_rows)
        protein_weights = _simplex_quadratic_solution(gram, linear)
        weights[protein] = protein_weights
        oof_blend[:, protein] = design @ protein_weights
        test_blend[:, protein] = test_design @ protein_weights

    if np.any(weights < -1e-12) or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10):
        raise FloatingPointError("per-protein OOF weights escaped the simplex")
    if not np.isfinite(oof_blend).all() or not np.isfinite(test_blend).all():
        raise FloatingPointError("per-protein OOF simplex produced non-finite predictions")
    return (
        oof_blend.astype(np.float32),
        test_blend.astype(np.float32),
        weights,
    )


def fit_predict_methods(
    *,
    x_train: sp.csr_matrix | None,
    x_test: sp.csr_matrix | None,
    y_train: object,
    seed: int,
    methods: Sequence[str],
    config: BMMCStuartConfig = BMMCStuartConfig(),
    train_embeddings: np.ndarray | None = None,
    test_embeddings: np.ndarray | None = None,
) -> dict[str, MethodFit]:
    """Fit requested methods without accepting outer-test protein truth."""

    config.validate()
    requested = tuple(dict.fromkeys(str(name) for name in methods))
    unknown = sorted(set(requested).difference(METHODS))
    if not requested or unknown:
        raise ValueError(f"methods must be a non-empty subset of {METHODS}; unknown={unknown}")
    target = np.asarray(y_train, dtype=np.float32)
    if target.ndim != 2 or not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError("y_train must be a finite nonnegative matrix")

    needs_simplex = "per_protein_oof_simplex" in requested
    if needs_simplex and config.ridge_cv_folds != PER_PROTEIN_OOF_SIMPLEX_FOLDS:
        raise ValueError("per_protein_oof_simplex freezes exactly three outer-train OOF folds")
    needs_rna = bool(
        set(requested).intersection(
            {
                "svd_ridge",
                "svd_knn",
                "tolt_residual",
                "oof_stack",
                "per_protein_oof_simplex",
            }
        )
    )
    needs_tolt = bool(
        set(requested).intersection({"tolt_ridge", "tolt_residual", "per_protein_oof_simplex"})
    )
    if "oof_stack" in requested and train_embeddings is not None:
        needs_tolt = True
    if needs_rna and (x_train is None or x_test is None):
        raise ValueError("requested methods require RNA matrices")
    if needs_tolt and (train_embeddings is None or test_embeddings is None):
        raise ValueError("requested TOLT methods require aligned train/test embeddings")

    result: dict[str, MethodFit] = {}
    if "mean" in requested:
        if x_test is not None:
            test_rows = x_test.shape[0]
        elif test_embeddings is not None:
            test_rows = test_embeddings.shape[0]
        else:
            raise ValueError("mean method needs x_test or test_embeddings to determine test rows")
        prediction = np.broadcast_to(target.mean(axis=0), (test_rows, target.shape[1])).copy()
        result["mean"] = MethodFit(
            prediction=prediction.astype(np.float32),
            recipe={"family": "outer_train_protein_mean", "test_truth_accessed": False},
        )

    rna_train: np.ndarray | None = None
    rna_test: np.ndarray | None = None
    rna_contract: dict[str, object] | None = None
    ridge_oof: np.ndarray | None = None
    ridge_test: np.ndarray | None = None
    ridge_recipe: dict[str, object] | None = None
    if needs_rna:
        assert x_train is not None and x_test is not None
        rna_train, rna_test, rna_contract = fit_rna_representation(
            x_train,
            x_test,
            seed=seed,
            config=config,
        )
        ridge_oof, ridge_test, alpha, screen = _ridge_oof_and_fit(
            rna_train,
            rna_test,
            target,
            seed=seed,
            alphas=config.ridge_alphas,
            folds=config.ridge_cv_folds,
        )
        ridge_recipe = {
            "family": "train_only_normalize_log1p_hvg_svd_ridge",
            "representation": rna_contract,
            "selected_alpha": alpha,
            "selection_scope": "outer_train_supervised_oof_only",
            "selection_screen": screen,
            "test_truth_accessed": False,
        }
        if "svd_ridge" in requested:
            result["svd_ridge"] = MethodFit(
                prediction=np.maximum(ridge_test, config.clip_min),
                oof_prediction=np.maximum(ridge_oof, config.clip_min),
                recipe=ridge_recipe,
            )

    knn_oof: np.ndarray | None = None
    knn_test: np.ndarray | None = None
    if "svd_knn" in requested or ("oof_stack" in requested and train_embeddings is None):
        assert rna_train is not None and rna_test is not None and rna_contract is not None
        knn_oof, knn_test = _knn_oof_and_fit(
            rna_train,
            rna_test,
            target,
            seed=seed,
            neighbors=config.knn_neighbors,
            folds=config.ridge_cv_folds,
            n_jobs=config.n_jobs,
        )
        knn_recipe = {
            "family": "train_only_normalize_log1p_hvg_svd_distance_knn",
            "representation": rna_contract,
            "neighbors": config.knn_neighbors,
            "weights": "distance",
            "oof_folds": config.ridge_cv_folds,
            "test_truth_accessed": False,
        }
        if "svd_knn" in requested:
            result["svd_knn"] = MethodFit(
                prediction=np.maximum(knn_test, config.clip_min),
                oof_prediction=np.maximum(knn_oof, config.clip_min),
                recipe=knn_recipe,
            )

    tolt_train: np.ndarray | None = None
    tolt_test: np.ndarray | None = None
    tolt_oof: np.ndarray | None = None
    tolt_test_prediction: np.ndarray | None = None
    tolt_recipe: dict[str, object] | None = None
    if needs_tolt:
        assert train_embeddings is not None and test_embeddings is not None
        if train_embeddings.shape[0] != target.shape[0]:
            raise ValueError("train embedding rows differ from y_train")
        scaler = StandardScaler().fit(train_embeddings)
        tolt_train = scaler.transform(train_embeddings).astype(np.float32)
        tolt_test = scaler.transform(test_embeddings).astype(np.float32)
        tolt_oof, tolt_test_prediction, alpha, screen = _ridge_oof_and_fit(
            tolt_train,
            tolt_test,
            target,
            seed=seed + 1_000,
            alphas=config.ridge_alphas,
            folds=config.ridge_cv_folds,
        )
        tolt_recipe = {
            "family": "TOLT_RNA_only_embedding_ridge",
            "embedding_scaler_scope": "outer_train_only",
            "selected_alpha": alpha,
            "selection_scope": "outer_train_supervised_oof_only",
            "selection_screen": screen,
            "test_truth_accessed": False,
        }
        if "tolt_ridge" in requested:
            result["tolt_ridge"] = MethodFit(
                prediction=np.maximum(tolt_test_prediction, config.clip_min),
                oof_prediction=np.maximum(tolt_oof, config.clip_min),
                recipe=tolt_recipe,
            )

    residual_oof: np.ndarray | None = None
    residual_test: np.ndarray | None = None
    if "tolt_residual" in requested or needs_simplex:
        assert ridge_oof is not None and ridge_test is not None
        assert tolt_train is not None and tolt_test is not None
        residual_target = target - ridge_oof
        residual_component_oof, residual_component_test, alpha, screen = _ridge_oof_and_fit(
            tolt_train,
            tolt_test,
            residual_target,
            seed=seed + 2_000,
            alphas=config.ridge_alphas,
            folds=config.ridge_cv_folds,
        )
        residual_oof = ridge_oof + residual_component_oof
        residual_test = ridge_test + residual_component_test
        if "tolt_residual" in requested:
            result["tolt_residual"] = MethodFit(
                prediction=np.maximum(residual_test, config.clip_min),
                oof_prediction=np.maximum(residual_oof, config.clip_min),
                recipe={
                    "family": "cross_fitted_RNA_SVD_ridge_plus_TOLT_embedding_residual_ridge",
                    "base": ridge_recipe,
                    "residual_selected_alpha": alpha,
                    "residual_selection_scope": "outer_train_supervised_oof_only",
                    "residual_selection_screen": screen,
                    "residual_targets": "y_train minus cross-fitted SVD-ridge OOF prediction",
                    "test_truth_accessed": False,
                },
            )

    if needs_simplex:
        assert ridge_oof is not None and ridge_test is not None
        assert tolt_oof is not None and tolt_test_prediction is not None
        assert residual_oof is not None and residual_test is not None
        simplex_oof, simplex_test, simplex_weights = _fit_per_protein_oof_simplex(
            oof_predictions=(
                np.maximum(ridge_oof, config.clip_min),
                np.maximum(tolt_oof, config.clip_min),
                np.maximum(residual_oof, config.clip_min),
            ),
            test_predictions=(
                np.maximum(ridge_test, config.clip_min),
                np.maximum(tolt_test_prediction, config.clip_min),
                np.maximum(residual_test, config.clip_min),
            ),
            y_train=target,
        )
        result["per_protein_oof_simplex"] = MethodFit(
            prediction=np.maximum(simplex_test, config.clip_min),
            oof_prediction=np.maximum(simplex_oof, config.clip_min),
            recipe={
                "family": "fit_only_per_protein_oof_simplex",
                "base_methods": list(PER_PROTEIN_OOF_SIMPLEX_BASES),
                "base_prediction_contract": "clip_min applied before convex blending",
                "weight_scope": "one weight vector per ordered protein",
                "weight_constraint": "nonnegative_simplex_sum_one",
                "weight_fit_objective": "outer-train OOF raw-scale squared error per protein",
                "weight_solver": "deterministic exact active-support quadratic solve",
                "oof_folds": PER_PROTEIN_OOF_SIMPLEX_FOLDS,
                "new_hyperparameters": False,
                "weights": simplex_weights.tolist(),
                "weights_sha256": sha256_json(simplex_weights.tolist()),
                "meta_fit_features": "outer-train cross-fitted base predictions only",
                "outer_test_protein_access": "evaluation only after frozen prediction",
                "test_truth_accessed": False,
            },
        )

    if "oof_stack" in requested:
        assert ridge_oof is not None and ridge_test is not None
        base_names = ["svd_ridge"]
        oof_parts = [ridge_oof]
        test_parts = [ridge_test]
        if tolt_oof is not None and tolt_test_prediction is not None:
            base_names.append("tolt_ridge")
            oof_parts.append(tolt_oof)
            test_parts.append(tolt_test_prediction)
        elif knn_oof is not None and knn_test is not None:
            base_names.append("svd_knn")
            oof_parts.append(knn_oof)
            test_parts.append(knn_test)
        if residual_oof is not None and residual_test is not None:
            base_names.append("tolt_residual")
            oof_parts.append(residual_oof)
            test_parts.append(residual_test)
        if len(oof_parts) < 2:
            raise ValueError("oof_stack requires at least two available base predictions")
        meta_train = np.concatenate(oof_parts, axis=1)
        meta_test = np.concatenate(test_parts, axis=1)
        meta = Ridge(alpha=config.stack_alpha, solver="lsqr").fit(meta_train, target)
        stack_test = meta.predict(meta_test).astype(np.float32)
        result["oof_stack"] = MethodFit(
            prediction=np.maximum(stack_test, config.clip_min),
            recipe={
                "family": "fit_only_OOF_prediction_stack",
                "base_methods": base_names,
                "meta_estimator": "multioutput_ridge",
                "meta_alpha": config.stack_alpha,
                "meta_fit_features": "outer-train cross-fitted base predictions only",
                "test_truth_accessed": False,
            },
        )

    return {name: result[name] for name in requested}


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        result = float(value)
        return result if math.isfinite(result) else None
    return value


def atomic_json(path: str | Path, payload: object, *, overwrite: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                _json_safe(payload),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_npz(path: str | Path, *, overwrite: bool = False, **arrays: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".npz", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def config_from_mapping(value: Mapping[str, object]) -> BMMCStuartConfig:
    fields = {
        "target_sum",
        "n_top_genes",
        "svd_components",
        "svd_iterations",
        "ridge_alphas",
        "ridge_cv_folds",
        "knn_neighbors",
        "stack_alpha",
        "clip_min",
        "n_jobs",
    }
    unknown = sorted(set(value).difference(fields))
    if unknown:
        raise ValueError(f"unknown BMMC-Stuart config keys: {unknown}")
    normalized = dict(value)
    if "ridge_alphas" in normalized:
        normalized["ridge_alphas"] = tuple(float(item) for item in normalized["ridge_alphas"])
    config = BMMCStuartConfig(**normalized)
    config.validate()
    return config


def config_receipt(config: BMMCStuartConfig) -> dict[str, object]:
    value = asdict(config)
    value["ridge_alphas"] = list(config.ridge_alphas)
    return value
