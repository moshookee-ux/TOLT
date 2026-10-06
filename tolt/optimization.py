from __future__ import annotations
import numpy as np

_TINY = np.finfo(np.float64).tiny

def _simplex_quadratic_solution(
    gram: np.ndarray,
    linear: np.ndarray,
) -> np.ndarray:
    """Exactly solve a small simplex-constrained convex quadratic by supports."""

    n_sources = int(linear.size)
    if gram.shape != (n_sources, n_sources):
        raise ValueError("quadratic Gram matrix shape mismatch")
    if n_sources > 8:
        raise ValueError("at most eight candidate sources are supported")
    best_objective: float | None = None
    best_weights: np.ndarray | None = None
    tolerance = 1e-10
    for mask in range(1, 1 << n_sources):
        support = np.flatnonzero([(mask >> index) & 1 for index in range(n_sources)])
        sub_gram = gram[np.ix_(support, support)]
        ones = np.ones(support.size, dtype=np.float64)
        kkt = np.block(
            [
                [sub_gram, ones[:, None]],
                [ones[None, :], np.zeros((1, 1), dtype=np.float64)],
            ]
        )
        right = np.concatenate([linear[support], np.ones(1, dtype=np.float64)])
        try:
            solution = np.linalg.solve(kkt, right)[:-1]
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(kkt, right, rcond=None)[0][:-1]
        if not np.isfinite(solution).all() or np.any(solution < -tolerance):
            continue
        solution = np.maximum(solution, 0.0)
        total = float(solution.sum())
        if total <= _TINY:
            continue
        solution /= total
        weights = np.zeros(n_sources, dtype=np.float64)
        weights[support] = solution
        objective = float(weights @ gram @ weights - 2.0 * linear @ weights)
        if (
            best_objective is None
            or objective < best_objective - 1e-14
            or (
                abs(objective - best_objective) <= 1e-14
                and tuple(weights.tolist()) < tuple(best_weights.tolist())  # type: ignore[union-attr]
            )
        ):
            best_objective = objective
            best_weights = weights
    if best_weights is None:  # pragma: no cover - singleton supports are always feasible
        raise FloatingPointError("simplex quadratic solver found no finite solution")
    return best_weights