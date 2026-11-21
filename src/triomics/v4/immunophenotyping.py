"""Training-only Bimodal GMM Surface Gating and 9-Lineage Human PBMC Flow Immunophenotyping."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, cohen_kappa_score, f1_score, roc_auc_score
from sklearn.mixture import GaussianMixture


def fit_bimodal_1d_gate(x_train: np.ndarray, nonzero_only: bool = True, seed: int = 42) -> tuple[float, float]:
    """Fit a zero-leakage 2-component 1D Gaussian Mixture gate strictly on reference training ADT values.
    Returns (threshold, train_positive_fraction).
    """
    arr = np.asarray(x_train, dtype=np.float64).ravel()
    fit_vals = arr[arr > 1e-4] if nonzero_only and (arr > 1e-4).sum() >= 100 else arr
    if len(fit_vals) < 50 or np.std(fit_vals) < 1e-6:
        thr = float(np.percentile(arr, 85.0))
        return thr, float((arr >= thr).mean())

    gmm = GaussianMixture(n_components=2, covariance_type="full", n_init=5, random_state=seed)
    gmm.fit(fit_vals.reshape(-1, 1))
    means = gmm.means_.ravel()
    low_c, high_c = int(np.argmin(means)), int(np.argmax(means))
    if means[high_c] - means[low_c] < 0.25:
        thr = float(np.percentile(arr, 85.0))
        return thr, float((arr >= thr).mean())

    grid = np.linspace(float(means[low_c]), float(means[high_c]), 500).reshape(-1, 1)
    post = gmm.predict_proba(grid)[:, high_c]
    cross = np.flatnonzero(post >= 0.5)
    thr = float(grid[cross[0], 0]) if len(cross) else float(0.5 * (means[low_c] + means[high_c]))
    pos_frac = float((arr >= thr).mean())
    # Guard against degenerate split if nonzero_only put both components inside positive peak
    if pos_frac > 0.70 or pos_frac < 0.01:
        gmm_all = GaussianMixture(n_components=2, covariance_type="full", n_init=5, random_state=seed).fit(arr.reshape(-1, 1))
        m2 = gmm_all.means_.ravel()
        lo2, hi2 = int(np.argmin(m2)), int(np.argmax(m2))
        g2 = np.linspace(float(m2[lo2]), float(m2[hi2]), 500).reshape(-1, 1)
        p2 = gmm_all.predict_proba(g2)[:, hi2]
        c2 = np.flatnonzero(p2 >= 0.5)
        thr = float(g2[c2[0], 0]) if len(c2) else float(0.5 * (m2[lo2] + m2[hi2]))
        pos_frac = float((arr >= thr).mean())
    return thr, pos_frac


def evaluate_gmm_marker_gates(
    y_train: np.ndarray,
    y_test: np.ndarray,
    pred_dict: dict[str, np.ndarray],
    protein_names: list[str],
    markers: list[str],
) -> list[dict[str, object]]:
    """Evaluate zero-leakage bimodal GMM gates vs legacy broken 75th-percentile gates across canonical & activation markers."""
    name_to_idx = {n: i for i, n in enumerate(protein_names)}
    rows: list[dict[str, object]] = []
    for mk in markers:
        if mk not in name_to_idx:
            continue
        j = name_to_idx[mk]
        tr_col = y_train[:, j]
        te_col = y_test[:, j]
        gmm_thr, gmm_pos_tr = fit_bimodal_1d_gate(tr_col, nonzero_only=(mk in {"CD8", "CD8a", "IgD", "CD19", "CD14", "CD169"}))
        q75_thr = float(np.percentile(tr_col, 75.0))
        y_true_gmm = (te_col >= gmm_thr).astype(int)
        y_true_q75 = (te_col >= q75_thr).astype(int)
        if y_true_gmm.sum() < 5 or (1 - y_true_gmm).sum() < 5:
            continue
        for method, yp in pred_dict.items():
            scores = yp[:, j]
            bin_pred = (scores >= gmm_thr).astype(int)
            rows.append({
                "marker": mk,
                "method": method,
                "gmm_threshold": round(gmm_thr, 4),
                "train_gmm_pos_pct": round(100.0 * gmm_pos_tr, 2),
                "test_gmm_pos_pct": round(100.0 * float(y_true_gmm.mean()), 2),
                "gmm_auroc": round(float(roc_auc_score(y_true_gmm, scores)), 4),
                "gmm_auprc": round(float(average_precision_score(y_true_gmm, scores)), 4),
                "gmm_f1": round(float(f1_score(y_true_gmm, bin_pred, zero_division=0)), 4),
                "legacy_q75_auroc": round(float(roc_auc_score(y_true_q75, scores)), 4),
            })
    return rows


NINE_LINEAGE_CLASSES: tuple[str, ...] = (
    "CD4_Naive_RA",
    "CD4_Mem_RO",
    "CD8_Naive",
    "CD8_EffMem_KLRG1",
    "B_Naive_IgD",
    "B_Mem_CD27",
    "NK_CD56_CD16",
    "Myeloid_Mono_cDC",
    "Other",
)


def _find_channel(protein_names: list[str], candidates: tuple[str, ...]) -> int:
    upper_map = {n.upper(): i for i, n in enumerate(protein_names)}
    for c in candidates:
        if c.upper() in upper_map:
            return upper_map[c.upper()]
    for c in candidates:
        for nm, idx in upper_map.items():
            if nm.startswith(c.upper()):
                return idx
    raise KeyError(f"Could not find any of {candidates} in panel")


def fit_pbmc_9lineage_gates(y_train: np.ndarray, protein_names: list[str]) -> dict[str, object]:
    """Fit hierarchical flow-cytometry 9-lineage PBMC surface gates strictly on reference training ADT."""
    idx = {
        "CD3": _find_channel(protein_names, ("CD3", "CD3-1", "CD3-2")),
        "CD4": _find_channel(protein_names, ("CD4", "CD4-1", "CD4-2")),
        "CD8a": _find_channel(protein_names, ("CD8a", "CD8")),
        "CD19": _find_channel(protein_names, ("CD19",)),
        "CD14": _find_channel(protein_names, ("CD14",)),
        "CD16": _find_channel(protein_names, ("CD16",)),
        "CD56": _find_channel(protein_names, ("CD56", "CD56-1", "NCAM1")),
        "CD11c": _find_channel(protein_names, ("CD11c", "ITGAX", "CD11b", "CD14")),
        "HLADR": _find_channel(protein_names, ("HLA-DR", "HLADR")),
        "CD45RA": _find_channel(protein_names, ("CD45RA",)),
        "CD45RO": _find_channel(protein_names, ("CD45RO",)),
        "CD27": _find_channel(protein_names, ("CD27",)),
        "IgD": _find_channel(protein_names, ("IgD", "IgM")),
        "KLRG1": _find_channel(protein_names, ("KLRG1", "MAFA", "CD57")),
    }
    gates: dict[str, float] = {}
    for k in ("CD3", "CD4", "CD8a", "CD19", "CD14", "CD16", "CD56", "CD11c", "HLADR", "KLRG1"):
        gates[k], _ = fit_bimodal_1d_gate(y_train[:, idx[k]], nonzero_only=(k in {"CD19", "CD14", "CD8a"}))

    # Within CD3+CD4+ T cells: fit RA-RO difference threshold
    t_cd4 = (y_train[:, idx["CD3"]] >= gates["CD3"]) & (y_train[:, idx["CD4"]] >= gates["CD4"])
    ra_minus_ro_cd4 = y_train[:, idx["CD45RA"]] - y_train[:, idx["CD45RO"]]
    gates["CD4_RA_minus_RO"], _ = fit_bimodal_1d_gate(ra_minus_ro_cd4[t_cd4] if t_cd4.sum() >= 50 else ra_minus_ro_cd4, nonzero_only=False)

    # Within CD3+CD8a+ T cells: Naive (CD45RA+ CD27+ KLRG1-) vs Eff/Mem
    t_cd8 = (y_train[:, idx["CD3"]] >= gates["CD3"]) & (y_train[:, idx["CD8a"]] >= gates["CD8a"])
    cd8_naive_score = y_train[:, idx["CD45RA"]] + 0.5 * y_train[:, idx["CD27"]] - y_train[:, idx["CD45RO"]] - 0.5 * y_train[:, idx["KLRG1"]]
    gates["CD8_Naive_Score"], _ = fit_bimodal_1d_gate(cd8_naive_score[t_cd8] if t_cd8.sum() >= 50 else cd8_naive_score, nonzero_only=False)

    # Conditional within CD19+ B cells: Memory CD27 vs Naive IgD (prevents abundant CD27+ T cells from skewing B gate!)
    b_mask = (y_train[:, idx["CD19"]] >= gates["CD19"]) & (y_train[:, idx["CD3"]] < gates["CD3"])
    b_mem_score = y_train[:, idx["CD27"]] - y_train[:, idx["IgD"]]
    gates["B_CD27_minus_IgD"], _ = fit_bimodal_1d_gate(b_mem_score[b_mask] if b_mask.sum() >= 40 else b_mem_score, nonzero_only=False)

    return {
        "indices": idx,
        "gates": gates,
        "train_mean": y_train.mean(axis=0).astype(np.float32),
        "train_std": np.maximum(y_train.std(axis=0), 1e-6).astype(np.float32),
    }


def assign_pbmc_9lineages(adt_mat: np.ndarray, gate_spec: dict[str, object]) -> np.ndarray:
    """Assign each cell to one of NINE_LINEAGE_CLASSES using fixed training-fitted surface gates."""
    idx: dict[str, int] = gate_spec["indices"]  # type: ignore[assignment]
    g: dict[str, float] = gate_spec["gates"]  # type: ignore[assignment]
    n = adt_mat.shape[0]
    labels = np.full(n, 8, dtype=np.int64)  # 8 = Other

    is_b = (adt_mat[:, idx["CD19"]] >= g["CD19"]) & (adt_mat[:, idx["CD3"]] < g["CD3"])
    is_myeloid = (~is_b) & (adt_mat[:, idx["CD3"]] < g["CD3"]) & (
        (adt_mat[:, idx["CD14"]] >= g["CD14"])
        | ((adt_mat[:, idx["CD11c"]] >= g["CD11c"]) & (adt_mat[:, idx["HLADR"]] >= g["HLADR"]))
    )
    is_nk = (~is_b) & (~is_myeloid) & (adt_mat[:, idx["CD3"]] < g["CD3"]) & (
        (adt_mat[:, idx["CD56"]] >= g["CD56"]) | (adt_mat[:, idx["CD16"]] >= g["CD16"])
    )
    is_t = (~is_b) & (~is_myeloid) & (~is_nk) & (adt_mat[:, idx["CD3"]] >= g["CD3"])

    # B subsets
    b_mem = is_b & ((adt_mat[:, idx["CD27"]] - adt_mat[:, idx["IgD"]]) >= g["B_CD27_minus_IgD"])
    labels[is_b & (~b_mem)] = 4  # B_Naive_IgD
    labels[b_mem] = 5            # B_Mem_CD27

    # Myeloid & NK
    labels[is_myeloid] = 7       # Myeloid_Mono_cDC
    labels[is_nk] = 6            # NK_CD56_CD16

    # T subsets: resolve CD4 vs CD8 by normalized excess over gate
    cd4_margin = adt_mat[:, idx["CD4"]] - g["CD4"]
    cd8_margin = adt_mat[:, idx["CD8a"]] - g["CD8a"]
    is_cd4 = is_t & (cd4_margin >= 0.0) & (cd4_margin >= cd8_margin)
    is_cd8 = is_t & (~is_cd4) & (cd8_margin >= -0.15)

    ra_ro = adt_mat[:, idx["CD45RA"]] - adt_mat[:, idx["CD45RO"]]
    labels[is_cd4 & (ra_ro >= g["CD4_RA_minus_RO"])] = 0   # CD4_Naive_RA
    labels[is_cd4 & (ra_ro < g["CD4_RA_minus_RO"])] = 1    # CD4_Mem_RO

    cd8_ns = adt_mat[:, idx["CD45RA"]] + 0.5 * adt_mat[:, idx["CD27"]] - adt_mat[:, idx["CD45RO"]] - 0.5 * adt_mat[:, idx["KLRG1"]]
    labels[is_cd8 & (cd8_ns >= g["CD8_Naive_Score"])] = 2  # CD8_Naive
    labels[is_cd8 & (cd8_ns < g["CD8_Naive_Score"])] = 3   # CD8_EffMem_KLRG1

    return labels


def calibrate_variance_shrinkage(
    pred_test: np.ndarray, pred_oof: np.ndarray | None, train_mean: np.ndarray, train_std: np.ndarray
) -> np.ndarray:
    """Protocol 2: training-only OOF variance rescaling (compensating linear shrinkage without touching test labels)."""
    ref = pred_oof if pred_oof is not None else pred_test
    mu_p = ref.mean(axis=0, keepdims=True)
    sd_p = np.maximum(ref.std(axis=0, keepdims=True), 1e-4)
    return ((pred_test - mu_p) / sd_p * train_std.reshape(1, -1) + train_mean.reshape(1, -1)).astype(np.float32)


def score_9lineage_predictions(
    y_test: np.ndarray,
    pred_test: np.ndarray,
    gate_spec: dict[str, object],
    pred_oof: np.ndarray | None = None,
) -> dict[str, object]:
    """Evaluate both Protocol 1 (Direct Reference Gates) and Protocol 2 (OOF Shrinkage-Calibrated Gates)."""
    true_cls = assign_pbmc_9lineages(y_test, gate_spec)
    p1_cls = assign_pbmc_9lineages(pred_test, gate_spec)
    cal_pred = calibrate_variance_shrinkage(
        pred_test, pred_oof, gate_spec["train_mean"], gate_spec["train_std"]  # type: ignore[arg-type]
    )
    p2_cls = assign_pbmc_9lineages(cal_pred, gate_spec)

    present_8 = [c for c in range(8) if (true_cls == c).sum() > 0]
    present_9 = [c for c in range(9) if (true_cls == c).sum() > 0]
    return {
        "n_test": int(len(true_cls)),
        "other_true_pct": round(100.0 * float((true_cls == 8).mean()), 2),
        "p1_correct": int((p1_cls == true_cls).sum()),
        "p1_accuracy_pct": round(100.0 * float(accuracy_score(true_cls, p1_cls)), 2),
        "p1_macro_f1_8lin": round(float(f1_score(true_cls, p1_cls, labels=present_8, average="macro", zero_division=0)), 4),
        "p1_macro_f1_9all": round(float(f1_score(true_cls, p1_cls, labels=present_9, average="macro", zero_division=0)), 4),
        "p1_kappa": round(float(cohen_kappa_score(true_cls, p1_cls)), 4),
        "p2_correct": int((p2_cls == true_cls).sum()),
        "p2_accuracy_pct": round(100.0 * float(accuracy_score(true_cls, p2_cls)), 2),
        "p2_macro_f1_8lin": round(float(f1_score(true_cls, p2_cls, labels=present_8, average="macro", zero_division=0)), 4),
        "p2_kappa": round(float(cohen_kappa_score(true_cls, p2_cls)), 4),
        "true_labels": true_cls,
        "p1_labels": p1_cls,
        "p2_labels": p2_cls,
    }
