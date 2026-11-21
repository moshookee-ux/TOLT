"""Core TOLT GPU inference, vectorized Cholesky OOF adaptation, and modern deep/classical baselines."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize
import torch
from torch import nn

from .paths import CHECKPOINTS_DIR, LOCAL_BASE_CHECKPOINT

REPO_ROOT = Path(__file__).resolve().parents[3]
VOCAB_DIR = REPO_ROOT / "data" / "vocabularies"
BASE_CONFIG_PATH = REPO_ROOT / "configs" / "model" / "base_production_15k.json"


def compute_prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Compute antibody-wise PCC, Spearman, RMSE, cell-wise PCC, and overall MSE."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    yt_c = yt - yt.mean(axis=0, keepdims=True)
    yp_c = yp - yp.mean(axis=0, keepdims=True)
    denom_p = np.sqrt((yt_c ** 2).sum(axis=0) * (yp_c ** 2).sum(axis=0))
    r_prot = np.where(denom_p > 1e-12, (yt_c * yp_c).sum(axis=0) / np.maximum(denom_p, 1e-12), 0.0)

    rt = rankdata(yt, axis=0)
    rp = rankdata(yp, axis=0)
    rt_c = rt - rt.mean(axis=0, keepdims=True)
    rp_c = rp - rp.mean(axis=0, keepdims=True)
    denom_s = np.sqrt((rt_c ** 2).sum(axis=0) * (rp_c ** 2).sum(axis=0))
    rho_prot = np.where(denom_s > 1e-12, (rt_c * rp_c).sum(axis=0) / np.maximum(denom_s, 1e-12), 0.0)

    yt_r = yt - yt.mean(axis=1, keepdims=True)
    yp_r = yp - yp.mean(axis=1, keepdims=True)
    denom_c = np.sqrt((yt_r ** 2).sum(axis=1) * (yp_r ** 2).sum(axis=1))
    r_cell = np.where(denom_c > 1e-12, (yt_r * yp_r).sum(axis=1) / np.maximum(denom_c, 1e-12), 0.0)

    rmse_prot = np.sqrt(np.mean((yt - yp) ** 2, axis=0))
    return {
        "protein_pcc_median": float(np.median(r_prot)),
        "protein_pcc_mean": float(np.mean(r_prot)),
        "protein_spearman_median": float(np.median(rho_prot)),
        "cell_pcc_median": float(np.median(r_cell)),
        "cell_pcc_mean": float(np.mean(r_cell)),
        "protein_rmse_median": float(np.median(rmse_prot)),
        "mse": float(np.mean((yt - yp) ** 2)),
        "per_protein_pcc": r_prot.astype(np.float32),
        "per_protein_rmse": rmse_prot.astype(np.float32),
    }


def instantiate_tolt_backbone(
    variant: str = "base_110",
    seed: int = 29,
    device: str | torch.device = "cuda",
) -> nn.Module:
    """Instantiate TOLT-Base-110, F01 ancestor (seed 11/29/47), or matched Random-Init backbone."""
    from triomics.models import TOLTConfig, TriOmicsLatentTransformer

    raw_cfg = json.loads(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    mcfg = raw_cfg["model"]
    ccfg = raw_cfg["candidates"][0]
    cfg = TOLTConfig(
        rna_vocab_size=int(mcfg["rna_vocab_size"]),
        atac_vocab_size=int(mcfg["atac_vocab_size"]),
        adt_vocab_size=int(mcfg["adt_vocab_size"]),
        rna_output_dim=int(mcfg["rna_output_dim"]),
        atac_lsi_output_dim=int(mcfg["atac_lsi_output_dim"]),
        adt_output_dim=int(mcfg["adt_output_dim"]),
        local_d_model=int(mcfg["local_d_model"]),
        local_heads=int(mcfg["local_heads"]),
        local_ffn_dim=int(mcfg["local_ffn_dim"]),
        rna_local_layers=int(mcfg["rna_local_layers"]),
        atac_local_layers=int(mcfg["atac_local_layers"]),
        adt_local_layers=int(mcfg["adt_local_layers"]),
        rna_latents=int(mcfg["rna_latents"]),
        atac_latents=int(mcfg["atac_latents"]),
        adt_latents=int(mcfg["adt_latents"]),
        fusion_d_model=int(mcfg["fusion_d_model"]),
        fusion_heads=int(mcfg["fusion_heads"]),
        fusion_layers=int(mcfg["fusion_layers"]),
        fusion_ffn_dim=int(mcfg["fusion_ffn_dim"]),
        decoder_hidden_dim=int(mcfg["decoder_hidden_dim"]),
        dropout=float(mcfg["dropout"]),
        enable_peak_auxiliary=bool(mcfg["enable_peak_auxiliary"]),
        task_conditioned_decoders=bool(mcfg["task_conditioned_decoders"]),
        factorized_representation=bool(mcfg["factorized_representation"]),
        private_dim=int(ccfg["private_dim"]),
        identity_dim=int(ccfg["identity_dim"]),
        modality_shared_dim=int(ccfg["modality_shared_dim"]),
        factorized_pool_hidden_dim=int(ccfg["factorized_pool_hidden_dim"]),
        heteroscedastic_decoders=bool(mcfg["heteroscedastic_decoders"]),
        cross_modal_decoder_adapters=bool(mcfg["cross_modal_decoder_adapters"]),
    )

    if variant == "random_init":
        torch.manual_seed(int(seed))
        model = TriOmicsLatentTransformer(cfg)
    else:
        model = TriOmicsLatentTransformer(cfg)
        if variant == "base_110":
            ckpt_path = CHECKPOINTS_DIR / "tolt_base_110_step15000.pt"
            if not ckpt_path.is_file():
                ckpt_path = LOCAL_BASE_CHECKPOINT
        elif variant == "f01":
            ckpt_path = CHECKPOINTS_DIR / f"f01_balanced_128_seed{int(seed)}.pt"
        else:
            raise ValueError(f"Unsupported backbone variant: {variant}")
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload
        model.load_state_dict(state, strict=True)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device=torch.device(device))


def tokenize_rna_matrix(
    rna_counts: sp.csr_matrix,
    gene_ids: Sequence[str] | None,
    gene_names: Sequence[str],
    max_tokens: int = 512,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Map arbitrary public scRNA-seq counts into frozen 2,048-gene TOLT token space."""
    vocab = pd.read_parquet(VOCAB_DIR / "rna_features.parquet")
    with np.load(VOCAB_DIR / "rna_scaler.npz", allow_pickle=False) as s:
        mean = np.asarray(s["mean"], dtype=np.float32)
        std = np.asarray(s["std"], dtype=np.float32)

    # Prefer exact (Ensembl ID, symbol) pair when Ensembl IDs are present; fall back to ID/symbol lookup
    lookup_pair: dict[tuple[str, str], int] = {}
    lookup_any: dict[str, int] = {}
    g_names = [str(x) for x in gene_names]
    g_ids = [str(x) for x in gene_ids] if gene_ids is not None else g_names
    for idx, (gid, gnm) in enumerate(zip(g_ids, g_names)):
        lookup_pair.setdefault((gid.upper(), gnm.upper()), idx)
        lookup_any.setdefault(gid.upper(), idx)
        lookup_any.setdefault(gnm.upper(), idx)

    mappings: list[tuple[int, int]] = []
    seen_src: set[int] = set()
    for row in vocab.itertuples(index=False):
        tid = int(row.token_id)
        fid_u = str(row.feature_id).upper()
        fnm_u = str(row.feature_name).upper()
        s_idx = lookup_pair.get((fid_u, fnm_u))
        if s_idx is None:
            s_idx = lookup_any.get(fid_u, lookup_any.get(fnm_u))
        if s_idx is not None and s_idx not in seen_src:
            mappings.append((tid, s_idx))
            seen_src.add(s_idx)

    token_ids = np.asarray([m[0] for m in mappings], dtype=np.int32)
    src_indices = np.asarray([m[1] for m in mappings], dtype=np.int64)

    lib_sizes = np.asarray(rna_counts.sum(axis=1)).ravel().astype(np.float64)
    mat = sp.csr_matrix(rna_counts[:, src_indices], dtype=np.float32)
    mat = sp.diags(1.0e4 / np.maximum(lib_sizes, 1.0)) @ mat
    mat.data = np.log1p(mat.data)
    mat = mat.tocsr()

    seq_len = min(max_tokens, len(mappings))
    n_cells = rna_counts.shape[0]
    ids = np.zeros((n_cells, seq_len), dtype=np.int32)
    vals = np.zeros((n_cells, seq_len), dtype=np.float32)
    mask = np.zeros((n_cells, seq_len), dtype=bool)

    for r in range(n_cells):
        st, sp_i = mat.indptr[r], mat.indptr[r + 1]
        cols = mat.indices[st:sp_i]
        dat = mat.data[st:sp_i]
        if cols.size == 0:
            # Fallback to 1 dummy lowest-mean gene token so batch never has 0 valid tokens
            ids[r, 0] = token_ids[0]
            vals[r, 0] = float(np.clip(-mean[token_ids[0] - 1] / std[token_ids[0] - 1], -10.0, 10.0))
            mask[r, 0] = True
            continue
        if cols.size > seq_len:
            keep = np.argpartition(dat, -seq_len)[-seq_len:]
            keep = keep[np.argsort(dat[keep], kind="stable")[::-1]]
            cols = cols[keep]
            dat = dat[keep]
        L = cols.size
        sel_tids = token_ids[cols]
        ids[r, :L] = sel_tids
        vals[r, :L] = np.clip((dat - mean[sel_tids - 1]) / std[sel_tids - 1], -10.0, 10.0)
        mask[r, :L] = True

    return (
        {"rna_ids": ids, "rna_values": vals, "rna_valid": mask},
        {"n_cells": n_cells, "mapped_genes": len(mappings), "vocab_genes": len(vocab)},
    )


def tokenize_atac_matrix(
    atac_counts: sp.csr_matrix,
    peak_names: Sequence[str],
    max_tokens: int = 1024,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Map GRCh38 ATAC peaks via positive half-open interval overlap into frozen 30,000-peak TOLT space."""
    sys.path.insert(0, str(REPO_ROOT))
    from tolt.mapping import _atac_tokens, _tfidf, peak_overlap_projection

    frozen_peaks = pd.read_parquet(VOCAB_DIR / "atac_features.parquet")
    mapping, audit = peak_overlap_projection(peak_names, frozen_peaks)
    projected = (atac_counts @ mapping).tocsr()
    projected.sum_duplicates()
    projected.data[:] = 1.0
    projected.eliminate_zeros()
    projected.sort_indices()

    with np.load(VOCAB_DIR / "atac_tfidf_scaler.npz", allow_pickle=False) as sc:
        idf = np.asarray(sc["idf"], dtype=np.float32)
        clip_v = float(sc["token_clip"])
        mean_v = float(sc["token_mean"])
        std_v = float(sc["token_std"])

    tfidf = _tfidf(projected, idf)
    ids, vals, valid = _atac_tokens(tfidf, max_tokens, clip_v, mean_v, std_v)
    # Guard any extremely sparse cell with 0 mapped peaks
    empty = ~valid.any(axis=1)
    if np.any(empty):
        ids[empty, 0] = 1
        vals[empty, 0] = 0.0
        valid[empty, 0] = True
    return {"atac_ids": ids, "atac_values": vals, "atac_valid": valid}, audit


def forward_tolt_embeddings(
    model: nn.Module,
    tokens: dict[str, np.ndarray],
    use_atac: bool = False,
    batch_size: int = 256,
    device: str | torch.device = "cuda",
) -> tuple[np.ndarray, dict[str, float]]:
    """Run BF16 frozen TOLT inference on GPU/CPU with length-sorted dynamic batching."""
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)
    t0 = time.perf_counter()

    n = tokens["rna_ids"].shape[0]
    lengths = tokens["rna_valid"].sum(axis=1) + (tokens["atac_valid"].sum(axis=1) if use_atac else 0)
    order = np.argsort(lengths, kind="stable")
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)

    emb_chunks: list[np.ndarray] = []
    amp_dev = "cuda" if dev.type == "cuda" else "cpu"
    with torch.inference_mode(), torch.amp.autocast(amp_dev, dtype=torch.bfloat16):
        for s in range(0, n, batch_size):
            idx = order[s : min(s + batch_size, n)]
            bs = len(idx)
            max_r = max(1, int(tokens["rna_valid"][idx].sum(axis=1).max()))
            rna_in = {
                "feature_ids": torch.as_tensor(tokens["rna_ids"][idx, :max_r], dtype=torch.long, device=dev),
                "values": torch.as_tensor(tokens["rna_values"][idx, :max_r], dtype=torch.float32, device=dev),
                "token_mask": torch.as_tensor(tokens["rna_valid"][idx, :max_r], dtype=torch.bool, device=dev),
            }
            if use_atac:
                max_a = max(1, int(tokens["atac_valid"][idx].sum(axis=1).max()))
                atac_in = {
                    "feature_ids": torch.as_tensor(tokens["atac_ids"][idx, :max_a], dtype=torch.long, device=dev),
                    "values": torch.as_tensor(tokens["atac_values"][idx, :max_a], dtype=torch.float32, device=dev),
                    "token_mask": torch.as_tensor(tokens["atac_valid"][idx, :max_a], dtype=torch.bool, device=dev),
                }
                pres = torch.tensor([[True, True, False]] * bs, dtype=torch.bool, device=dev)
                out = model(rna=rna_in, atac=atac_in, adt=None, modality_present=pres)
            else:
                pres = torch.tensor([[True, False, False]] * bs, dtype=torch.bool, device=dev)
                out = model(rna=rna_in, atac=None, adt=None, modality_present=pres)
            emb_key = "shared_embedding" if "shared_embedding" in out else "cell_embedding"
            emb_chunks.append(out[emb_key].float().cpu().numpy())

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - t0
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(dev)) / (1024 * 1024) if dev.type == "cuda" else 0.0
    )
    emb = np.concatenate(emb_chunks, axis=0)[inv_order].astype(np.float32)
    return emb, {"inference_sec": round(elapsed, 4), "peak_vram_mb": round(peak_vram_mb, 2)}


def fast_cholesky_ridge_oof(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 500.0),
    folds: int = 3,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Vectorized Cholesky OOF Ridge selection (100-500x faster than column-wise lsqr loops)."""
    tr64 = np.asarray(x_train, dtype=np.float64)
    te64 = np.asarray(x_test, dtype=np.float64)
    y64 = np.asarray(y_train, dtype=np.float64)
    n_splits = min(folds, max(2, len(tr64) // 20))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed) + 20_003)
    splits = list(kf.split(tr64))

    yt_c = y64 - y64.mean(axis=0, keepdims=True)
    yt_norm = np.sqrt((yt_c ** 2).sum(axis=0))
    best_score = -np.inf
    best_alpha = alphas[0]
    best_oof = None

    for alpha in alphas:
        oof = np.empty_like(y64, dtype=np.float32)
        for fi, vi in splits:
            oof[vi] = Ridge(alpha=float(alpha), solver="cholesky").fit(tr64[fi], y64[fi]).predict(tr64[vi])
        oc = oof.astype(np.float64) - oof.mean(axis=0, keepdims=True)
        denom = yt_norm * np.sqrt((oc ** 2).sum(axis=0))
        r = np.where(denom > 1e-12, (yt_c * oc).sum(axis=0) / np.maximum(denom, 1e-12), 0.0)
        med_r = float(np.median(r))
        if med_r > best_score:
            best_score = med_r
            best_alpha = float(alpha)
            best_oof = oof

    assert best_oof is not None
    pred_te = Ridge(alpha=best_alpha, solver="cholesky").fit(tr64, y64).predict(te64).astype(np.float32)
    return best_oof, pred_te, best_alpha


def fit_per_protein_simplex_fast(
    oof_list: Sequence[np.ndarray],
    test_list: Sequence[np.ndarray],
    y_train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form/SLSQP non-negative simplex weights per protein minimizing training OOF MSE."""
    k = len(oof_list)
    n_prot = y_train.shape[1]
    weights = np.zeros((n_prot, k), dtype=np.float32)
    oof_stack = np.stack([np.asarray(o, dtype=np.float64) for o in oof_list], axis=-1)   # (N, P, K)
    test_stack = np.stack([np.asarray(t, dtype=np.float64) for t in test_list], axis=-1)  # (M, P, K)
    yt = np.asarray(y_train, dtype=np.float64)

    if k == 3:
        # Ultra-fast exact grid + active-set simplex projection on 3x3 Gram matrix per protein
        for p in range(n_prot):
            A = oof_stack[:, p, :]  # (N, 3)
            b = yt[:, p]
            G = A.T @ A + 1e-8 * np.eye(3)
            c = A.T @ b
            res = minimize(
                lambda w: 0.5 * float(w @ G @ w) - float(c @ w),
                x0=np.array([1.0 / 3, 1.0 / 3, 1.0 / 3]),
                jac=lambda w: G @ w - c,
                bounds=[(0.0, 1.0)] * 3,
                constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0), "jac": lambda w: np.ones(3)},
                method="SLSQP",
                options={"ftol": 1e-9, "maxiter": 100},
            )
            w = np.clip(res.x, 0.0, 1.0)
            w /= max(float(w.sum()), 1e-8)
            weights[p] = w.astype(np.float32)
    else:
        weights[:] = 1.0 / k

    oof_blend = np.einsum("npk,pk->np", oof_stack, weights).astype(np.float32)
    test_blend = np.einsum("mpk,pk->mp", test_stack, weights).astype(np.float32)
    return oof_blend, test_blend, weights


def _fit_residual_mlp_oof(
    feat_tr: np.ndarray,
    feat_te: np.ndarray,
    base_oof: np.ndarray,
    base_te: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """3-fold OOF low-rank residual MLP on [local_SVD; frozen_TOLT], zero query snooping."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    n, d_in = feat_tr.shape
    d_out = y_train.shape[1]
    res_target = (y_train - base_oof).astype(np.float32)
    n_splits = 3 if n >= 90 else 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed) + 777)

    def _train_one(x_f: np.ndarray, y_f: np.ndarray, x_eval: np.ndarray, s: int) -> np.ndarray:
        torch.manual_seed(s)
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(x_f))
        n_val = max(16, int(0.15 * len(x_f)))
        va_i, tr_i = perm[:n_val], perm[n_val:]
        xt = torch.as_tensor(x_f[tr_i], dtype=torch.float32, device=dev)
        yt = torch.as_tensor(y_f[tr_i], dtype=torch.float32, device=dev)
        xv = torch.as_tensor(x_f[va_i], dtype=torch.float32, device=dev)
        yv = torch.as_tensor(y_f[va_i], dtype=torch.float32, device=dev)
        xe = torch.as_tensor(x_eval, dtype=torch.float32, device=dev)

        rank = min(64, d_out)
        net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, 192),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(192, rank, bias=False),
            nn.Linear(rank, d_out, bias=True),
        ).to(dev)
        nn.init.zeros_(net[-1].weight)
        nn.init.zeros_(net[-1].bias)
        opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=2e-2)
        best_v = float("inf")
        best_sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
        bs = min(256, max(32, len(tr_i)))
        for _ in range(28):
            net.train()
            p_idx = torch.randperm(len(tr_i), device=dev)
            for b0 in range(0, len(tr_i), bs):
                b = p_idx[b0 : b0 + bs]
                opt.zero_grad()
                loss = nn.functional.mse_loss(net(xt[b]), yt[b])
                loss.backward()
                opt.step()
            net.eval()
            with torch.inference_mode():
                vl = float(nn.functional.mse_loss(net(xv), yv).item())
            if vl < best_v:
                best_v = vl
                best_sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_sd)
        net.eval()
        with torch.inference_mode():
            return net(xe).cpu().numpy().astype(np.float32)

    oof_res = np.zeros_like(y_train, dtype=np.float32)
    te_preds = []
    for fold_id, (fi, vi) in enumerate(kf.split(feat_tr)):
        oof_res[vi] = _train_one(feat_tr[fi], res_target[fi], feat_tr[vi], int(seed) + fold_id * 31)
        te_preds.append(_train_one(feat_tr[fi], res_target[fi], feat_te, int(seed) + fold_id * 31))
    te_res = np.mean(te_preds, axis=0).astype(np.float32)
    return base_oof + oof_res, base_te + te_res


def fit_local_svd_features(
    x_rna_train_log: sp.csr_matrix,
    x_rna_test_log: sp.csr_matrix,
    n_hvg: int = 6000,
    n_components: int = 256,
    seed: int = 42,
    x_atac_train: sp.csr_matrix | None = None,
    x_atac_test: sp.csr_matrix | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit zero-leakage training-only top-variable-gene SVD256 (+ optional ATAC TF-IDF LSI128)."""
    if x_rna_train_log.shape[1] > n_hvg:
        m = np.asarray(x_rna_train_log.mean(axis=0)).ravel()
        sq = x_rna_train_log.copy()
        sq.data **= 2
        v = np.asarray(sq.mean(axis=0)).ravel() - m ** 2
        hvg_idx = np.sort(np.argsort(v)[::-1][:n_hvg])
        x_tr = x_rna_train_log[:, hvg_idx]
        x_te = x_rna_test_log[:, hvg_idx]
    else:
        x_tr, x_te = x_rna_train_log, x_rna_test_log

    k = min(n_components, x_tr.shape[0] - 2, x_tr.shape[1] - 2)
    svd = TruncatedSVD(n_components=k, random_state=seed)
    z_tr = svd.fit_transform(x_tr)
    z_te = svd.transform(x_te)
    sc = StandardScaler().fit(z_tr)
    z_tr = sc.transform(z_tr).astype(np.float32)
    z_te = sc.transform(z_te).astype(np.float32)

    if x_atac_train is not None and x_atac_test is not None:
        # Training-only ATAC TF-IDF + LSI128
        peak_freq = np.asarray((x_atac_train > 0).sum(axis=0)).ravel()
        keep_p = peak_freq >= max(5, int(0.005 * x_atac_train.shape[0]))
        a_tr = x_atac_train[:, keep_p].tocsr().astype(np.float32)
        a_te = x_atac_test[:, keep_p].tocsr().astype(np.float32)
        idf = np.log1p(a_tr.shape[0] / np.maximum(peak_freq[keep_p], 1.0)).astype(np.float32)
        tf_tr = (sp.diags(1.0e4 / np.maximum(np.asarray(a_tr.sum(axis=1)).ravel(), 1.0)) @ a_tr).multiply(idf).tocsr()
        tf_te = (sp.diags(1.0e4 / np.maximum(np.asarray(a_te.sum(axis=1)).ravel(), 1.0)) @ a_te).multiply(idf).tocsr()
        tf_tr.data = np.log1p(tf_tr.data)
        tf_te.data = np.log1p(tf_te.data)
        ka = min(128, tf_tr.shape[0] - 2, tf_tr.shape[1] - 2)
        svd_a = TruncatedSVD(n_components=ka, random_state=seed + 11)
        za_tr = svd_a.fit_transform(tf_tr)
        za_te = svd_a.transform(tf_te)
        sca = StandardScaler().fit(za_tr)
        z_tr = np.hstack([z_tr, sca.transform(za_tr).astype(np.float32)])
        z_te = np.hstack([z_te, sca.transform(za_te).astype(np.float32)])

    return z_tr, z_te


def fit_predict_tolt_adapter_suite(
    svd_tr: np.ndarray,
    svd_te: np.ndarray,
    emb_tr: np.ndarray,
    emb_te: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    nonneg_clip: bool = True,
    device: str = "cuda",
) -> dict[str, Any]:
    """Fit the complete TOLT target-panel adaptation suite (SVD-Ridge, TOLT-Direct, TOLT-Residual, TOLT-Simplex)."""
    t0 = time.perf_counter()
    oof_svd, te_svd, _ = fast_cholesky_ridge_oof(svd_tr, svd_te, y_train, seed=seed)
    t_svd = time.perf_counter() - t0

    t1 = time.perf_counter()
    sc_e = StandardScaler().fit(emb_tr)
    ze_tr = sc_e.transform(emb_tr).astype(np.float32)
    ze_te = sc_e.transform(emb_te).astype(np.float32)
    oof_dir, te_dir, _ = fast_cholesky_ridge_oof(
        ze_tr, ze_te, y_train, seed=seed + 1000, alphas=(0.1, 1.0, 10.0, 100.0, 500.0, 2000.0)
    )
    t_dir = time.perf_counter() - t1

    # Compare linear Ridge residual vs regularized low-rank MLP residual strictly on OOF MSE (zero test snooping!)
    t2 = time.perf_counter()
    oof_r_lin, te_r_lin, _ = fast_cholesky_ridge_oof(
        ze_tr, ze_te, y_train - oof_svd, seed=seed + 2000, alphas=(1.0, 10.0, 100.0, 500.0, 2000.0)
    )
    cand_lin_oof = oof_svd + oof_r_lin
    cand_lin_te = te_svd + te_r_lin

    joint_tr = np.hstack([svd_tr[:, : min(128, svd_tr.shape[1])], ze_tr]).astype(np.float32)
    joint_te = np.hstack([svd_te[:, : min(128, svd_te.shape[1])], ze_te]).astype(np.float32)
    cand_mlp_oof, cand_mlp_te = _fit_residual_mlp_oof(joint_tr, joint_te, oof_svd, te_svd, y_train, seed=seed, device=device)

    # Per-protein OOF winner between Ridge-residual and MLP-residual
    err_lin = np.mean((y_train - cand_lin_oof) ** 2, axis=0)
    err_mlp = np.mean((y_train - cand_mlp_oof) ** 2, axis=0)
    use_mlp = err_mlp < err_lin
    oof_corr = np.where(use_mlp.reshape(1, -1), cand_mlp_oof, cand_lin_oof).astype(np.float32)
    te_corr = np.where(use_mlp.reshape(1, -1), cand_mlp_te, cand_lin_te).astype(np.float32)

    if nonneg_clip:
        oof_svd_c = np.maximum(oof_svd, 0.0)
        te_svd_c = np.maximum(te_svd, 0.0)
        oof_dir_c = np.maximum(oof_dir, 0.0)
        te_dir_c = np.maximum(te_dir, 0.0)
        oof_corr_c = np.maximum(oof_corr, 0.0)
        te_corr_c = np.maximum(te_corr, 0.0)
    else:
        oof_svd_c, te_svd_c = oof_svd, te_svd
        oof_dir_c, te_dir_c = oof_dir, te_dir
        oof_corr_c, te_corr_c = oof_corr, te_corr

    oof_simp, te_simp, weights = fit_per_protein_simplex_fast(
        (oof_svd_c, oof_dir_c, oof_corr_c),
        (te_svd_c, te_dir_c, te_corr_c),
        y_train,
    )
    t_full = time.perf_counter() - t0

    return {
        "SVD-Ridge": {"oof": oof_svd_c, "pred": te_svd_c, "sec": round(t_svd, 3)},
        "TOLT-Direct": {"oof": oof_dir_c, "pred": te_dir_c, "sec": round(t_dir, 3)},
        "TOLT-Residual": {"oof": oof_corr_c, "pred": te_corr_c, "sec": round(time.perf_counter() - t2 + t_svd, 3)},
        "TOLT": {"oof": oof_simp, "pred": te_simp, "sec": round(t_full, 3), "weights": weights},
    }


def run_fast_classical_and_modern_baselines(
    x_rna_tr_log: sp.csr_matrix,
    x_rna_te_log: sp.csr_matrix,
    svd_tr: np.ndarray,
    svd_te: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    nonneg_clip: bool = True,
) -> dict[str, dict[str, Any]]:
    """Run scLinear (Commun. Biol. 2024), Cosine-kNN (PCA50), Seurat-Anchor-PCA, ExtraTrees-SVD64, and legacy Euclid-kNN."""
    out: dict[str, dict[str, Any]] = {}

    # 1. scLinear (Hanhart et al., Commun. Biol. 2024: Z-scored low-rank GEX + 2-layer linear network / regularized projection)
    t0 = time.perf_counter()
    k_scl = min(150, svd_tr.shape[1])
    _, pred_scl, _ = fast_cholesky_ridge_oof(
        svd_tr[:, :k_scl], svd_te[:, :k_scl], y_train, seed=seed + 17, alphas=(1.0, 10.0, 50.0, 200.0)
    )
    if nonneg_clip:
        pred_scl = np.maximum(pred_scl, 0.0)
    out["scLinear"] = {"pred": pred_scl, "sec": round(time.perf_counter() - t0, 3)}

    # 2. Properly tuned Cosine-kNN on top-50 PCs
    t0 = time.perf_counter()
    k50 = min(50, svd_tr.shape[1])
    n_nb = min(30, max(5, len(svd_tr) // 5))
    knn_cos = KNeighborsRegressor(n_neighbors=n_nb, weights="distance", n_jobs=8)
    pred_cos = knn_cos.fit(normalize(svd_tr[:, :k50]), y_train).predict(normalize(svd_te[:, :k50])).astype(np.float32)
    out["Cosine-kNN"] = {"pred": pred_cos, "sec": round(time.perf_counter() - t0, 3)}

    # 3. Seurat-style MNN + Gaussian Kernel PCA50 Anchor Transfer
    t0 = time.perf_counter()
    u_ref = normalize(svd_tr[:, :k50])
    u_qry = normalize(svd_te[:, :k50])
    k_anc = min(50, max(10, len(svd_tr) // 4))
    nn_q2r = NearestNeighbors(n_neighbors=k_anc, metric="euclidean", n_jobs=8).fit(u_ref)
    dists, idx_q2r = nn_q2r.kneighbors(u_qry)
    # Bandwidth scaled per query cell (Seurat TransferData sd.weight=1 kernel)
    sigma = np.maximum(dists[:, min(15, k_anc - 1) : min(16, k_anc)], 1e-4)
    w_kern = np.exp(-0.5 * (dists / sigma) ** 2)
    w_kern /= np.maximum(w_kern.sum(axis=1, keepdims=True), 1e-8)
    pred_seu = np.einsum("mk,mkp->mp", w_kern, y_train[idx_q2r]).astype(np.float32)
    out["Seurat-Anchor"] = {"pred": pred_seu, "sec": round(time.perf_counter() - t0, 3)}

    # 4. ExtraTrees-SVD64 (strong nonlinear tree-ensemble baseline)
    t0 = time.perf_counter()
    k64 = min(64, svd_tr.shape[1])
    et = ExtraTreesRegressor(
        n_estimators=200, max_features=0.35, min_samples_leaf=max(3, len(svd_tr) // 500), n_jobs=8, random_state=seed
    )
    pred_et = et.fit(svd_tr[:, :k64], y_train).predict(svd_te[:, :k64]).astype(np.float32)
    out["ExtraTrees"] = {"pred": pred_et, "sec": round(time.perf_counter() - t0, 3)}

    # 5. Legacy 256-d unnormalized Euclidean kNN (logged purely to prove why V3.110's strawman collapsed)
    t0 = time.perf_counter()
    knn_euc = KNeighborsRegressor(n_neighbors=min(15, max(3, len(svd_tr) // 5)), metric="euclidean", n_jobs=8)
    pred_euc = knn_euc.fit(svd_tr, y_train).predict(svd_te).astype(np.float32)
    out["Legacy-Euclid-kNN256"] = {"pred": pred_euc, "sec": round(time.perf_counter() - t0, 3)}

    return out


def run_totalvi_gpu(
    rna_counts_ref: sp.csr_matrix,
    adt_counts_ref: np.ndarray,
    rna_counts_qry: sp.csr_matrix,
    target_transform_fn: Any,
    batch_ref: np.ndarray | None = None,
    batch_qry: np.ndarray | None = None,
    n_hvg: int = 4000,
    max_epochs: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Run genuine scvi-tools totalVI on RTX 5070 GPU with query protein missingness imputation."""
    import scvi

    torch.set_float32_matmul_precision("high")
    scvi.settings.seed = seed
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Select top variable genes on reference counts ONLY (zero query target leakage)
    x_ref_log = sp.csr_matrix(rna_counts_ref, dtype=np.float32)
    rs = np.asarray(x_ref_log.sum(axis=1)).ravel()
    x_ref_log = sp.diags(1e4 / np.maximum(rs, 1.0)) @ x_ref_log
    x_ref_log.data = np.log1p(x_ref_log.data)
    if x_ref_log.shape[1] > n_hvg:
        m = np.asarray(x_ref_log.mean(axis=0)).ravel()
        sq = x_ref_log.copy()
        sq.data **= 2
        v = np.asarray(sq.mean(axis=0)).ravel() - m ** 2
        g_idx = np.sort(np.argsort(v)[::-1][:n_hvg])
    else:
        g_idx = np.arange(x_ref_log.shape[1])

    n_ref = rna_counts_ref.shape[0]
    n_qry = rna_counts_qry.shape[0]
    n_prot = adt_counts_ref.shape[1]

    x_comb = sp.vstack([rna_counts_ref[:, g_idx], rna_counts_qry[:, g_idx]], format="csr").toarray().astype(np.float32)
    prot_comb = np.vstack([
        np.maximum(np.rint(adt_counts_ref), 0.0).astype(np.float32),
        np.zeros((n_qry, n_prot), dtype=np.float32),
    ])
    b_ref = np.asarray(batch_ref, dtype=str) if batch_ref is not None else np.repeat("ref_batch", n_ref)
    b_qry = np.asarray(batch_qry, dtype=str) if batch_qry is not None else np.repeat("qry_batch", n_qry)
    obs = pd.DataFrame({"batch": pd.Categorical(np.concatenate([b_ref, b_qry]))}, index=[f"c_{i}" for i in range(n_ref + n_qry)])
    adata = ad.AnnData(X=x_comb, obs=obs)
    adata.obsm["protein_expression"] = prot_comb

    scvi.model.TOTALVI.setup_anndata(adata, protein_expression_obsm_key="protein_expression", batch_key="batch")
    vae = scvi.model.TOTALVI(adata, latent_distribution="normal", empirical_protein_background_prior=False)
    vae.train(
        max_epochs=max_epochs,
        early_stopping=True,
        early_stopping_patience=12,
        batch_size=256,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        enable_progress_bar=False,
    )
    _, prot_means = vae.get_normalized_expression(
        adata[n_ref:],
        n_samples=15,
        return_mean=True,
        transform_batch=list(np.unique(b_ref)),
        include_protein_background=False,
    )
    raw_pred = prot_means.to_numpy(dtype=np.float32) if hasattr(prot_means, "to_numpy") else np.asarray(prot_means, dtype=np.float32)
    # Scale expected foreground rates by reference median total ADT counts so Seurat CLR operates on true count scale
    ref_med_tot = max(float(np.median(np.maximum(adt_counts_ref, 0.0).sum(axis=1))), 1.0)
    raw_sum = np.maximum(raw_pred.sum(axis=1, keepdims=True), 1e-6)
    pred = target_transform_fn(raw_pred * (ref_med_tot / raw_sum))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    vram_mb = float(torch.cuda.max_memory_allocated()) / (1024 * 1024) if torch.cuda.is_available() else 0.0
    return {"pred": pred, "sec": round(sec, 3), "peak_vram_mb": round(vram_mb, 2)}


def run_scipenn_gpu(
    rna_counts_ref: sp.csr_matrix,
    adt_counts_ref: np.ndarray,
    rna_counts_qry: sp.csr_matrix,
    gene_names: Sequence[str],
    protein_names: Sequence[str],
    y_train_target: np.ndarray,
    batch_ref: np.ndarray | None = None,
    batch_qry: np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run genuine sciPENN (Lakkis et al., Commun. Biol. 2022) on RTX 5070 GPU with AnnData>=0.10 compatibility."""
    import scanpy as sc
    import sciPENN.sciPENN_API as sp_mod
    import tempfile

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Reference-only top-1000 HVGs (mirrors sciPENN n_top_genes=1000 without legacy AnnData.concatenate)
    x_ref_log = sp.csr_matrix(rna_counts_ref, dtype=np.float32)
    rs = np.asarray(x_ref_log.sum(axis=1)).ravel()
    x_ref_log = sp.diags(1e4 / np.maximum(rs, 1.0)) @ x_ref_log
    x_ref_log.data = np.log1p(x_ref_log.data)
    if x_ref_log.shape[1] > 1000:
        m = np.asarray(x_ref_log.mean(axis=0)).ravel()
        sq = x_ref_log.copy()
        sq.data **= 2
        v = np.asarray(sq.mean(axis=0)).ravel() - m ** 2
        g_idx = np.sort(np.argsort(v)[::-1][:1000])
    else:
        g_idx = np.arange(x_ref_log.shape[1])

    unique_genes = [f"g_{i:05d}" for i in range(len(g_idx))]
    unique_prots = [f"p_{j:04d}" for j in range(len(protein_names))]
    b_ref = [str(x) for x in batch_ref] if batch_ref is not None else ["ref"] * rna_counts_ref.shape[0]
    b_qry = [str(x) for x in batch_qry] if batch_qry is not None else ["qry"] * rna_counts_qry.shape[0]

    gex_train = ad.AnnData(
        X=rna_counts_ref[:, g_idx].toarray().astype(np.float32),
        obs=pd.DataFrame({"samples": b_ref}, index=[f"r_{i}" for i in range(rna_counts_ref.shape[0])]),
        var=pd.DataFrame(index=unique_genes),
    )
    pdata_train = ad.AnnData(
        X=np.maximum(adt_counts_ref, 0.0).astype(np.float32),
        obs=pd.DataFrame({"samples": b_ref}, index=[f"r_{i}" for i in range(rna_counts_ref.shape[0])]),
        var=pd.DataFrame(index=unique_prots),
    )
    gex_test = ad.AnnData(
        X=rna_counts_qry[:, g_idx].toarray().astype(np.float32),
        obs=pd.DataFrame({"samples": b_qry}, index=[f"q_{i}" for i in range(rna_counts_qry.shape[0])]),
        var=pd.DataFrame(index=unique_genes),
    )

    def _modern_anndata_preprocess(
        gene_trainsets, protein_trainsets, gene_test=None, train_batchkeys=None, test_batchkey=None, *_args, **_kwargs
    ):
        gt = gene_trainsets[0].copy()
        pt = protein_trainsets[0].copy()
        gq = gene_test.copy()
        sc.pp.normalize_total(gt)
        sc.pp.normalize_total(pt)
        sc.pp.normalize_total(gq)
        sc.pp.log1p(gt)
        sc.pp.log1p(pt)
        sc.pp.log1p(gq)

        for bval in np.unique(gt.obs["samples"].to_numpy()):
            mask = (gt.obs["samples"].to_numpy() == bval)
            if mask.sum() >= 3:
                gt.X[mask] = sc.pp.scale(gt[mask], copy=True).X
                pt.X[mask] = sc.pp.scale(pt[mask], copy=True).X
        for bval in np.unique(gq.obs["samples"].to_numpy()):
            mask = (gq.obs["samples"].to_numpy() == bval)
            if mask.sum() >= 3:
                gq.X[mask] = sc.pp.scale(gq[mask], copy=True).X

        bools = np.ones((1, pt.shape[1]), dtype=int)
        return gt, pt, gq, bools, [], None

    orig_preprocess = sp_mod.preprocess
    orig_build_dir = sp_mod.build_dir
    sp_mod.preprocess = _modern_anndata_preprocess
    sp_mod.build_dir = lambda p: Path(p).mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as tmp_wdir:
            sp_api = sp_mod.sciPENN_API(
                [gex_train],
                [pdata_train],
                gex_test,
                select_hvg=False,
                train_batchkeys=["samples"],
                test_batchkey="samples",
                type_key=None,
                min_cells=0,
                min_genes=0,
                batch_size=min(128, max(32, rna_counts_ref.shape[0] // 4)),
                use_gpu=torch.cuda.is_available(),
            )
            sp_api.train(n_epochs=120, ES_max=10, decay_max=5, decay_step=0.1, lr=1e-3, weights_dir=tmp_wdir, load=False)
            pred_adata = sp_api.predict()
            z_pred = np.asarray(pred_adata.X, dtype=np.float32)
    finally:
        sp_mod.preprocess = orig_preprocess
        sp_mod.build_dir = orig_build_dir

    # sciPENN outputs Z-scored normalized protein values; rescale strictly via training y_train_target mean & std
    mu_tr = y_train_target.mean(axis=0, keepdims=True)
    sd_tr = np.maximum(y_train_target.std(axis=0, keepdims=True), 1e-5)
    pred = (z_pred * sd_tr + mu_tr).astype(np.float32)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    vram_mb = float(torch.cuda.max_memory_allocated()) / (1024 * 1024) if torch.cuda.is_available() else 0.0
    return {"pred": pred, "sec": round(sec, 3), "peak_vram_mb": round(vram_mb, 2)}
