"""Head-to-head single-cell foundation model benchmark (TOLT vs scGPT vs Geneformer) + Extended Data S1-S6 diagnostics.

Executes:
  1. Exact `strict=True` GPU BF16 embedding extraction for:
     - `scGPT_human` (51.3M params, 33M human cells, 512-d L2-normalized `<cls>`, Cui et al., Nature Methods 2024)
     - `Geneformer` (10.3M params, 30M human cells, 256-d penultimate-layer mean pool, Theodoris et al., Nature 2023)
     across all 4 unseen cohorts:
     - GSE194122 Multi-Site Bone Marrow (12,949 unseen cells, 134 ADTs)
     - GSE128639 BMMC Stuart (33,454 cells, 25 ADTs)
     - GSE166489 Pediatric MIS-C CITE-seq (5 donors, 43,724 cells, 188 clean ADTs)
     - GSE158013 Trimodal TEA-seq Cross-Donor (X066_W3 & X061_W1, 14,920 cells, 45 clean ADTs)
  2. Symmetrical head-to-head evaluation under BOTH:
     - Pure Frozen Linear Probe (`Direct`: 3-fold Cholesky Ridge on standardized frozen embeddings)
     - Identical 3-Fold Cholesky OOF + Residual + Delta^2-Simplex Adapter (`Adapter`)
     including sample-complexity curves, 5-donor 188-AB LODO, decoupled/spliced surface markers, and 9-lineage TEA-seq gating.
  3. Complete empirical diagnostics for Extended Data Figures S1-S6:
     - S1d: Pairwise distance concentration (d_NN / d_median) in 256-d Euclidean vs unit-sphere PCA50 Cosine space
     - S2c,d: Single-cell CD8 (0046) vs CD8a (0080) concordance & IgGFc vs CD64 (Fc-gamma-RI) binding on MIS-C P1.1
     - S4a-n: Full 14-marker 1D CLR histograms + 2-GMM Bayesian decision thresholds on A.HD3
     - S5a-d: Complete 18-sample GSE166489 single-cell QC metrics & distributions across 132,324 PBMC cells
     - S6c,d: RTX 5070 8 GB stage-by-stage latency breakdown & batch-size (32..1024) peak VRAM / throughput scaling
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from safetensors.torch import load_file
from scipy.stats import norm, wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, normalize
import torch
from torch import nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from triomics.v4.paths import EMB_DIR, PRED_DIR, COHORTS_DIR, EXTERNAL_MODELS_DIR

from triomics.v4.cohort_loaders import (
    D_BMMC90K,
    D_STUART,
    GSE166489_METADATA,
    log1p_cp10k_dense,
    log1p_cp10k_sparse,
    seurat_clr,
)
from triomics.v4.core_methods import (
    compute_prediction_metrics,
    fast_cholesky_ridge_oof,
    fit_local_svd_features,
    fit_predict_tolt_adapter_suite,
    forward_tolt_embeddings,
    instantiate_tolt_backbone,
    tokenize_atac_matrix,
    tokenize_rna_matrix,
)
from triomics.v4.immunophenotyping import fit_pbmc_9lineage_gates, score_9lineage_predictions

TABLES_DIR = REPO_ROOT / "results" / "tables"
RECEIPTS_DIR = REPO_ROOT / "results" / "receipts"


# =============================================================================
# 1. Exact Pure-PyTorch Official scGPT_human (33M cells, 51.3M params)
# =============================================================================
class ScGPTGeneEncoder(nn.Module):
    def __init__(self, num_embeddings: int, d_model: int, padding_idx: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, d_model, padding_idx=padding_idx)
        self.enc_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_norm(self.embedding(x))


class ScGPTContinuousValueEncoder(nn.Module):
    def __init__(self, d_model: int, max_value: float = 512.0) -> None:
        super().__init__()
        self.linear1 = nn.Linear(1, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = torch.clamp(x.unsqueeze(-1), max=self.max_value)
        v = F.relu(self.linear1(v))
        return self.norm(self.linear2(v))


class ScGPTExprDecoder(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, d_model),
            nn.LeakyReLU(),
            nn.Linear(d_model, 1),
        )


class OfficialScGPTModel(nn.Module):
    """Exact tensor-compatible architecture matching `tdc/scGPT` (`scGPT_human`, 159 tensors, strict=True)."""

    def __init__(self, ntoken: int = 60697, d_model: int = 512, nhead: int = 8, d_hid: int = 512, nlayers: int = 12, pad_id: int = 60694) -> None:
        super().__init__()
        self.gene_encoder = ScGPTGeneEncoder(ntoken, d_model, padding_idx=pad_id)
        self.value_encoder = ScGPTContinuousValueEncoder(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_hid,
            dropout=0.0,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=nlayers, enable_nested_tensor=False)
        self.expr_decoder = ScGPTExprDecoder(d_model)
        self.pad_id = pad_id

    def encode_cls(self, gene_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        pad_mask = gene_ids.eq(self.pad_id)
        h = self.gene_encoder(gene_ids) + self.value_encoder(values)
        out = self.transformer(h, src_key_padding_mask=pad_mask)
        return F.normalize(out[:, 0, :].float(), p=2, dim=-1)


def load_official_scgpt(device: str = "cuda") -> tuple[OfficialScGPTModel, dict[str, int], int]:
    sc_dir = EXTERNAL_MODELS_DIR / "scgpt_human"
    vocab: dict[str, int] = json.loads((sc_dir / "vocab.json").read_text(encoding="utf-8"))
    for sp_tok in ["<pad>", "<cls>", "<eoc>"]:
        if sp_tok not in vocab:
            vocab[sp_tok] = len(vocab)
    assert len(vocab) == 60697, f"Expected 60697 scGPT tokens, got {len(vocab)}"
    model = OfficialScGPTModel(ntoken=len(vocab), pad_id=vocab["<pad>"])
    st = load_file(str(sc_dir / "model.safetensors"))
    model.load_state_dict(st, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    return model.to(device), vocab, n_params


def _scgpt_bin_nonzero(vals: np.ndarray, rng: np.random.Generator, n_bins: int = 51) -> np.ndarray:
    """Exact Bowang Lab `_digitize` + `binning` on strictly positive gene counts per cell."""
    if vals.size == 0:
        return vals.astype(np.float32)
    bins = np.quantile(vals, np.linspace(0.0, 1.0, n_bins - 1))
    left_d = np.digitize(vals, bins, right=False)
    right_d = np.digitize(vals, bins, right=True)
    rands = rng.random(vals.size)
    digits = np.ceil(rands * (right_d - left_d) + left_d).astype(np.float32)
    return np.maximum(digits, 1.0)


def extract_scgpt_embeddings(
    model: OfficialScGPTModel,
    vocab: dict[str, int],
    rna_counts: sp.csr_matrix,
    gene_names: Sequence[str],
    max_length: int = 1200,
    batch_size: int = 128,
    device: str = "cuda",
) -> tuple[np.ndarray, dict[str, Any]]:
    dev = torch.device(device)
    t_tok0 = time.perf_counter()

    vocab_upper = {k.upper(): v for k, v in vocab.items() if not k.startswith("<")}
    keep_cols: list[int] = []
    mapped_ids: list[int] = []
    seen_v: set[int] = set()
    for col_i, gnm in enumerate(gene_names):
        vid = vocab_upper.get(str(gnm).upper())
        if vid is None and str(gnm) in vocab:
            vid = vocab[str(gnm)]
        if vid is not None and vid not in seen_v:
            keep_cols.append(col_i)
            mapped_ids.append(vid)
            seen_v.add(vid)

    sub_csr = rna_counts[:, np.asarray(keep_cols, dtype=np.int64)].tocsr()
    gene_vocab_arr = np.asarray(mapped_ids, dtype=np.int32)
    pad_id = vocab["<pad>"]
    cls_id = vocab["<cls>"]
    pad_val = -2.0

    n_cells = sub_csr.shape[0]
    rng = np.random.default_rng(42)
    ids_mat = np.full((n_cells, max_length), pad_id, dtype=np.int32)
    val_mat = np.full((n_cells, max_length), pad_val, dtype=np.float32)
    lengths = np.ones(n_cells, dtype=np.int32)
    ids_mat[:, 0] = cls_id
    val_mat[:, 0] = pad_val

    max_genes = max_length - 1
    for r in range(n_cells):
        st, ed = sub_csr.indptr[r], sub_csr.indptr[r + 1]
        cols = sub_csr.indices[st:ed]
        dat = sub_csr.data[st:ed]
        nz = dat > 0
        cols = cols[nz]
        dat = dat[nz]
        if cols.size == 0:
            continue
        binned = _scgpt_bin_nonzero(dat, rng, n_bins=51)
        if cols.size > max_genes:
            sel = rng.choice(cols.size, size=max_genes, replace=False)
            cols = cols[sel]
            binned = binned[sel]
        L = cols.size
        ids_mat[r, 1 : 1 + L] = gene_vocab_arr[cols]
        val_mat[r, 1 : 1 + L] = binned
        lengths[r] = 1 + L

    tok_sec = time.perf_counter() - t_tok0

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)
    t_inf0 = time.perf_counter()

    order = np.argsort(lengths, kind="stable")
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n_cells)
    chunks: list[np.ndarray] = []
    amp_dev = "cuda" if dev.type == "cuda" else "cpu"
    with torch.inference_mode(), torch.amp.autocast(amp_dev, dtype=torch.bfloat16):
        for s in range(0, n_cells, batch_size):
            idx = order[s : min(s + batch_size, n_cells)]
            ml = int(lengths[idx].max())
            g_t = torch.as_tensor(ids_mat[idx, :ml], dtype=torch.long, device=dev)
            v_t = torch.as_tensor(val_mat[idx, :ml], dtype=torch.float32, device=dev)
            chunks.append(model.encode_cls(g_t, v_t).cpu().numpy())

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    inf_sec = time.perf_counter() - t_inf0
    peak_mb = float(torch.cuda.max_memory_allocated(dev)) / (1024 * 1024) if dev.type == "cuda" else 0.0
    emb = np.concatenate(chunks, axis=0)[inv_order].astype(np.float32)
    return emb, {
        "mapped_genes": len(keep_cols),
        "tokenize_sec": round(tok_sec, 3),
        "gpu_inference_sec": round(inf_sec, 3),
        "peak_vram_mb": round(peak_mb, 2),
        "cells_per_sec": round(n_cells / max(inf_sec, 1e-4), 1),
    }


# =============================================================================
# 2. Exact Pure-PyTorch Official Geneformer V1-30M (10.3M params, 256-d)
# =============================================================================
class BertEmbeddingsExact(nn.Module):
    def __init__(self, vocab_size: int = 25426, hidden_size: int = 256, max_pos: int = 2048, eps: float = 1e-12) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_pos, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)
        self.register_buffer("position_ids", torch.arange(max_pos).expand((1, -1)), persistent=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        pos_ids = self.position_ids[:, :seq_len]
        tok_type = torch.zeros_like(input_ids)
        h = self.word_embeddings(input_ids) + self.position_embeddings(pos_ids) + self.token_type_embeddings(tok_type)
        return self.LayerNorm(h)


class BertSelfAttentionExact(nn.Module):
    def __init__(self, hidden_size: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q = self.query(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        # attn_mask: True where valid token -> shape (B, 1, 1, S)
        mask_4d = attn_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_4d, dropout_p=0.0)
        return out.transpose(1, 2).contiguous().view(b, s, d)


class BertSelfOutputExact(nn.Module):
    def __init__(self, hidden_size: int = 256, eps: float = 1e-12) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dense(hidden) + residual)


class BertAttentionExact(nn.Module):
    def __init__(self, hidden_size: int = 256, num_heads: int = 4, eps: float = 1e-12) -> None:
        super().__init__()
        self.self = BertSelfAttentionExact(hidden_size, num_heads)
        self.output = BertSelfOutputExact(hidden_size, eps)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        return self.output(self.self(x, attn_mask), x)


class BertIntermediateExact(nn.Module):
    def __init__(self, hidden_size: int = 256, intermediate_size: int = 512) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.dense(x))


class BertOutputExact(nn.Module):
    def __init__(self, intermediate_size: int = 512, hidden_size: int = 256, eps: float = 1e-12) -> None:
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dense(hidden) + residual)


class BertLayerExact(nn.Module):
    def __init__(self, hidden_size: int = 256, num_heads: int = 4, intermediate_size: int = 512, eps: float = 1e-12) -> None:
        super().__init__()
        self.attention = BertAttentionExact(hidden_size, num_heads, eps)
        self.intermediate = BertIntermediateExact(hidden_size, intermediate_size)
        self.output = BertOutputExact(intermediate_size, hidden_size, eps)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        a = self.attention(x, attn_mask)
        return self.output(self.intermediate(a), a)


class BertEncoderExact(nn.Module):
    def __init__(self, num_layers: int = 6) -> None:
        super().__init__()
        self.layer = nn.ModuleList([BertLayerExact() for _ in range(num_layers)])


class BertCoreExact(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = BertEmbeddingsExact()
        self.encoder = BertEncoderExact(num_layers=6)


class BertPredictionHeadTransformExact(nn.Module):
    def __init__(self, hidden_size: int = 256, eps: float = 1e-12) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=eps)


class BertLMPredictionHeadExact(nn.Module):
    def __init__(self, vocab_size: int = 25426, hidden_size: int = 256) -> None:
        super().__init__()
        self.transform = BertPredictionHeadTransformExact(hidden_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))


class BertOnlyMLMHeadExact(nn.Module):
    def __init__(self, vocab_size: int = 25426, hidden_size: int = 256) -> None:
        super().__init__()
        self.predictions = BertLMPredictionHeadExact(vocab_size, hidden_size)


class OfficialGeneformerModel(nn.Module):
    """Exact tensor-compatible architecture matching `ctheodoris/Geneformer` (`30M` cells, 107 tensors, strict=True)."""

    def __init__(self) -> None:
        super().__init__()
        self.bert = BertCoreExact()
        self.cls = BertOnlyMLMHeadExact()

    def encode_penultimate_mean(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Official Geneformer `EmbExtractor(emb_mode='cell', emb_layer=-1)` penultimate layer mean pooling."""
        attn_mask = input_ids.ne(0)
        h = self.bert.embeddings(input_ids)
        penultimate = h
        for li, layer in enumerate(self.bert.encoder.layer):
            h = layer(h, attn_mask)
            if li == len(self.bert.encoder.layer) - 2:
                penultimate = h
        mask_f = attn_mask.unsqueeze(-1).float()
        denom = torch.clamp(mask_f.sum(dim=1), min=1.0)
        return (penultimate.float() * mask_f).sum(dim=1) / denom


def load_official_geneformer(device: str = "cuda") -> tuple[OfficialGeneformerModel, dict[str, int], dict[str, str], dict[str, float], int]:
    gf_dir = EXTERNAL_MODELS_DIR / "geneformer_30m"
    with open(gf_dir / "token_dict.pkl", "rb") as f:
        token_dict = {str(k): int(v) for k, v in pickle.load(f).items()}
    with open(gf_dir / "gene_name_id_dict.pkl", "rb") as f:
        gene_name_id = {str(k).upper(): str(v) for k, v in pickle.load(f).items()}
    with open(gf_dir / "gene_median_dict.pkl", "rb") as f:
        gene_median = {str(k): float(v) for k, v in pickle.load(f).items()}

    model = OfficialGeneformerModel()
    st = load_file(str(gf_dir / "model.safetensors"))
    model.load_state_dict(st, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    return model.to(device), token_dict, gene_name_id, gene_median, n_params


def extract_geneformer_embeddings(
    model: OfficialGeneformerModel,
    token_dict: dict[str, int],
    gene_name_id: dict[str, str],
    gene_median: dict[str, float],
    rna_counts: sp.csr_matrix,
    gene_ids: Sequence[str] | None,
    gene_names: Sequence[str],
    max_tokens: int = 2048,
    batch_size: int = 96,
    device: str = "cuda",
) -> tuple[np.ndarray, dict[str, Any]]:
    dev = torch.device(device)
    t_tok0 = time.perf_counter()

    keep_cols: list[int] = []
    tok_ids: list[int] = []
    med_scales: list[float] = []
    seen_tok: set[int] = set()

    g_ids_seq = [str(x).split(".")[0] for x in gene_ids] if gene_ids is not None else [None] * len(gene_names)
    for col_i, (ens_raw, gnm) in enumerate(zip(g_ids_seq, gene_names)):
        ens = ens_raw if (ens_raw is not None and ens_raw in token_dict) else gene_name_id.get(str(gnm).upper())
        if ens is not None and ens in token_dict and ens in gene_median:
            tid = token_dict[ens]
            if tid not in seen_tok:
                keep_cols.append(col_i)
                tok_ids.append(tid)
                med_scales.append(max(gene_median[ens], 1e-6))
                seen_tok.add(tid)

    total_counts = np.maximum(np.asarray(rna_counts.sum(axis=1)).ravel().astype(np.float32), 1.0)
    sub_csr = rna_counts[:, np.asarray(keep_cols, dtype=np.int64)].tocsr()
    tok_arr = np.asarray(tok_ids, dtype=np.int32)
    inv_med = (1.0 / np.asarray(med_scales, dtype=np.float32))

    n_cells = sub_csr.shape[0]
    ids_mat = np.zeros((n_cells, max_tokens), dtype=np.int32)
    lengths = np.ones(n_cells, dtype=np.int32)

    for r in range(n_cells):
        st, ed = sub_csr.indptr[r], sub_csr.indptr[r + 1]
        cols = sub_csr.indices[st:ed]
        dat = sub_csr.data[st:ed]
        nz = dat > 0
        cols = cols[nz]
        dat = dat[nz]
        if cols.size == 0:
            ids_mat[r, 0] = tok_arr[0]
            continue
        rank_val = (dat * (10000.0 / total_counts[r])) * inv_med[cols]
        if cols.size > max_tokens:
            top_k = np.argpartition(rank_val, -max_tokens)[-max_tokens:]
            order_k = top_k[np.argsort(-rank_val[top_k], kind="stable")]
        else:
            order_k = np.argsort(-rank_val, kind="stable")
        L = order_k.size
        ids_mat[r, :L] = tok_arr[cols[order_k]]
        lengths[r] = L

    tok_sec = time.perf_counter() - t_tok0

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize(dev)
    t_inf0 = time.perf_counter()

    order = np.argsort(lengths, kind="stable")
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n_cells)
    chunks: list[np.ndarray] = []
    amp_dev = "cuda" if dev.type == "cuda" else "cpu"
    with torch.inference_mode(), torch.amp.autocast(amp_dev, dtype=torch.bfloat16):
        for s in range(0, n_cells, batch_size):
            idx = order[s : min(s + batch_size, n_cells)]
            ml = int(lengths[idx].max())
            inp = torch.as_tensor(ids_mat[idx, :ml], dtype=torch.long, device=dev)
            chunks.append(model.encode_penultimate_mean(inp).cpu().numpy())

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    inf_sec = time.perf_counter() - t_inf0
    peak_mb = float(torch.cuda.max_memory_allocated(dev)) / (1024 * 1024) if dev.type == "cuda" else 0.0
    emb = np.concatenate(chunks, axis=0)[inv_order].astype(np.float32)
    return emb, {
        "mapped_genes": len(keep_cols),
        "tokenize_sec": round(tok_sec, 3),
        "gpu_inference_sec": round(inf_sec, 3),
        "peak_vram_mb": round(peak_mb, 2),
        "cells_per_sec": round(n_cells / max(inf_sec, 1e-4), 1),
    }


# =============================================================================
# 3. Symmetrical Foundation-Model Head-to-Head Evaluator
# =============================================================================
def evaluate_foundation_backbones_on_split(
    cohort: str,
    split: str,
    budget: str,
    x_ref_log: sp.csr_matrix,
    x_qry_log: sp.csr_matrix,
    y_ref: np.ndarray,
    y_qry: np.ndarray,
    backbone_embs: dict[str, tuple[np.ndarray, np.ndarray]],
    x_atac_ref: sp.csr_matrix | None = None,
    x_atac_qry: sp.csr_matrix | None = None,
    expressed_mask: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    rows: list[dict[str, Any]] = []
    per_prot: dict[str, dict[str, np.ndarray]] = {}

    svd_r_tr, svd_r_te = fit_local_svd_features(x_ref_log, x_qry_log, seed=seed)
    svd_d_tr, svd_d_te = (
        fit_local_svd_features(x_ref_log, x_qry_log, seed=seed, x_atac_train=x_atac_ref, x_atac_test=x_atac_qry)
        if (x_atac_ref is not None and x_atac_qry is not None)
        else (svd_r_tr, svd_r_te)
    )

    # First record pure local SVD-Ridge baseline once
    _, te_svd, _ = fast_cholesky_ridge_oof(svd_r_tr, svd_r_te, y_ref, seed=seed)
    te_svd = np.maximum(te_svd, 0.0)
    m_svd = compute_prediction_metrics(y_qry, te_svd)
    rec_svd: dict[str, Any] = {
        "cohort": cohort,
        "split": split,
        "budget": budget,
        "n_ref": int(y_ref.shape[0]),
        "n_query": int(y_qry.shape[0]),
        "n_proteins": int(y_qry.shape[1]),
        "backbone": "Local_HVG_SVD",
        "regime": "SVD-Ridge",
        "method": "SVD-Ridge",
        "protein_pcc_median": round(m_svd["protein_pcc_median"], 4),
        "protein_pcc_mean": round(m_svd["protein_pcc_mean"], 4),
        "protein_spearman_median": round(m_svd["protein_spearman_median"], 4),
        "cell_pcc_median": round(m_svd["cell_pcc_median"], 4),
        "protein_rmse_median": round(m_svd["protein_rmse_median"], 4),
        "mse": round(m_svd["mse"], 4),
    }
    if expressed_mask is not None and expressed_mask.any():
        me = compute_prediction_metrics(y_qry[:, expressed_mask], te_svd[:, expressed_mask])
        rec_svd["expressed_pcc_median"] = round(me["protein_pcc_median"], 4)
    rows.append(rec_svd)
    per_prot["SVD-Ridge"] = {"pcc": m_svd["per_protein_pcc"], "pred": te_svd}

    for bname, (e_tr, e_te) in backbone_embs.items():
        use_dual_svd = bname.endswith("_RNA+ATAC")
        s_tr, s_te = (svd_d_tr, svd_d_te) if use_dual_svd else (svd_r_tr, svd_r_te)
        suite = fit_predict_tolt_adapter_suite(
            s_tr, s_te, e_tr, e_te, y_ref, seed=seed, nonneg_clip=True, device="cuda"
        )
        for reg_key, reg_label in [("TOLT-Direct", "Direct_Linear_Probe"), ("TOLT", "OOF_Simplex_Adapter")]:
            pred = suite[reg_key]["pred"]
            m = compute_prediction_metrics(y_qry, pred)
            m_tag = f"{bname}__{reg_label}"
            rec: dict[str, Any] = {
                "cohort": cohort,
                "split": split,
                "budget": budget,
                "n_ref": int(y_ref.shape[0]),
                "n_query": int(y_qry.shape[0]),
                "n_proteins": int(y_qry.shape[1]),
                "backbone": bname,
                "regime": reg_label,
                "method": m_tag,
                "protein_pcc_median": round(m["protein_pcc_median"], 4),
                "protein_pcc_mean": round(m["protein_pcc_mean"], 4),
                "protein_spearman_median": round(m["protein_spearman_median"], 4),
                "cell_pcc_median": round(m["cell_pcc_median"], 4),
                "protein_rmse_median": round(m["protein_rmse_median"], 4),
                "mse": round(m["mse"], 4),
            }
            if expressed_mask is not None and expressed_mask.any():
                me = compute_prediction_metrics(y_qry[:, expressed_mask], pred[:, expressed_mask])
                rec["expressed_pcc_median"] = round(me["protein_pcc_median"], 4)
            rows.append(rec)
            per_prot[m_tag] = {"pcc": m["per_protein_pcc"], "pred": pred, "oof": suite[reg_key].get("oof")}

    return rows, per_prot


def main() -> None:
    t_global = time.perf_counter()
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"
    print(f"=== Running Step 10: External Foundation Models + S1-S6 Diagnostics on {gpu_name} ===", flush=True)

    # Instantiate scGPT, Geneformer, and TOLT-Base-110 (to verify exact parameter counts & throughput side-by-side)
    scgpt_model, scgpt_vocab, scgpt_params = load_official_scgpt(device=device)
    gf_model, gf_tok, gf_name2id, gf_med, gf_params = load_official_geneformer(device=device)
    tolt_model = instantiate_tolt_backbone("base_110", seed=29, device=device)
    tolt_params = sum(p.numel() for p in tolt_model.parameters())
    print(
        f"Loaded strict=True weights! Params -> TOLT={tolt_params/1e6:.2f}M | "
        f"scGPT={scgpt_params/1e6:.2f}M | Geneformer={gf_params/1e6:.2f}M",
        flush=True,
    )

    tp_rows: list[dict[str, Any]] = []
    ext_emb_cache: dict[str, np.ndarray] = {}

    # -------------------------------------------------------------------------
    # A. Load Cohorts & Extract (or Reuse Cached) Frozen scGPT + Geneformer Embeddings
    # -------------------------------------------------------------------------
    cached_ext_path = EMB_DIR / "external_foundation_embeddings.npz"
    use_cached_ext = cached_ext_path.is_file() and (TABLES_DIR / "v42_foundation_throughput.csv").is_file()
    cached_ext = np.load(cached_ext_path, allow_pickle=True) if use_cached_ext else None
    if use_cached_ext:
        print("Reusing verified cached external_foundation_embeddings.npz!", flush=True)

    # 1. GSE194122 (12,949 cells)
    print("[1/4] Loading GSE194122 (12,949 unseen BMMC cells)...", flush=True)
    ad_90k = ad.read_h5ad(D_BMMC90K)
    mask_90k = ad_90k.obs["DonorID"].astype(str).isin(["12710", "13272"]).to_numpy()
    sub_90k = ad_90k[mask_90k].copy()
    x_90k = sp.csr_matrix(sub_90k.X, dtype=np.float32)
    adt_90k = np.asarray(sub_90k.obsm["protein_expression"], dtype=np.float32)
    y_90k = seurat_clr(adt_90k)
    gnames_90k = [str(x) for x in sub_90k.var_names]
    gids_90k = [str(x) for x in sub_90k.var["gene_id"]] if "gene_id" in sub_90k.var.columns else None
    del ad_90k, sub_90k

    tok_tolt_90k, aud_tolt_90k = tokenize_rna_matrix(x_90k, gids_90k, gnames_90k)
    if use_cached_ext and cached_ext is not None:
        scgpt_90k = cached_ext["gse194122_scgpt"]
        gf_90k = cached_ext["gse194122_geneformer"]
    else:
        scgpt_90k, st_sc_90k = extract_scgpt_embeddings(scgpt_model, scgpt_vocab, x_90k, gnames_90k, device=device)
        gf_90k, st_gf_90k = extract_geneformer_embeddings(gf_model, gf_tok, gf_name2id, gf_med, x_90k, gids_90k, gnames_90k, device=device)
        _, st_tl_90k = forward_tolt_embeddings(tolt_model, tok_tolt_90k, use_atac=False, batch_size=512, device=device)
        ext_emb_cache["gse194122_scgpt"] = scgpt_90k
        ext_emb_cache["gse194122_geneformer"] = gf_90k
        for m_nm, p_cnt, c_sz, mod_s, st_d in [
            ("TOLT-Base-110", tolt_params, 0.325, "RNA+ATAC+ADT", {"mapped_genes": aud_tolt_90k["mapped_genes"], "gpu_inference_sec": st_tl_90k["inference_sec"], "peak_vram_mb": st_tl_90k["peak_vram_mb"], "cells_per_sec": round(x_90k.shape[0] / max(st_tl_90k["inference_sec"], 1e-4), 1)}),
            ("scGPT_human", scgpt_params, 33.0, "RNA-only", st_sc_90k),
            ("Geneformer_30M", gf_params, 30.0, "RNA-only", st_gf_90k),
        ]:
            tp_rows.append({"cohort": "GSE194122_BMMC", "model": m_nm, "params_M": round(p_cnt / 1e6, 2), "pretrain_cells_M": c_sz, "modalities": mod_s, "n_cells": int(x_90k.shape[0]), **st_d})

    # 2. GSE128639 Stuart (33,454 cells)
    print("[2/4] Loading GSE128639 Stuart (33,454 BMMC cells)...", flush=True)
    ad_stu = ad.read_h5ad(D_STUART)
    x_stu = sp.csr_matrix(ad_stu.X, dtype=np.float32)
    prot_stu = ad_stu.obsm["protein_expression"]
    prot_stu = prot_stu.toarray() if sp.issparse(prot_stu) else np.asarray(prot_stu, dtype=np.float32)
    y_stu = seurat_clr(prot_stu)
    gnames_stu = [str(x) for x in ad_stu.var_names]
    del ad_stu

    if use_cached_ext and cached_ext is not None:
        scgpt_stu = cached_ext["gse128639_scgpt"]
        gf_stu = cached_ext["gse128639_geneformer"]
    else:
        scgpt_stu, st_sc_stu = extract_scgpt_embeddings(scgpt_model, scgpt_vocab, x_stu, gnames_stu, device=device)
        gf_stu, st_gf_stu = extract_geneformer_embeddings(gf_model, gf_tok, gf_name2id, gf_med, x_stu, None, gnames_stu, device=device)
        ext_emb_cache["gse128639_scgpt"] = scgpt_stu
        ext_emb_cache["gse128639_geneformer"] = gf_stu
        tp_rows.append({"cohort": "GSE128639_Stuart", "model": "scGPT_human", "params_M": round(scgpt_params / 1e6, 2), "pretrain_cells_M": 33.0, "modalities": "RNA-only", "n_cells": int(x_stu.shape[0]), **st_sc_stu})
        tp_rows.append({"cohort": "GSE128639_Stuart", "model": "Geneformer_30M", "params_M": round(gf_params / 1e6, 2), "pretrain_cells_M": 30.0, "modalities": "RNA-only", "n_cells": int(x_stu.shape[0]), **st_gf_stu})

    # 3. GSE166489 5 CITE-seq donors (43,724 cells, 188 ADTs)
    print("[3/4] Loading GSE166489 5 CITE-seq donors (43,724 PBMC cells)...", flush=True)
    cite_ids = ["A.HD1", "A.HD2", "A.HD3", "P1.1", "P2.1"]
    gschema_misc = np.load(EMB_DIR / "gse166489_gene_schema.npz", allow_pickle=True)
    gids_misc = [str(x) for x in gschema_misc["gene_ids"]]
    gnames_misc = [str(x) for x in gschema_misc["gene_names"]]

    misc_cite: dict[str, dict[str, Any]] = {}
    tot_misc_sc_sec, tot_misc_gf_sec, tot_misc_n = 0.0, 0.0, 0
    max_sc_vram, max_gf_vram = 0.0, 0.0
    for sid in cite_ids:
        xr = sp.load_npz(EMB_DIR / f"gse166489_{sid}_rna_counts.npz")
        bd = np.load(EMB_DIR / f"gse166489_{sid}_bundle.npz", allow_pickle=True)
        if use_cached_ext and cached_ext is not None:
            sc_e = cached_ext[f"gse166489_{sid}_scgpt"]
            gf_e = cached_ext[f"gse166489_{sid}_geneformer"]
        else:
            sc_e, st_sc = extract_scgpt_embeddings(scgpt_model, scgpt_vocab, xr, gnames_misc, device=device)
            gf_e, st_gf = extract_geneformer_embeddings(gf_model, gf_tok, gf_name2id, gf_med, xr, gids_misc, gnames_misc, device=device)
            ext_emb_cache[f"gse166489_{sid}_scgpt"] = sc_e
            ext_emb_cache[f"gse166489_{sid}_geneformer"] = gf_e
            tot_misc_sc_sec += st_sc["gpu_inference_sec"]
            tot_misc_gf_sec += st_gf["gpu_inference_sec"]
            tot_misc_n += xr.shape[0]
            max_sc_vram = max(max_sc_vram, st_sc["peak_vram_mb"])
            max_gf_vram = max(max_gf_vram, st_gf["peak_vram_mb"])
        adt_c = bd["adt_counts"].astype(np.float32)
        misc_cite[sid] = {
            "xr_log": log1p_cp10k_sparse(xr),
            "y_clr": seurat_clr(adt_c),
            "adt_raw": adt_c,
            "iso_raw": bd["isotype_counts"].astype(np.float32),
            "adt_names": [str(x) for x in bd["adt_names"]],
            "tolt": bd["base_110"],
            "f01": bd["f01_s29"],
            "rand": bd["rand_s29"],
            "scgpt": sc_e,
            "geneformer": gf_e,
        }
    if not use_cached_ext:
        tp_rows.append({"cohort": "GSE166489_5CITE", "model": "scGPT_human", "params_M": round(scgpt_params / 1e6, 2), "pretrain_cells_M": 33.0, "modalities": "RNA-only", "n_cells": tot_misc_n, "mapped_genes": st_sc["mapped_genes"], "gpu_inference_sec": round(tot_misc_sc_sec, 3), "peak_vram_mb": round(max_sc_vram, 2), "cells_per_sec": round(tot_misc_n / max(tot_misc_sc_sec, 1e-4), 1)})
        tp_rows.append({"cohort": "GSE166489_5CITE", "model": "Geneformer_30M", "params_M": round(gf_params / 1e6, 2), "pretrain_cells_M": 30.0, "modalities": "RNA-only", "n_cells": tot_misc_n, "mapped_genes": st_gf["mapped_genes"], "gpu_inference_sec": round(tot_misc_gf_sec, 3), "peak_vram_mb": round(max_gf_vram, 2), "cells_per_sec": round(tot_misc_n / max(tot_misc_gf_sec, 1e-4), 1)})

    # 4. GSE158013 Cross-Donor TEA-seq (X066_W3 & X061_W1, 14,920 cells, 45 ADTs)
    print("[4/4] Loading GSE158013 TEA-seq (X066_W3 & X061_W1)...", flush=True)
    tea_h5 = next((COHORTS_DIR / "GSE158013_TEAseq").glob("GSM5123951*filtered_feature_bc_matrix.h5"))
    with h5py.File(tea_h5, "r") as h:
        ft_tea = np.asarray([x.decode("utf-8") for x in h["matrix/features/feature_type"][:]])
        gnames_tea = [x.decode("utf-8") for x in h["matrix/features/name"][:][ft_tea == "Gene Expression"]]
        gids_tea = [x.decode("utf-8") for x in h["matrix/features/id"][:][ft_tea == "Gene Expression"]]

    tea_wells: dict[str, dict[str, Any]] = {}
    for wid in ["X066_W3", "X061_W1"]:
        xr = sp.load_npz(EMB_DIR / f"gse158013_{wid}_rna_counts.npz")
        xa = sp.load_npz(EMB_DIR / f"gse158013_{wid}_atac_consensus30k.npz")
        bd = np.load(EMB_DIR / f"gse158013_{wid}_bundle.npz", allow_pickle=True)
        if use_cached_ext and cached_ext is not None:
            sc_e = cached_ext[f"gse158013_{wid}_scgpt"]
            gf_e = cached_ext[f"gse158013_{wid}_geneformer"]
        else:
            sc_e, st_sc = extract_scgpt_embeddings(scgpt_model, scgpt_vocab, xr, gnames_tea, device=device)
            gf_e, st_gf = extract_geneformer_embeddings(gf_model, gf_tok, gf_name2id, gf_med, xr, gids_tea, gnames_tea, device=device)
            ext_emb_cache[f"gse158013_{wid}_scgpt"] = sc_e
            ext_emb_cache[f"gse158013_{wid}_geneformer"] = gf_e
        raw45 = bd["adt_counts_45"].astype(np.float32)
        tot46 = np.maximum(bd["adt_counts_all46"].astype(np.float32).sum(axis=1, keepdims=True), 1.0)
        y_iso = np.log1p(bd["isotype_counts"].astype(np.float32) / tot46 * 10000.0).astype(np.float32)
        tea_wells[wid] = {
            "xr_raw": xr,
            "xr_log": log1p_cp10k_sparse(xr),
            "xa": xa,
            "y45": log1p_cp10k_dense(raw45),
            "y_iso": y_iso,
            "names45": [str(x) for x in bd["adt_names_45"]],
            "tolt_rna": bd["base_110_rna"],
            "tolt_dual": bd["base_110_rna_atac"],
            "rand_dual": bd["rand_s29_rna_atac"],
            "scgpt": sc_e,
            "geneformer": gf_e,
        }

    if not use_cached_ext:
        np.savez_compressed(EMB_DIR / "external_foundation_embeddings.npz", **ext_emb_cache)
        pd.DataFrame(tp_rows).to_csv(TABLES_DIR / "v42_foundation_throughput.csv", index=False)
        print("Saved external_foundation_embeddings.npz & v42_foundation_throughput.csv!", flush=True)

    with np.load(EMB_DIR / "gse194122_unseen_12949_embeddings.npz", allow_pickle=True) as d90:
        dids_90 = d90["donor_ids"]
        tolt_90 = d90["base_110"]
        rand_90 = d90["rand_s29"]
    x_90k_log = log1p_cp10k_sparse(x_90k)
    i12 = np.flatnonzero(dids_90 == "12710")
    i13 = np.flatnonzero(dids_90 == "13272")
    pnames_188 = misc_cite["A.HD3"]["adt_names"]

    steady_csv = TABLES_DIR / "v42_foundation_steady_throughput.csv"
    if not steady_csv.is_file():
        print("=== Measuring warmed-up steady-state RTX 5070 GPU throughput (TOLT vs scGPT vs Geneformer) ===", flush=True)
        _, _ = forward_tolt_embeddings(tolt_model, {k: v[:256] for k, v in tok_tolt_90k.items()}, use_atac=False, batch_size=256, device=device)
        _, st_tl_warm = forward_tolt_embeddings(tolt_model, tok_tolt_90k, use_atac=False, batch_size=512, device=device)
        _, st_sc_warm = extract_scgpt_embeddings(scgpt_model, scgpt_vocab, x_90k, gnames_90k, batch_size=128, device=device)
        _, st_gf_warm = extract_geneformer_embeddings(gf_model, gf_tok, gf_name2id, gf_med, x_90k, gids_90k, gnames_90k, batch_size=96, device=device)
        pd.DataFrame([
            {"cohort": "GSE194122_SteadyState", "model": "TOLT-Base-110", "params_M": round(tolt_params / 1e6, 2), "pretrain_cells_M": 0.325, "modalities": "RNA+ATAC+ADT", "vocab_size": 2048, "n_cells": int(x_90k.shape[0]), "mapped_genes": aud_tolt_90k["mapped_genes"], "gpu_inference_sec": st_tl_warm["inference_sec"], "peak_vram_mb": st_tl_warm["peak_vram_mb"], "cells_per_sec": round(x_90k.shape[0] / max(st_tl_warm["inference_sec"], 1e-4), 1)},
            {"cohort": "GSE194122_SteadyState", "model": "scGPT_human", "params_M": round(scgpt_params / 1e6, 2), "pretrain_cells_M": 33.0, "modalities": "RNA-only", "vocab_size": 60697, "n_cells": int(x_90k.shape[0]), **st_sc_warm},
            {"cohort": "GSE194122_SteadyState", "model": "Geneformer_30M", "params_M": round(gf_params / 1e6, 2), "pretrain_cells_M": 30.0, "modalities": "RNA-only", "vocab_size": 25426, "n_cells": int(x_90k.shape[0]), **st_gf_warm},
        ]).to_csv(steady_csv, index=False)

    fm_csv = TABLES_DIR / "v42_foundation_model_comparison.csv"
    dm_csv = TABLES_DIR / "v42_foundation_decoupled_markers.csv"
    tp_csv = TABLES_DIR / "v42_foundation_teaseq_phenotyping.csv"

    if not (fm_csv.is_file() and dm_csv.is_file() and tp_csv.is_file()):
        print("=== Evaluating TOLT vs scGPT vs Geneformer across all 4 cohorts ===", flush=True)
        fm_rows: list[dict[str, Any]] = []
        decoupled_rows: list[dict[str, Any]] = []
        tea_pheno_fm_rows: list[dict[str, Any]] = []

        for dir_nm, ref_pool, qry_idx in [("12710_Site2_to_13272_Site4", i12, i13), ("13272_Site4_to_12710_Site2", i13, i12)]:
            exp_m = y_90k[ref_pool].std(axis=0) >= 0.35
            for b in [50, 100, 250, 500, 1000, "full"]:
                rng = np.random.default_rng(1001 + (0 if b == "full" else int(b)))
                r_idx = ref_pool if (b == "full" or int(b) >= len(ref_pool)) else np.sort(rng.choice(ref_pool, size=int(b), replace=False))
                blab = str(b)
                rows_b, _ = evaluate_foundation_backbones_on_split(
                    "GSE194122_BMMC_CrossSite",
                    dir_nm,
                    blab,
                    x_90k_log[r_idx],
                    x_90k_log[qry_idx],
                    y_90k[r_idx],
                    y_90k[qry_idx],
                    {
                        "TOLT": (tolt_90[r_idx], tolt_90[qry_idx]),
                        "scGPT": (scgpt_90k[r_idx], scgpt_90k[qry_idx]),
                        "Geneformer": (gf_90k[r_idx], gf_90k[qry_idx]),
                        "RandomInit": (rand_90[r_idx], rand_90[qry_idx]),
                    },
                    expressed_mask=exp_m,
                )
                fm_rows.extend(rows_b)

        # 2. GSE128639 Stuart (Donor A <-> Donor B)
        with np.load(EMB_DIR / "gse128639_stuart_33454_embeddings.npz", allow_pickle=True) as dstu:
            donors_stu = dstu["donors"]
            tolt_stu = dstu["base_110"]
            rand_stu = dstu["rand_s29"]
        x_stu_log = log1p_cp10k_sparse(x_stu)
        da = np.flatnonzero(donors_stu == "Donor_A")
        db = np.flatnonzero(donors_stu == "Donor_B")
        for dir_nm, ref_pool, qry_idx in [("DonorA_to_DonorB", da, db), ("DonorB_to_DonorA", db, da)]:
            for b in [50, 100, 250, 500, 1000, "full"]:
                rng = np.random.default_rng(2002 + (0 if b == "full" else int(b)))
                r_idx = ref_pool if (b == "full" or int(b) >= len(ref_pool)) else np.sort(rng.choice(ref_pool, size=int(b), replace=False))
                blab = str(b)
                rows_b, _ = evaluate_foundation_backbones_on_split(
                    "GSE128639_BMMC_Stuart",
                    dir_nm,
                    blab,
                    x_stu_log[r_idx],
                    x_stu_log[qry_idx],
                    y_stu[r_idx],
                    y_stu[qry_idx],
                    {
                        "TOLT": (tolt_stu[r_idx], tolt_stu[qry_idx]),
                        "scGPT": (scgpt_stu[r_idx], scgpt_stu[qry_idx]),
                        "Geneformer": (gf_stu[r_idx], gf_stu[qry_idx]),
                        "RandomInit": (rand_stu[r_idx], rand_stu[qry_idx]),
                    },
                )
                fm_rows.extend(rows_b)

        # 3. GSE166489 5-Donor 188-Antibody LODO (budgets 250, 500, 6000) + Decoupled Marker Audit
        target_decoupled = ["CD45RA", "CD45RO", "CD95", "CD24", "CD82", "CD44", "CD4", "CD279", "TIGIT", "CD38", "CD62L", "CX3CR1", "CD16", "CD56"]
        for held_out in cite_ids:
            tr_d = [d for d in cite_ids if d != held_out]
            xr_tr_all = sp.vstack([misc_cite[d]["xr_log"] for d in tr_d], format="csr")
            y_tr_all = np.vstack([misc_cite[d]["y_clr"] for d in tr_d])
            tl_tr_all = np.vstack([misc_cite[d]["tolt"] for d in tr_d])
            sc_tr_all = np.vstack([misc_cite[d]["scgpt"] for d in tr_d])
            gf_tr_all = np.vstack([misc_cite[d]["geneformer"] for d in tr_d])
            rn_tr_all = np.vstack([misc_cite[d]["rand"] for d in tr_d])
            exp_mask = (y_tr_all.std(axis=0) >= 0.35) | (np.percentile(y_tr_all, 99, axis=0) >= 1.50)

            for b_lodo in [250, 500, 6000]:
                rng = np.random.default_rng(3003 + b_lodo)
                sub_i = np.sort(rng.choice(len(y_tr_all), size=min(b_lodo, len(y_tr_all)), replace=False))
                rows_l, pp_l = evaluate_foundation_backbones_on_split(
                    "GSE166489_188AB_LODO",
                    f"LODO_Holdout_{held_out}",
                    str(b_lodo),
                    xr_tr_all[sub_i],
                    misc_cite[held_out]["xr_log"],
                    y_tr_all[sub_i],
                    misc_cite[held_out]["y_clr"],
                    {
                        "TOLT": (tl_tr_all[sub_i], misc_cite[held_out]["tolt"]),
                        "scGPT": (sc_tr_all[sub_i], misc_cite[held_out]["scgpt"]),
                        "Geneformer": (gf_tr_all[sub_i], misc_cite[held_out]["geneformer"]),
                        "RandomInit": (rn_tr_all[sub_i], misc_cite[held_out]["rand"]),
                    },
                    expressed_mask=exp_mask,
                )
                fm_rows.extend(rows_l)
                if b_lodo in {500, 6000}:
                    for mk in target_decoupled:
                        if mk in pnames_188:
                            mi = pnames_188.index(mk)
                            for m_key, pdict in pp_l.items():
                                decoupled_rows.append({
                                    "held_out_donor": held_out,
                                    "budget": str(b_lodo),
                                    "antibody": mk,
                                    "method": m_key,
                                    "pcc": round(float(pdict["pcc"][mi]), 4),
                                })

        # 4. GSE158013 Cross-Donor Trimodal TEA-seq (X066_W3 <-> X061_W1)
        for pair_nm, w_ref, w_qry in [("X066_W3_to_X061_W1", "X066_W3", "X061_W1"), ("X061_W1_to_X066_W3", "X061_W1", "X066_W3")]:
            ref_w, qry_w = tea_wells[w_ref], tea_wells[w_qry]
            for b in [250, 500, "full"]:
                rng = np.random.default_rng(4004 + (0 if b == "full" else int(b)))
                n_pool = ref_w["y45"].shape[0]
                r_idx = np.arange(n_pool) if (b == "full" or int(b) >= n_pool) else np.sort(rng.choice(n_pool, size=int(b), replace=False))
                blab = str(b)
                rows_t, pp_t = evaluate_foundation_backbones_on_split(
                    "GSE158013_TEAseq_CrossDonor",
                    pair_nm,
                    blab,
                    ref_w["xr_log"][r_idx],
                    qry_w["xr_log"],
                    ref_w["y45"][r_idx],
                    qry_w["y45"],
                    {
                        "TOLT_RNA+ATAC": (ref_w["tolt_dual"][r_idx], qry_w["tolt_dual"]),
                        "TOLT_RNA": (ref_w["tolt_rna"][r_idx], qry_w["tolt_rna"]),
                        "scGPT": (ref_w["scgpt"][r_idx], qry_w["scgpt"]),
                        "Geneformer": (ref_w["geneformer"][r_idx], qry_w["geneformer"]),
                        "RandomInit_RNA+ATAC": (ref_w["rand_dual"][r_idx], qry_w["rand_dual"]),
                    },
                    x_atac_ref=ref_w["xa"][r_idx],
                    x_atac_qry=qry_w["xa"],
                )
                fm_rows.extend(rows_t)
                if blab in {"250", "full"}:
                    gate_spec = fit_pbmc_9lineage_gates(ref_w["y45"][r_idx], ref_w["names45"])
                    for m_key, pdict in pp_t.items():
                        sc_ph = score_9lineage_predictions(
                            qry_w["y45"],
                            pdict["pred"],
                            gate_spec,
                            pred_oof=pdict.get("oof"),
                        )
                        tea_pheno_fm_rows.append({
                            "pair": pair_nm,
                            "budget": blab,
                            "method": m_key,
                            "p1_accuracy_pct": sc_ph["p1_accuracy_pct"],
                            "p1_macro_f1_8lin": sc_ph["p1_macro_f1_8lin"],
                            "p1_kappa": sc_ph["p1_kappa"],
                            "p2_accuracy_pct": sc_ph["p2_accuracy_pct"],
                            "p2_macro_f1_8lin": sc_ph["p2_macro_f1_8lin"],
                            "p2_kappa": sc_ph["p2_kappa"],
                        })

        df_fm = pd.DataFrame(fm_rows)
        df_fm.to_csv(fm_csv, index=False)
        pd.DataFrame(decoupled_rows).to_csv(dm_csv, index=False)
        pd.DataFrame(tea_pheno_fm_rows).to_csv(tp_csv, index=False)
    else:
        print("Reusing verified cached foundation benchmark CSVs!", flush=True)
        df_fm = pd.read_csv(fm_csv)
        # Ensure CD279, TIGIT, CD38 are also appended to decoupled_markers if missing
        dm_existing = pd.read_csv(dm_csv)
        if "CD279" not in set(dm_existing["antibody"].astype(str)):
            extra_dm: list[dict[str, Any]] = []
            for held_out in cite_ids:
                tr_d = [d for d in cite_ids if d != held_out]
                xr_tr_all = sp.vstack([misc_cite[d]["xr_log"] for d in tr_d], format="csr")
                y_tr_all = np.vstack([misc_cite[d]["y_clr"] for d in tr_d])
                tl_tr_all = np.vstack([misc_cite[d]["tolt"] for d in tr_d])
                sc_tr_all = np.vstack([misc_cite[d]["scgpt"] for d in tr_d])
                gf_tr_all = np.vstack([misc_cite[d]["geneformer"] for d in tr_d])
                rn_tr_all = np.vstack([misc_cite[d]["rand"] for d in tr_d])
                for b_lodo in [500, 6000]:
                    rng = np.random.default_rng(3003 + b_lodo)
                    sub_i = np.sort(rng.choice(len(y_tr_all), size=min(b_lodo, len(y_tr_all)), replace=False))
                    _, pp_l = evaluate_foundation_backbones_on_split(
                        "GSE166489_188AB_LODO", f"LODO_Holdout_{held_out}", str(b_lodo),
                        xr_tr_all[sub_i], misc_cite[held_out]["xr_log"], y_tr_all[sub_i], misc_cite[held_out]["y_clr"],
                        {"TOLT": (tl_tr_all[sub_i], misc_cite[held_out]["tolt"]), "scGPT": (sc_tr_all[sub_i], misc_cite[held_out]["scgpt"]), "Geneformer": (gf_tr_all[sub_i], misc_cite[held_out]["geneformer"]), "RandomInit": (rn_tr_all[sub_i], misc_cite[held_out]["rand"])},
                    )
                    for mk in ["CD279", "TIGIT", "CD38"]:
                        mi = pnames_188.index(mk)
                        for m_key, pdict in pp_l.items():
                            extra_dm.append({"held_out_donor": held_out, "budget": str(b_lodo), "antibody": mk, "method": m_key, "pcc": round(float(pdict["pcc"][mi]), 4)})
            pd.concat([dm_existing, pd.DataFrame(extra_dm)], ignore_index=True).to_csv(dm_csv, index=False)

    # Print crisp summary of Direct Linear Probe & OOF Simplex Adapter!
    print("\n--- Summary: Direct Linear Probe at n_ref=250 ---", flush=True)
    sub_250 = df_fm[(df_fm["budget"] == "250") & (df_fm["regime"] == "Direct_Linear_Probe")]
    print(sub_250.groupby(["cohort", "backbone"])["protein_pcc_median"].mean().unstack(), flush=True)
    print("\n--- Summary: OOF Simplex Adapter at Full / 6000 ---", flush=True)
    sub_full = df_fm[(df_fm["budget"].isin(["full", "6000"])) & (df_fm["regime"].isin(["OOF_Simplex_Adapter", "SVD-Ridge"]))]
    print(sub_full.groupby(["cohort", "method"])["protein_pcc_median"].mean().unstack(), flush=True)

    # -------------------------------------------------------------------------
    # C. Generate Extended Data S1-S6 Diagnostics
    # -------------------------------------------------------------------------
    print("\n=== Computing Extended Data S1-S6 Diagnostics ===", flush=True)
    supp_diag: dict[str, np.ndarray] = {}

    # 1. S1d: Pairwise distance concentration (d_NN / d_median) in 256-d Euclidean vs PCA50 unit-sphere Cosine
    svd_tr_s1, svd_te_s1 = fit_local_svd_features(x_90k_log[i12], x_90k_log[i13][:2000], n_components=256, seed=42)
    d_euc = pairwise_distances(svd_te_s1, svd_tr_s1, metric="euclidean")
    ratio_euc = np.min(d_euc, axis=1) / np.maximum(np.median(d_euc, axis=1), 1e-8)

    pca50 = PCA(n_components=50, random_state=42).fit(svd_tr_s1)
    u_tr = normalize(pca50.transform(svd_tr_s1), norm="l2")
    u_te = normalize(pca50.transform(svd_te_s1), norm="l2")
    d_cos = pairwise_distances(u_te, u_tr, metric="cosine")
    ratio_cos = np.min(d_cos, axis=1) / np.maximum(np.median(d_cos, axis=1), 1e-8)
    supp_diag["s1_ratio_euc256"] = ratio_euc.astype(np.float32)
    supp_diag["s1_ratio_cos50"] = ratio_cos.astype(np.float32)

    # 2. S2c,d: Single-cell CD8 (0046) vs CD8a (0080) & IgGFc vs CD64 on MIS-C P1.1
    y_p11 = misc_cite["P1.1"]["y_clr"]
    iso_p11 = np.log1p(misc_cite["P1.1"]["iso_raw"][:, 0] / np.maximum(np.exp(np.log1p(misc_cite["P1.1"]["adt_raw"]).mean(axis=1)), 1e-6)).astype(np.float32)
    supp_diag["s2_p11_cd8_0046"] = y_p11[:, pnames_188.index("CD8")].astype(np.float32)
    supp_diag["s2_p11_cd8a_0080"] = y_p11[:, pnames_188.index("CD8a")].astype(np.float32)
    supp_diag["s2_p11_cd64"] = y_p11[:, pnames_188.index("CD64")].astype(np.float32)
    supp_diag["s2_p11_cd14"] = y_p11[:, pnames_188.index("CD14")].astype(np.float32)
    supp_diag["s2_p11_iggfc"] = iso_p11

    # 3. S4a-n: Full 14-marker 1D CLR histogram + 2-GMM Bayesian gate gallery on A.HD3
    gmm_14_markers = ["CD3", "CD4", "CD8a", "CD19", "CD20", "IgD", "CD14", "CD16", "CD56", "CD45RA", "CD45RO", "HLA-DR", "CD27", "CD127"]
    y_hd3 = misc_cite["A.HD3"]["y_clr"]
    supp_diag["s4_marker_names"] = np.asarray(gmm_14_markers, dtype=str)
    x_grid = np.linspace(0.0, 5.2, 260, dtype=np.float32)
    supp_diag["s4_x_grid"] = x_grid
    for m_i, mk in enumerate(gmm_14_markers):
        mk_key = mk if mk in pnames_188 else mk.replace("-", "")
        vals = y_hd3[:, pnames_188.index(mk_key)].astype(np.float64)
        gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42, n_init=5).fit(vals.reshape(-1, 1))
        means = gmm.means_.ravel()
        order_g = np.argsort(means)
        w0, w1 = float(gmm.weights_[order_g[0]]), float(gmm.weights_[order_g[1]])
        mu0, mu1 = float(means[order_g[0]]), float(means[order_g[1]])
        sd0 = float(np.sqrt(gmm.covariances_.ravel()[order_g[0]]))
        sd1 = float(np.sqrt(gmm.covariances_.ravel()[order_g[1]]))
        c0 = w0 * norm.pdf(x_grid, loc=mu0, scale=max(sd0, 1e-3))
        c1 = w1 * norm.pdf(x_grid, loc=mu1, scale=max(sd1, 1e-3))
        # Posterior 0.5 crossing between mu0 and mu1
        mid_mask = (x_grid >= mu0) & (x_grid <= mu1)
        if mid_mask.any():
            sub_x = x_grid[mid_mask]
            tau = float(sub_x[np.argmin(np.abs(c1[mid_mask] - c0[mid_mask]))])
        else:
            tau = 0.5 * (mu0 + mu1)
        supp_diag[f"s4_vals_{m_i}"] = vals.astype(np.float32)
        supp_diag[f"s4_c0_{m_i}"] = c0.astype(np.float32)
        supp_diag[f"s4_c1_{m_i}"] = c1.astype(np.float32)
        supp_diag[f"s4_tau_{m_i}"] = np.array([tau, mu0, mu1, w1], dtype=np.float32)

    # 4. S5a-d: Complete 18-Donor GSE166489 Single-Cell QC Metrics & Distributions (132,324 cells)
    mt_mask_misc = np.array([g.upper().startswith("MT-") for g in gnames_misc], dtype=bool)
    qc_rows: list[dict[str, Any]] = []
    for s_idx, meta in enumerate(GSE166489_METADATA):
        sid = meta["sample_id"]
        xr = sp.load_npz(EMB_DIR / f"gse166489_{sid}_rna_counts.npz")
        bd = np.load(EMB_DIR / f"gse166489_{sid}_bundle.npz", allow_pickle=True)
        umis = np.asarray(xr.sum(axis=1)).ravel().astype(np.float32)
        ngenes = np.diff(xr.indptr).astype(np.float32)
        mt_sum = np.asarray(xr[:, mt_mask_misc].sum(axis=1)).ravel().astype(np.float32)
        mt_pct = (100.0 * mt_sum / np.maximum(umis, 1.0)).astype(np.float32)
        has_tcr = bd["has_tcr"] if "has_tcr" in bd else np.zeros(xr.shape[0], dtype=bool)
        is_v11 = bd["is_trbv11_2"] if "is_trbv11_2" in bd else np.zeros(xr.shape[0], dtype=bool)
        qc_rows.append({
            "sample_id": sid,
            "condition": meta["condition"],
            "age_group": meta["age_group"],
            "has_adt": meta["has_adt"] == "true",
            "n_cells": int(xr.shape[0]),
            "umi_q25": round(float(np.percentile(umis, 25)), 1),
            "umi_median": round(float(np.median(umis)), 1),
            "umi_q75": round(float(np.percentile(umis, 75)), 1),
            "genes_q25": round(float(np.percentile(ngenes, 25)), 1),
            "genes_median": round(float(np.median(ngenes)), 1),
            "genes_q75": round(float(np.percentile(ngenes, 75)), 1),
            "mt_pct_q25": round(float(np.percentile(mt_pct, 25)), 2),
            "mt_pct_median": round(float(np.median(mt_pct)), 2),
            "mt_pct_q75": round(float(np.percentile(mt_pct, 75)), 2),
            "tcr_pct": round(100.0 * float(has_tcr.mean()), 2),
            "trbv11_2_pct_of_tcr": round(100.0 * float(is_v11.sum()) / max(int(has_tcr.sum()), 1), 2),
        })
        # Subsample up to 1,500 cells per donor for smooth S5 violin rendering
        rng_s = np.random.default_rng(500 + s_idx)
        take = np.arange(xr.shape[0]) if xr.shape[0] <= 1500 else np.sort(rng_s.choice(xr.shape[0], size=1500, replace=False))
        supp_diag[f"s5_{sid}_log10_umi"] = np.log10(np.maximum(umis[take], 1.0)).astype(np.float32)
        supp_diag[f"s5_{sid}_genes"] = ngenes[take].astype(np.float32)
        supp_diag[f"s5_{sid}_mt_pct"] = mt_pct[take].astype(np.float32)
    pd.DataFrame(qc_rows).to_csv(TABLES_DIR / "v42_gse166489_18sample_qc.csv", index=False)

    # 5. S6c,d: RTX 5070 8 GB Stage-by-Stage Latency Breakdown & Batch-Size VRAM/Throughput Scaling
    scale_rows: list[dict[str, Any]] = []
    # Prepare TEA-seq X066_W3 dual tokens for trimodal batch scaling
    tok_tea_r, _ = tokenize_rna_matrix(tea_wells["X066_W3"]["xr_raw"], gids_tea, gnames_tea)
    # Load peak names from H5 for exact ATAC tokenization timing
    with h5py.File(tea_h5, "r") as h:
        ft_t = np.asarray([x.decode("utf-8") for x in h["matrix/features/feature_type"][:]])
        pnames_tea_raw = [x.decode("utf-8") for x in h["matrix/features/name"][:][ft_t == "Peaks"]]
    xa_raw_w3 = sp.load_npz(EMB_DIR / "gse158013_X066_W3_atac_counts.npz")
    t_atak0 = time.perf_counter()
    tok_tea_a, _ = tokenize_atac_matrix(xa_raw_w3, pnames_tea_raw)
    t_atak_sec = time.perf_counter() - t_atak0
    tok_tea_both = {**tok_tea_r, **tok_tea_a}

    for bs in [32, 64, 128, 256, 512, 1024]:
        _, st_r_bs = forward_tolt_embeddings(tolt_model, tok_tolt_90k, use_atac=False, batch_size=bs, device=device)
        scale_rows.append({
            "kind": "batch_scaling",
            "workload": "GSE194122_RNA_512tok (12,949 cells)",
            "batch_size": bs,
            "sec": st_r_bs["inference_sec"],
            "cells_per_sec": round(x_90k.shape[0] / max(st_r_bs["inference_sec"], 1e-4), 1),
            "peak_vram_mb": st_r_bs["peak_vram_mb"],
        })
        if bs <= 512:
            _, st_d_bs = forward_tolt_embeddings(tolt_model, tok_tea_both, use_atac=True, batch_size=bs, device=device)
            scale_rows.append({
                "kind": "batch_scaling",
                "workload": "GSE158013_RNA+ATAC_1536tok (7,377 cells)",
                "batch_size": bs,
                "sec": st_d_bs["inference_sec"],
                "cells_per_sec": round(xa_raw_w3.shape[0] / max(st_d_bs["inference_sec"], 1e-4), 1),
                "peak_vram_mb": st_d_bs["peak_vram_mb"],
            })

    # Stage-by-stage end-to-end breakdown on GSE194122 (n_ref=500 -> 6,941 query cells, 134 ADTs)
    rng_st = np.random.default_rng(1501)
    r500 = np.sort(rng_st.choice(i12, size=500, replace=False))
    t0 = time.perf_counter()
    _, _ = tokenize_rna_matrix(x_90k[np.concatenate([r500, i13])], gids_90k, gnames_90k)
    s_tok = time.perf_counter() - t0
    _, st_enc = forward_tolt_embeddings(tolt_model, tok_tolt_90k, use_atac=False, batch_size=512, device=device)
    s_enc = st_enc["inference_sec"] * (len(r500) + len(i13)) / x_90k.shape[0]
    t0 = time.perf_counter()
    sv_tr, sv_te = fit_local_svd_features(x_90k_log[r500], x_90k_log[i13], seed=42)
    s_svd = time.perf_counter() - t0
    t0 = time.perf_counter()
    oof_s, te_s, _ = fast_cholesky_ridge_oof(sv_tr, sv_te, y_90k[r500], seed=42)
    sc_e = StandardScaler().fit(tolt_90[r500])
    oof_d, te_d, _ = fast_cholesky_ridge_oof(sc_e.transform(tolt_90[r500]), sc_e.transform(tolt_90[i13]), y_90k[r500], seed=1042)
    s_chol = time.perf_counter() - t0
    t0 = time.perf_counter()
    suite_full = fit_predict_tolt_adapter_suite(sv_tr, sv_te, tolt_90[r500], tolt_90[i13], y_90k[r500], seed=42, device=device)
    s_res_qp = max(time.perf_counter() - t0 - s_chol, 0.05)

    for stg_nm, stg_sec in [
        ("1. Frozen Vocab Tokenization", s_tok),
        ("2. BF16 TOLT Encoder (GPU)", s_enc),
        ("3. Reference HVG-SVD256", s_svd),
        ("4. 3-Fold Cholesky OOF Ridge", s_chol),
        ("5. Residual + 3x3 Simplex QP", s_res_qp),
    ]:
        scale_rows.append({
            "kind": "stage_breakdown",
            "workload": stg_nm,
            "batch_size": 512,
            "sec": round(stg_sec, 4),
            "cells_per_sec": 0.0,
            "peak_vram_mb": st_enc["peak_vram_mb"] if "GPU" in stg_nm else 0.0,
        })

    pd.DataFrame(scale_rows).to_csv(TABLES_DIR / "v42_laptop_latency_vram_scaling.csv", index=False)
    np.savez_compressed(PRED_DIR / "v42_supp_diagnostics.npz", **supp_diag)

    receipt = {
        "schema": "tolt.v42.foundation_and_supp_receipt.v1",
        "device": device,
        "gpu_name": gpu_name,
        "tolt_params": tolt_params,
        "scgpt_params": scgpt_params,
        "geneformer_params": gf_params,
        "atac_tokenize_w3_sec": round(t_atak_sec, 3),
        "elapsed_sec": round(time.perf_counter() - t_global, 2),
    }
    (RECEIPTS_DIR / "10_foundation_models_and_qc_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nALL STEP 10 BENCHMARKS & S1-S6 DIAGNOSTICS COMPLETED IN {receipt['elapsed_sec']}s!", flush=True)


if __name__ == "__main__":
    main()
