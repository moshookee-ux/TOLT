"""Run official scVAEIT v1.1.1 (Python 3.11 .venv-scvaeit) across cross-well & true cross-donor TEA-seq splits."""
from __future__ import annotations

import gc
import os
import sysconfig
import time
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "6"
os.environ["MKL_NUM_THREADS"] = "6"
os.environ["OPENBLAS_NUM_THREADS"] = "6"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from triomics.v4.paths import EMB_DIR, PRED_DIR
VOCAB_DIR = REPO_ROOT / "data" / "vocabularies"


def ensure_scvaeit_val_patch() -> None:
    p = Path(sysconfig.get_paths()["purelib"]) / "scVAEIT" / "train.py"
    if p.exists():
        txt = p.read_text(encoding="utf-8")
        bad = "loss_monitor['val'][0].result()"
        good = "loss_monitor.losses['val']['total'].result()"
        if bad in txt:
            p.write_text(txt.replace(bad, good), encoding="utf-8")


def log1p_cp10k_dense(c: np.ndarray) -> np.ndarray:
    c = np.maximum(np.asarray(c, dtype=np.float32), 0.0)
    return np.log1p(c / np.maximum(c.sum(axis=1, keepdims=True), 1.0) * 10000.0).astype(np.float32)


def load_well_raw(wid: str) -> dict[str, np.ndarray | sp.csr_matrix]:
    xr = sp.load_npz(EMB_DIR / f"gse158013_{wid}_rna_counts.npz").tocsr().astype(np.float32)
    xa = sp.load_npz(EMB_DIR / f"gse158013_{wid}_atac_consensus30k.npz").tocsr().astype(np.float32)
    bd = np.load(EMB_DIR / f"gse158013_{wid}_bundle.npz", allow_pickle=True)
    raw45 = bd["adt_counts_45"].astype(np.float32)
    return {"xr": xr, "xa": xa, "adt_raw": raw45, "y45": log1p_cp10k_dense(raw45)}


def run_one_scvaeit_split(
    xr_tr: sp.csr_matrix,
    xa_tr: sp.csr_matrix,
    adt_tr: np.ndarray,
    xr_te: sp.csr_matrix,
    xa_te: sp.csr_matrix,
    chrom_arr: np.ndarray,
    seed: int = 42,
) -> np.ndarray:
    import tensorflow as tf
    from scVAEIT import VAEIT

    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(seed)

    # Reference-only feature selection (top 1500 variable RNA genes + top 4000 accessible ATAC peaks ordered by chromosome)
    r_lib_tr = np.maximum(np.asarray(xr_tr.sum(axis=1)).ravel(), 1.0)[:, None]
    r_lib_te = np.maximum(np.asarray(xr_te.sum(axis=1)).ravel(), 1.0)[:, None]
    # Fast sparse mean/var for top 1500 RNA genes
    g_mean = np.asarray(xr_tr.mean(axis=0)).ravel()
    g_sq = np.asarray(xr_tr.power(2).mean(axis=0)).ravel()
    g_var = g_sq - g_mean ** 2
    r_sel = np.sort(np.argsort(g_var)[-1500:])

    xr_tr_log = np.log1p(xr_tr[:, r_sel].toarray() / r_lib_tr * 10000.0).astype(np.float32)
    xr_te_log = np.log1p(xr_te[:, r_sel].toarray() / r_lib_te * 10000.0).astype(np.float32)

    p_freq = np.asarray((xa_tr > 0).mean(axis=0)).ravel()
    valid_p = np.flatnonzero((p_freq >= 0.01) & (p_freq <= 0.80))
    if len(valid_p) > 4000:
        top_p = valid_p[np.argsort(p_freq[valid_p] * (1.0 - p_freq[valid_p]))[-4000:]]
    else:
        top_p = valid_p if len(valid_p) >= 500 else np.argsort(p_freq)[-3000:]
    # Group selected peaks by chromosome for scVAEIT block structure
    df_p = pd.DataFrame({"idx": top_p, "chrom": chrom_arr[top_p]}).sort_values(["chrom", "idx"])
    p_sel = df_p["idx"].to_numpy(dtype=np.int64)
    chunks = df_p.groupby("chrom", sort=False).size().to_numpy(dtype=np.int32)

    xa_tr_bin = (xa_tr[:, p_sel].toarray() > 0).astype(np.float32)
    xa_te_bin = (xa_te[:, p_sel].toarray() > 0).astype(np.float32)
    y_tr_log = log1p_cp10k_dense(adt_tr)

    nr, ny, na = xr_tr_log.shape[1], y_tr_log.shape[1], xa_tr_bin.shape[1]
    config = {
        "dim_input_arr": np.array([nr, ny, na]),
        "dim_block": np.r_[nr, ny, chunks],
        "dist_block": ["NB", "NB"] + ["Bernoulli"] * len(chunks),
        "dim_block_enc": np.array([256, 128] + [16] * len(chunks)),
        "dim_block_dec": np.array([256, 128] + [16] * len(chunks)),
        "dim_block_embed": np.array([16, 8] + [1] * len(chunks)) * 2,
        "block_names": np.array(["rna", "adt"] + ["atac"] * len(chunks)),
        "uni_block_names": np.array(["rna", "adt", "atac"]),
        "dimensions": [256],
        "dim_latent": 32,
        "beta_unobs": 2.0 / 3.0,
        "beta_modal": np.array([0.14, 0.85, 0.01]),
        "p_feat": 0.2,
        "p_modal": np.ones(3) / 3.0,
        "gamma": 0.0,
    }
    data = np.concatenate([xr_tr_log, y_tr_log, xa_tr_bin], axis=1)
    masks = np.zeros((1, data.shape[1]), np.float32)
    ds_id = np.zeros(len(xr_tr_log), dtype=np.int32)
    bs = min(256, max(32, len(xr_tr_log) // 4))

    model = VAEIT(config, data, masks, ds_id)
    n_ep = 90 if len(xr_tr_log) <= 500 else 110
    hist = model.train(
        valid=True,
        test_size=0.12,
        random_state=seed,
        learning_rate=0.001,
        batch_size=bs,
        batch_size_inference=128,
        num_repeat=1,
        L=1,
        num_epoch=n_ep,
        early_stopping_patience=10,
        checkpoint_dir=None,
        verbose=False,
    )
    best_ep = max(10, int(np.argmin(hist["val"]["total"])) + 1)
    del model
    tf.keras.backend.clear_session()
    gc.collect()

    tf.keras.utils.set_random_seed(seed)
    model = VAEIT(config, data, masks, ds_id)
    model.train(
        valid=False,
        learning_rate=0.001,
        batch_size=bs,
        num_repeat=1,
        L=1,
        num_epoch=best_ep,
        checkpoint_dir=None,
        verbose=False,
    )

    qry = np.concatenate([xr_te_log, np.zeros((len(xr_te_log), ny), np.float32), xa_te_bin], axis=1)
    model.data = tf.constant(qry)
    model.batches = tf.zeros((len(xr_te_log), 0), dtype=tf.float32)
    model.id_dataset = tf.zeros((len(xr_te_log),), dtype=np.int32)
    if hasattr(model, "dataset_full"):
        del model.dataset_full
    qmask = np.zeros((1, nr + ny + na), np.float32)
    qmask[:, nr : nr + ny] = 1.0
    pred_raw = model.get_denoised_data(
        masks=tf.constant(qmask),
        zero_out=True,
        return_mean=True,
        num_repeat=2,
        L=20,
        batch_size_inference=256,
        training=True,
    )[:, nr : nr + ny].astype(np.float32)

    del model
    tf.keras.backend.clear_session()
    gc.collect()

    # Rescale non-negative denoised rates to CP10k log1p scale
    return log1p_cp10k_dense(np.maximum(pred_raw, 0.0))


def main() -> None:
    out_npz = PRED_DIR / "scvaeit_cross_well_donor.npz"
    if out_npz.is_file():
        print("Cached scvaeit_cross_well_donor.npz already exists.", flush=True)
        return

    ensure_scvaeit_val_patch()
    chrom_arr = np.load(PRED_DIR / "atac_30k_chroms.npy", allow_pickle=True).astype(str)

    wells = {w: load_well_raw(w) for w in ["X066_W3", "X066_W4", "X061_W1"]}
    payload: dict[str, np.ndarray] = {}

    tasks = [
        ("X066_W3_to_X066_W4_250", "X066_W3", "X066_W4", 250, 700 + 250),
        ("X066_W3_to_X066_W4_full", "X066_W3", "X066_W4", "full", 42),
        ("X066_W3_to_X061_W1_250", "X066_W3", "X061_W1", 250, 800 + 250),
        ("X066_W3_to_X061_W1_full", "X066_W3", "X061_W1", "full", 42),
        ("X061_W1_to_X066_W3_250", "X061_W1", "X066_W3", 250, 800 + 250),
        ("X061_W1_to_X066_W3_full", "X061_W1", "X066_W3", "full", 42),
    ]

    for key, src, dst, b, rseed in tasks:
        t0 = time.perf_counter()
        n_src = wells[src]["xr"].shape[0]
        r_idx = np.arange(n_src) if b == "full" else np.sort(np.random.default_rng(rseed).choice(n_src, size=int(b), replace=False))
        pred = run_one_scvaeit_split(
            wells[src]["xr"][r_idx],  # type: ignore[index]
            wells[src]["xa"][r_idx],  # type: ignore[index]
            wells[src]["adt_raw"][r_idx],  # type: ignore[index]
            wells[dst]["xr"],  # type: ignore[arg-type]
            wells[dst]["xa"],  # type: ignore[arg-type]
            chrom_arr,
            seed=42,
        )
        payload[key] = pred
        print(f"  [scVAEIT] {key}: shape={pred.shape} in {time.perf_counter()-t0:.1f}s", flush=True)

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload)
    print("Saved scvaeit_cross_well_donor.npz successfully.", flush=True)


if __name__ == "__main__":
    main()
