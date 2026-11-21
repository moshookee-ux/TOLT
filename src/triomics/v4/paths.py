"""Locate cohort inputs, pretrained weights, embeddings and predictions."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ASSETS_ROOT = REPO_ROOT / "data" / "assets"
OFFLINE_ASSETS_ROOT = REPO_ROOT / "data" / "repro_cache"
configured_root = os.environ.get("TOLT_ASSETS_ROOT")
ASSETS_ROOT = Path(configured_root) if configured_root else DEFAULT_ASSETS_ROOT

COHORTS_DIR = ASSETS_ROOT / "01_cohorts_raw"
CHECKPOINTS_DIR = ASSETS_ROOT / "02_checkpoints" / "tolt"
EXTERNAL_MODELS_DIR = ASSETS_ROOT / "02_checkpoints" / "external_foundation"
EMB_DIR = ASSETS_ROOT / "03_embeddings"
PRED_DIR = ASSETS_ROOT / "04_predictions"
TEASEQ_REFERENCE_DIR = PRED_DIR / "teaseq_reference"

D_BMMC90K = COHORTS_DIR / "GSE194122_bmmc_multisite" / "CITE_BMMC_paired.h5ad"
D_STUART = COHORTS_DIR / "GSE128639_bmmc_stuart" / "CITE-BMMC-Stuart.h5ad"
LOCAL_BASE_CHECKPOINT = REPO_ROOT / "model" / "tolt_base_110.pt"

# Prefer the complete local figure cache for routine manuscript work.
figure_root = (
    OFFLINE_ASSETS_ROOT
    if not configured_root and (OFFLINE_ASSETS_ROOT / "manifest.json").is_file()
    else ASSETS_ROOT
)
FIGURE_EMB_DIR = figure_root / "03_embeddings"
FIGURE_PRED_DIR = figure_root / "04_predictions"
