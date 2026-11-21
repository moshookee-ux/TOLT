# Model Card: TOLT-Base-110

- **Model:** TriOmics-Latent Transformer (`TOLT-Base-110`)
- **Author:** Xiao Jia (The Chinese University of Hong Kong, Shenzhen; ORCID: `0009-0006-7711-4067`)
- **Version:** `4.5.0`
- **License:** MIT

## Architecture summary

- **Parameters:** `30,989,097` across `391` frozen model tensors (`tolt_base_110.pt`, SHA-256: `d5392634350225b2bbcdc3f9a560ff0b4abf645d6f7f1ba77d729bd23196c082`).
- **Frozen vocabularies:** `2,048` human immune genes (`rna_features.parquet`), `30,000` GRCh38 cis-regulatory chromatin peaks (`atac_features.parquet`), and `210` surface antibody-derived tags (`ADTs`).
- **Tokenization and local encoders:** Rank-sparse per-cell active token caps (`K_RNA <= 512`, `K_ATAC <= 1,024`) with explicit observation-state embeddings (`observed`, `valid zero`, `unmeasured`, `missing modality`), encoded by Pre-LayerNorm Transformer blocks (`d_loc = 256`; 2 layers for RNA, 2 for ATAC, 1 for ADT).
- **Perceiver bottleneck and shared fusion:** Cross-attention query banks of `48` (RNA), `64` (ATAC), and `24` (ADT) latents plus `1` global summary token (`137` tokens total) processed by a 6-layer, 8-head Shared Fusion Transformer (`d_fus = 384`, feed-forward width `1,536`).
- **Output heads:** Global cell embedding (`z in R^384`), three factorized `128`-dimensional attentive subspace poolers (`z_id` biological identity, `z_sh` cross-modal shared, `z_pr` modality-private), and heteroscedastic Gaussian negative log-likelihood variance heads (`log sigma^2`).

## Pretraining data and disjointness

`TOLT-Base-110` was pretrained once on `325,275` human immune cells from four public multi-omic atlases:
1. NeurIPS 2021 multi-site bone marrow (`GSE194122`, `137,659` cells across 7 pretraining donors; evaluation donors `12710` and `13272` held out prior to vocabulary and scaler fitting).
2. Seurat v4 adult PBMC CITE-seq (`GSE164378`, `161,764` cells).
3. Stimulated human PBMC DOGMA-seq (`GSE166188`, `13,763` cells).
4. 10x Genomics PBMC10k/5k CITE-seq (`12,089` cells).

All downstream evaluation cohorts (`246,925` unseen cells across held-out `GSE194122`, `GSE128639`, `GSE166489`, and `GSE158013`) have zero cell or donor overlap with the pretraining corpus.

## Intended use and scope

`TOLT-Base-110` provides frozen multi-omic cell representations and per-cell uncertainty estimates for downstream surface-antibody panel calibration (`TOLTPanelCalibrator` / three-fold cross-fitted Cholesky ridge, residual, and $\Delta^2$-simplex regression). Downstream panel reconstruction requires paired reference cells (`n_ref >= 50`) for the target antibody panel rather than uncalibrated zero-shot decoding on arbitrary panels. Because the frozen vocabularies were constructed on human bone marrow and peripheral blood mononuclear cells, applications to solid tumors or non-hematopoietic stromal tissues should establish domain coverage on representative reference samples.

## Provenance

Weight checksums and end-to-end verification records are stored in [provenance/model_and_example.json](provenance/model_and_example.json) and [provenance/verification.json](provenance/verification.json).
