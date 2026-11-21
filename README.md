# TriOmics-Latent Transformer (TOLT)

TOLT reconstructs single-cell surface-protein abundance from scRNA-seq or paired scRNA-seq and scATAC-seq measurements. A paired reference sample calibrates a frozen 30,989,097-parameter Perceiver encoder (`TOLT-Base-110`) to any target antibody panel via three-fold cross-fitted Cholesky ridge, residual, and probability-simplex regression, providing per-receptor pathway attribution, calibrated heteroscedastic uncertainty, and two-component Gaussian mixture (2-GMM) surface gates.

## Quickstart calibration example

Using Python 3.11 or 3.12 from the repository root:

```sh
python -m pip install -e ".[dev]"
python examples/quickstart_fewshot_calibration.py
python -m pytest tests/test_v4_smoke.py
```

`examples/quickstart_fewshot_calibration.py` simulates paired reference and unseen query cells on CPU, fits the cross-fitted Cholesky $\Delta^2$-simplex calibrator, and reports query correlations, simplex attribution weights $(w_{\mathrm{loc}}, w_{\mathrm{dir}}, w_{\mathrm{cor}})$, and the equal-posterior 2-GMM decision boundary. The public Python API is defined in [src/triomics/inference.py](src/triomics/inference.py).

## Run the pretrained `TOLT-Base-110` encoder

Download [`TOLT-v4.5.0-model.zip`](https://github.com/moshookee-ux/TOLT/releases/download/v4.5.0/TOLT-v4.5.0-model.zip) from the [v4.5.0 release](https://github.com/moshookee-ux/TOLT/releases/tag/v4.5.0) and extract it in the repository root so that the weights reside at `model/tolt_base_110.pt`. Then run:

```sh
python demo.py
```

`demo.py` maps raw RNA and ATAC counts from public trimodal TEA-seq data onto TOLT's frozen GRCh38 vocabularies, extracts `TOLT-Base-110` embeddings, calibrates a 46-target panel on 128 reference cells, and writes query predictions to `results/demo/predicted_proteins.csv`. To run on custom paired data, pass `--data <dir>` containing `rna_counts.npz`, optional `atac_counts.npz`, and `features.npz` (`gene_ids`, `gene_names`, `peak_names`, `reference_cells`, `query_cells`, `reference_adt_counts`, `protein_names`, `cell_ids`). Architecture details and weight checksums are documented in [MODEL_CARD.md](MODEL_CARD.md).

## Study benchmarks and clinical analyses

The numbered pipeline in [scripts/](scripts/) reproduces the donor-, site-, and well-disjoint evaluations across multi-site bone marrow (`GSE194122`), two-donor bone marrow (`GSE128639`), 18-sample pediatric MIS-C and $\alpha\beta$ TCR blood (`GSE166489`), and five-well trimodal TEA-seq (`GSE158013`), including head-to-head comparisons against `totalVI`, `sciPENN`, `scVAEIT`, `scLinear`, `Seurat-Anchor`, `scGPT`, and `Geneformer`. Numerical output tables and execution summaries are archived in [results/tables/](results/tables/) and [results/receipts/](results/receipts/).

For file-backed cohort analyses, set `TOLT_ASSETS_ROOT` to your local data directory (defaulting to `data/assets/`):

| Directory | Contents |
| --- | --- |
| `01_cohorts_raw/` | Public cohort matrices and sample annotations |
| `02_checkpoints/` | TOLT ablation checkpoints and external foundation model weights |
| `03_embeddings/` | Extracted cell embeddings used by benchmark scripts |
| `04_predictions/` | Out-of-donor predictions and reference splits |

`scripts/01_download_public_cohorts.py` downloads the public GEO archives for `GSE166489` and `GSE158013`; deep generative baselines (`scvi-tools`, `sciPENN`, `scVAEIT`) use optional GPU environments.

## Citation and license

Source code and released `TOLT-Base-110` model weights are distributed under the [MIT License](LICENSE). Citation metadata is provided in [CITATION.cff](CITATION.cff). Public single-cell datasets and external baseline packages remain under their respective original licenses.
