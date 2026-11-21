# TriOmics-Latent Transformer (TOLT)

TOLT estimates surface-protein abundance in single cells from RNA or paired RNA and chromatin-accessibility measurements. A reference sample with measured proteins calibrates a frozen 30,989,097-parameter encoder to the target antibody panel.

Version **4.5.0** includes the calibration API, donor- and site-disjoint benchmarks, trimodal TEA-seq transfer, and comparisons with scGPT and Geneformer. The frozen TOLT-Base-110 encoder uses the weights distributed with v3.110.1.

## Run the calibration example

From the repository root, use Python 3.11 or 3.12 and run the following commands.

```sh
python -m pip install -e ".[dev]"
python examples/quickstart_fewshot_calibration.py
python -m pytest tests/test_v4_smoke.py
```

The example generates a synthetic reference and query, fits a protein-specific calibrator, and reports prediction correlations and a two-component gating threshold. It runs on CPU using the generated data. The public Python interface is defined in [src/triomics/inference.py](src/triomics/inference.py).

## Run the pretrained encoder

Extract the [pretrained model archive](https://github.com/moshookee-ux/TOLT/releases/download/v3.110.1/TOLT-v3.110.1-model.zip) in the repository root so that the weights are at `model/tolt_base_110.pt`. Then run the bundled example.

```sh
python demo.py
```

The example maps RNA and ATAC counts from public TEA-seq data, fits the target panel on 128 reference cells, and writes predictions for 16 query cells to `results/demo/predicted_proteins.csv`. The model card describes the encoder and its training. The original release documents the input format and provides archive checksums. [Model card](MODEL_CARD.md) · [Input format](https://github.com/moshookee-ux/TOLT/blob/v3.110.1/README.md#supply-another-paired-reference-and-query) · [Weights and checksums](https://github.com/moshookee-ux/TOLT/releases/tag/v3.110.1)

## Evaluate the study cohorts

The scripts evaluate bone marrow (GSE194122 and GSE128639), peripheral blood (GSE166489), and trimodal TEA-seq (GSE158013). The repository includes the numerical benchmark tables, run summaries, model configurations, and feature vocabularies. [Evaluation scripts](scripts/) · [Benchmark tables](results/tables/) · [Run summaries](results/receipts/)

For file-backed analyses, set `TOLT_ASSETS_ROOT` to your data directory. It defaults to `data/assets/` in this checkout. The scripts read and write the following directories beneath that root.

| Directory | Contents |
| --- | --- |
| `01_cohorts_raw/` | Cohort matrices and sample annotations |
| `02_checkpoints/` | TOLT checkpoints for ablations and external model weights |
| `03_embeddings/` | Embeddings used by the benchmark scripts |
| `04_predictions/` | Predictions and reference splits for downstream analyses and figures |

The download script retrieves the listed GEO files for GSE166489 and GSE158013. Full benchmark runs also require the two preprocessed bone-marrow matrices, the remaining TEA-seq inputs, and the checkpoints and reference splits used by each comparison. The deep baselines use separate dependencies, including scvi-tools, sciPENN, and scVAEIT. The full pipeline was developed on Windows with CUDA. [Download manifest](scripts/01_download_public_cohorts.py) · [Asset paths](src/triomics/v4/paths.py) · [Cross-donor benchmarks](scripts/04_run_cross_donor_benchmarks.py)

## Citation and license

The software and released TOLT model tensors use the MIT license. Source datasets and third-party models retain their respective terms. [Software citation](CITATION.cff) · [License](LICENSE)
