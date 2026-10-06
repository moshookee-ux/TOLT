# TriOmics-Latent Transformer (TOLT)

TOLT is the triomic neural model developed by Jia Xiao for prediction from incomplete single-cell RNA, chromatin-accessibility and surface-protein measurements. This compact distribution supplies the original TOLT-Base-110 model, frozen feature transforms, and the target-assay adaptation used in the manuscript.

Version: **3.110.1**. Backbone: **30,989,097 parameters**, 391 parameter tensors, step 15,000 of the reported continuation. The model contains exactly the trained model tensors, without optimizer state. The 137-token fusion interface and all model parameters are unchanged.

## Run the example

Use Python 3.11 or 3.12. Download `TOLT-v3.110.1-model.zip` from the [versioned release](https://github.com/moshookee-ux/TOLT/releases/tag/v3.110.1) and extract it in this directory to populate `model/`. Install the dependencies and run:

```shell
python -m pip install -r requirements.txt
python demo.py
```

The second command performs raw-feature mapping → frozen RNA+ATAC embedding → target-domain fitting → protein prediction. It uses 128 reference cells and 16 query cells from public TEA-seq data. The example includes reference ADT only; query ADT is absent from the distributed input. The backbone remains frozen. The program writes:

- `results/demo/predicted_proteins.csv`: query rows × 46 antibody columns, in log1p(CP10k) units.
- `results/demo/run_receipt.json`: actual input sizes, feature-mapping coverage, device, timing and fitting boundary.
- `results/demo/adapter_recipe.json`: training-side selections and protein-specific mixture weights.
- `results/demo/embeddings_and_tokens.npz`: mapped inputs and the common 384-dimensional embedding.

This small example verifies the software workflow. It is not the full 7,069-cell C2 benchmark and should not be used to reproduce its scores. Full benchmark source tables and the fixed protocol accompany the manuscript.

## Supply another paired reference and query

```shell
python demo.py --data my_data --model model --output my_predictions --threads 6
```

`my_data` contains:

| File | Contents |
|---|---|
| `rna_counts.npz` | SciPy CSR matrix, cells × source genes; unnormalized nonnegative counts |
| `atac_counts.npz` | SciPy CSR matrix with the same rows, cells × source peaks |
| `features.npz` | `gene_ids`, `gene_names`, `peak_names`, `protein_names`, `cell_ids`, `reference_cells`, `query_cells`, `reference_adt_counts` |

Gene and peak arrays follow their corresponding matrix columns. `reference_cells` and `query_cells` are disjoint zero-based row indices. `reference_adt_counts` contains only the reference rows and follows `protein_names`. Target panels may differ from the backbone's protein vocabulary because the target-domain heads are fitted to the reference panel.

The supplied example mapping requires exact frozen Ensembl-ID/name matches and GRCh38 peak coordinates. It maps every positive half-open interval overlap, excludes boundary touching, and uses the released frozen scaler values. Datasets with other genome builds or incompatible identifiers need an explicit harmonization step. The compact demo currently uses RNA+ATAC input; the model API in `tolt/tolt.py` supports all seven non-empty input combinations.

Target adaptation uses RNA-SVD/ridge, a ridge head on TOLT embeddings, a TOLT residual predictor, and a nonnegative protein-specific OOF mixture. The supervised fitting interface never accepts query protein truth. It uses training-fitted unsupervised transforms during supervised OOF selection, as in the manuscript.

## Model and evidence provenance

`provenance/model_and_example.json` records the original checkpoint hash, distribution hash, exact model size and example origin. `provenance/source_extraction.json` identifies the unchanged mapping functions carried into this compact distribution. The model implementation and adaptation code retain the research implementation. Release verification checks mapped tokens and embeddings against the original experiment.

Example data originate from [TEA-seq GSE158013, GSM5123951](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM5123951). Cite the original TEA-seq study when reusing the example. Third-party datasets and dependencies retain their respective terms; the software licence does not relicense them.

The paper's principal quantitative evidence uses matched PBMC/BMMC studies, paired-sample experiments, an external 189-antibody panel, and the bounded triomic input comparison. This distribution does not imply unrestricted cross-tissue or zero-shot antibody-panel prediction.

## Licence and citation

The original software and released TOLT model tensors are available under the MIT licence. Public source data and third-party dependencies retain their own terms. The model card describes the intended use and the evaluated scope. See `CITATION.cff` for software citation metadata. The accompanying scientific manuscript is in preparation; this release does not claim journal acceptance.
