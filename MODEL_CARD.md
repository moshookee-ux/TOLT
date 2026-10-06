# TOLT-Base-110

**Developer:** Jia Xiao. **Model:** TriOmics-Latent Transformer, originally developed in this study.

- RNA / ATAC / ADT latent arrays: 48 / 64 / 24 tokens plus a cell token.
- Six shared fusion layers, width 384; 30,989,097 parameters.
- Observation-state masks distinguish absent modalities, unmeasured features, valid zeros and padding.
- Supports seven non-empty combinations of RNA, ATAC and ADT in the model API.
- This release reuses the step-15,000 checkpoint; it performs no new backbone training.

The target-domain recipe learns assay-specific heads from paired reference data. It is not zero-shot inference for arbitrary antibody panels. The example and new C2 analysis use human immune-cell measurements; evidence for unrelated tissues should be established for the intended application.

**Outputs:** a fused 384-dimensional embedding and model-level modality outputs; the example adds an assay-specific 46-protein predictor in log1p(CP10k) units.

**Training lineage:** the architecture was designed and initially trained within this research project. The reported checkpoint inherits all 391 model tensors from the study's previous training state and receives 15,000 continuation steps. The recorded 3.43-hour continuation cost does not represent the cumulative initial training and development cost.

**License:** MIT for the original code and released model tensors, as approved by the author. Public source data and third-party dependencies remain under their original terms.

**Provenance:** see `provenance/model_and_example.json` and `provenance/verification.json`. The original checkpoint hash is preserved even though the distribution file contains only model tensors and therefore has its own hash.
