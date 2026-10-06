"""Run raw-feature mapping, frozen TOLT embedding, target fitting and query prediction."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from threadpoolctl import threadpool_limits

from tolt.tolt import TOLTConfig, TriOmicsLatentTransformer
from tolt.mapping import map_inputs
from tolt.adaptation import BMMCStuartConfig, fit_predict_methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--data", type=Path, default=root / "example")
    parser.add_argument("--model", type=Path, default=root / "model")
    parser.add_argument("--output", type=Path, default=root / "results/demo")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261007)
    with np.load(args.data / "features.npz") as d:
        features = {name: d[name] for name in d.files}
    ref, query = features["reference_cells"], features["query_cells"]
    assert not set(ref) & set(query)
    y = features["reference_adt_counts"].astype(np.float32)
    assert y.shape[0] == len(ref) and "query_adt_counts" not in features
    y = np.log1p(y / y.sum(1, keepdims=True) * 10000).astype(np.float32)
    rna, atac = [sp.load_npz(args.data / f"{name}_counts.npz") for name in ("rna", "atac")]
    tokens, mapping = map_inputs(rna, atac, features["gene_ids"], features["gene_names"], features["peak_names"], args.model)
    model = TriOmicsLatentTransformer(TOLTConfig(**json.loads((args.model / "config.json").read_text())))
    model.load_state_dict(torch.load(args.model / "tolt_base_110.pt", map_location="cpu", weights_only=True), strict=True)
    model.eval().requires_grad_(False)
    embeds = []
    with threadpool_limits(limits=args.threads), torch.inference_mode():
        for start in range(0, rna.shape[0], 16):
            stop = min(start + 16, rna.shape[0])
            def view(m):
                return {"feature_ids": torch.as_tensor(tokens[m + "_ids"][start:stop], dtype=torch.long),
                        "values": torch.as_tensor(tokens[m + "_values"][start:stop], dtype=torch.float32),
                        "token_mask": torch.as_tensor(tokens[m + "_valid"][start:stop], dtype=torch.bool)}
            observed = torch.tensor([[True, True, False]] * (stop - start), dtype=torch.bool)
            output = model(rna=view("rna"), atac=view("atac"), adt=None, modality_present=observed)
            embeds.append(output["shared_embedding"].numpy())
        z = np.concatenate(embeds)
        fits = fit_predict_methods(x_train=rna[ref], x_test=rna[query], y_train=y, seed=20261007,
            methods=["svd_ridge", "tolt_ridge", "tolt_residual", "per_protein_oof_simplex"],
            config=BMMCStuartConfig(n_jobs=args.threads), train_embeddings=z[ref], test_embeddings=z[query])
    pred = fits["per_protein_oof_simplex"].prediction
    assert pred.shape == (len(query), len(features["protein_names"])) and np.isfinite(pred).all()
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred, index=features["cell_ids"][query], columns=features["protein_names"]).to_csv(args.output / "predicted_proteins.csv", index_label="cell_id")
    np.savez_compressed(args.output / "embeddings_and_tokens.npz", embeddings=z, **tokens)
    receipt = {"status": "complete", "reference_cells": len(ref), "query_cells": len(query), "proteins": pred.shape[1],
               "target_scale": "log1p all-panel CP10k", "seconds": time.perf_counter() - started,
               "query_ADT_available_to_demo": False, "backbone_frozen": True, "mapping": mapping,
               "seed": 20261007, "threads": args.threads, "device": "CPU"}
    (args.output / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    (args.output / "adapter_recipe.json").write_text(json.dumps(fits["per_protein_oof_simplex"].recipe, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
