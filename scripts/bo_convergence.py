import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from monty.json import MontyDecoder
from tqdm import tqdm
from regfgw.structure_to_graph import GraphEncoder
from regfgw.fgw_metric import FGWBuilder, FGWBuildParams, FGWScorer, FGWScoreParams
from regfgw.registry_bo import RegistryPriorBO, BOParams

def main():
    p = argparse.ArgumentParser(description="BO and random-search convergence benchmark")
    p.add_argument("--record-json", required=True, help="Interface record JSON")
    p.add_argument("--embedding", required=True, help="Element embedding JSON/CSV")
    p.add_argument("--n-trials", type=int, default=80, help="Number of BO/random trials")
    p.add_argument("--seed", type=int, default=0, help="First random seed used for trials")
    p.add_argument("--out-dir", required=True, help="Output directory")
    args = p.parse_args()

    if args.n_trials < 1:
        raise ValueError("Number of trials must be at least 1.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load interface record
    with open(args.record_json, "r", encoding="utf-8") as f:
        record = json.load(f, cls=MontyDecoder)

    # Build FGW scorer
    encoder = GraphEncoder(embedding_path=args.embedding)
    fgw_builder = FGWBuilder(FGWBuildParams(feature_metric="euclidean"))
    scorer = FGWScorer(
        builder=fgw_builder,
        score_params=FGWScoreParams(
            alpha=0.5,
            n_starts=80,
            init_seed=0,
        ),
    )
    bo = RegistryPriorBO(
        encoder=encoder,
        scorer=scorer,
        bo_params=BOParams(),
        structure_check=False,
    )
    # Suggest near-contact gap
    shift_c = float(bo.suggest_shift_c(record))
    bo_curves = []
    random_curves = []
    n_init = 8
    n_iter = 50

    for trial_index in tqdm(range(args.n_trials), desc="BO & random trials"):
        seed = args.seed + trial_index
        params = BOParams(
            n_init=n_init,
            n_iter=n_iter,
            acq_candidates=4096,
            seed=seed,
            xi=1e-4,
            penalty=1e6,
        )
        bo = RegistryPriorBO(
            encoder=encoder,
            scorer=scorer,
            bo_params=params,
            structure_check=False,
        )
        _, records = bo.bayes_optimize_registry(record, shift_c=shift_c)
        bo_scores = np.asarray(
            [record.score for record in records],
            dtype=float,
        )
        bo_curves.append(np.minimum.accumulate(bo_scores))
        # BO and random evaluations have the same initial design.
        random_scores = list(bo_scores[:n_init])
        rng = np.random.default_rng(seed + 10000)
        random_search = RegistryPriorBO(
            encoder=encoder,
            scorer=scorer,
            bo_params=params,
            structure_check=False,
        )
        for _ in tqdm(range(n_iter), desc="Random refinement", leave=False):
            shift_a = float(rng.uniform(0.0, 1.0))
            shift_b = float(rng.uniform(0.0, 1.0))
            score, _, _ = random_search.score_registry(
                record,
                shift_a=shift_a,
                shift_b=shift_b,
                shift_c=shift_c,
            )
            random_scores.append(score)
        random_scores = np.asarray(random_scores, dtype=float)
        random_curves.append(np.minimum.accumulate(random_scores))

    bo_curves = np.asarray(bo_curves, dtype=float)
    random_curves = np.asarray(random_curves, dtype=float)
    evaluations = np.arange(n_iter+1)
    bo_values = bo_curves[:, n_init-1:]
    random_values = random_curves[:, n_init-1:]
    bo_summary = pd.DataFrame(
        {
            "evaluation": evaluations,
            "fgw_distance_median": np.median(bo_values, axis=0),
            "fgw_distance_q25": np.quantile(bo_values, 0.25, axis=0),
            "fgw_distance_q75": np.quantile(bo_values, 0.75, axis=0),
        }
    )
    random_summary = pd.DataFrame(
        {
            "evaluation": evaluations,
            "fgw_distance_median": np.median(random_values, axis=0),
            "fgw_distance_q25": np.quantile(random_values, 0.25, axis=0),
            "fgw_distance_q75": np.quantile(random_values, 0.75, axis=0),
        }
    )
    bo_output = out_dir / "bo_convergence.csv"
    random_output = out_dir / "random_convergence.csv"
    bo_summary.to_csv(bo_output, index=False)
    random_summary.to_csv(random_output, index=False)
    print(f"[Done] Write BO and random convergence curves in {out_dir}.")

if __name__ == "__main__":
    main()
