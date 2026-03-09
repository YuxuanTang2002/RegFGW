import argparse
from pymatgen.core import Structure
from regfgw.interface_construction import InterfaceBuilder, ZSLParams, InterfaceParams
from regfgw.structure_to_graph import GraphEncoder
from regfgw.fgw_metric import FGWBuilder, FGWBuildParams, FGWScorer, FGWScoreParams
from regfgw.registry_bo import RegistryPriorBO, BOParams

def main():
    p = argparse.ArgumentParser(description="Interface construction and FGW-based registry optimization pipeline")
    # Global mode
    p.add_argument(
        "--mode",
        choices=["build","optimize"],
        default="optimize",
        help=(
            "build: build interface candidates only and exit. "
            "optimize: run Bayesian optimization on built candidates."
        ),
    )
    # Inputs
    p.add_argument("--substrate", required=True, help="Substrate unit cell CIF")
    p.add_argument("--film", required=True, help="Film unit cell CIF")
    p.add_argument("--embedding", default=None, help="Element embedding file (required in optimize mode)")
    # Interface construction
    p.add_argument("--max-miller", type=int, default=1)
    p.add_argument("--film-layers", type=int, default=3)
    p.add_argument("--substrate-layers", type=int, default=3)
    p.add_argument("--gap", type=float, default=5.0)
    p.add_argument("--vacuum", type=float, default=20.0)
    # ZSL tolerances
    p.add_argument("--zsl-max-area", type=float, default=150.0)
    p.add_argument("--zsl-area-ratio", type=float, default=0.06)
    p.add_argument("--zsl-length", type=float, default=0.03)
    p.add_argument("--zsl-angle", type=float, default=0.02)
    # FGW settings
    p.add_argument("--fgw-metric", type=str, default="euclidean")
    p.add_argument("--fgw-alpha", type=float, default=0.5)
    p.add_argument("--fgw-n-starts", type=int, default=80)
    p.add_argument("--fgw-seed", type=int, default=0)
    # BO settings
    p.add_argument("--bo-n-init", type=int, default=20)
    p.add_argument("--bo-n-iter", type=int, default=60)
    p.add_argument("--bo-candidates", type=int, default=3000)
    p.add_argument("--bo-seed", type=int, default=0)
    p.add_argument("--bo-xi", type=float, default=0.01)
    p.add_argument("--bo-penalty", type=float, default=1e6)
    # Output switches
    p.add_argument("--out-traj", action="store_true",
                   help="Write BO sampled registries with FGW scores to .traj files.")
    p.add_argument("--pipeline-check", action="store_true",
                   help="Check intermediate structures generated in the pipeline.")
    args = p.parse_args()

    if args.mode == "build":
        print("[Mode] Build mode: interface construction only. Bayesian optimization will be skipped.")
    else:
        print("[Mode] Optimize mode: Bayesian optimization will be run.")
        if not args.embedding:
            p.error("--embedding is required when --mode optimize is used.")
        if args.pipeline_check:
            print("[Note] Pipeline check enabled: intermediate structures will be dumped.")
        if args.out_traj:
            print("[Note] BO trajectory output enabled: .traj files will be written.")

    # -------------------------------------------------------------------------
    # Build interface candidates
    # -------------------------------------------------------------------------

    substrate = Structure.from_file(args.substrate)
    film = Structure.from_file(args.film)
    zsl_params = ZSLParams(
        max_area=args.zsl_max_area,
        max_area_ratio_tol=args.zsl_area_ratio,
        max_length_tol=args.zsl_length,
        max_angle_tol=args.zsl_angle,
    )
    interface_params = InterfaceParams(
        film_layers=args.film_layers,
        substrate_layers=args.substrate_layers,
        gap=args.gap,
        vacuum=args.vacuum,
    )
    interface_builder = InterfaceBuilder(
        substrate=substrate,
        film=film,
        max_miller_idx=args.max_miller,
        zsl_params=zsl_params,
        interface_params=interface_params,
    )
    records = interface_builder.sum_interface_records(
        build_bulk_refs=True,
        structure_check=(args.mode=="build"),
    )
    print(f"[Note] {len(records)} interface candidates are built.")

    if not records:
        raise RuntimeError("No valid interface candidates generated.")

    if args.mode == "build":
        return

    # -------------------------------------------------------------------------
    # Bayesian optimization of interface registries
    # -------------------------------------------------------------------------

    encoder = GraphEncoder(embedding_path=args.embedding)
    fgw_builder = FGWBuilder(FGWBuildParams(feature_metric=args.fgw_metric))
    scorer = FGWScorer(
        builder=fgw_builder,
        score_params=FGWScoreParams(
            alpha=args.fgw_alpha,
            n_starts=args.fgw_n_starts,
            init_seed=args.fgw_seed,
        ),
    )
    bo = RegistryPriorBO(
        encoder=encoder,
        scorer=scorer,
        bo_params=BOParams(
            n_init=args.bo_n_init,
            n_iter=args.bo_n_iter,
            acq_candidates=args.bo_candidates,
            seed=args.bo_seed,
            xi=args.bo_xi,
            penalty=args.bo_penalty,
        ),
        structure_check=args.pipeline_check,
    )

    # -------------------------------------------------------------------------
    # BO for each interface candidate
    # -------------------------------------------------------------------------

    for interface in records:
        try:
            best_record, _ = bo.bayes_optimize_registry(
                interface,
                out_best=True,
                out_traj=args.out_traj,
            )
            print(
                f"[Done] miller={interface['substrate_miller']}/"
                f"{interface['film_miller']} "
                f"term={interface['termination']} "
                f"cand={interface['cand_id']} "
            )

        except Exception as e:
            print(
                f"[Error] miller={interface['substrate_miller']}/"
                f"{interface['film_miller']} "
                f"term={interface['termination']} "
                f"cand={interface['cand_id']} "
                f"failed: {type(e).__name__}: {e}"
            )
            continue

if __name__ == "__main__":
    main()
