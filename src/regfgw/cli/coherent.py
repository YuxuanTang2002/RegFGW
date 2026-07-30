import argparse
import configargparse
from tqdm import tqdm
from typing import List, Dict, Any
from pymatgen.core import Structure
from regfgw.interface_construction import InterfaceBuilder, ZSLParams, InterfaceParams
from regfgw.structure_to_graph import GraphEncoder
from regfgw.fgw_metric import FGWBuilder, FGWBuildParams, FGWScorer, FGWScoreParams
from regfgw.registry_bo import RegistryPriorBO, BOParams

def build_parser():
    """Build the command-line and YAML configuration parser."""
    parser = configargparse.ArgumentParser(
        prog="regfgw_coherent",
        description=(
            "Construct coherent interface candidates and optionally perform "
            "FGW-guided Bayesian optimization of their registries."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument(
        "--config",
        is_config_file=True,
        metavar="FILE",
        help="Path to a YAML configuration file",
    )
    # Global mode
    parser.add_argument(
        "--mode",
        choices=["build", "optimize"],
        default="optimize",
        help=(
            "build: build interface candidates only and exit. "
            "optimize: run Bayesian optimization (BO) on built candidates."
        ),
    )
    # Inputs
    parser.add_argument("--substrate", required=True, help="Path to the substrate bulk CIF file")
    parser.add_argument("--film", required=True, help="Path to the film bulk CIF file")
    parser.add_argument("--embedding", default=None, help="Path to the element embedding CSV or JSON file (required in optimize mode)")
    # Interface construction
    parser.add_argument("--max-miller-idx", type=int, default=1, help="Maximum absolute Miller index used for surface enumeration")
    parser.add_argument("--film-layers", type=int, default=3, help="Number of film layers in each interface candidate")
    parser.add_argument("--substrate-layers", type=int, default=3, help="Number of substrate layers in each interface candidate")
    parser.add_argument("--gap", type=float, default=5.0, help="Initial interfacial gap in Å")
    parser.add_argument("--dft-gap-offset", type=float, default=0.0, help="Additional normal gap offset in Å applied before structure output")
    parser.add_argument("--vacuum", type=float, default=20.0, help="Vacuum thickness in Å")
    # ZSL tolerances
    parser.add_argument("--zsl-max-area", type=float, default=150.0, help="Maximum matched interface area in Å²")
    parser.add_argument("--zsl-area-ratio", type=float, default=0.06, help="Maximum relative area mismatch")
    parser.add_argument("--zsl-length", type=float, default=0.03, help="Maximum relative lattice-vector length mismatch")
    parser.add_argument("--zsl-angle", type=float, default=0.02, help="Maximum relative lattice-vector angle mismatch")
    # FGW settings
    parser.add_argument("--fgw-metric", default="euclidean", help="Feature-space metric for FGW distance calculation")
    parser.add_argument("--fgw-alpha", type=float, default=0.5, help="Weight of the structural cost in FGW, the weight of the feature cost is 1-alpha")
    parser.add_argument("--fgw-n-starts", type=int, default=80, help="Number of FGW initializations")
    parser.add_argument("--fgw-seed", type=int, default=0, help="Random seed for FGW initialization")
    # BO settings
    parser.add_argument("--bo-n-init", type=int, default=8, help="Number of initial BO observations")
    parser.add_argument("--bo-n-iter", type=int, default=50, help="Number of BO refinement iterations")
    parser.add_argument("--bo-candidates", type=int, default=4096, help="Number of acquisition-function candidates per BO iteration")
    parser.add_argument("--bo-seed", type=int, default=0, help="Random seed for BO")
    parser.add_argument("--bo-xi", type=float, default=1e-4, help="Exploration parameter for expected improvement")
    parser.add_argument("--bo-penalty", type=float, default=1e6, help="Penalty assigned to invalid registries. Must be substantially larger than valid FGW values.")
    # Outputs and diagnosis
    parser.add_argument("--budget", type=int, default=3, help="Number of best registries to output per interface candidate")
    parser.add_argument("--unique", action="store_true", help="Only output structurally inequivalent registries per interface candidate.")
    parser.add_argument("--out-traj", action="store_true", help="Write BO sampled registries with FGW scores to .traj files.")
    parser.add_argument("--pipeline-check", action="store_true", help="Write intermediate structures for pipeline inspection.")
    return parser

def validate_args(args: argparse.Namespace, parser: configargparse.ArgumentParser):
    """Validate parsed configuration before starting the workflow."""
    if args.mode == "optimize" and not args.embedding:
        parser.error("--embedding is required when --mode optimize is used.")

def report_configuration(args: argparse.Namespace):
    """Report the run mode and enabled outputs."""
    if args.mode == "build":
        print("[Mode] Construct coherent interfaces only. Bayesian optimization will be skipped.")
        return

    print("[Mode] Construct coherent interfaces and run FGW-based Bayesian optimization.")

    if args.pipeline_check:
        print("[Info] Pipeline check is enabled.")

    if args.out_traj:
        print("[Info] BO trajectory output is enabled.")

    if args.dft_gap_offset != 0.0:
        print(f"[Info] DFT gap offset: {args.dft_gap_offset} Å")

def print_interface_candidates(interfaces: List[Dict[str, Any]]):
    """Print coherent interface candidates for BO optimization."""
    for index, interface in enumerate(interfaces, start=1):
        print(
            f"[{index}] "
            f"sub={interface['substrate_miller']} "
            f"film={interface['film_miller']} "
            f"term={interface['termination']} "
            f"cand={interface['cand_id']} "
            f"area={interface['area']:.2f} Å²"
        )

def select_interface_candidates(interfaces: List[Dict[str, Any]]):
    """Prompt the user to select interface candidates for BO optimization."""
    print("[Input] Enter 'all' or candidate numbers, e.g. 1,3,5")

    while True:
        selection = input("[Input] Select candidates: ").strip().lower()
        if selection == "all":
            return interfaces
        try:
            indices = sorted({int(item.strip()) for item in selection.split(",")})
        except ValueError:
            print("[Error] Enter 'all' or comma-separated candidate numbers, e.g. 1,3,5")
            continue
        if not all(1<=index<=len(interfaces) for index in indices):
            print(f"[Error] Enter candidate numbers from 1 to {len(interfaces)}.")
            continue
        return [interfaces[index-1] for index in indices]

def prompt_continuity_check():
    """Prompt whether to keep interface registry continuity filtering."""
    while True:
        selection = input(
            "[Input] This interface may not be compatible with the default continuity criteria. "
            "Keep the continuity check enabled? [y/n/q]: "
        ).strip().lower()
        if selection in {"y", "yes"}:
            return True
        if selection in {"n", "no"}:
            return False
        if selection in {"q", "quit"}:
            raise SystemExit("Interface registry optimization cancelled.")
        print("[Error] Enter 'y' to keep the continuity check enabled, 'n' to disable it, or 'q' to quit.")

def prompt_shift_c():
    """Prompt for a manual normal shift or exit the workflow."""
    while True:
        selection = input(
            "[Input] Enter a manual shift_c in Å "
            "(positive increases the interfacial gap, negative decreases it) "
            "or 'q' to quit: "
        ).strip().lower()
        if selection in {"q", "quit"}:
            raise SystemExit("Interface registry optimization cancelled.")
        try:
            return float(selection)
        except ValueError:
            print("[Error] Enter a numeric shift_c, such as -1.25, or 'q' to quit.")

def run(args: argparse.Namespace):
    report_configuration(args)
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
        max_miller_idx=args.max_miller_idx,
        zsl_params=zsl_params,
        interface_params=interface_params,
    )
    records = interface_builder.sum_interface_records(
        build_bulk_refs=True,
        structure_check = args.mode == "build",
    )

    if not records:
        raise RuntimeError(
            "No valid coherent interface candidates were generated."
        )

    print(f"[Info] Built {len(records)} coherent interface candidates.")
    print(
        "[Note] Miller-index pairs are symmetry-unique. Negative indices "
        "indicate substrate or film flips used to enumerate all terminations."
    )

    if args.mode == "build":
        return

    print_interface_candidates(records)
    selected_candidates = select_interface_candidates(records)
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

    for index, interface in enumerate(selected_candidates, start=1):
        print(f"[Info] Optimizing interface {index}/{len(selected_candidates)}.")
        shift_c = None
        continuity_check = True
        while True:
            try:
                bo.bayes_optimize_registry(
                    interface,
                    budget=args.budget,
                    unique=args.unique,
                    out_traj=args.out_traj,
                    dft_gap_offset=args.dft_gap_offset,
                    shift_c=shift_c,
                    continuity_check=continuity_check,
                )
                break
            except RuntimeError as e:
                print(f"[Error] {str(e)}")
                continuity_check = prompt_continuity_check()
                shift_c = prompt_shift_c()
                print(f"[Info] Retry shift_c={shift_c:.6f} Å with continuity_check={continuity_check}.")
        
    print("[Done] Interface optimization completed.")

def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    run(args)

if __name__ == "__main__":
    main()
