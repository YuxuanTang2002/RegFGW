import argparse
import json
import csv
import numpy as np
from pathlib import Path
from numpy.linalg import LinAlgError
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from dataclasses import dataclass
from monty.json import MontyDecoder
from ase import Atoms
from ase.optimize import FIRE, BFGS
from pymatgen.io.ase import AseAtomsAdaptor
from regfgw.fgw_metric import FGWBuilder, FGWBuildParams, FGWScorer, FGWScoreParams
from regfgw.registry_bo import  BOParams, RegistryPriorBO
from regfgw.structure_to_graph import GraphEncoder
from mace.calculators import mace_mp

@dataclass(frozen=True)
class GridSampleConfig:
    grid_a: int = 24 # 24x24 grid for GaP/GaAs and GaN/Al2O3 interfaces. 23x23 grid for KI/NaCl interface.
    grid_b: int = 24

def main():
    p = argparse.ArgumentParser(description="Sample uniform registry grid to check the correlation of fgw distance and mace relaxed energy.")
    p.add_argument("--record-json", required=True, help="Interface record JSON")
    p.add_argument("--embedding", required=True, help="Element embedding JSON/CSV")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--mode", choices=["opt", "scf"], default="opt", help="opt mode for relaxation, scf mode for single-point energy calculation.")
    p.add_argument("--mace-model", type=str, default="medium")
    p.add_argument("--mace-device", type=str, default="cuda")
    p.add_argument("--mace-dtype", type=str, default="float32")
    p.add_argument("--mace-gap-offset", type=float, default=0.0)
    p.add_argument("--optimizer", default="fire", choices=["fire", "bfgs"])
    p.add_argument("--fmax", type=float, default=0.03)
    p.add_argument("--max-steps", type=int, default=300)
    args = p.parse_args()

    if args.mode == "scf" and args.mace_gap_offset != 0.0:
        raise ValueError(
            "--mace-gap-offset is for opt mode. "
            "In scf mode, single-point energies must be calculated at the same gap used in FGW distance computation "
            "(i.e., 0 additional gap offset)."
        )

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
        bo_params=BOParams(
            n_init=20,
            n_iter=60,
            acq_candidates=3000,
            seed=0,
            xi=0.01,
            penalty=1e6,
        ),
        structure_check=False,
    )
    # Suggest near-contact gap
    shift_c = float(bo.suggest_shift_c(record))
    # Build uniform registry grid
    grid_a = np.arange(GridSampleConfig.grid_a, dtype=float) / float(GridSampleConfig.grid_a)
    grid_b = np.arange(GridSampleConfig.grid_b, dtype=float) / float(GridSampleConfig.grid_b)
    # Build MACE calculator
    calc = mace_mp(model=args.mace_model, device=args.mace_device, default_dtype=args.mace_dtype)
    total = len(grid_a) * len(grid_b)
    summary_path = out_dir / "fgw_mace_correlation.csv"
    fgw_ds = []
    energies = []
    opt_class = FIRE if args.optimizer == "fire" else BFGS

    with summary_path.open(mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if args.mode == "opt":
            energy_col = "relaxed_energy"
        else:
            energy_col = "single_point_energy"
        writer.writerow(["shift_a", "shift_b", "shift_c", "fgw_distance", energy_col])
        # Loop over registry grid
        with tqdm(total=total, desc="FGW & MACE calculating over uniform grid") as pbar:
            for a in grid_a:
                for b in grid_b:
                    score, reg = bo.score_registry(
                        record,
                        shift_a=float(a),
                        shift_b=float(b),
                        shift_c=shift_c,
                    )
                    # FGW distance invalid: skip relaxtion.
                    if not np.isfinite(score):
                        writer.writerow([
                            f"{float(a):.3f}",
                            f"{float(b):.3f}",
                            f"{float(shift_c + args.mace_gap_offset):.3f}",
                            "",
                            "",
                        ])
                        pbar.update(1)
                        continue
                    reg_relax = RegistryPriorBO.shift_film(reg, shift_c=args.mace_gap_offset)
                    atoms: Atoms = AseAtomsAdaptor.get_atoms(reg_relax)
                    atoms.pbc = (True, True, False)
                    atoms.wrap()
                    atoms.calc = calc
                    if args.mode == "opt":
                        optimizer = opt_class(atoms, logfile=None)
                        try:
                            with tqdm(total=int(args.max_steps), desc=f"Relax a={a:.3f}, b={b:.3f}", leave=False) as pbar_relax:
                                optimizer.attach(lambda: pbar_relax.update(1), interval=1)
                                converged = optimizer.run(fmax=float(args.fmax), steps=int(args.max_steps))
                                energy = float(atoms.get_potential_energy()) if converged else None
                        except (RuntimeError, ValueError, LinAlgError):
                            energy = None
                    else:
                        try:
                            energy = float(atoms.get_potential_energy())
                        except (RuntimeError, ValueError, LinAlgError):
                            energy = None
                    # If relaxtion failed, discard both FGW distance and MACE energy.
                    if energy is None:
                        writer.writerow([
                            f"{float(a):.3f}",
                            f"{float(b):.3f}",
                            f"{float(shift_c + args.mace_gap_offset):.3f}",
                            "",
                            "",
                        ])
                    else:
                        writer.writerow([
                            f"{float(a):.3f}",
                            f"{float(b):.3f}",
                            f"{float(shift_c + args.mace_gap_offset):.3f}",
                            f"{float(score):.12f}" if np.isfinite(score) else "",
                            f"{energy:.12f}",
                        ])
                        fgw_ds.append(float(score))
                        energies.append(float(energy))
                    pbar.update(1)
        writer.writerow([])
        writer.writerow(["metric", "value"])
        n_valid = len(fgw_ds)
        writer.writerow(["total_points", total])
        writer.writerow(["valid_points", n_valid])
        if n_valid < 3:
            raise RuntimeError(f"Too few valid points to compute correlation coefficients: {n_valid}")
        x = np.asarray(fgw_ds, dtype=float)
        y = np.asarray(energies, dtype=float)
        pearson = float(pearsonr(x, y).statistic)
        spearman = float(spearmanr(x, y).statistic)
        writer.writerow(["pearson", f"{pearson:.12f}"])
        writer.writerow(["spearman", f"{spearman:.12f}"])

    if args.mode == "opt":
        print(f"[Done] Write FGW distances and MACE relaxed energies in {out_dir}")
    else:
        print(f"[Done] Write FGW distances and MACE single-point energies in {out_dir}")

if __name__ == "__main__":
    main()






